"""Phase 22 — ingestion worker: retry budget, dead-lettering, idempotency.

Driven against the real SQLite queue, a real registry and a temp-directory
object store, so the retry and dead-letter paths from the architecture are
exercised end to end. Only the parse/embed step is stubbed — it is the
part that would call OpenAI.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.events.bus import Event
from src.events.sqlite_bus import SQLiteEventBus
from src.ingestion.pipeline import DocumentParseError, IngestionResult
from src.ingestion.registry import (
    STATUS_DEAD_LETTER,
    STATUS_FAILED,
    STATUS_PROCESSED,
    DocumentRegistry,
    compute_checksum,
)
from src.ingestion.worker import (
    OUTCOME_DEAD_LETTERED,
    OUTCOME_PROCESSED,
    OUTCOME_RETRIED,
    OUTCOME_SKIPPED,
    OUTCOME_UNKNOWN_DOCUMENT,
    IngestionWorker,
)
from src.storage.object_store import LocalObjectStore, ObjectNotFoundError

TOPIC = "document.uploaded"
DLQ = "document.uploaded.dlq"
CONTENT = b"# Handbook\nRemote work is allowed three days a week.\n"


@pytest.fixture(autouse=True)
def _worker_settings():
    """Deterministic, fast retry settings for every worker test."""
    with (
        patch("src.ingestion.worker.settings") as ws,
        patch("src.events.sqlite_bus.settings") as bs,
        patch("src.ingestion.registry.settings") as rs,
    ):
        ws.ingest_max_attempts = 3
        ws.ingest_retry_backoff_s = 0.0
        ws.worker_batch_size = 1
        ws.worker_concurrency = 1  # sequential, so outcome order is stable
        ws.worker_poll_interval_s = 0.01
        bs.ingest_visibility_timeout_s = 300
        rs.ingest_visibility_timeout_s = 300
        yield ws


@pytest.fixture
def bus(tmp_path):
    b = SQLiteEventBus(db_path=tmp_path / "events.db")
    yield b
    b.close()


@pytest.fixture
def registry(tmp_path):
    r = DocumentRegistry(db_path=tmp_path / "documents.db")
    yield r
    r.close()


@pytest.fixture
def store(tmp_path):
    return LocalObjectStore(root=tmp_path / "objects")


@pytest.fixture
def worker(bus, registry, store):
    return IngestionWorker(
        bus=bus, registry=registry, object_store=store,
        topic=TOPIC, dlq_topic=DLQ, group="test-workers", max_attempts=3,
    )


def enqueue(bus, registry, store, *, content=CONTENT, filename="handbook.md", department="hr"):
    """Register, store and publish a document the way the API does."""
    checksum = compute_checksum(content)
    record, _ = registry.register_upload(
        checksum=checksum, filename=filename, department=department,
        content_type="text/markdown", size_bytes=len(content),
    )
    key = f"documents/{department}/{record.document_id}/{filename}"
    uri = store.put(key, content)
    registry.attach_storage(record.document_id, key, uri)
    bus.publish(
        TOPIC,
        Event(
            event_type="document.uploaded",
            document_id=record.document_id,
            payload={"storage_key": key, "filename": filename},
        ),
    )
    return record.document_id


def _result(document_id, chunks=4):
    return IngestionResult(
        document_id=document_id, documents_loaded=1,
        chunks_created=chunks, chunks_added=chunks,
    )


class TestHappyPath:
    def test_document_is_indexed_and_marked_processed(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document", return_value=_result(doc_id, 4)
        ) as proc:
            assert worker.poll_once() == [OUTCOME_PROCESSED]

        proc.assert_called_once()
        stored = registry.get(doc_id)
        assert stored.status == STATUS_PROCESSED
        assert stored.chunks_indexed == 4

    def test_processed_event_is_acked(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch("src.ingestion.worker.process_document", return_value=_result(doc_id)):
            worker.poll_once()
        assert bus.depth(TOPIC) == 0

    def test_stats_track_progress(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch("src.ingestion.worker.process_document", return_value=_result(doc_id, 6)):
            worker.poll_once()
        assert worker.stats.processed == 1
        assert worker.stats.chunks_indexed == 6

    def test_worker_receives_the_registered_record(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store, filename="policy.md", department="legal")
        with patch(
            "src.ingestion.worker.process_document", return_value=_result(doc_id)
        ) as proc:
            worker.poll_once()
        record = proc.call_args.args[0]
        assert record.document_id == doc_id
        assert record.filename == "policy.md"
        assert record.department == "legal"


class TestIdempotency:
    def test_redelivered_event_does_not_reindex(self, worker, bus, registry, store):
        """At-least-once delivery must not mean at-least-once embedding."""
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document", return_value=_result(doc_id)
        ) as proc:
            worker.poll_once()
            # Same event arrives again (broker redelivery).
            bus.publish(TOPIC, Event(event_type="document.uploaded", document_id=doc_id))
            assert worker.poll_once() == [OUTCOME_SKIPPED]

        assert proc.call_count == 1
        assert registry.get(doc_id).status == STATUS_PROCESSED

    def test_dead_lettered_document_is_not_reprocessed(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        registry.claim_for_processing(doc_id)
        registry.mark_dead_letter(doc_id, "previously gave up")
        with patch("src.ingestion.worker.process_document") as proc:
            assert worker.poll_once() == [OUTCOME_SKIPPED]
        proc.assert_not_called()

    def test_event_for_unknown_document_is_acked(self, worker, bus):
        bus.publish(TOPIC, Event(event_type="document.uploaded", document_id="doc_ghost"))
        assert worker.poll_once() == [OUTCOME_UNKNOWN_DOCUMENT]
        assert bus.depth(TOPIC) == 0  # not left to cycle forever


class TestTransientFailureRetries:
    def test_first_failure_is_retried(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document",
            side_effect=RuntimeError("rate limited"),
        ):
            assert worker.poll_once() == [OUTCOME_RETRIED]

        assert registry.get(doc_id).status == STATUS_FAILED
        # A fresh event carrying attempt=2 is now queued.
        queued = bus.peek(TOPIC)
        assert len(queued) == 1
        assert queued[0].attempt == 2

    def test_retry_preserves_the_event_id_for_tracing(self, worker, bus, registry, store):
        enqueue(bus, registry, store)
        original = bus.peek(TOPIC)[0].event_id
        with patch(
            "src.ingestion.worker.process_document", side_effect=RuntimeError("blip")
        ):
            worker.poll_once()
        assert bus.peek(TOPIC)[0].event_id == original

    def test_retry_succeeds_on_a_later_attempt(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        attempts = {"n": 0}

        def flaky(record, object_store=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient")
            return _result(doc_id, 5)

        with patch("src.ingestion.worker.process_document", side_effect=flaky):
            assert worker.poll_once() == [OUTCOME_RETRIED]
            assert worker.poll_once() == [OUTCOME_PROCESSED]

        assert registry.get(doc_id).status == STATUS_PROCESSED
        assert registry.get(doc_id).chunks_indexed == 5

    def test_exhausting_three_attempts_dead_letters(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document",
            side_effect=RuntimeError("always fails"),
        ):
            outcomes = [worker.poll_once()[0] for _ in range(3)]

        assert outcomes == [OUTCOME_RETRIED, OUTCOME_RETRIED, OUTCOME_DEAD_LETTERED]
        assert registry.get(doc_id).status == STATUS_DEAD_LETTER
        assert bus.depth(TOPIC) == 0  # no longer cycling

    def test_dead_lettered_event_lands_on_the_dlq_topic(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document", side_effect=RuntimeError("nope")
        ):
            for _ in range(3):
                worker.poll_once()

        dlq = bus.peek(DLQ)
        assert len(dlq) == 1
        assert dlq[0].document_id == doc_id
        assert "nope" in dlq[0].payload["error"]
        assert dlq[0].event_type.endswith(".dlq")

    def test_max_attempts_is_configurable(self, bus, registry, store):
        one_shot = IngestionWorker(
            bus=bus, registry=registry, object_store=store,
            topic=TOPIC, dlq_topic=DLQ, group="g", max_attempts=1,
        )
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document", side_effect=RuntimeError("x")
        ):
            assert one_shot.poll_once() == [OUTCOME_DEAD_LETTERED]
        assert registry.get(doc_id).status == STATUS_DEAD_LETTER


class TestPermanentFailures:
    """Errors that cannot succeed on a retry skip the budget entirely."""

    def test_unparseable_document_is_dead_lettered_immediately(
        self, worker, bus, registry, store
    ):
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document",
            side_effect=DocumentParseError("corrupt PDF"),
        ):
            assert worker.poll_once() == [OUTCOME_DEAD_LETTERED]

        assert registry.get(doc_id).status == STATUS_DEAD_LETTER
        assert bus.depth(TOPIC) == 0

    def test_missing_object_is_dead_lettered_immediately(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document",
            side_effect=ObjectNotFoundError("gone from storage"),
        ):
            assert worker.poll_once() == [OUTCOME_DEAD_LETTERED]
        assert registry.get(doc_id).status == STATUS_DEAD_LETTER

    def test_permanent_failure_records_the_reason(self, worker, bus, registry, store):
        enqueue(bus, registry, store)
        with patch(
            "src.ingestion.worker.process_document",
            side_effect=DocumentParseError("corrupt PDF"),
        ):
            worker.poll_once()
        assert bus.peek(DLQ)[0].payload["reason"] == "unrecoverable"


class TestFailureIsolation:
    def test_a_failing_document_does_not_block_the_queue(self, worker, bus, registry, store):
        bad = enqueue(bus, registry, store, content=b"bad", filename="bad.md")
        good = enqueue(bus, registry, store, content=b"good", filename="good.md")

        def selective(record, object_store=None):
            if record.document_id == bad:
                raise DocumentParseError("corrupt")
            return _result(record.document_id, 3)

        with patch("src.ingestion.worker.process_document", side_effect=selective):
            outcomes = [worker.poll_once()[0], worker.poll_once()[0]]

        assert outcomes == [OUTCOME_DEAD_LETTERED, OUTCOME_PROCESSED]
        assert registry.get(bad).status == STATUS_DEAD_LETTER
        assert registry.get(good).status == STATUS_PROCESSED

    def test_failed_dlq_publish_still_records_dead_letter(self, worker, bus, registry, store):
        """Losing the DLQ message must not lose the document's state."""
        doc_id = enqueue(bus, registry, store)
        from src.events.bus import EventBusError

        real_publish = bus.publish

        def fail_dlq(topic, event, delay_s=0.0):
            if topic == DLQ:
                raise EventBusError("broker down")
            return real_publish(topic, event, delay_s)

        with (
            patch.object(bus, "publish", side_effect=fail_dlq),
            patch(
                "src.ingestion.worker.process_document",
                side_effect=DocumentParseError("corrupt"),
            ),
        ):
            assert worker.poll_once() == [OUTCOME_DEAD_LETTERED]

        assert registry.get(doc_id).status == STATUS_DEAD_LETTER

    def test_failed_retry_publish_leaves_event_for_redelivery(
        self, worker, bus, registry, store
    ):
        """If the retry cannot be queued, the original must not be dropped."""
        enqueue(bus, registry, store)
        from src.events.bus import EventBusError

        with (
            patch.object(bus, "publish", side_effect=EventBusError("broker down")),
            patch(
                "src.ingestion.worker.process_document", side_effect=RuntimeError("blip")
            ),
        ):
            assert worker.poll_once() == [OUTCOME_RETRIED]

        # Still on the queue rather than silently lost.
        assert bus.depth(TOPIC) == 1


class TestRunLoop:
    def test_run_drains_then_stops_on_the_stop_event(self, worker, bus, registry, store):
        doc_id = enqueue(bus, registry, store)
        stop = threading.Event()

        with patch("src.ingestion.worker.process_document", return_value=_result(doc_id)):
            thread = threading.Thread(target=worker.run, kwargs={"stop_event": stop})
            thread.start()
            for _ in range(200):
                if registry.get(doc_id).status == STATUS_PROCESSED:
                    break
                threading.Event().wait(0.02)
            stop.set()
            thread.join(timeout=10)

        assert thread.is_alive() is False
        assert registry.get(doc_id).status == STATUS_PROCESSED

    def test_run_survives_a_bus_error(self, worker, bus):
        from src.events.bus import EventBusError

        stop = threading.Event()
        calls = {"n": 0}

        def flaky_poll(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] >= 3:
                stop.set()
            raise EventBusError("transient broker error")

        with patch.object(bus, "poll", side_effect=flaky_poll):
            worker.run(stop_event=stop)

        assert calls["n"] >= 3  # kept going instead of dying

    def test_unhandled_error_releases_the_delivery(self, worker, bus, registry, store):
        enqueue(bus, registry, store)
        with patch.object(
            worker, "handle", side_effect=RuntimeError("bug in handle")
        ):
            assert worker.poll_once() == []
        # Released rather than held invisible until the timeout lapses.
        assert bus.depth(TOPIC) == 1


class TestWorkerStats:
    def test_stats_serialise_for_health_output(self, worker):
        payload = worker.stats.to_dict()
        assert set(payload) >= {
            "polled", "processed", "skipped", "retried",
            "dead_lettered", "chunks_indexed", "uptime_s",
        }


class TestScriptEntrypoint:
    def test_once_mode_drains_and_exits(self, bus, registry, store, monkeypatch):
        doc_id = enqueue(bus, registry, store)
        fake_worker = IngestionWorker(
            bus=bus, registry=registry, object_store=store,
            topic=TOPIC, dlq_topic=DLQ, group="g", max_attempts=3,
        )
        # Settings is a frozen dataclass, so stub the module-level reference
        # rather than trying to patch an attribute on it.
        with (
            patch("src.ingestion.worker.IngestionWorker", return_value=fake_worker),
            patch("src.ingestion.worker.process_document", return_value=_result(doc_id)),
            patch("scripts.worker.settings", MagicMock()),
            patch("scripts.worker.setup_logging", MagicMock()),
        ):
            from scripts.worker import main

            assert main(["--once"]) == 0

        assert registry.get(doc_id).status == STATUS_PROCESSED
