"""Phase 27 — P1-9, P1-10 and worker concurrency.

Three defects in the ingestion path, all about what happens when the
pipeline is under pressure or mid-upgrade:

* Uploads were accepted regardless of queue depth, so a stalled worker
  meant unbounded growth and a stream of ``202 Accepted`` responses
  promising work that would not happen for hours.
* The event envelope carried no schema version, so a rolling deploy had
  no way to distinguish "a field I do not know" from "a field that should
  be there" — the only safe reaction to any change was to guess.
* A worker indexed one document at a time, so throughput scaled only by
  adding processes.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.events.bus import (
    LEGACY_SCHEMA_VERSION,
    SCHEMA_MAJOR,
    SCHEMA_VERSION,
    Event,
    EventBusError,
    IncompatibleSchemaError,
)
from src.ingestion.backpressure import (
    QueueSaturated,
    check_capacity,
    queue_depth,
    reset_cache,
    retry_after_seconds,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# P1-10 — schema version on the envelope
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_published_events_carry_a_version(self):
        event = Event(event_type="document.uploaded", document_id="doc1")
        assert json.loads(event.to_json())["schema_version"] == SCHEMA_VERSION

    def test_a_roundtrip_preserves_it(self):
        original = Event(event_type="document.uploaded", document_id="doc1")
        assert Event.from_json(original.to_json()).schema_version == SCHEMA_VERSION

    def test_an_event_without_a_version_is_read_as_1_0(self):
        """Every envelope published before this field existed IS 1.0."""
        legacy = json.dumps({
            "event_type": "document.uploaded",
            "document_id": "doc1",
            "payload": {"key": "value"},
            "attempt": 2,
        })
        event = Event.from_json(legacy)
        assert event.schema_version == LEGACY_SCHEMA_VERSION
        assert event.document_id == "doc1"
        assert event.attempt == 2

    def test_a_same_major_version_is_accepted(self):
        """Adding an optional field must not need a coordinated deploy."""
        forward = json.dumps({
            "schema_version": f"{SCHEMA_MAJOR}.9",
            "event_type": "document.uploaded",
            "document_id": "doc1",
            "a_field_from_the_future": "ignored",
        })
        assert Event.from_json(forward).document_id == "doc1"

    def test_a_newer_major_version_is_refused(self):
        """At a major bump a field's meaning may have changed."""
        newer = json.dumps({
            "schema_version": f"{SCHEMA_MAJOR + 1}.0",
            "event_type": "document.uploaded",
            "document_id": "doc1",
        })
        with pytest.raises(IncompatibleSchemaError) as exc:
            Event.from_json(newer)
        assert exc.value.found == f"{SCHEMA_MAJOR + 1}.0"

    def test_the_refusal_says_what_to_do(self):
        newer = json.dumps({
            "schema_version": "99.0", "event_type": "x", "document_id": "d",
        })
        with pytest.raises(IncompatibleSchemaError, match="Upgrade the worker"):
            Event.from_json(newer)

    def test_an_unparseable_version_is_treated_as_current(self):
        """Better to try than to strand an event over a malformed string."""
        odd = json.dumps({
            "schema_version": "not-a-version", "event_type": "x", "document_id": "d",
        })
        assert Event.from_json(odd).document_id == "d"

    def test_incompatible_is_a_kind_of_bus_error(self):
        assert issubclass(IncompatibleSchemaError, EventBusError)

    def test_a_retry_keeps_the_version_it_arrived_with(self):
        """Rewriting it would let a retry silently upgrade the envelope."""
        original = Event(
            event_type="document.uploaded", document_id="doc1", schema_version="1.0"
        )
        assert original.next_attempt().schema_version == "1.0"

    def test_a_retry_still_advances_the_attempt(self):
        original = Event(event_type="x", document_id="d", attempt=1)
        retried = original.next_attempt()
        assert retried.attempt == 2
        assert retried.event_id == original.event_id


class TestSchemaHandlingInTheQueue:
    """A too-new event must be preserved, not discarded like a bad one."""

    def test_sqlite_leaves_an_incompatible_event_for_a_newer_worker(self, tmp_path, caplog):
        from src.events.sqlite_bus import SQLiteEventBus

        bus = SQLiteEventBus(db_path=tmp_path / "events.db")
        try:
            with bus._lock:
                bus._conn.execute(
                    "INSERT INTO events (topic, event_id, body, status, available_at, created_at) "
                    "VALUES (?, ?, ?, 'pending', 0, ?)",
                    (
                        "t",
                        "evt-future",
                        json.dumps({
                            "schema_version": "99.0",
                            "event_type": "document.uploaded",
                            "document_id": "doc-future",
                        }),
                        time.time(),
                    ),
                )
                bus._conn.commit()

            with caplog.at_level(logging.ERROR):
                assert bus.poll("t", "g", timeout_s=0.1) == []

            # Still there, and released rather than held or deleted.
            with bus._lock:
                row = bus._conn.execute(
                    "SELECT status FROM events WHERE topic = 't'"
                ).fetchone()
            assert row is not None, "an incompatible event must not be discarded"
            assert row["status"] == "pending"
            assert "upgraded worker" in caplog.text
        finally:
            bus.close()

    def test_sqlite_still_discards_a_genuinely_malformed_event(self, tmp_path):
        """Undecodable is different from unreadable-by-this-version."""
        from src.events.sqlite_bus import SQLiteEventBus

        bus = SQLiteEventBus(db_path=tmp_path / "events.db")
        try:
            with bus._lock:
                bus._conn.execute(
                    "INSERT INTO events (topic, event_id, body, status, available_at, created_at) "
                    "VALUES ('t', 'evt-bad', '{not json', 'pending', 0, ?)",
                    (time.time(),),
                )
                bus._conn.commit()

            assert bus.poll("t", "g", timeout_s=0.1) == []
            with bus._lock:
                remaining = bus._conn.execute(
                    "SELECT COUNT(*) AS n FROM events WHERE topic = 't'"
                ).fetchone()["n"]
            assert remaining == 0
        finally:
            bus.close()

    def test_a_compatible_event_still_polls_normally(self, tmp_path):
        from src.events.sqlite_bus import SQLiteEventBus

        bus = SQLiteEventBus(db_path=tmp_path / "events.db")
        try:
            bus.publish("t", Event(event_type="document.uploaded", document_id="d1"))
            deliveries = bus.poll("t", "g", timeout_s=0.5)
            assert [d.event.document_id for d in deliveries] == ["d1"]
        finally:
            bus.close()


# ---------------------------------------------------------------------------
# P1-9 — backpressure
# ---------------------------------------------------------------------------

class TestQueueDepth:
    def test_depth_is_read_from_the_bus(self):
        bus = MagicMock()
        bus.depth.return_value = 42
        with patch("src.events.bus.get_event_bus", return_value=bus):
            assert queue_depth(force=True) == 42

    def test_an_unreadable_depth_reports_zero(self):
        """A monitoring failure must not become an ingestion outage."""
        with patch("src.events.bus.get_event_bus", side_effect=RuntimeError("bus down")):
            assert queue_depth(force=True) == 0

    def test_the_depth_is_cached(self):
        """A count per upload would put the queue on the request path."""
        bus = MagicMock()
        bus.depth.return_value = 7
        with patch("src.events.bus.get_event_bus", return_value=bus):
            queue_depth(force=True)
            queue_depth()
            queue_depth()
        assert bus.depth.call_count == 1


class TestCheckCapacity:
    def _settings(self, limit):
        s = MagicMock()
        s.ingest_max_queue_depth = limit
        s.kafka_topic_ingestion = "document.uploaded"
        s.ingest_drain_rate_per_minute = 30
        return s

    def test_a_zero_limit_disables_the_check(self):
        with patch("src.ingestion.backpressure.settings", self._settings(0)):
            with patch("src.ingestion.backpressure.queue_depth", return_value=999_999):
                check_capacity()  # must not raise

    def test_below_the_limit_is_accepted(self):
        with patch("src.ingestion.backpressure.settings", self._settings(1000)):
            with patch("src.ingestion.backpressure.queue_depth", return_value=999):
                check_capacity()

    def test_at_the_limit_is_refused(self, caplog):
        with patch("src.ingestion.backpressure.settings", self._settings(1000)):
            with patch("src.ingestion.backpressure.queue_depth", return_value=1000):
                with caplog.at_level(logging.WARNING):
                    with pytest.raises(QueueSaturated) as exc:
                        check_capacity()
        assert exc.value.depth == 1000
        assert exc.value.limit == 1000
        assert "Rejecting upload" in caplog.text

    def test_the_error_reports_the_backlog(self):
        error = QueueSaturated(depth=5000, limit=1000)
        assert "5000" in str(error) and "1000" in str(error)


class TestRetryAfter:
    def _settings(self, rate):
        s = MagicMock()
        s.ingest_drain_rate_per_minute = rate
        return s

    def test_a_deep_backlog_asks_for_a_longer_wait(self):
        """Telling everyone 60s when the queue is hours deep just re-stampedes."""
        with patch("src.ingestion.backpressure.settings", self._settings(30)):
            shallow = retry_after_seconds(depth=1001, limit=1000)
            deep = retry_after_seconds(depth=10_000, limit=1000)
        assert deep > shallow

    def test_it_is_never_zero(self):
        with patch("src.ingestion.backpressure.settings", self._settings(30)):
            assert retry_after_seconds(depth=1000, limit=1000) >= 5

    def test_it_is_capped_at_an_hour(self):
        with patch("src.ingestion.backpressure.settings", self._settings(1)):
            assert retry_after_seconds(depth=10_000_000, limit=1) <= 3600

    def test_a_zero_drain_rate_does_not_divide_by_zero(self):
        with patch("src.ingestion.backpressure.settings", self._settings(0)):
            assert retry_after_seconds(depth=2000, limit=1000) <= 3600


class TestUploadBackpressure:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        s = MagicMock()
        s.validate = MagicMock()
        s.auth_enabled = False
        s.chroma_collection = "c"
        s.log_level = "WARNING"
        s.debug_mode = False
        s.max_upload_size_mb = 10
        s.cors_origins = "http://localhost:8501"
        s.cors_allow_methods = "GET,POST,OPTIONS"
        s.cors_allow_headers = "Content-Type"
        s.async_ingestion = True
        s.guardrails_enabled = False
        s.rate_limit_storage_uri = ""
        with patch("api.app.settings", s), patch("src.security.auth.settings", s):
            from api.app import app

            with TestClient(app) as tc:
                yield tc

    def _upload(self, client):
        return client.post(
            "/upload",
            files={"file": ("policy.txt", b"hello world", "text/plain")},
            data={"department": "general"},
        )

    def test_a_saturated_queue_refuses_the_upload(self, client):
        with patch("api.app.check_capacity", side_effect=QueueSaturated(1000, 1000)):
            resp = self._upload(client)
        assert resp.status_code == 503

    def test_the_refusal_says_when_to_retry(self, client):
        with patch("api.app.check_capacity", side_effect=QueueSaturated(1000, 1000)):
            resp = self._upload(client)
        assert int(resp.headers["Retry-After"]) > 0

    def test_the_bytes_are_not_stored_when_refused(self, client):
        """Accepting into a full queue costs storage for work that won't run."""
        with (
            patch("api.app.check_capacity", side_effect=QueueSaturated(1000, 1000)),
            patch("api.app._upload_async") as store,
        ):
            self._upload(client)
        store.assert_not_called()

    def test_an_upload_proceeds_when_there_is_room(self, client):
        from api.models import UploadAcceptedResponse

        accepted = UploadAcceptedResponse(
            document_id="doc1", filename="policy.txt", department="general",
            status="PENDING", status_url="/documents/doc1", message="queued",
            checksum="abc123", size_bytes=11,
        )
        with (
            patch("api.app.check_capacity"),
            patch("api.app._upload_async", return_value=(accepted, 202)),
        ):
            resp = self._upload(client)
        assert resp.status_code == 202

    def test_the_sync_path_is_not_gated(self):
        """Inline indexing builds no backlog, so there is nothing to shed."""
        import inspect

        import api.app

        source = inspect.getsource(api.app.upload_endpoint)
        gate = source.index("check_capacity")
        guard = source.rindex("if settings.async_ingestion", 0, gate)
        assert guard < gate


# ---------------------------------------------------------------------------
# Worker concurrency
# ---------------------------------------------------------------------------

class TestWorkerConcurrency:
    def _worker(self, deliveries):
        from src.ingestion.worker import IngestionWorker

        bus = MagicMock()
        bus.poll.return_value = deliveries
        worker = IngestionWorker(bus=bus, registry=MagicMock(), object_store=MagicMock())
        return worker

    def _deliveries(self, n):
        out = []
        for i in range(n):
            d = MagicMock()
            d.event = Event(event_type="document.uploaded", document_id=f"doc{i}")
            out.append(d)
        return out

    def test_sequential_by_default(self):
        worker = self._worker(self._deliveries(4))
        seen_threads = set()

        def handle(delivery):
            seen_threads.add(threading.current_thread().name)
            return "processed"

        with patch("src.ingestion.worker.settings") as s:
            s.worker_batch_size = 4
            s.worker_concurrency = 1
            s.worker_poll_interval_s = 0.01
            with patch.object(type(worker), "handle", side_effect=handle):
                outcomes = worker.poll_once()

        assert outcomes == ["processed"] * 4
        assert len(seen_threads) == 1

    def test_a_batch_is_processed_in_parallel(self):
        worker = self._worker(self._deliveries(4))
        seen_threads = set()
        barrier = threading.Barrier(4, timeout=5)

        def handle(delivery):
            seen_threads.add(threading.current_thread().name)
            barrier.wait()  # deadlocks unless genuinely concurrent
            return "processed"

        with patch("src.ingestion.worker.settings") as s:
            s.worker_batch_size = 4
            s.worker_concurrency = 4
            s.worker_poll_interval_s = 0.01
            with patch.object(type(worker), "handle", side_effect=handle):
                outcomes = worker.poll_once()

        assert len(outcomes) == 4
        assert len(seen_threads) == 4

    def test_one_failing_document_does_not_lose_the_others(self):
        worker = self._worker(self._deliveries(3))
        calls = []

        def handle(delivery):
            calls.append(delivery.event.document_id)
            if delivery.event.document_id == "doc1":
                raise RuntimeError("indexing blew up")
            return "processed"

        with patch("src.ingestion.worker.settings") as s:
            s.worker_batch_size = 3
            s.worker_concurrency = 3
            s.worker_retry_backoff_s = 0.0
            s.ingest_retry_backoff_s = 0.0
            s.worker_poll_interval_s = 0.01
            with patch.object(type(worker), "handle", side_effect=handle):
                outcomes = worker.poll_once()

        assert sorted(calls) == ["doc0", "doc1", "doc2"]
        assert outcomes == ["processed", "processed"]

    def test_a_failed_document_is_released_for_redelivery(self):
        deliveries = self._deliveries(1)
        worker = self._worker(deliveries)

        with patch("src.ingestion.worker.settings") as s:
            s.worker_batch_size = 1
            s.worker_concurrency = 2
            s.ingest_retry_backoff_s = 0.0
            s.worker_poll_interval_s = 0.01
            with patch.object(type(worker), "handle", side_effect=RuntimeError("boom")):
                worker.poll_once()

        deliveries[0].nack.assert_called_once()

    def test_a_single_delivery_does_not_start_a_pool(self):
        """Not worth a thread pool, and it keeps the common path simple."""
        worker = self._worker(self._deliveries(1))
        with patch("src.ingestion.worker.settings") as s:
            s.worker_batch_size = 4
            s.worker_concurrency = 8
            s.worker_poll_interval_s = 0.01
            with patch("src.ingestion.worker.ThreadPoolExecutor") as pool:
                with patch.object(type(worker), "handle", return_value="processed"):
                    assert worker.poll_once() == ["processed"]
        pool.assert_not_called()


class TestIngestionConfigDefaults:
    def test_backpressure_and_concurrency_defaults(self):
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        # Concurrency ships at 1 so throughput behaviour is unchanged
        # until a deployment opts in.
        assert re.search(r'worker_concurrency:.*"WORKER_CONCURRENCY",\s*"1"', source)
        # Backpressure ships enabled: silently unbounded is the defect.
        assert re.search(r'ingest_max_queue_depth:.*"INGEST_MAX_QUEUE_DEPTH",\s*"1000"', source)

    def test_a_negative_queue_limit_is_just_disabled_not_inverted(self):
        s = MagicMock()
        s.ingest_max_queue_depth = -5
        with patch("src.ingestion.backpressure.settings", s):
            check_capacity()  # must not raise
