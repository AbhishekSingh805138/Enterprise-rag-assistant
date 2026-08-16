"""Phase 25 — capability-aware validation and dead-letter replay.

Both fix defects found by auditing the running system:

  #1  .docx was accepted by validation while its parser dependency was
      absent, so the upload returned 202 and dead-lettered two seconds
      later — the caller never learned why.
  #2  Dead-lettered documents had no way back into the pipeline. Their
      bytes were still in object storage, but the only recovery was
      asking the user to upload the file again.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.events.bus import Event, EventBusError
from src.events.sqlite_bus import SQLiteEventBus
from src.ingestion import loader
from src.ingestion.registry import (
    STATUS_DEAD_LETTER,
    STATUS_PENDING,
    STATUS_PROCESSED,
    DocumentRegistry,
    compute_checksum,
)
from src.ingestion.replay import (
    OUTCOME_ALREADY_PROCESSED,
    OUTCOME_MISSING_OBJECT,
    OUTCOME_NOT_REPLAYABLE,
    OUTCOME_PUBLISH_FAILED,
    OUTCOME_REQUEUED,
    OUTCOME_UNKNOWN_DOCUMENT,
    redrive_dlq,
    replay_document,
)
from src.storage.object_store import LocalObjectStore

TOPIC = "document.uploaded"
DLQ = "document.uploaded.dlq"


# ---------------------------------------------------------------------------
# #1 Capability-aware validation
# ---------------------------------------------------------------------------

class TestLoaderCapabilities:
    @pytest.fixture(autouse=True)
    def _clear(self):
        loader.reset_availability_cache()
        yield
        loader.reset_availability_cache()

    def test_types_with_no_optional_dependency_are_always_available(self):
        available = loader.available_suffixes()
        assert {".txt", ".md", ".csv"} <= available

    def test_missing_dependency_is_none_when_installed(self):
        assert loader.missing_dependency(".txt") is None

    def test_missing_dependency_names_the_package(self):
        with patch.object(loader, "_dependency_installed", return_value=False):
            assert loader.missing_dependency(".docx") == "docx2txt"

    def test_unavailable_type_is_dropped_from_available_suffixes(self):
        def installed(module):
            return module != "docx2txt"

        with patch.object(loader, "_dependency_installed", side_effect=installed):
            available = loader.available_suffixes()
        assert ".docx" not in available
        assert ".docx" in loader.SUPPORTED_SUFFIXES  # still known in principle
        assert ".pdf" in available

    def test_probe_result_is_cached(self):
        with patch("importlib.util.find_spec", return_value=object()) as find:
            loader._dependency_installed("docx2txt")
            loader._dependency_installed("docx2txt")
        find.assert_called_once()

    def test_reset_clears_the_cache(self):
        with patch("importlib.util.find_spec", return_value=object()) as find:
            loader._dependency_installed("docx2txt")
            loader.reset_availability_cache()
            loader._dependency_installed("docx2txt")
        assert find.call_count == 2

    def test_broken_probe_counts_as_unavailable(self):
        with patch("importlib.util.find_spec", side_effect=ValueError("bad spec")):
            assert loader._dependency_installed("docx2txt") is False

    def test_case_insensitive_suffix_lookup(self):
        with patch.object(loader, "_dependency_installed", return_value=False):
            assert loader.missing_dependency(".DOCX") == "docx2txt"


class TestUploadRejectsUnavailableTypes:
    """The API must not accept a file it cannot parse."""

    @pytest.fixture
    def client(self, tmp_path):
        with (
            patch("api.app.settings") as s,
            patch("src.storage.object_store.settings") as ss,
        ):
            s.validate = MagicMock()
            s.chroma_collection = "test"
            s.log_level = "WARNING"
            s.debug_mode = False
            s.max_upload_size_mb = 10
            s.cors_origins = "http://localhost:8501"
            s.cors_allow_methods = "GET,POST,OPTIONS"
            s.cors_allow_headers = "Content-Type"
            s.ingest_root = "./data"
            s.async_ingestion = False
            ss.s3_prefix = "documents"
            from fastapi.testclient import TestClient

            from api.app import app

            with TestClient(app) as tc:
                yield tc

    def test_unavailable_type_is_rejected_with_415(self, client):
        with patch("src.ingestion.loader._dependency_installed", return_value=False):
            resp = client.post(
                "/upload",
                files={"file": ("policy.docx", b"x", "application/vnd.openxmlformats-"
                                                     "officedocument.wordprocessingml.document")},
            )
        assert resp.status_code == 415

    def test_rejection_names_the_missing_package(self, client):
        with patch("src.ingestion.loader._dependency_installed", return_value=False):
            resp = client.post(
                "/upload",
                files={"file": ("policy.docx", b"x", "application/vnd.openxmlformats-"
                                                     "officedocument.wordprocessingml.document")},
            )
        assert "docx2txt" in resp.json()["detail"]

    def test_rejection_lists_what_is_accepted(self, client):
        with patch("src.ingestion.loader._dependency_installed", return_value=False):
            detail = client.post(
                "/upload",
                files={"file": ("p.docx", b"x", "application/vnd.openxmlformats-"
                                                "officedocument.wordprocessingml.document")},
            ).json()["detail"]
        assert ".md" in detail and ".txt" in detail

    def test_available_type_is_still_accepted(self, client):
        with (
            patch("src.ingestion.loader.load_path", return_value=[MagicMock()]),
            patch("src.ingestion.chunker.chunk_documents", return_value=[MagicMock()]),
            patch("src.vectorstore.chroma_store.add_chunks", return_value=1),
            patch("src.vectorstore.chroma_store.collection_stats",
                  return_value={"document_count": 1, "collection": "t"}),
        ):
            resp = client.post("/upload", files={"file": ("n.md", b"# hi", "text/markdown")})
        assert resp.status_code == 200

    def test_unknown_extension_is_still_400_not_415(self, client):
        """415 means 'known but unavailable'; 400 means 'not a supported type'."""
        resp = client.post("/upload", files={"file": ("i.jpg", b"x", "image/jpeg")})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# #2 Replay
# ---------------------------------------------------------------------------

@pytest.fixture
def infra(tmp_path):
    registry = DocumentRegistry(db_path=tmp_path / "documents.db")
    store = LocalObjectStore(root=tmp_path / "objects")
    with patch("src.events.sqlite_bus.settings") as bs:
        bs.ingest_visibility_timeout_s = 300
        bus = SQLiteEventBus(db_path=tmp_path / "events.db")
        yield registry, bus, store
    bus.close()
    registry.close()


def make_dead_lettered(registry, store, *, filename="report.pdf", with_bytes=True):
    """Register a document, store its bytes, and drive it to DEAD_LETTER."""
    content = b"payload for " + filename.encode()
    record, _ = registry.register_upload(
        checksum=compute_checksum(content), filename=filename,
        department="hr", size_bytes=len(content),
    )
    key = f"documents/hr/{record.document_id}/{filename}"
    if with_bytes:
        store.put(key, content)
    registry.attach_storage(record.document_id, key, f"file://{key}")
    registry.claim_for_processing(record.document_id)
    registry.mark_dead_letter(record.document_id, "boom")
    return record.document_id


def replay(document_id, infra, **kw):
    registry, bus, store = infra
    return replay_document(
        document_id, registry=registry, bus=bus, store=store, topic=TOPIC, **kw
    )


class TestReplayDocument:
    def test_dead_lettered_document_is_requeued(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        result = replay(doc, infra)
        assert result.outcome == OUTCOME_REQUEUED
        assert result.requeued is True

    def test_status_returns_to_pending(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        replay(doc, infra)
        assert registry.get(doc).status == STATUS_PENDING

    def test_retry_budget_is_restored(self, infra):
        """A replay must get the full policy, not one attempt from exhaustion."""
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        replay(doc, infra)
        assert registry.get(doc).attempts == 0

    def test_a_fresh_event_is_published(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        replay(doc, infra)
        queued = bus.peek(TOPIC)
        assert len(queued) == 1
        assert queued[0].document_id == doc
        assert queued[0].attempt == 1
        assert queued[0].payload["replayed"] is True

    def test_the_user_does_not_have_to_re_upload(self, infra):
        """The stored bytes are what make replay possible at all."""
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        replay(doc, infra)
        assert store.exists(registry.get(doc).storage_key)

    def test_missing_object_is_reported_not_requeued(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store, with_bytes=False)
        result = replay(doc, infra)
        assert result.outcome == OUTCOME_MISSING_OBJECT
        assert bus.peek(TOPIC) == []

    def test_already_processed_is_skipped(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        registry.mark_processed(doc, 5)
        result = replay(doc, infra)
        assert result.outcome == OUTCOME_ALREADY_PROCESSED
        assert bus.peek(TOPIC) == []

    def test_pending_document_is_not_replayed(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        registry.reset_for_retry(doc)  # already live
        assert replay(doc, infra).outcome == OUTCOME_NOT_REPLAYABLE

    def test_force_overrides_state_checks(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        registry.mark_processed(doc, 5)
        assert replay(doc, infra, force=True).outcome == OUTCOME_REQUEUED

    def test_unknown_document(self, infra):
        assert replay("doc_nope", infra).outcome == OUTCOME_UNKNOWN_DOCUMENT

    def test_publish_failure_leaves_the_document_failed(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        with patch.object(bus, "publish", side_effect=EventBusError("broker down")):
            result = replay(doc, infra)
        assert result.outcome == OUTCOME_PUBLISH_FAILED
        assert registry.get(doc).status != STATUS_PENDING  # not left looking healthy

    def test_replaying_twice_queues_one_unit_of_work(self, infra):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        replay(doc, infra)
        second = replay(doc, infra)
        assert second.outcome == OUTCOME_NOT_REPLAYABLE
        assert len(bus.peek(TOPIC)) == 1


class TestRedriveDlq:
    def _dead_letter_with_event(self, registry, bus, store, filename):
        doc = make_dead_lettered(registry, store, filename=filename)
        bus.publish(DLQ, Event(event_type="document.uploaded.dlq", document_id=doc))
        return doc

    def test_drains_the_dlq(self, infra):
        registry, bus, store = infra
        self._dead_letter_with_event(registry, bus, store, "a.pdf")
        self._dead_letter_with_event(registry, bus, store, "b.pdf")

        results = redrive_dlq(registry=registry, bus=bus, store=store,
                              topic=TOPIC, dlq_topic=DLQ)
        assert len(results) == 2
        assert all(r.requeued for r in results)
        assert bus.depth(DLQ) == 0   # the queue is actually drained

    def test_documents_are_requeued_for_indexing(self, infra):
        registry, bus, store = infra
        doc = self._dead_letter_with_event(registry, bus, store, "a.pdf")
        redrive_dlq(registry=registry, bus=bus, store=store, topic=TOPIC, dlq_topic=DLQ)
        assert registry.get(doc).status == STATUS_PENDING
        assert [e.document_id for e in bus.peek(TOPIC)] == [doc]

    def test_empty_dlq_is_a_no_op(self, infra):
        registry, bus, store = infra
        assert redrive_dlq(registry=registry, bus=bus, store=store,
                           topic=TOPIC, dlq_topic=DLQ) == []

    def test_limit_is_respected(self, infra):
        registry, bus, store = infra
        for i in range(5):
            self._dead_letter_with_event(registry, bus, store, f"f{i}.pdf")
        results = redrive_dlq(limit=2, registry=registry, bus=bus,
                              store=store, topic=TOPIC, dlq_topic=DLQ)
        assert len(results) == 2
        assert bus.depth(DLQ) == 3   # the rest stay for the next run

    def test_dry_run_drains_nothing(self, infra):
        registry, bus, store = infra
        self._dead_letter_with_event(registry, bus, store, "a.pdf")
        results = redrive_dlq(registry=registry, bus=bus, store=store,
                              topic=TOPIC, dlq_topic=DLQ, dry_run=True)
        assert results[0].outcome == "would_replay"
        assert bus.depth(DLQ) == 1
        assert bus.peek(TOPIC) == []

    def test_dry_run_reports_each_event_exactly_once(self, infra):
        """Releasing an event inside the loop makes it visible again, so a
        naive implementation re-polls the same message until it hits --limit."""
        registry, bus, store = infra
        self._dead_letter_with_event(registry, bus, store, "a.pdf")
        self._dead_letter_with_event(registry, bus, store, "b.pdf")

        results = redrive_dlq(limit=50, registry=registry, bus=bus, store=store,
                              topic=TOPIC, dlq_topic=DLQ, dry_run=True)
        assert len(results) == 2
        assert len({r.document_id for r in results}) == 2

    def test_dry_run_leaves_events_claimable_afterwards(self, infra):
        registry, bus, store = infra
        self._dead_letter_with_event(registry, bus, store, "a.pdf")
        redrive_dlq(registry=registry, bus=bus, store=store,
                    topic=TOPIC, dlq_topic=DLQ, dry_run=True)
        # A real redrive immediately after must still find the event.
        results = redrive_dlq(registry=registry, bus=bus, store=store,
                              topic=TOPIC, dlq_topic=DLQ)
        assert len(results) == 1
        assert results[0].requeued is True

    def test_unreplayable_events_are_still_acked(self, infra):
        """Otherwise an unfixable document is redriven forever."""
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store, with_bytes=False)
        bus.publish(DLQ, Event(event_type="document.uploaded.dlq", document_id=doc))

        results = redrive_dlq(registry=registry, bus=bus, store=store,
                              topic=TOPIC, dlq_topic=DLQ)
        assert results[0].outcome == OUTCOME_MISSING_OBJECT
        assert bus.depth(DLQ) == 0

    def test_a_bad_document_does_not_block_the_others(self, infra):
        registry, bus, store = infra
        bad = make_dead_lettered(registry, store, filename="bad.pdf", with_bytes=False)
        bus.publish(DLQ, Event(event_type="x", document_id=bad))
        good = self._dead_letter_with_event(registry, bus, store, "good.pdf")

        outcomes = {r.document_id: r.outcome for r in redrive_dlq(
            registry=registry, bus=bus, store=store, topic=TOPIC, dlq_topic=DLQ)}
        assert outcomes[bad] == OUTCOME_MISSING_OBJECT
        assert outcomes[good] == OUTCOME_REQUEUED


class TestReplayCli:
    def test_list_reports_stuck_documents(self, infra, capsys):
        registry, bus, store = infra
        make_dead_lettered(registry, store, filename="stuck.pdf")
        from scripts.replay_dlq import main

        with (
            patch("src.ingestion.registry.get_registry", return_value=registry),
            patch("config.setup_logging", MagicMock()),
        ):
            assert main(["--list"]) == 0
        assert "stuck.pdf" in capsys.readouterr().out

    def test_list_on_empty_dlq(self, infra, capsys):
        registry, _bus, _store = infra
        from scripts.replay_dlq import main

        with (
            patch("src.ingestion.registry.get_registry", return_value=registry),
            patch("config.setup_logging", MagicMock()),
        ):
            main(["--list"])
        assert "No dead-lettered documents" in capsys.readouterr().out

    def test_single_document_replay(self, infra, capsys):
        registry, bus, store = infra
        doc = make_dead_lettered(registry, store)
        from scripts.replay_dlq import main

        with (
            patch("src.ingestion.replay.get_registry", return_value=registry),
            patch("src.ingestion.replay.get_event_bus", return_value=bus),
            patch("src.ingestion.replay.get_object_store", return_value=store),
            patch("config.setup_logging", MagicMock()),
        ):
            assert main(["--document-id", doc]) == 0
        assert registry.get(doc).status == STATUS_PENDING

    def test_nonzero_exit_when_replay_does_not_happen(self, infra):
        registry, bus, store = infra
        from scripts.replay_dlq import main

        with (
            patch("src.ingestion.replay.get_registry", return_value=registry),
            patch("src.ingestion.replay.get_event_bus", return_value=bus),
            patch("src.ingestion.replay.get_object_store", return_value=store),
            patch("config.setup_logging", MagicMock()),
        ):
            assert main(["--document-id", "doc_missing"]) == 1
