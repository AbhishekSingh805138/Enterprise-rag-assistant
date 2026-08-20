"""Phase 22 — the async upload path and document status endpoints.

Verifies the top lane of the ingestion architecture at the HTTP boundary
(validate -> idempotency check -> store -> publish -> 202) and closes the
loop with an end-to-end test that runs a real document through storage,
the queue, the worker and the parse/chunk step.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.events.sqlite_bus import SQLiteEventBus
from src.ingestion.registry import (
    STATUS_DEAD_LETTER,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSED,
    DocumentRegistry,
)
from src.storage.object_store import LocalObjectStore

TOPIC = "document.uploaded"
DLQ = "document.uploaded.dlq"


@pytest.fixture
def infra(tmp_path):
    """Real registry, queue and object store wired into the API."""
    registry = DocumentRegistry(db_path=tmp_path / "documents.db")
    store = LocalObjectStore(root=tmp_path / "objects")
    with patch("src.events.sqlite_bus.settings") as bus_settings:
        bus_settings.ingest_visibility_timeout_s = 300
        bus = SQLiteEventBus(db_path=tmp_path / "events.db")
        yield registry, bus, store
    bus.close()
    registry.close()


@pytest.fixture
def client(infra):
    """TestClient with ASYNC_INGESTION on and the fixtures above injected."""
    registry, bus, store = infra
    with (
        patch("api.app.settings") as mock_settings,
        patch("src.ingestion.registry.get_registry", return_value=registry),
        patch("src.events.bus.get_event_bus", return_value=bus),
        patch("src.storage.object_store.get_object_store", return_value=store),
        patch("src.storage.object_store.settings") as storage_settings,
    ):
        mock_settings.validate = MagicMock()
        mock_settings.chroma_collection = "test_collection"
        mock_settings.log_level = "WARNING"
        mock_settings.debug_mode = False
        mock_settings.max_upload_size_mb = 10
        mock_settings.cors_origins = "http://localhost:8501"
        mock_settings.cors_allow_methods = "GET,POST,OPTIONS"
        mock_settings.cors_allow_headers = "Content-Type,Authorization,X-Request-ID"
        mock_settings.ingest_root = "./data"
        mock_settings.async_ingestion = True
        mock_settings.kafka_topic_ingestion = TOPIC
        mock_settings.kafka_topic_dlq = DLQ
        storage_settings.s3_prefix = "documents"

        from api.app import app

        with TestClient(app) as tc:
            yield tc


def upload(client, content=b"# Handbook\nPTO is 20 days.\n", name="handbook.md",
           mime="text/markdown", department="hr"):
    return client.post(
        "/upload",
        files={"file": (name, content, mime)},
        data={"department": department},
    )


class TestAcceptedResponse:
    def test_new_upload_returns_202(self, client):
        resp = upload(client)
        assert resp.status_code == 202

    def test_response_carries_a_document_id_and_status_url(self, client):
        body = upload(client).json()
        assert body["document_id"].startswith("doc_")
        assert body["status"] == STATUS_PENDING
        assert body["duplicate"] is False
        assert body["status_url"] == f"/documents/{body['document_id']}"

    def test_response_reports_size_and_checksum(self, client):
        content = b"exactly-this-many-bytes"
        body = upload(client, content=content).json()
        assert body["size_bytes"] == len(content)
        assert len(body["checksum"]) == 64  # sha256 hex

    def test_upload_does_not_parse_or_embed_on_the_request_path(self, client):
        """The whole point of 202: no embedding cost inside the request."""
        with (
            patch("src.vectorstore.chroma_store.add_chunks") as add,
            patch("src.ingestion.loader.load_path") as load,
        ):
            assert upload(client).status_code == 202
        add.assert_not_called()
        load.assert_not_called()


class TestDurableStorageAndQueue:
    def test_bytes_are_persisted_before_acknowledging(self, client, infra):
        registry, _bus, store = infra
        content = b"# Durable\nThis must survive the response.\n"
        doc_id = upload(client, content=content).json()["document_id"]

        record = registry.get(doc_id)
        assert record.storage_key
        assert store.get(record.storage_key) == content

    def test_event_is_published_for_the_worker(self, client, infra):
        _registry, bus, _store = infra
        doc_id = upload(client).json()["document_id"]

        queued = bus.peek(TOPIC)
        assert len(queued) == 1
        assert queued[0].document_id == doc_id
        assert queued[0].attempt == 1

    def test_event_carries_the_request_id_for_tracing(self, client, infra):
        _registry, bus, _store = infra
        resp = client.post(
            "/upload",
            files={"file": ("h.md", b"# x", "text/markdown")},
            headers={"X-Request-ID": "trace-abc-123"},
        )
        assert resp.status_code == 202
        assert resp.headers["X-Request-ID"] == "trace-abc-123"
        assert bus.peek(TOPIC)[0].request_id == "trace-abc-123"

    def test_storage_key_includes_department_and_document_id(self, client, infra):
        registry, _bus, _store = infra
        doc_id = upload(client, department="legal").json()["document_id"]
        key = registry.get(doc_id).storage_key
        assert key.startswith("documents/legal/")
        assert doc_id in key


class TestIdempotency:
    def test_identical_reupload_returns_the_same_id_without_requeueing(self, client, infra):
        _registry, bus, _store = infra
        first = upload(client).json()
        second_resp = upload(client)
        second = second_resp.json()

        assert second_resp.status_code == 200  # accepted earlier, not re-accepted
        assert second["document_id"] == first["document_id"]
        assert second["duplicate"] is True
        assert len(bus.peek(TOPIC)) == 1  # exactly one unit of work

    def test_same_file_in_another_department_is_a_new_document(self, client, infra):
        _registry, bus, _store = infra
        first = upload(client, department="hr").json()
        second = upload(client, department="legal").json()
        assert first["document_id"] != second["document_id"]
        assert len(bus.peek(TOPIC)) == 2

    def test_different_content_is_a_new_document(self, client):
        first = upload(client, content=b"one").json()
        second = upload(client, content=b"two")
        assert second.status_code == 202
        assert second.json()["document_id"] != first["document_id"]

    def test_reuploading_a_dead_lettered_document_requeues_it(self, client, infra):
        registry, bus, _store = infra
        doc_id = upload(client).json()["document_id"]
        registry.claim_for_processing(doc_id)
        registry.mark_dead_letter(doc_id, "gave up")
        bus.purge(TOPIC)

        resp = upload(client)
        assert resp.status_code == 202
        assert resp.json()["duplicate"] is False
        assert registry.get(doc_id).status == STATUS_PENDING
        assert len(bus.peek(TOPIC)) == 1

    def test_already_processed_document_is_not_requeued(self, client, infra):
        registry, bus, _store = infra
        doc_id = upload(client).json()["document_id"]
        registry.mark_processed(doc_id, chunks_indexed=4)
        bus.purge(TOPIC)

        resp = upload(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == STATUS_PROCESSED
        assert bus.peek(TOPIC) == []


class TestFailureHandling:
    def test_publish_failure_marks_the_document_failed(self, client, infra):
        registry, bus, _store = infra
        from src.events.bus import EventBusError

        with patch.object(bus, "publish", side_effect=EventBusError("broker down")):
            resp = upload(client)
        assert resp.status_code == 500

        # Recorded as FAILED, so a re-upload retries rather than being
        # mistaken for an already-accepted duplicate.
        failed = registry.list_documents(status=STATUS_FAILED)
        assert len(failed) == 1

    def test_retry_after_a_publish_failure_succeeds(self, client, infra):
        _registry, bus, _store = infra
        from src.events.bus import EventBusError

        with patch.object(bus, "publish", side_effect=EventBusError("broker down")):
            upload(client)

        resp = upload(client)  # broker is back
        assert resp.status_code == 202
        assert len(bus.peek(TOPIC)) == 1

    def test_storage_failure_marks_the_document_failed(self, client, infra):
        registry, _bus, store = infra
        from src.storage.object_store import ObjectStoreError

        with patch.object(store, "put", side_effect=ObjectStoreError("disk full")):
            assert upload(client).status_code == 500
        assert len(registry.list_documents(status=STATUS_FAILED)) == 1


class TestValidationStillApplies:
    """The async path must not weaken any upload guard."""

    def test_unsupported_extension_is_rejected(self, client):
        resp = upload(client, name="image.jpg", mime="image/jpeg")
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_invalid_department_is_rejected(self, client):
        resp = upload(client, department="marketing")
        assert resp.status_code == 400

    def test_oversized_file_is_rejected(self, client):
        resp = upload(client, content=b"x" * (11 * 1024 * 1024))
        assert resp.status_code == 413

    def test_path_traversal_filename_is_rejected(self, client):
        resp = upload(client, name="../../etc/passwd")
        assert resp.status_code == 400

    def test_nothing_is_queued_for_a_rejected_upload(self, client, infra):
        _registry, bus, _store = infra
        upload(client, name="image.jpg", mime="image/jpeg")
        assert bus.peek(TOPIC) == []

    @pytest.mark.parametrize(
        "name,mime",
        [
            ("doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("data.csv", "text/csv"),
            ("data.csv", "application/vnd.ms-excel"),
            ("report.pdf", "application/pdf"),
        ],
    )
    def test_architecture_file_types_are_accepted(self, client, name, mime):
        assert upload(client, name=name, mime=mime, content=b"payload").status_code == 202


class TestDepartmentBinding:
    """Regression: department must bind from the form field, not only the query.

    Declared as a bare `str` default, FastAPI read `department` from the
    query string only. The UI sends it as a multipart form field, so every
    upload silently landed in "general" — breaking department-filtered
    retrieval and downgrading legal/security documents from "confidential"
    to "internal".
    """

    def test_form_field_is_honoured(self, client, infra):
        registry, _bus, _store = infra
        doc_id = client.post(
            "/upload",
            files={"file": ("terms.md", b"# Terms", "text/markdown")},
            data={"department": "legal"},
        ).json()["document_id"]
        assert registry.get(doc_id).department == "legal"

    def test_query_parameter_still_works(self, client, infra):
        registry, _bus, _store = infra
        doc_id = client.post(
            "/upload",
            files={"file": ("terms.md", b"# Terms", "text/markdown")},
            params={"department": "finance"},
        ).json()["document_id"]
        assert registry.get(doc_id).department == "finance"

    def test_defaults_to_general_when_absent(self, client, infra):
        registry, _bus, _store = infra
        doc_id = client.post(
            "/upload",
            files={"file": ("notes.md", b"# Notes", "text/markdown")},
        ).json()["document_id"]
        assert registry.get(doc_id).department == "general"

    def test_invalid_department_in_form_is_rejected(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("x.md", b"# x", "text/markdown")},
            data={"department": "marketing"},
        )
        assert resp.status_code == 400

    def test_confidential_department_reaches_the_indexed_metadata(self, client, infra):
        """access_level is derived from department, so binding matters here."""
        registry, bus, store = infra
        doc_id = client.post(
            "/upload",
            files={"file": ("contract.md", b"# Vendor Terms\nNet 30.\n", "text/markdown")},
            data={"department": "legal"},
        ).json()["document_id"]

        from src.ingestion.worker import IngestionWorker

        worker = IngestionWorker(
            bus=bus, registry=registry, object_store=store,
            topic=TOPIC, dlq_topic=DLQ, group="dept", max_attempts=3,
        )
        with (
            patch("src.vectorstore.chroma_store.add_chunks", return_value=1) as add_chunks,
            patch("src.events.sqlite_bus.settings") as bs,
            patch("src.ingestion.worker.settings") as ws,
        ):
            bs.ingest_visibility_timeout_s = 300
            ws.ingest_max_attempts = 3
            ws.ingest_retry_backoff_s = 0.0
            ws.worker_batch_size = 1
            ws.worker_concurrency = 1
            ws.worker_poll_interval_s = 0.01
            worker.poll_once()

        metadata = add_chunks.call_args.args[0][0].metadata
        assert metadata["department"] == "legal"
        assert metadata["access_level"] == "confidential"
        assert metadata["document_id"] == doc_id


class TestDocumentStatusEndpoints:
    def test_status_reflects_the_pending_document(self, client):
        doc_id = upload(client).json()["document_id"]
        body = client.get(f"/documents/{doc_id}").json()
        assert body["document_id"] == doc_id
        assert body["status"] == STATUS_PENDING
        assert body["filename"] == "handbook.md"
        assert body["department"] == "hr"

    def test_status_reflects_completion(self, client, infra):
        registry, _bus, _store = infra
        doc_id = upload(client).json()["document_id"]
        registry.mark_processed(doc_id, chunks_indexed=9)

        body = client.get(f"/documents/{doc_id}").json()
        assert body["status"] == STATUS_PROCESSED
        assert body["chunks_indexed"] == 9

    def test_unknown_document_returns_404(self, client):
        assert client.get("/documents/doc_missing").status_code == 404

    def test_error_detail_is_hidden_when_not_in_debug_mode(self, client, infra):
        registry, _bus, _store = infra
        doc_id = upload(client).json()["document_id"]
        registry.mark_dead_letter(doc_id, "OSError: /secret/path/on/server denied")

        body = client.get(f"/documents/{doc_id}").json()
        assert body["status"] == STATUS_DEAD_LETTER
        assert "/secret/path" not in body["error"]

    def test_list_returns_documents_and_stats(self, client):
        upload(client, content=b"one", name="a.md")
        upload(client, content=b"two", name="b.md")

        body = client.get("/documents").json()
        assert body["count"] == 2
        assert body["stats"]["TOTAL"] == 2
        assert body["stats"][STATUS_PENDING] == 2

    def test_list_filters_by_status(self, client, infra):
        registry, _bus, _store = infra
        doc_id = upload(client, content=b"one", name="a.md").json()["document_id"]
        upload(client, content=b"two", name="b.md")
        registry.mark_processed(doc_id, 3)

        body = client.get(f"/documents?status={STATUS_PROCESSED}").json()
        assert body["count"] == 1
        assert body["documents"][0]["document_id"] == doc_id

    def test_list_rejects_an_invalid_status(self, client):
        assert client.get("/documents?status=BOGUS").status_code == 400

    def test_list_filters_by_department(self, client):
        upload(client, content=b"one", department="hr")
        upload(client, content=b"two", department="legal")
        body = client.get("/documents?department=legal").json()
        assert body["count"] == 1


class TestEndToEnd:
    """Upload through HTTP, then let a real worker index it."""

    def test_uploaded_document_is_indexed_by_the_worker(self, client, infra):
        registry, bus, store = infra
        content = b"# Leave Policy\nEmployees receive 20 days of PTO each year.\n"
        doc_id = upload(client, content=content, name="policy.md").json()["document_id"]

        from src.ingestion.worker import OUTCOME_PROCESSED, IngestionWorker

        worker = IngestionWorker(
            bus=bus, registry=registry, object_store=store,
            topic=TOPIC, dlq_topic=DLQ, group="e2e", max_attempts=3,
        )

        # Real download, parse and chunk; only the embedding write is stubbed.
        with (
            patch("src.vectorstore.chroma_store.add_chunks", return_value=2) as add_chunks,
            patch("src.events.sqlite_bus.settings") as bus_settings,
            patch("src.ingestion.worker.settings") as worker_settings,
        ):
            bus_settings.ingest_visibility_timeout_s = 300
            worker_settings.ingest_max_attempts = 3
            worker_settings.ingest_retry_backoff_s = 0.0
            worker_settings.worker_batch_size = 1
            worker_settings.worker_concurrency = 1
            worker_settings.worker_poll_interval_s = 0.01
            assert worker.poll_once() == [OUTCOME_PROCESSED]

        # The document really was parsed: chunks reached the vector store.
        chunks = add_chunks.call_args.args[0]
        assert chunks
        assert "20 days of PTO" in chunks[0].page_content

        # Citations point at a stable source, not the worker's temp dir.
        assert chunks[0].metadata["source"] == "uploads/hr/policy.md"
        assert chunks[0].metadata["document_id"] == doc_id
        assert chunks[0].metadata["department"] == "hr"

        # And the status endpoint now reports completion.
        body = client.get(f"/documents/{doc_id}").json()
        assert body["status"] == STATUS_PROCESSED
        assert body["chunks_indexed"] == 2
        assert bus.depth(TOPIC) == 0
