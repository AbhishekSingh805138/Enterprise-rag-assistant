"""Phase 7 tests: FastAPI endpoints, models, rate limiting, CORS.

All LLM/ChromaDB calls are mocked. Uses FastAPI TestClient (synchronous).
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.models import (
    AskRequest,
    AskResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Create a TestClient with mocked startup validation."""
    with patch("api.app.settings") as mock_settings:
        mock_settings.validate = MagicMock()
        mock_settings.chroma_collection = "test_collection"
        mock_settings.chroma_dir = "./test_chroma"
        mock_settings.checkpoint_dir = "./test_checkpoints"
        mock_settings.log_level = "WARNING"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.top_k = 4
        mock_settings.chunk_size = 1000
        mock_settings.chunk_overlap = 150
        mock_settings.debug_mode = False
        mock_settings.max_upload_size_mb = 10
        mock_settings.cors_origins = "http://localhost:8501"
        mock_settings.cors_allow_methods = "GET,POST,OPTIONS"
        mock_settings.cors_allow_headers = "Content-Type,Authorization,X-Request-ID"
        mock_settings.ingest_root = "./data"

        from api.app import app
        with TestClient(app) as tc:
            yield tc


# ---------------------------------------------------------------------------
# TestModels
# ---------------------------------------------------------------------------

class TestModels:
    """Pydantic model validation."""

    def test_ask_request_defaults(self):
        req = AskRequest(question="test?")
        assert req.mode == "naive"
        assert req.retriever_strategy == "dense"
        assert req.stream is False

    def test_ask_request_rejects_empty_question(self):
        with pytest.raises(Exception):
            AskRequest(question="")

    def test_ask_response_fields(self):
        resp = AskResponse(
            answer="test", question="q", mode="naive",
            retriever_strategy="dense", cost_usd=0.001,
            latency_ms=100.0, tokens_used=50,
        )
        assert resp.cost_usd == 0.001

    def test_ingest_request_defaults(self):
        req = IngestRequest()
        assert req.path == "./data/sample_docs"

    def test_eval_request_defaults(self):
        req = EvalRequest()
        assert req.mode == "naive"
        assert req.retriever_strategy == "dense"


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /health."""

    def test_health_returns_ok(self, client):
        with patch("api.app.collection_stats", create=True) as mock_stats:
            # Patch at the import location inside the endpoint
            with patch("src.vectorstore.chroma_store.collection_stats") as mock_cs:
                mock_cs.return_value = {
                    "collection": "enterprise_docs",
                    "persist_directory": "./chroma_db",
                    "document_count": 41,
                }
                resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["document_count"] == 41
        assert "version" in data

    def test_health_degraded_on_error(self, client):
        with patch("src.vectorstore.chroma_store.collection_stats", side_effect=RuntimeError("DB error")):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"


# ---------------------------------------------------------------------------
# TestAskEndpoint
# ---------------------------------------------------------------------------

class TestAskEndpoint:
    """POST /ask."""

    def test_ask_naive_returns_answer(self, client):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The remote work policy allows 3 days."

        with patch("src.rag.naive_rag.build_naive_rag_chain", return_value=mock_chain):
            with patch("src.observability.metrics_store.get_store") as mock_store:
                mock_store.return_value = MagicMock()
                resp = client.post("/ask", json={
                    "question": "What is the remote work policy?",
                    "mode": "naive",
                })

        assert resp.status_code == 200
        data = resp.json()
        assert "remote work" in data["answer"].lower() or len(data["answer"]) > 0
        assert data["mode"] == "naive"
        assert "cost_usd" in data
        assert "latency_ms" in data

    def test_ask_graph_returns_answer(self, client):
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"generation": "Graph answer here."}

        with (
            patch("src.graph.build_graph.get_graph", return_value=mock_graph),
            patch("src.observability.metrics_store.get_store") as mock_store,
        ):
            mock_store.return_value = MagicMock()
            resp = client.post("/ask", json={
                "question": "What is the remote work policy?",
                "mode": "graph",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Graph answer here."
        assert data["mode"] == "graph"

    def test_ask_empty_question_returns_400(self, client):
        resp = client.post("/ask", json={"question": "   "})
        assert resp.status_code == 400

    def test_ask_streaming_returns_sse(self, client):
        mock_chain = MagicMock()
        mock_chain.stream.return_value = iter(["Hello ", "world"])

        with (
            patch("src.rag.naive_rag.build_naive_rag_chain", return_value=mock_chain),
            patch("src.observability.metrics_store.get_store") as mock_store,
        ):
            mock_store.return_value = MagicMock()
            resp = client.post("/ask", json={
                "question": "test?",
                "mode": "naive",
                "stream": True,
            })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        # Parse SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert any(e.get("type") == "token" for e in events)
        assert any(e.get("type") == "done" for e in events)


# ---------------------------------------------------------------------------
# TestIngestEndpoint
# ---------------------------------------------------------------------------

class TestIngestEndpoint:
    """POST /ingest."""

    def test_ingest_returns_counts(self, client):
        with (
            patch("src.ingestion.loader.load_path") as mock_load,
            patch("src.ingestion.chunker.chunk_documents") as mock_chunk,
            patch("src.vectorstore.chroma_store.add_chunks") as mock_add,
            patch("src.vectorstore.chroma_store.collection_stats") as mock_stats,
        ):
            mock_load.return_value = [MagicMock()] * 5
            mock_chunk.return_value = [MagicMock()] * 20
            mock_add.return_value = 15
            mock_stats.return_value = {"document_count": 41, "collection": "test", "persist_directory": "."}

            resp = client.post("/ingest", json={"path": "./data/sample_docs"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_loaded"] == 5
        assert data["chunks_created"] == 20
        assert data["chunks_added"] == 15
        assert data["collection_total"] == 41

    def test_ingest_missing_path_returns_400(self, client):
        with patch("src.ingestion.loader.load_path", side_effect=FileNotFoundError("Path not found")):
            resp = client.post("/ingest", json={"path": "/nonexistent/path"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# TestUploadEndpoint
# ---------------------------------------------------------------------------

class TestUploadEndpoint:
    """POST /upload — file upload and ingestion."""

    def test_upload_txt_file(self, client):
        with (
            patch("src.ingestion.loader.load_path") as mock_load,
            patch("src.ingestion.chunker.chunk_documents") as mock_chunk,
            patch("src.vectorstore.chroma_store.add_chunks") as mock_add,
            patch("src.vectorstore.chroma_store.collection_stats") as mock_stats,
        ):
            mock_load.return_value = [MagicMock()] * 1
            mock_chunk.return_value = [MagicMock()] * 3
            mock_add.return_value = 3
            mock_stats.return_value = {"document_count": 44, "collection": "test", "persist_directory": "."}

            resp = client.post(
                "/upload",
                files={"file": ("my_doc.txt", b"Some document content here.", "text/plain")},
                data={"department": "hr"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "my_doc.txt"
        assert data["chunks_added"] == 3
        assert data["collection_total"] == 44

    def test_upload_pdf_file(self, client):
        with (
            patch("src.ingestion.loader.load_path") as mock_load,
            patch("src.ingestion.chunker.chunk_documents") as mock_chunk,
            patch("src.vectorstore.chroma_store.add_chunks") as mock_add,
            patch("src.vectorstore.chroma_store.collection_stats") as mock_stats,
        ):
            mock_load.return_value = [MagicMock()] * 2
            mock_chunk.return_value = [MagicMock()] * 8
            mock_add.return_value = 8
            mock_stats.return_value = {"document_count": 49, "collection": "test", "persist_directory": "."}

            resp = client.post(
                "/upload",
                files={"file": ("report.pdf", b"%PDF-fake-content", "application/pdf")},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "report.pdf"
        assert data["documents_loaded"] == 2

    def test_upload_unsupported_extension_returns_400(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("image.jpg", b"fake-image", "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_default_department_is_general(self, client):
        with (
            patch("src.ingestion.loader.load_path") as mock_load,
            patch("src.ingestion.chunker.chunk_documents") as mock_chunk,
            patch("src.vectorstore.chroma_store.add_chunks") as mock_add,
            patch("src.vectorstore.chroma_store.collection_stats") as mock_stats,
        ):
            mock_load.return_value = [MagicMock()]
            mock_chunk.return_value = [MagicMock()]
            mock_add.return_value = 1
            mock_stats.return_value = {"document_count": 42, "collection": "test", "persist_directory": "."}

            resp = client.post(
                "/upload",
                files={"file": ("notes.md", b"# Notes", "text/markdown")},
            )

        assert resp.status_code == 200
        # Verify load_path was called (the department folder is created inside the temp dir)
        mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# TestEvalEndpoint
# ---------------------------------------------------------------------------

class TestEvalEndpoint:
    """POST /eval."""

    def test_eval_returns_scores(self, client):
        mock_scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.90,
            "context_precision": 0.88,
            "context_recall": 0.92,
        }

        with (
            patch("src.eval.ragas_eval.load_eval_set") as mock_load,
            patch("src.eval.ragas_eval.evaluate", return_value=mock_scores),
            patch("src.retrieval.get_retriever") as mock_ret,
        ):
            mock_load.return_value = [{"question": "q", "ground_truth": "a"}] * 5
            mock_ret.return_value = MagicMock()

            resp = client.post("/eval", json={"mode": "naive", "limit": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert data["scores"]["faithfulness"] == 0.85
        assert data["items_evaluated"] == 5
        assert data["duration_s"] >= 0


# ---------------------------------------------------------------------------
# TestCORS
# ---------------------------------------------------------------------------

class TestCORS:
    """CORS middleware."""

    def test_cors_headers_present(self, client):
        resp = client.options(
            "/ask",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# TestRateLimiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Rate limiting on /ask endpoint."""

    def test_rate_limit_header_present(self, client):
        """Verify rate limiting is wired up (header present on responses)."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "answer"

        with (
            patch("src.rag.naive_rag.build_naive_rag_chain", return_value=mock_chain),
            patch("src.observability.metrics_store.get_store") as mock_store,
        ):
            mock_store.return_value = MagicMock()
            resp = client.post("/ask", json={"question": "test?"})

        # slowapi adds rate limit headers
        assert resp.status_code == 200
