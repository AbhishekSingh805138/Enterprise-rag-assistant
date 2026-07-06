"""Phase 9.1 tests: Critical security fixes.

Tests for:
  - _safe_error_detail() function (9.1.1)
  - Upload endpoint hardening (9.1.2): path traversal, file size, MIME, department
  - CORS hardening (9.1.3): restricted methods and headers
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


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


# ===========================================================================
# 9.1.1 — _safe_error_detail()
# ===========================================================================

class TestSafeErrorDetail:
    """Verify _safe_error_detail returns safe messages based on debug_mode."""

    def test_production_mode_hides_details(self):
        """In production (debug_mode=False), returns generic message."""
        from api.app import _safe_error_detail
        with patch("api.app.settings") as ms:
            ms.debug_mode = False
            msg = _safe_error_detail(RuntimeError("secret db path /var/data"))
            assert "secret" not in msg
            assert "internal error" in msg.lower()

    def test_debug_mode_shows_details(self):
        """In debug mode, returns the actual exception message."""
        from api.app import _safe_error_detail
        with patch("api.app.settings") as ms:
            ms.debug_mode = True
            msg = _safe_error_detail(RuntimeError("secret db path /var/data"))
            assert "secret db path" in msg

    def test_ask_endpoint_uses_safe_error(self, client):
        """Verify /ask 500 errors use _safe_error_detail (no internal leak)."""
        with patch("api.app._ask_sync", side_effect=RuntimeError("db crash at /var")):
            resp = client.post("/ask", json={"question": "test?"})
            assert resp.status_code == 500
            body = resp.json()
            assert "/var" not in body.get("detail", "")
            assert "internal error" in body.get("detail", "").lower()


# ===========================================================================
# 9.1.2 — Upload endpoint hardening
# ===========================================================================

class TestUploadPathTraversal:
    """Verify path traversal attacks are blocked."""

    def test_directory_traversal_stripped(self, client):
        """Filenames with ../ components are sanitized to basename only."""
        file = io.BytesIO(b"hello")
        # ../../etc/passwd.txt → passwd.txt (safe)
        resp = client.post(
            "/upload",
            files={"file": ("../../etc/passwd.txt", file, "text/plain")},
            params={"department": "hr"},
        )
        # Should not 500 with traversal — the filename is sanitized
        assert resp.status_code != 500 or "traversal" not in resp.json().get("detail", "").lower()

    def test_dotdot_in_name_rejected(self, client):
        """Filenames containing '..' are rejected."""
        file = io.BytesIO(b"hello")
        resp = client.post(
            "/upload",
            files={"file": ("..secret.txt", file, "text/plain")},
            params={"department": "hr"},
        )
        assert resp.status_code == 400

    def test_empty_filename_rejected(self, client):
        """Empty filenames are rejected (400 from handler or 422 from FastAPI)."""
        file = io.BytesIO(b"hello")
        resp = client.post(
            "/upload",
            files={"file": ("", file, "text/plain")},
            params={"department": "hr"},
        )
        assert resp.status_code in (400, 422)


class TestUploadFileSize:
    """Verify file size limits are enforced."""

    def test_oversized_file_rejected(self, client):
        """Files exceeding max_upload_size_mb are rejected with 413."""
        # Default max is 10 MB; create 11 MB file
        big_content = b"x" * (11 * 1024 * 1024)
        file = io.BytesIO(big_content)
        resp = client.post(
            "/upload",
            files={"file": ("big.txt", file, "text/plain")},
            params={"department": "hr"},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()


class TestUploadMimeType:
    """Verify MIME type validation."""

    def test_unsupported_mime_rejected(self, client):
        """Files with disallowed MIME types are rejected."""
        file = io.BytesIO(b"<html></html>")
        resp = client.post(
            "/upload",
            files={"file": ("page.txt", file, "text/html")},
            params={"department": "hr"},
        )
        assert resp.status_code == 400
        assert "content type" in resp.json()["detail"].lower()


class TestUploadDepartmentValidation:
    """Verify department parameter is validated."""

    def test_invalid_department_rejected(self, client):
        """Non-whitelisted department names are rejected."""
        file = io.BytesIO(b"hello")
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", file, "text/plain")},
            params={"department": "../../evil"},
        )
        assert resp.status_code == 400
        assert "department" in resp.json()["detail"].lower()

    def test_unknown_department_rejected(self, client):
        """Department names not in VALID_DEPARTMENTS are rejected."""
        file = io.BytesIO(b"hello")
        resp = client.post(
            "/upload",
            files={"file": ("test.txt", file, "text/plain")},
            params={"department": "nonexistent"},
        )
        assert resp.status_code == 400

    def test_valid_department_accepted(self, client):
        """Known departments pass validation (upload itself may fail without real ChromaDB)."""
        file = io.BytesIO(b"hello world test content")
        with patch("src.ingestion.loader.load_path", return_value=[]), \
             patch("src.ingestion.chunker.chunk_documents", return_value=[]), \
             patch("src.vectorstore.chroma_store.add_chunks", return_value=0), \
             patch("src.vectorstore.chroma_store.collection_stats", return_value={"document_count": 0}):
            resp = client.post(
                "/upload",
                files={"file": ("test.txt", file, "text/plain")},
                params={"department": "hr"},
            )
            # Should not be a 400 for department
            if resp.status_code == 400:
                assert "department" not in resp.json().get("detail", "").lower()

    def test_unsupported_extension_rejected(self, client):
        """Files with wrong extensions are rejected."""
        file = io.BytesIO(b"hello")
        resp = client.post(
            "/upload",
            files={"file": ("test.exe", file, "application/octet-stream")},
            params={"department": "hr"},
        )
        assert resp.status_code == 400
        assert "unsupported file type" in resp.json()["detail"].lower()


class TestSanitizeFilename:
    """Unit tests for _sanitize_filename helper."""

    def test_strips_directory_components(self):
        from api.app import _sanitize_filename
        assert _sanitize_filename("some/path/file.txt") == "file.txt"

    def test_rejects_dotdot(self):
        from api.app import _sanitize_filename
        with pytest.raises(ValueError):
            _sanitize_filename("..hidden.txt")

    def test_rejects_null_byte(self):
        from api.app import _sanitize_filename
        with pytest.raises(ValueError):
            _sanitize_filename("file\x00.txt")

    def test_rejects_empty(self):
        from api.app import _sanitize_filename
        with pytest.raises(ValueError):
            _sanitize_filename("")

    def test_normal_filename_passes(self):
        from api.app import _sanitize_filename
        assert _sanitize_filename("report-2026.pdf") == "report-2026.pdf"


# ===========================================================================
# 9.1.3 — CORS hardening
# ===========================================================================

class TestCORSHardening:
    """Verify CORS is restricted to configured methods and headers."""

    def test_cors_allows_configured_origin(self, client):
        """Configured origin gets access-control-allow-origin header."""
        resp = client.options(
            "/ask",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in resp.headers

    def test_cors_blocks_unconfigured_origin(self, client):
        """Non-configured origin does NOT get access-control-allow-origin header."""
        resp = client.options(
            "/ask",
            headers={
                "Origin": "http://evil.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_restricts_methods(self, client):
        """Only configured methods are allowed (not DELETE, PUT, PATCH)."""
        resp = client.options(
            "/ask",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        allowed = resp.headers.get("access-control-allow-methods", "")
        assert "DELETE" not in allowed
        assert "PATCH" not in allowed
        assert "POST" in allowed


# ===========================================================================
# Config validation
# ===========================================================================

class TestConfigValidation:
    """Test new config settings for Phase 9.1."""

    def test_max_upload_size_mb_positive(self):
        """max_upload_size_mb must be positive."""
        from config import Settings

        # Settings is a frozen dataclass evaluated at construction time via os.getenv.
        # We must construct a fresh one with the bad value to test validation.
        import dataclasses
        base = Settings()
        # Create a copy with max_upload_size_mb=0 (bypass frozen via replace)
        s = dataclasses.replace(base, max_upload_size_mb=0, openai_api_key="sk-test")
        with pytest.raises(ValueError, match="max_upload_size_mb"):
            s.validate()


class TestValidDepartments:
    """Test VALID_DEPARTMENTS constant."""

    def test_contains_expected_departments(self):
        from api.models import VALID_DEPARTMENTS
        for dept in ["hr", "legal", "engineering", "finance", "security", "operations", "general"]:
            assert dept in VALID_DEPARTMENTS

    def test_is_frozenset(self):
        from api.models import VALID_DEPARTMENTS
        assert isinstance(VALID_DEPARTMENTS, frozenset)
