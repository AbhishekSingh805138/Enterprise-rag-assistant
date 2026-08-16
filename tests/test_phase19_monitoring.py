"""Phase 19: Monitoring & Production Infrastructure tests.

Tests for:
- Deep health checker (individual checks + aggregation)
- Deep health API endpoint
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Health checker unit tests
# ---------------------------------------------------------------------------


class TestHealthChecker:
    def test_check_chromadb_ok(self):
        with patch("src.vectorstore.chroma_store.collection_stats") as mock:
            mock.return_value = {"collection": "test", "document_count": 42}

            from src.observability.health_checker import _check_chromadb

            result = _check_chromadb()
            assert result.status == "ok"
            assert result.name == "chromadb"
            assert "42" in result.detail

    def test_check_chromadb_error(self):
        with patch("src.vectorstore.chroma_store.collection_stats") as mock:
            mock.side_effect = RuntimeError("Connection failed")

            from src.observability.health_checker import _check_chromadb

            result = _check_chromadb()
            assert result.status == "error"

    def test_check_sqlite_no_db(self):
        with patch("src.observability.health_checker.Path") as mock_path:
            mock_instance = MagicMock()
            mock_instance.__truediv__ = MagicMock(return_value=mock_instance)
            mock_instance.exists.return_value = False
            mock_path.return_value = mock_instance

            from src.observability.health_checker import _check_sqlite

            result = _check_sqlite()
            assert result.status == "ok"

    def test_check_memory_without_psutil(self):
        with patch.dict("sys.modules", {"psutil": None}):
            from src.observability.health_checker import _check_memory

            result = _check_memory()
            assert result.status == "ok"

    @pytest.fixture(autouse=True)
    def _core_checks_only(self):
        """Pin async ingestion off for the aggregation tests.

        deep_health_check() adds object-store, queue and registry checks
        when ASYNC_INGESTION is on. Left unpinned, these tests would both
        change shape depending on the developer's .env and reach out to
        those real subsystems. The async variants are covered explicitly
        in test_phase23_health_checks.py.
        """
        with patch("src.observability.health_checker.settings") as mock_settings:
            mock_settings.async_ingestion = False
            mock_settings.debug_mode = False
            yield mock_settings

    def test_deep_health_all_ok(self):
        with patch("src.observability.health_checker._check_chromadb") as mock_chroma, \
             patch("src.observability.health_checker._check_sqlite") as mock_sqlite, \
             patch("src.observability.health_checker._check_memory") as mock_mem:

            from src.observability.health_checker import HealthCheck, deep_health_check

            mock_chroma.return_value = HealthCheck(name="chromadb", status="ok")
            mock_sqlite.return_value = HealthCheck(name="sqlite", status="ok")
            mock_mem.return_value = HealthCheck(name="memory", status="ok")

            result = deep_health_check()
            assert result.status == "ok"
            assert len(result.checks) == 3

    def test_deep_health_degraded(self):
        with patch("src.observability.health_checker._check_chromadb") as mock_chroma, \
             patch("src.observability.health_checker._check_sqlite") as mock_sqlite, \
             patch("src.observability.health_checker._check_memory") as mock_mem:

            from src.observability.health_checker import HealthCheck, deep_health_check

            mock_chroma.return_value = HealthCheck(name="chromadb", status="ok")
            mock_sqlite.return_value = HealthCheck(name="sqlite", status="ok")
            mock_mem.return_value = HealthCheck(name="memory", status="degraded")

            result = deep_health_check()
            assert result.status == "degraded"

    def test_deep_health_error(self):
        with patch("src.observability.health_checker._check_chromadb") as mock_chroma, \
             patch("src.observability.health_checker._check_sqlite") as mock_sqlite, \
             patch("src.observability.health_checker._check_memory") as mock_mem:

            from src.observability.health_checker import HealthCheck, deep_health_check

            mock_chroma.return_value = HealthCheck(name="chromadb", status="error", detail="down")
            mock_sqlite.return_value = HealthCheck(name="sqlite", status="ok")
            mock_mem.return_value = HealthCheck(name="memory", status="ok")

            result = deep_health_check()
            assert result.status == "error"

    def test_deep_health_to_dict(self):
        from src.observability.health_checker import DeepHealthResult, HealthCheck

        result = DeepHealthResult(
            status="ok",
            checks=[HealthCheck(name="test", status="ok", latency_ms=1.5, detail="fine")],
        )
        d = result.to_dict()
        assert d["status"] == "ok"
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "test"
        assert d["checks"][0]["latency_ms"] == 1.5


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestDeepHealthEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from api.app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_shallow_health(self, client):
        with patch("src.vectorstore.chroma_store.collection_stats") as mock:
            mock.return_value = {"collection": "test", "document_count": 10}
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "checks" not in data  # shallow mode

    def test_deep_health(self, client):
        with patch("src.observability.health_checker.deep_health_check") as mock:
            from src.observability.health_checker import DeepHealthResult, HealthCheck

            mock.return_value = DeepHealthResult(
                status="ok",
                checks=[
                    HealthCheck(name="chromadb", status="ok", latency_ms=5.0, detail="10 docs"),
                    HealthCheck(name="sqlite", status="ok", latency_ms=1.0, detail="ok"),
                ],
            )
            resp = client.get("/health?deep=true")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "checks" in data
            assert len(data["checks"]) == 2
