"""Phase 23 — deep health checks (src/observability/health_checker.py).

Health output drives alerting, so the assertions here are about the two
things that make it useful: a subsystem failure must never take the health
endpoint itself down, and internal error strings must not leak to
unauthenticated callers outside debug mode.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.observability.health_checker import (
    DeepHealthResult,
    HealthCheck,
    _check_chromadb,
    _check_document_registry,
    _check_ingestion_queue,
    _check_memory,
    _check_object_store,
    _check_sqlite,
    deep_health_check,
)


@pytest.fixture
def hc_settings():
    with patch("src.observability.health_checker.settings") as s:
        s.debug_mode = False
        s.async_ingestion = True
        s.storage_backend = "local"
        s.event_bus = "sqlite"
        s.kafka_topic_ingestion = "document.uploaded"
        s.kafka_topic_dlq = "document.uploaded.dlq"
        s.checkpoint_dir = "./checkpoints"
        yield s


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

class TestResultShaping:
    def test_health_check_defaults(self):
        c = HealthCheck(name="x", status="ok")
        assert c.latency_ms == 0.0 and c.detail == ""

    def test_to_dict_rounds_latency(self):
        result = DeepHealthResult(
            status="ok", checks=[HealthCheck("db", "ok", 12.3456, "fine")]
        )
        assert result.to_dict()["checks"][0]["latency_ms"] == 12.3

    def test_to_dict_carries_every_field(self):
        result = DeepHealthResult(status="ok", checks=[HealthCheck("db", "ok", 1.0, "d")])
        assert result.to_dict()["checks"][0] == {
            "name": "db", "status": "ok", "latency_ms": 1.0, "detail": "d",
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

class TestChromaCheck:
    def test_ok_reports_document_count(self, hc_settings):
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            return_value={"document_count": 42, "collection": "docs"},
        ):
            check = _check_chromadb()
        assert check.status == "ok"
        assert "42 documents" in check.detail

    def test_failure_is_reported_not_raised(self, hc_settings):
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            side_effect=RuntimeError("connection refused to 10.0.0.5"),
        ):
            check = _check_chromadb()
        assert check.status == "error"

    def test_internal_detail_is_hidden_outside_debug_mode(self, hc_settings):
        hc_settings.debug_mode = False
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            side_effect=RuntimeError("connection refused to 10.0.0.5"),
        ):
            assert "10.0.0.5" not in _check_chromadb().detail

    def test_internal_detail_is_shown_in_debug_mode(self, hc_settings):
        hc_settings.debug_mode = True
        with patch(
            "src.vectorstore.chroma_store.collection_stats",
            side_effect=RuntimeError("connection refused to 10.0.0.5"),
        ):
            assert "10.0.0.5" in _check_chromadb().detail


class TestSqliteCheck:
    def test_missing_db_is_ok_not_an_error(self, hc_settings, tmp_path):
        """A fresh install has no checkpoint DB yet; that is not a failure."""
        hc_settings.checkpoint_dir = str(tmp_path)
        check = _check_sqlite()
        assert check.status == "ok"
        assert "No checkpoint DB yet" in check.detail

    def test_existing_db_is_probed(self, hc_settings, tmp_path):
        import sqlite3

        db = tmp_path / "graph_checkpoints.db"
        sqlite3.connect(str(db)).close()
        hc_settings.checkpoint_dir = str(tmp_path)
        assert _check_sqlite().status == "ok"

    def test_corrupt_db_is_an_error(self, hc_settings, tmp_path):
        db = tmp_path / "graph_checkpoints.db"
        db.write_bytes(b"this is not a sqlite file at all" * 10)
        hc_settings.checkpoint_dir = str(tmp_path)
        assert _check_sqlite().status == "error"

    def test_error_detail_is_hidden_outside_debug_mode(self, hc_settings, tmp_path):
        db = tmp_path / "graph_checkpoints.db"
        db.write_bytes(b"corrupt" * 50)
        hc_settings.checkpoint_dir = str(tmp_path)
        hc_settings.debug_mode = False
        assert _check_sqlite().detail == "SQLite unavailable"


class TestMemoryCheck:
    def test_reports_resident_set_size(self, hc_settings):
        check = _check_memory()
        assert check.status in {"ok", "degraded"}
        assert "RSS" in check.detail

    def test_high_memory_is_degraded_not_error(self, hc_settings):
        proc = MagicMock()
        proc.memory_info.return_value = MagicMock(rss=2048 * 1024 * 1024)
        with patch("psutil.Process", return_value=proc):
            assert _check_memory().status == "degraded"

    def test_low_memory_is_ok(self, hc_settings):
        proc = MagicMock()
        proc.memory_info.return_value = MagicMock(rss=200 * 1024 * 1024)
        with patch("psutil.Process", return_value=proc):
            assert _check_memory().status == "ok"

    def test_psutil_failure_is_non_critical(self, hc_settings):
        with patch("psutil.Process", side_effect=RuntimeError("no /proc")):
            check = _check_memory()
        assert check.status == "ok"
        assert "non-critical" in check.detail


class TestObjectStoreCheck:
    def test_reachable_store_is_ok(self, hc_settings):
        store = MagicMock()
        store.exists.return_value = False
        with patch("src.storage.object_store.get_object_store", return_value=store):
            check = _check_object_store()
        assert check.status == "ok"
        assert "local" in check.detail

    def test_probe_does_not_write_anything(self, hc_settings):
        """A health check must not leave objects behind in the bucket."""
        store = MagicMock()
        with patch("src.storage.object_store.get_object_store", return_value=store):
            _check_object_store()
        store.put.assert_not_called()
        store.delete.assert_not_called()

    def test_unreachable_store_is_an_error(self, hc_settings):
        with patch(
            "src.storage.object_store.get_object_store",
            side_effect=RuntimeError("s3 credentials rejected for arn:aws:iam::123"),
        ):
            check = _check_object_store()
        assert check.status == "error"
        assert "arn:aws:iam" not in check.detail


class TestIngestionQueueCheck:
    def _bus(self, backlog=0, dlq=0):
        bus = MagicMock()
        bus.depth.side_effect = lambda topic: dlq if topic.endswith(".dlq") else backlog
        return bus

    def test_empty_queue_is_ok(self, hc_settings):
        with patch("src.events.bus.get_event_bus", return_value=self._bus()):
            check = _check_ingestion_queue()
        assert check.status == "ok"
        assert "backlog=0 dlq=0" in check.detail

    def test_backlog_alone_is_still_ok(self, hc_settings):
        """A backlog means workers are behind, not that anything was lost."""
        with patch("src.events.bus.get_event_bus", return_value=self._bus(backlog=25)):
            check = _check_ingestion_queue()
        assert check.status == "ok"
        assert "backlog=25" in check.detail

    def test_any_dlq_depth_degrades_health(self, hc_settings):
        """Dead-lettered documents were accepted from users and never indexed."""
        with patch("src.events.bus.get_event_bus", return_value=self._bus(dlq=1)):
            check = _check_ingestion_queue()
        assert check.status == "degraded"
        assert "dlq=1" in check.detail

    def test_bus_without_depth_support_is_still_ok(self, hc_settings):
        class NoDepthBus:
            pass

        with patch("src.events.bus.get_event_bus", return_value=NoDepthBus()):
            check = _check_ingestion_queue()
        assert check.status == "ok"
        assert "depth not reported" in check.detail

    def test_unreachable_bus_is_an_error(self, hc_settings):
        with patch("src.events.bus.get_event_bus", side_effect=RuntimeError("broker down")):
            assert _check_ingestion_queue().status == "error"


class TestDocumentRegistryCheck:
    def _registry(self, **counts):
        base = {"TOTAL": 0, "PROCESSED": 0, "PENDING": 0, "FAILED": 0, "DEAD_LETTER": 0}
        base.update(counts)
        registry = MagicMock()
        registry.stats.return_value = base
        return registry

    def test_healthy_registry_is_ok(self, hc_settings):
        with patch(
            "src.ingestion.registry.get_registry",
            return_value=self._registry(TOTAL=10, PROCESSED=10),
        ):
            check = _check_document_registry()
        assert check.status == "ok"
        assert "processed=10" in check.detail

    def test_dead_letters_degrade_health(self, hc_settings):
        with patch(
            "src.ingestion.registry.get_registry",
            return_value=self._registry(TOTAL=5, PROCESSED=4, DEAD_LETTER=1),
        ):
            assert _check_document_registry().status == "degraded"

    def test_pending_documents_do_not_degrade_health(self, hc_settings):
        with patch(
            "src.ingestion.registry.get_registry",
            return_value=self._registry(TOTAL=5, PENDING=5),
        ):
            assert _check_document_registry().status == "ok"

    def test_unreachable_registry_is_an_error(self, hc_settings):
        with patch("src.ingestion.registry.get_registry", side_effect=RuntimeError("locked")):
            assert _check_document_registry().status == "error"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestDeepHealthCheck:
    def _all_ok(self):
        return patch.multiple(
            "src.observability.health_checker",
            _check_chromadb=MagicMock(return_value=HealthCheck("chromadb", "ok")),
            _check_sqlite=MagicMock(return_value=HealthCheck("sqlite", "ok")),
            _check_memory=MagicMock(return_value=HealthCheck("memory", "ok")),
            _check_object_store=MagicMock(return_value=HealthCheck("object_store", "ok")),
            _check_ingestion_queue=MagicMock(return_value=HealthCheck("ingestion_queue", "ok")),
            _check_document_registry=MagicMock(return_value=HealthCheck("document_registry", "ok")),
        )

    def test_all_ok_aggregates_to_ok(self, hc_settings):
        with self._all_ok():
            assert deep_health_check().status == "ok"

    def test_ingestion_checks_run_only_when_async_is_enabled(self, hc_settings):
        hc_settings.async_ingestion = False
        with self._all_ok():
            names = {c.name for c in deep_health_check().checks}
        assert names == {"chromadb", "sqlite", "memory"}

    def test_ingestion_checks_are_included_when_async_is_enabled(self, hc_settings):
        hc_settings.async_ingestion = True
        with self._all_ok():
            names = {c.name for c in deep_health_check().checks}
        assert "ingestion_queue" in names and "document_registry" in names

    def test_any_error_makes_the_whole_result_error(self, hc_settings):
        with self._all_ok(), patch(
            "src.observability.health_checker._check_chromadb",
            return_value=HealthCheck("chromadb", "error"),
        ):
            assert deep_health_check().status == "error"

    def test_degraded_wins_over_ok_but_loses_to_error(self, hc_settings):
        with self._all_ok(), patch(
            "src.observability.health_checker._check_ingestion_queue",
            return_value=HealthCheck("ingestion_queue", "degraded"),
        ):
            assert deep_health_check().status == "degraded"

    def test_error_beats_degraded(self, hc_settings):
        with self._all_ok(), patch(
            "src.observability.health_checker._check_ingestion_queue",
            return_value=HealthCheck("ingestion_queue", "degraded"),
        ), patch(
            "src.observability.health_checker._check_sqlite",
            return_value=HealthCheck("sqlite", "error"),
        ):
            assert deep_health_check().status == "error"
