"""Phase 23 — config validation, structured logging, and ingestion edge paths.

Covers the failure branches that only fire in production: misconfigured
env vars caught at startup, the JSON log formatter that log aggregators
depend on, request-id propagation across threads, and the error handling
in the loader, pipeline and queue that keeps a bad document from taking
down an ingest.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import sqlite3
import threading
from unittest.mock import MagicMock, patch

import pytest

from config import Settings, _JsonFormatter, setup_logging


def make_settings(**overrides) -> Settings:
    """A valid Settings, with the given fields overridden."""
    base = Settings()
    defaults = {"openai_api_key": "sk-test"}
    defaults.update(overrides)
    return dataclasses.replace(base, **defaults)


# ---------------------------------------------------------------------------
# Settings.validate — startup guards
# ---------------------------------------------------------------------------

class TestSettingsValidation:
    def test_a_sane_config_validates(self):
        make_settings().validate()  # must not raise

    def test_missing_api_key_is_rejected(self):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            make_settings(openai_api_key="").validate()

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("log_level", "CHATTY", "Invalid LOG_LEVEL"),
            ("chunk_size", 0, "chunk_size must be positive"),
            ("chunk_overlap", -1, "chunk_overlap must be non-negative"),
            ("top_k", 0, "top_k must be positive"),
            ("llm_timeout", 0, "llm_timeout must be positive"),
            ("llm_max_retries", -1, "llm_max_retries must be non-negative"),
            ("cost_alert_threshold", 0, "cost_alert_threshold must be positive"),
            ("max_upload_size_mb", 0, "max_upload_size_mb must be positive"),
            ("critic_mode", "sometimes", "Invalid CRITIC_MODE"),
            ("storage_backend", "gcs", "Invalid STORAGE_BACKEND"),
            ("event_bus", "rabbitmq", "Invalid EVENT_BUS"),
            ("ingest_max_attempts", 0, "ingest_max_attempts must be at least 1"),
            ("ingest_visibility_timeout_s", 0, "ingest_visibility_timeout_s must be positive"),
        ],
    )
    def test_invalid_values_are_rejected(self, field, value, match):
        with pytest.raises((ValueError, RuntimeError), match=match):
            make_settings(**{field: value}).validate()

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError, match="must be less than chunk_size"):
            make_settings(chunk_size=500, chunk_overlap=500).validate()

    def test_s3_backend_requires_a_bucket(self):
        """Caught at startup rather than on the first user upload."""
        with pytest.raises(ValueError, match="requires S3_BUCKET"):
            make_settings(storage_backend="s3", s3_bucket="").validate()

    def test_s3_backend_with_a_bucket_is_accepted(self):
        make_settings(storage_backend="s3", s3_bucket="my-bucket").validate()

    def test_backend_names_are_case_insensitive(self):
        make_settings(storage_backend="LOCAL", event_bus="SQLite").validate()

    def test_async_with_sqlite_bus_logs_a_scaling_note(self, caplog):
        with caplog.at_level(logging.INFO, logger="config"):
            make_settings(async_ingestion=True, event_bus="sqlite").validate()
        assert any("EVENT_BUS_PATH" in r.message for r in caplog.records)

    def test_async_with_kafka_logs_no_such_note(self, caplog):
        with caplog.at_level(logging.INFO, logger="config"):
            make_settings(async_ingestion=True, event_bus="kafka").validate()
        assert not any("EVENT_BUS_PATH" in r.message for r in caplog.records)

    def test_langsmith_without_a_key_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="config"):
            make_settings(langsmith_tracing="true", langsmith_api_key="").validate()
        assert any("LANGSMITH_API_KEY" in r.message for r in caplog.records)

    def test_ingestion_defaults_are_safe(self):
        """Defaults must not enable async ingestion or require a broker."""
        s = Settings()
        assert s.async_ingestion is False
        assert s.storage_backend == "local"
        assert s.event_bus == "sqlite"
        assert s.ingest_max_attempts == 3


# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    def _record(self, msg="hello", level=logging.INFO, **extra):
        record = logging.LogRecord(
            name="src.test", level=level, pathname=__file__, lineno=1,
            msg=msg, args=(), exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_output_is_one_json_object(self):
        payload = json.loads(_JsonFormatter().format(self._record()))
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "src.test"

    def test_timestamp_is_iso_utc(self):
        payload = json.loads(_JsonFormatter().format(self._record()))
        assert payload["ts"].endswith("+00:00")

    def test_message_args_are_interpolated(self):
        record = logging.LogRecord(
            name="n", level=logging.INFO, pathname=__file__, lineno=1,
            msg="indexed %d chunks", args=(5,), exc_info=None,
        )
        assert json.loads(_JsonFormatter().format(record))["message"] == "indexed 5 chunks"

    def test_request_id_is_attached_when_bound(self):
        from src.observability.request_context import reset_request_id, set_request_id

        token = set_request_id("trace-xyz")
        try:
            payload = json.loads(_JsonFormatter().format(self._record()))
        finally:
            reset_request_id(token)
        assert payload["request_id"] == "trace-xyz"

    def test_request_id_is_omitted_outside_a_request(self):
        from src.observability.request_context import _request_id

        token = _request_id.set("")
        try:
            payload = json.loads(_JsonFormatter().format(self._record()))
        finally:
            _request_id.reset(token)
        assert "request_id" not in payload

    def test_exceptions_are_serialised(self):
        try:
            raise ValueError("kaboom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="n", level=logging.ERROR, pathname=__file__, lineno=1,
                msg="failed", args=(), exc_info=sys.exc_info(),
            )
        payload = json.loads(_JsonFormatter().format(record))
        assert "kaboom" in payload["exception"]

    def test_ctx_prefixed_extras_are_promoted_to_fields(self):
        payload = json.loads(_JsonFormatter().format(self._record(ctx_document_id="doc_1")))
        assert payload["document_id"] == "doc_1"

    def test_non_ctx_attributes_are_not_promoted(self):
        payload = json.loads(_JsonFormatter().format(self._record(secret="do-not-log")))
        assert "secret" not in payload

    def test_unserialisable_values_do_not_break_logging(self):
        payload = json.loads(_JsonFormatter().format(self._record(ctx_obj=object())))
        assert "obj" in payload


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def _restore_root(self):
        root = logging.getLogger()
        handlers, level = root.handlers[:], root.level
        yield
        root.handlers[:] = handlers
        root.setLevel(level)

    def test_json_logs_flag_selects_the_json_formatter(self):
        logging.getLogger().handlers.clear()
        with patch("config.settings") as s:
            s.json_logs = True
            s.log_level = "INFO"
            setup_logging()
        assert isinstance(logging.getLogger().handlers[0].formatter, _JsonFormatter)

    def test_plain_formatter_is_the_default(self):
        logging.getLogger().handlers.clear()
        with patch("config.settings") as s:
            s.json_logs = False
            s.log_level = "INFO"
            setup_logging()
        assert not isinstance(logging.getLogger().handlers[0].formatter, _JsonFormatter)

    def test_repeated_calls_do_not_duplicate_handlers(self):
        logging.getLogger().handlers.clear()
        with patch("config.settings") as s:
            s.json_logs = False
            s.log_level = "INFO"
            setup_logging()
            setup_logging()
        assert len(logging.getLogger().handlers) == 1

    def test_noisy_third_party_loggers_are_quieted(self):
        with patch("config.settings") as s:
            s.json_logs = False
            s.log_level = "DEBUG"
            setup_logging()
        assert logging.getLogger("httpx").level == logging.WARNING


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------

class TestRequestContext:
    def test_set_and_get_roundtrip(self):
        from src.observability.request_context import (
            get_request_id,
            reset_request_id,
            set_request_id,
        )

        token = set_request_id("abc")
        try:
            assert get_request_id() == "abc"
        finally:
            reset_request_id(token)

    def test_setting_none_generates_an_id(self):
        from src.observability.request_context import (
            get_request_id,
            reset_request_id,
            set_request_id,
        )

        token = set_request_id(None)
        try:
            assert get_request_id()
        finally:
            reset_request_id(token)

    def test_generated_ids_are_unique(self):
        from src.observability.request_context import new_request_id

        assert len({new_request_id() for _ in range(100)}) == 100

    def test_reset_restores_the_previous_value(self):
        from src.observability.request_context import (
            get_request_id,
            reset_request_id,
            set_request_id,
        )

        outer = set_request_id("outer")
        inner = set_request_id("inner")
        reset_request_id(inner)
        try:
            assert get_request_id() == "outer"
        finally:
            reset_request_id(outer)

    def test_reset_with_a_foreign_token_is_tolerated(self):
        """Tokens set inside a thread cannot be reset from the parent."""
        from src.observability.request_context import reset_request_id, set_request_id

        captured = {}

        def _in_thread():
            captured["token"] = set_request_id("thread-local")

        t = threading.Thread(target=_in_thread)
        t.start()
        t.join()
        reset_request_id(captured["token"])  # must not raise

    def test_id_propagates_into_worker_threads(self):
        """asyncio.to_thread and copy_context carry the id; verify the premise."""
        import contextvars

        from src.observability.request_context import (
            get_request_id,
            reset_request_id,
            set_request_id,
        )

        token = set_request_id("carried")
        seen = {}
        try:
            ctx = contextvars.copy_context()
            t = threading.Thread(target=lambda: seen.update(id=ctx.run(get_request_id)))
            t.start()
            t.join()
        finally:
            reset_request_id(token)
        assert seen["id"] == "carried"


# ---------------------------------------------------------------------------
# Loader edge paths
# ---------------------------------------------------------------------------

class TestLoaderEdges:
    def test_confidential_departments_get_a_higher_access_level(self, tmp_path):
        from src.ingestion.loader import load_path

        for dept in ("legal", "security"):
            (tmp_path / dept).mkdir()
            (tmp_path / dept / "policy.md").write_text("# P\nbody\n", encoding="utf-8")
        docs = load_path(str(tmp_path))
        assert {d.metadata["access_level"] for d in docs} == {"confidential"}

    def test_other_departments_are_internal(self, tmp_path):
        from src.ingestion.loader import load_path

        (tmp_path / "hr").mkdir()
        (tmp_path / "hr" / "policy.md").write_text("# P\nbody\n", encoding="utf-8")
        assert load_path(str(tmp_path))[0].metadata["access_level"] == "internal"

    def test_root_level_files_default_to_general(self, tmp_path):
        from src.ingestion.loader import load_path

        (tmp_path / "readme.md").write_text("# R\nbody\n", encoding="utf-8")
        assert load_path(str(tmp_path))[0].metadata["department"] == "general"

    def test_a_single_file_is_loadable_directly(self, tmp_path):
        from src.ingestion.loader import load_path

        f = tmp_path / "notes.md"
        f.write_text("# Notes\nbody\n", encoding="utf-8")
        assert len(load_path(str(f))) == 1

    def test_unreadable_file_is_skipped_not_fatal(self, tmp_path):
        """One corrupt file must not abort a whole directory ingest."""
        from src.ingestion.loader import load_path

        (tmp_path / "good.md").write_text("# Good\nbody\n", encoding="utf-8")
        (tmp_path / "bad.pdf").write_bytes(b"not really a pdf")
        docs = load_path(str(tmp_path))
        assert [d.metadata["filename"] for d in docs] == ["good.md"]

    def test_all_files_failing_raises(self, tmp_path):
        from src.ingestion.loader import load_path

        (tmp_path / "bad.pdf").write_bytes(b"not really a pdf")
        with pytest.raises(FileNotFoundError):
            load_path(str(tmp_path))

    def test_docx_without_docx2txt_gives_an_install_hint(self, tmp_path):
        import builtins

        from src.ingestion.loader import _load_docx

        real_import = builtins.__import__

        def _no_docx2txt(name, *args, **kwargs):
            if name == "docx2txt":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", _no_docx2txt):
            with pytest.raises(RuntimeError, match="pip install docx2txt"):
                _load_docx(tmp_path / "x.docx")

    def test_docx_is_loaded_when_the_dependency_is_present(self, tmp_path):
        import sys
        import types

        from src.ingestion.loader import _load_docx

        fake = types.ModuleType("docx2txt")
        loader_instance = MagicMock()
        loader_instance.load.return_value = ["doc"]
        with (
            patch.dict(sys.modules, {"docx2txt": fake}),
            patch(
                "langchain_community.document_loaders.Docx2txtLoader",
                return_value=loader_instance,
            ),
        ):
            assert _load_docx(tmp_path / "x.docx") == ["doc"]


# ---------------------------------------------------------------------------
# Pipeline error paths
# ---------------------------------------------------------------------------

class TestPipelineErrors:
    def _record(self, tmp_path):
        from src.ingestion.registry import DocumentRecord

        return DocumentRecord(
            document_id="doc_1", checksum="abc", filename="f.md",
            department="hr", storage_key="documents/hr/doc_1/f.md",
        )

    def test_unparseable_document_raises_a_permanent_error(self, tmp_path):
        from src.ingestion.pipeline import DocumentParseError, process_document
        from src.storage.object_store import LocalObjectStore

        store = LocalObjectStore(root=tmp_path / "objects")
        store.put("documents/hr/doc_1/f.md", b"body")
        with patch(
            "src.ingestion.loader.load_path", side_effect=FileNotFoundError("no docs")
        ):
            with pytest.raises(DocumentParseError, match="Could not parse"):
                process_document(self._record(tmp_path), object_store=store)

    def test_document_producing_no_chunks_is_a_permanent_error(self, tmp_path):
        from src.ingestion.pipeline import DocumentParseError, process_document
        from src.storage.object_store import LocalObjectStore

        store = LocalObjectStore(root=tmp_path / "objects")
        store.put("documents/hr/doc_1/f.md", b"body")
        with (
            patch("src.ingestion.loader.load_path", return_value=[MagicMock(metadata={})]),
            patch("src.ingestion.chunker.chunk_documents", return_value=[]),
        ):
            with pytest.raises(DocumentParseError, match="no chunks"):
                process_document(self._record(tmp_path), object_store=store)

    def test_missing_object_propagates(self, tmp_path):
        from src.ingestion.pipeline import process_document
        from src.storage.object_store import LocalObjectStore, ObjectNotFoundError

        store = LocalObjectStore(root=tmp_path / "objects")
        with pytest.raises(ObjectNotFoundError):
            process_document(self._record(tmp_path), object_store=store)

    def test_stable_source_matches_the_sync_upload_path(self):
        """Divergence here would double-index the same file across paths."""
        from src.ingestion.pipeline import stable_source

        assert stable_source("hr", "handbook.md") == "uploads/hr/handbook.md"


# ---------------------------------------------------------------------------
# Queue internals
# ---------------------------------------------------------------------------

class TestSqliteQueueInternals:
    @pytest.fixture
    def bus(self, tmp_path):
        from src.events.sqlite_bus import SQLiteEventBus

        with patch("src.events.sqlite_bus.settings") as s:
            s.ingest_visibility_timeout_s = 300
            b = SQLiteEventBus(db_path=tmp_path / "events.db")
            yield b
            b.close()

    def _event(self, doc="doc_1"):
        from src.events.bus import Event

        return Event(event_type="document.uploaded", document_id=doc)

    @staticmethod
    @contextlib.contextmanager
    def broken(bus, message="disk I/O error"):
        """Swap in a connection that fails.

        sqlite3.Connection methods are read-only C attributes, so the whole
        object has to be replaced rather than patched.
        """
        original = bus._conn
        fake = MagicMock()
        fake.execute.side_effect = sqlite3.Error(message)
        fake.close.side_effect = sqlite3.Error(message)
        bus._conn = fake
        try:
            yield
        finally:
            bus._conn = original

    def test_publish_failure_is_wrapped(self, bus):
        from src.events.bus import EventBusError

        with self.broken(bus):
            with pytest.raises(EventBusError, match="Failed to publish"):
                bus.publish("t", self._event())

    def test_poll_failure_is_wrapped(self, bus):
        from src.events.bus import EventBusError

        bus.publish("t", self._event())
        with self.broken(bus, "locked"):
            with pytest.raises(EventBusError, match="Failed to poll"):
                bus.poll("t", "g")

    def test_ack_failure_does_not_raise(self, bus):
        bus.publish("t", self._event())
        delivery = bus.poll("t", "g")[0]
        with self.broken(bus, "locked"):
            delivery.ack()  # must not raise
        assert delivery.settled is True

    def test_nack_failure_does_not_raise(self, bus):
        bus.publish("t", self._event())
        delivery = bus.poll("t", "g")[0]
        with self.broken(bus, "locked"):
            delivery.nack()
        assert delivery.settled is True

    def test_depth_failure_reports_unknown(self, bus):
        with self.broken(bus):
            assert bus.depth("t") == -1

    def test_peek_failure_returns_empty(self, bus):
        with self.broken(bus):
            assert bus.peek("t") == []

    def test_peek_skips_undecodable_rows(self, bus):
        bus.publish("t", self._event())
        with bus._lock:
            bus._conn.execute("UPDATE events SET body = ?", ("{corrupt",))
            bus._conn.commit()
        assert bus.peek("t") == []

    def test_purge_all_topics(self, bus):
        bus.publish("a", self._event())
        bus.publish("b", self._event())
        assert bus.purge() == 2
        assert bus.depth("a") == 0 and bus.depth("b") == 0

    def test_purge_failure_reports_zero(self, bus):
        with self.broken(bus, "locked"):
            assert bus.purge("t") == 0

    def test_reclaim_failure_is_swallowed(self, bus):
        """A failed reclaim must not take the poll loop down."""
        with self.broken(bus, "locked"):
            bus._reclaim_expired("t")  # must not raise

    def test_close_is_tolerant_of_a_broken_connection(self, bus):
        with self.broken(bus, "already closed"):
            bus.close()  # must not raise

    def test_creating_the_bus_makes_parent_directories(self, tmp_path):
        from src.events.sqlite_bus import SQLiteEventBus

        with patch("src.events.sqlite_bus.settings") as s:
            s.ingest_visibility_timeout_s = 300
            b = SQLiteEventBus(db_path=tmp_path / "nested" / "deep" / "events.db")
            try:
                assert (tmp_path / "nested" / "deep").is_dir()
            finally:
                b.close()
