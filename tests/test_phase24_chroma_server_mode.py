"""Phase 24 — ChromaDB server mode and cross-process index freshness.

Regression suite for a bug found by running the app: with
ASYNC_INGESTION=true the worker indexes in its own process, and embedded
ChromaDB caches its index per process — so the API served answers from a
corpus that predated every uploaded document, indefinitely, until restart.

Two independent staleness sources had to be closed, and both are pinned
here:

  dense   embedded mode cannot see another process's writes at all;
          server mode routes every read through one server.
  sparse  the BM25 index is an in-memory copy of the corpus whose
          invalidation hook only fires in the writing process.
"""
from __future__ import annotations

import dataclasses
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import src.retrieval.hybrid as hybrid
import src.vectorstore.chroma_store as cs
from config import Settings


@pytest.fixture(autouse=True)
def _reset():
    cs.reset_store()
    hybrid.reset_bm25_cache()
    yield
    cs.reset_store()
    hybrid.reset_bm25_cache()


# ---------------------------------------------------------------------------
# Configuration guards
# ---------------------------------------------------------------------------

class TestChromaModeConfig:
    def _settings(self, **kw):
        return dataclasses.replace(Settings(), openai_api_key="sk-test", **kw)

    def test_embedded_is_the_shipped_default(self):
        """Guard the default in the source, not in this machine's environment.

        `Settings()` reads the environment, and config.py always loads the
        project's .env — so a developer who sets CHROMA_MODE=server locally
        would otherwise "fail" this test. What matters is that the value
        shipped to someone with no configuration stays embedded, so that a
        single-process install works with no server to run.
        """
        import inspect
        import re

        source = inspect.getsource(Settings)
        match = re.search(
            r'chroma_mode:\s*str\s*=\s*os\.getenv\(\s*"CHROMA_MODE"\s*,\s*"([^"]+)"',
            source,
        )
        assert match, "chroma_mode default not found in config.Settings"
        assert match.group(1) == "embedded"

    def test_valid_modes_pass(self):
        self._settings(chroma_mode="embedded").validate()
        self._settings(chroma_mode="server", chroma_host="chroma").validate()

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid CHROMA_MODE"):
            self._settings(chroma_mode="clustered").validate()

    def test_server_mode_requires_a_host(self):
        with pytest.raises(ValueError, match="requires CHROMA_HOST"):
            self._settings(chroma_mode="server", chroma_host="").validate()

    def test_async_ingestion_on_embedded_chroma_warns(self, caplog):
        """The exact misconfiguration that produced the bug."""
        import logging

        with caplog.at_level(logging.WARNING, logger="config"):
            self._settings(async_ingestion=True, chroma_mode="embedded").validate()
        assert any("CHROMA_MODE=server" in r.message for r in caplog.records)

    def test_async_ingestion_on_server_chroma_is_silent(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="config"):
            self._settings(
                async_ingestion=True, chroma_mode="server", chroma_host="chroma"
            ).validate()
        assert not any("CHROMA_MODE=server" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

class TestServerModeClient:
    @pytest.fixture
    def server_settings(self):
        with patch("src.vectorstore.chroma_store.settings") as s:
            s.chroma_mode = "server"
            s.chroma_host = "chroma"
            s.chroma_port = 8000
            s.chroma_ssl = False
            s.chroma_auth_token = ""
            s.chroma_collection = "enterprise_docs"
            s.chroma_dir = "./chroma_db"
            s.chroma_refresh_interval = 300
            s.embedding_model = "text-embedding-3-small"
            s.openai_api_key = "sk-test"
            yield s

    def test_http_client_targets_the_configured_server(self, server_settings):
        with patch("chromadb.HttpClient") as http:
            cs._build_http_client()
        assert http.call_args.kwargs["host"] == "chroma"
        assert http.call_args.kwargs["port"] == 8000
        assert http.call_args.kwargs["ssl"] is False

    def test_ssl_is_honoured(self, server_settings):
        server_settings.chroma_ssl = True
        with patch("chromadb.HttpClient") as http:
            cs._build_http_client()
        assert http.call_args.kwargs["ssl"] is True

    def test_auth_token_is_attached_when_set(self, server_settings):
        server_settings.chroma_auth_token = "secret-token"
        with patch("chromadb.HttpClient") as http:
            cs._build_http_client()
        assert "settings" in http.call_args.kwargs

    def test_no_auth_settings_when_token_is_absent(self, server_settings):
        with patch("chromadb.HttpClient") as http:
            cs._build_http_client()
        assert "settings" not in http.call_args.kwargs

    def test_server_mode_passes_a_client_not_a_directory(self, server_settings):
        """persist_directory would silently re-create the embedded store."""
        with (
            patch("chromadb.HttpClient", return_value=MagicMock()),
            patch("src.vectorstore.chroma_store.Chroma") as chroma,
            patch("src.vectorstore.chroma_store.build_embeddings"),
        ):
            cs.get_vectorstore()
        kwargs = chroma.call_args.kwargs
        assert "client" in kwargs
        assert "persist_directory" not in kwargs

    def test_embedded_mode_passes_a_directory_not_a_client(self, server_settings):
        server_settings.chroma_mode = "embedded"
        with (
            patch("src.vectorstore.chroma_store.Chroma") as chroma,
            patch("src.vectorstore.chroma_store.build_embeddings"),
        ):
            cs.get_vectorstore()
        kwargs = chroma.call_args.kwargs
        assert kwargs["persist_directory"] == "./chroma_db"
        assert "client" not in kwargs

    def test_server_client_is_not_recycled_by_the_refresh_interval(self, server_settings):
        """The server is authoritative — reconnecting buys nothing."""
        server_settings.chroma_refresh_interval = 0  # would recycle in embedded mode
        with (
            patch("chromadb.HttpClient", return_value=MagicMock()),
            patch("src.vectorstore.chroma_store.Chroma") as chroma,
            patch("src.vectorstore.chroma_store.build_embeddings"),
        ):
            cs.get_vectorstore()
            cs.get_vectorstore()
            cs.get_vectorstore()
        chroma.assert_called_once()

    def test_embedded_store_is_still_recycled_on_interval(self, server_settings):
        server_settings.chroma_mode = "embedded"
        server_settings.chroma_refresh_interval = 0
        with (
            patch("src.vectorstore.chroma_store.Chroma") as chroma,
            patch("src.vectorstore.chroma_store.build_embeddings"),
        ):
            cs.get_vectorstore()
            cs.get_vectorstore()
        assert chroma.call_count == 2


# ---------------------------------------------------------------------------
# BM25 staleness
# ---------------------------------------------------------------------------

class TestBm25Staleness:
    @pytest.fixture
    def bm25_settings(self):
        with patch("src.retrieval.hybrid.settings") as s:
            s.bm25_cache_ttl = 300
            s.rrf_k = 60
            yield s

    def _entry(self, built_at=None, source_count=10):
        return hybrid._CachedIndex(
            bm25=MagicMock(),
            docs=[Document(page_content="cached")],
            built_at=built_at if built_at is not None else time.monotonic(),
            source_count=source_count,
        )

    def test_unchanged_corpus_serves_the_cache(self, bm25_settings):
        with patch("src.retrieval.hybrid._collection_count", return_value=10):
            assert hybrid._is_stale(self._entry(source_count=10)) is False

    def test_growing_corpus_invalidates_immediately(self, bm25_settings):
        """A worker indexing a document in another process must be noticed."""
        with patch("src.retrieval.hybrid._collection_count", return_value=13):
            assert hybrid._is_stale(self._entry(source_count=10)) is True

    def test_shrinking_corpus_invalidates(self, bm25_settings):
        with patch("src.retrieval.hybrid._collection_count", return_value=4):
            assert hybrid._is_stale(self._entry(source_count=10)) is True

    def test_expired_ttl_invalidates_even_at_the_same_count(self, bm25_settings):
        """Backstop for an add and a delete that cancel out."""
        bm25_settings.bm25_cache_ttl = 60
        old = self._entry(built_at=time.monotonic() - 3600, source_count=10)
        with patch("src.retrieval.hybrid._collection_count", return_value=10):
            assert hybrid._is_stale(old) is True

    def test_unreadable_count_falls_back_to_the_ttl(self, bm25_settings):
        with patch("src.retrieval.hybrid._collection_count", return_value=-1):
            assert hybrid._is_stale(self._entry()) is False

    def test_collection_count_failure_is_not_fatal(self, bm25_settings):
        with patch(
            "src.vectorstore.chroma_store.get_vectorstore",
            side_effect=RuntimeError("chroma unreachable"),
        ):
            assert hybrid._collection_count() == -1

    def test_stale_entry_triggers_a_rebuild(self, bm25_settings):
        """End-to-end: a stale cache must not be served."""
        retriever = hybrid.HybridRetriever(k=2)
        store = MagicMock()
        store._collection.get.return_value = {
            "ids": ["1", "2"],
            "documents": ["alpha document", "beta document"],
            "metadatas": [{}, {}],
        }
        store._collection.count.return_value = 2

        hybrid._bm25_cache["__no_filter__"] = self._entry(source_count=99)
        with patch("src.vectorstore.chroma_store.get_vectorstore", return_value=store):
            retriever._build_bm25_index()

        assert [d.page_content for d in retriever._corpus_docs] == [
            "alpha document", "beta document",
        ]

    def test_fresh_entry_is_reused_without_a_rebuild(self, bm25_settings):
        retriever = hybrid.HybridRetriever(k=2)
        hybrid._bm25_cache["__no_filter__"] = self._entry(source_count=7)
        store = MagicMock()
        with (
            patch("src.retrieval.hybrid._collection_count", return_value=7),
            patch("src.vectorstore.chroma_store.get_vectorstore", return_value=store),
        ):
            retriever._build_bm25_index()
        store._collection.get.assert_not_called()
        assert retriever._corpus_docs[0].page_content == "cached"

    def test_rebuild_records_the_whole_collection_size(self, bm25_settings):
        """Filtered corpora must still compare against the full count."""
        retriever = hybrid.HybridRetriever(k=2, filter={"department": "hr"})
        store = MagicMock()
        store._collection.get.return_value = {
            "ids": ["1", "2", "3"],
            "documents": ["remote work policy", "vendor payment terms", "leave entitlement"],
            "metadatas": [
                {"department": "hr"}, {"department": "legal"}, {"department": "hr"},
            ],
        }
        with patch("src.vectorstore.chroma_store.get_vectorstore", return_value=store):
            retriever._build_bm25_index()

        entry = hybrid._bm25_cache[retriever._filter_cache_key({"department": "hr"})]
        assert len(entry.docs) == 2      # filtered corpus
        assert entry.source_count == 3   # whole collection

    def test_corpus_with_no_indexable_terms_does_not_crash(self, bm25_settings):
        """All-stop-word documents would divide by zero inside BM25Okapi."""
        retriever = hybrid.HybridRetriever(k=2)
        store = MagicMock()
        store._collection.get.return_value = {
            "ids": ["1", "2"],
            "documents": ["the a of", "and to it"],
            "metadatas": [{}, {}],
        }
        with patch("src.vectorstore.chroma_store.get_vectorstore", return_value=store):
            retriever._build_bm25_index()
        assert retriever._bm25 is None  # degrades to dense-only

    def test_no_indexable_terms_yields_no_sparse_results(self, bm25_settings):
        retriever = hybrid.HybridRetriever(k=2)
        store = MagicMock()
        store._collection.get.return_value = {
            "ids": ["1"], "documents": ["the a of"], "metadatas": [{}],
        }
        with patch("src.vectorstore.chroma_store.get_vectorstore", return_value=store):
            assert retriever._get_bm25_results("anything", 5) == []

    def test_reset_still_clears_everything(self, bm25_settings):
        hybrid._bm25_cache["__no_filter__"] = self._entry()
        hybrid.reset_bm25_cache()
        assert hybrid._bm25_cache == {}
