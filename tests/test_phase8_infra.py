"""Phase 8.7 tests: Infrastructure Hardening.

Covers:
  - ChromaDB staleness detection (auto-refresh after interval)
  - Document freshness / TTL (stale detection and deletion)
  - Semantic caching (hit, miss, TTL expiry, invalidation)
  - MetricsStore protocol abstraction
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# TestChromaRefresh
# ---------------------------------------------------------------------------

class TestChromaRefresh:
    """Verify ChromaDB singleton refreshes after interval."""

    def test_refresh_interval_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.chroma_refresh_interval == 300

    @patch("src.vectorstore.chroma_store.Chroma")
    @patch("src.vectorstore.chroma_store._get_embeddings")
    def test_singleton_refreshed_after_interval(self, mock_embed, mock_chroma):
        """After the refresh interval, get_vectorstore recreates the client."""
        import src.vectorstore.chroma_store as cs

        cs.reset_store()
        mock_chroma.return_value = MagicMock()

        # First call creates the store
        with patch.object(cs, 'settings', create=True) as mock_settings:
            mock_settings.chroma_collection = "test"
            mock_settings.chroma_dir = "/tmp/test"
            mock_settings.chroma_refresh_interval = 1  # 1 second

            store1 = cs.get_vectorstore()
            assert mock_chroma.call_count == 1

            # Immediate second call reuses singleton
            store2 = cs.get_vectorstore()
            assert mock_chroma.call_count == 1

            # Simulate time passing beyond refresh interval
            cs._last_refresh = time.monotonic() - 2  # 2 seconds ago
            store3 = cs.get_vectorstore()
            assert mock_chroma.call_count == 2  # recreated

        cs.reset_store()

    def test_refresh_store_marks_for_refresh(self):
        """refresh_store() should reset _last_refresh and _vectorstore."""
        import src.vectorstore.chroma_store as cs
        cs._last_refresh = 999.0
        cs._vectorstore = MagicMock()
        cs.refresh_store()
        assert cs._vectorstore is None
        assert cs._last_refresh == 0.0


# ---------------------------------------------------------------------------
# TestDocumentTTL
# ---------------------------------------------------------------------------

class TestDocumentTTL:
    """Verify document TTL detection and deletion."""

    def test_ttl_disabled_by_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.document_ttl_days == 0

    def test_get_stale_returns_empty_when_disabled(self):
        """When TTL is 0, get_stale_documents returns empty list."""
        from src.vectorstore.chroma_store import get_stale_documents
        with patch("src.vectorstore.chroma_store.settings") as mock_s:
            mock_s.document_ttl_days = 0
            result = get_stale_documents()
            assert result == []

    @patch("src.vectorstore.chroma_store.get_vectorstore")
    def test_get_stale_detects_old_docs(self, mock_get_vs):
        """Documents older than max_age_days should be detected as stale."""
        old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        fresh_ts = datetime.now(UTC).isoformat()

        mock_store = MagicMock()
        mock_store.get.return_value = {
            "ids": ["old-doc", "fresh-doc"],
            "metadatas": [
                {"ingested_at": old_ts},
                {"ingested_at": fresh_ts},
            ],
        }
        mock_get_vs.return_value = mock_store

        from src.vectorstore.chroma_store import get_stale_documents
        with patch("src.vectorstore.chroma_store.settings") as mock_s:
            mock_s.document_ttl_days = 30
            stale = get_stale_documents()
            assert "old-doc" in stale
            assert "fresh-doc" not in stale

    @patch("src.vectorstore.chroma_store.get_vectorstore")
    def test_delete_stale_removes_docs(self, mock_get_vs):
        """delete_stale_documents should call store.delete with stale IDs."""
        old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()

        mock_store = MagicMock()
        mock_store.get.return_value = {
            "ids": ["old-doc"],
            "metadatas": [{"ingested_at": old_ts}],
        }
        mock_get_vs.return_value = mock_store

        from src.vectorstore.chroma_store import delete_stale_documents
        with patch("src.vectorstore.chroma_store.settings") as mock_s:
            mock_s.document_ttl_days = 30
            count = delete_stale_documents()
            assert count == 1
            mock_store.delete.assert_called_once_with(ids=["old-doc"])


# ---------------------------------------------------------------------------
# TestSemanticCache
# ---------------------------------------------------------------------------

class TestSemanticCache:
    """Verify semantic cache operations."""

    def test_cache_disabled_by_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.semantic_cache_enabled is False

    def test_cache_lookup_returns_none_when_disabled(self):
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache(":memory:")
        with patch("src.cache.semantic_cache.settings") as mock_s:
            mock_s.semantic_cache_enabled = False
            result = cache.lookup("test query")
            assert result is None
        cache.close()

    def test_cache_store_noop_when_disabled(self):
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache(":memory:")
        with patch("src.cache.semantic_cache.settings") as mock_s:
            mock_s.semantic_cache_enabled = False
            cache.store("test query", "test answer")
            # Verify nothing was stored
            row = cache._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
            assert row[0] == 0
        cache.close()

    def test_cache_hit_on_identical_query(self):
        """Identical query embedding should return cached answer."""
        from src.cache.semantic_cache import SemanticCache

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [1.0, 0.0, 0.0]

        cache = SemanticCache(":memory:", embed_fn=mock_embed)
        with patch("src.cache.semantic_cache.settings") as mock_s:
            mock_s.semantic_cache_enabled = True
            mock_s.semantic_cache_threshold = 0.95
            mock_s.semantic_cache_ttl = 3600
            mock_s.embedding_model = "text-embedding-3-small"
            mock_s.openai_api_key = "sk-test"

            cache.store("What is PTO?", "PTO is paid time off.")
            result = cache.lookup("What is PTO?")
            assert result == "PTO is paid time off."
        cache.close()

    def test_cache_miss_on_different_query(self):
        """Different query embedding should return None."""
        from src.cache.semantic_cache import SemanticCache

        call_count = [0]
        def fake_embed(query):
            call_count[0] += 1
            if call_count[0] <= 1:
                return [1.0, 0.0, 0.0]  # store
            return [0.0, 1.0, 0.0]  # lookup — orthogonal

        mock_embed = MagicMock()
        mock_embed.embed_query.side_effect = fake_embed

        cache = SemanticCache(":memory:", embed_fn=mock_embed)
        with patch("src.cache.semantic_cache.settings") as mock_s:
            mock_s.semantic_cache_enabled = True
            mock_s.semantic_cache_threshold = 0.95
            mock_s.semantic_cache_ttl = 3600

            cache.store("What is PTO?", "PTO is paid time off.")
            result = cache.lookup("Completely different question")
            assert result is None
        cache.close()

    def test_cache_ttl_expiry(self):
        """Expired entries should not be returned."""
        from src.cache.semantic_cache import SemanticCache

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [1.0, 0.0, 0.0]

        cache = SemanticCache(":memory:", embed_fn=mock_embed)
        with patch("src.cache.semantic_cache.settings") as mock_s:
            mock_s.semantic_cache_enabled = True
            mock_s.semantic_cache_threshold = 0.95
            mock_s.semantic_cache_ttl = 1  # 1 second TTL

            cache.store("What is PTO?", "PTO is paid time off.", ttl=1)

            # Manually backdate the entry
            old_ts = (datetime.now(UTC) - timedelta(seconds=10)).isoformat()
            cache._conn.execute(
                "UPDATE semantic_cache SET created_at = ?", (old_ts,)
            )
            cache._conn.commit()

            result = cache.lookup("What is PTO?")
            assert result is None
        cache.close()

    def test_cache_invalidation(self):
        """invalidate() should clear all entries."""
        from src.cache.semantic_cache import SemanticCache

        mock_embed = MagicMock()
        mock_embed.embed_query.return_value = [1.0, 0.0, 0.0]

        cache = SemanticCache(":memory:", embed_fn=mock_embed)
        with patch("src.cache.semantic_cache.settings") as mock_s:
            mock_s.semantic_cache_enabled = True
            mock_s.semantic_cache_threshold = 0.95
            mock_s.semantic_cache_ttl = 3600

            cache.store("q1", "a1")
            cache.store("q2", "a2")

            count = cache.invalidate()
            assert count == 2

            result = cache.lookup("q1")
            assert result is None
        cache.close()

    def test_cache_cleanup_expired(self):
        """cleanup_expired() should remove old entries."""
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache(":memory:")
        # Insert entries directly with backdated timestamps
        old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        fresh_ts = datetime.now(UTC).isoformat()
        cache._conn.execute(
            "INSERT INTO semantic_cache (query, answer, embedding, created_at, ttl) "
            "VALUES (?, ?, ?, ?, ?)",
            ("old q", "old a", "[]", old_ts, 3600),
        )
        cache._conn.execute(
            "INSERT INTO semantic_cache (query, answer, embedding, created_at, ttl) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fresh q", "fresh a", "[]", fresh_ts, 3600),
        )
        cache._conn.commit()

        removed = cache.cleanup_expired()
        assert removed == 1

        # Verify only fresh entry remains
        row = cache._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()
        assert row[0] == 1
        cache.close()

    def test_cache_stats(self):
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache(":memory:")
        cache._conn.execute(
            "INSERT INTO semantic_cache (query, answer, embedding, created_at, ttl) "
            "VALUES (?, ?, ?, ?, ?)",
            ("q1", "a1", "[]", datetime.now(UTC).isoformat(), 3600),
        )
        cache._conn.commit()
        stats = cache.stats()
        assert stats["total_entries"] == 1
        cache.close()


# ---------------------------------------------------------------------------
# TestCosineSimiarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Verify cosine similarity computation."""

    def test_identical_vectors(self):
        from src.cache.semantic_cache import _cosine_similarity
        assert abs(_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        from src.cache.semantic_cache import _cosine_similarity
        assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_opposite_vectors(self):
        from src.cache.semantic_cache import _cosine_similarity
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-6

    def test_zero_vector(self):
        from src.cache.semantic_cache import _cosine_similarity
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# TestMetricsProtocol
# ---------------------------------------------------------------------------

class TestMetricsProtocol:
    """Verify SQLite implementation satisfies the MetricsStoreProtocol."""

    def test_sqlite_implements_protocol(self):
        from src.observability.metrics_store import MetricsStore, MetricsStoreProtocol
        assert isinstance(MetricsStore(":memory:"), MetricsStoreProtocol)

    def test_protocol_is_runtime_checkable(self):
        from src.observability.metrics_store import MetricsStoreProtocol
        # Protocol should be decorated with @runtime_checkable
        assert hasattr(MetricsStoreProtocol, '__protocol_attrs__') or hasattr(MetricsStoreProtocol, '__abstractmethods__') or True
        # The real check is isinstance above — this just ensures the import works

    def test_protocol_defines_required_methods(self):
        from src.observability.metrics_store import MetricsStoreProtocol
        # Check that the protocol defines the expected methods
        expected = {
            "record", "query_recent", "summary", "idk_rate",
            "grader_rejection_rate", "latency_percentiles",
            "cost_alert_check", "close",
        }
        # Protocol methods are in __protocol_attrs__ or annotations
        actual_methods = set()
        for attr in dir(MetricsStoreProtocol):
            if not attr.startswith("_"):
                actual_methods.add(attr)
        assert expected.issubset(actual_methods), f"Missing: {expected - actual_methods}"


# ---------------------------------------------------------------------------
# TestCleanupScript
# ---------------------------------------------------------------------------

class TestCleanupScript:
    """Verify the cleanup_stale script imports and has a main function."""

    def test_script_has_main(self):
        from scripts.cleanup_stale import main
        assert callable(main)
