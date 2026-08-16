"""Phase 9.2.1 — Thread-safety tests for all singleton modules.

Verifies that concurrent access to module-level singletons and shared
mutable state does not cause data corruption or race conditions.
"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_THREADS = 10
ITERATIONS_PER_THREAD = 5


def _run_concurrent(fn, n_threads=NUM_THREADS):
    """Run *fn* concurrently from *n_threads* threads, return list of results."""
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(fn) for _ in range(n_threads)]
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as exc:
                errors.append(exc)
    if errors:
        raise errors[0]
    return results


# ---------------------------------------------------------------------------
# 1. MetricsStore singleton (get_store / reset_store)
# ---------------------------------------------------------------------------

class TestMetricsStoreSingleton:
    """Concurrent calls to get_store() must return the same instance."""

    def test_concurrent_get_store_returns_same_instance(self, tmp_path):
        from src.observability.metrics_store import _store_lock, get_store, reset_store

        db_path = str(tmp_path / "metrics.db")
        reset_store()

        instances = _run_concurrent(lambda: get_store(db_path))
        # All threads must get the exact same object
        assert len(set(id(i) for i in instances)) == 1
        reset_store()

    def test_concurrent_record_no_corruption(self, tmp_path):
        """Concurrent record() calls must not lose or corrupt rows."""
        from src.observability.metrics_store import MetricsStore

        db_path = str(tmp_path / "metrics_record.db")
        store = MetricsStore(db_path)

        @dataclass
        class FakeMetrics:
            thread_id: str = "t1"
            question_preview: str = "test?"
            mode: str = "graph"
            retriever_strategy: str = "dense"
            prompt_tokens: int = 10
            completion_tokens: int = 20
            total_tokens: int = 30
            estimated_cost_usd: float = 0.001
            latency_ms: float = 100.0
            is_idk: bool = False
            grader_rejected: int = 0

        total = NUM_THREADS * ITERATIONS_PER_THREAD

        def _record_batch():
            for _ in range(ITERATIONS_PER_THREAD):
                store.record(FakeMetrics())

        _run_concurrent(_record_batch)

        rows = store.query_recent(n=total + 10)
        assert len(rows) == total
        store.close()

    def test_concurrent_record_and_read(self, tmp_path):
        """Interleaved record() and query_recent() must not raise."""
        from src.observability.metrics_store import MetricsStore

        db_path = str(tmp_path / "metrics_rw.db")
        store = MetricsStore(db_path)

        @dataclass
        class FakeMetrics:
            thread_id: str = "t1"
            question_preview: str = "test?"
            mode: str = "graph"
            retriever_strategy: str = "dense"
            prompt_tokens: int = 10
            completion_tokens: int = 20
            total_tokens: int = 30
            estimated_cost_usd: float = 0.001
            latency_ms: float = 100.0
            is_idk: bool = False
            grader_rejected: int = 0

        errors = []

        def _writer():
            try:
                for _ in range(ITERATIONS_PER_THREAD):
                    store.record(FakeMetrics())
            except Exception as exc:
                errors.append(exc)

        def _reader():
            try:
                for _ in range(ITERATIONS_PER_THREAD):
                    store.query_recent(5)
                    store.summary()
            except Exception as exc:
                errors.append(exc)

        threads = []
        for _ in range(NUM_THREADS // 2):
            threads.append(threading.Thread(target=_writer))
            threads.append(threading.Thread(target=_reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        store.close()


# ---------------------------------------------------------------------------
# 2. SemanticCache singleton (get_cache / reset_cache)
# ---------------------------------------------------------------------------

class TestSemanticCacheSingleton:
    """Concurrent calls to get_cache() must return the same instance."""

    def test_concurrent_get_cache_returns_same_instance(self, tmp_path):
        from src.cache.semantic_cache import get_cache, reset_cache

        db_path = str(tmp_path / "cache.db")
        reset_cache()

        instances = _run_concurrent(lambda: get_cache(db_path))
        assert len(set(id(i) for i in instances)) == 1
        reset_cache()

    def test_concurrent_store_no_corruption(self, tmp_path):
        """Concurrent store() calls must not lose rows."""
        from src.cache.semantic_cache import SemanticCache

        db_path = str(tmp_path / "cache_store.db")

        # Use a fake embed_fn that returns a fixed vector
        fake_embed = MagicMock()
        fake_embed.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])

        cache = SemanticCache(db_path, embed_fn=fake_embed)

        total = NUM_THREADS * ITERATIONS_PER_THREAD

        def _store_batch():
            for i in range(ITERATIONS_PER_THREAD):
                tid = threading.current_thread().ident
                cache.store(f"q-{tid}-{i}", f"a-{tid}-{i}")

        with patch.object(type(cache), '_SemanticCache__class__', None, create=True):
            # Enable cache for this test
            with patch("src.cache.semantic_cache.settings") as mock_settings:
                mock_settings.semantic_cache_enabled = True
                mock_settings.semantic_cache_ttl = 3600
                _run_concurrent(_store_batch)

        stats = cache.stats()
        assert stats["total_entries"] == total
        cache.close()


# ---------------------------------------------------------------------------
# 3. BM25 cache (hybrid retriever)
# ---------------------------------------------------------------------------

class TestBM25CacheThreadSafety:
    """Concurrent BM25 cache access must not corrupt shared state."""

    def test_concurrent_reset_no_error(self):
        from src.retrieval.hybrid import _bm25_cache, reset_bm25_cache

        # Just verify concurrent resets don't raise
        _run_concurrent(reset_bm25_cache)
        assert _bm25_cache == {}


# ---------------------------------------------------------------------------
# 4. Node metrics (tracing)
# ---------------------------------------------------------------------------

class TestNodeMetricsThreadSafety:
    """Concurrent metric recording must not lose data."""

    def test_concurrent_metric_append(self):
        from src.graph.tracing import _lock, _node_metrics, reset_node_metrics

        reset_node_metrics()
        total = NUM_THREADS * ITERATIONS_PER_THREAD

        def _append():
            for _ in range(ITERATIONS_PER_THREAD):
                with _lock:
                    _node_metrics.setdefault("test_node", []).append(1.0)

        _run_concurrent(_append)
        assert len(_node_metrics["test_node"]) == total
        reset_node_metrics()

    def test_concurrent_reset_and_append(self):
        """Interleaved reset and append must not raise."""
        from src.graph.tracing import _lock, _node_metrics, reset_node_metrics

        reset_node_metrics()
        errors = []

        def _appender():
            try:
                for _ in range(ITERATIONS_PER_THREAD):
                    with _lock:
                        _node_metrics.setdefault("node_a", []).append(42.0)
            except Exception as exc:
                errors.append(exc)

        def _resetter():
            try:
                for _ in range(ITERATIONS_PER_THREAD):
                    reset_node_metrics()
            except Exception as exc:
                errors.append(exc)

        threads = []
        for _ in range(NUM_THREADS // 2):
            threads.append(threading.Thread(target=_appender))
            threads.append(threading.Thread(target=_resetter))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        reset_node_metrics()


# ---------------------------------------------------------------------------
# 5. Graph singleton (get_graph / reset_graph)
# ---------------------------------------------------------------------------

class TestGraphSingletonThreadSafety:
    """Concurrent get_graph() must return the same compiled graph."""

    @patch("src.graph.build_graph.build_graph")
    def test_concurrent_get_graph_returns_same_instance(self, mock_build):
        from src.graph.build_graph import get_graph, reset_graph

        fake_graph = MagicMock()
        mock_build.return_value = fake_graph
        reset_graph()

        instances = _run_concurrent(get_graph)
        # All threads get the same object
        assert len(set(id(i) for i in instances)) == 1
        # build_graph was called exactly once
        assert mock_build.call_count == 1
        reset_graph()


# ---------------------------------------------------------------------------
# 6. ChromaDB vectorstore singleton
# ---------------------------------------------------------------------------

class TestVectorstoreSingletonThreadSafety:
    """Concurrent get_vectorstore() must return the same instance."""

    @patch("src.vectorstore.chroma_store.Chroma")
    @patch("src.vectorstore.chroma_store._get_embeddings")
    def test_concurrent_get_vectorstore_returns_same_instance(
        self, mock_embed, mock_chroma
    ):
        """Embedded mode: concurrent callers share one store.

        The mode is pinned rather than inherited from the environment. In
        server mode get_vectorstore() opens an HTTP client eagerly, so a
        developer whose .env points at a Chroma server would otherwise see
        this locking test fail on connectivity — testing the wrong thing.
        """
        from src.vectorstore.chroma_store import get_vectorstore, reset_store

        fake_store = MagicMock()
        mock_chroma.return_value = fake_store

        with patch("src.vectorstore.chroma_store.settings") as s:
            s.chroma_mode = "embedded"
            s.chroma_collection = "test_docs"
            s.chroma_dir = "./test_chroma"
            s.chroma_refresh_interval = 300
            reset_store()

            instances = _run_concurrent(get_vectorstore)
            assert len({id(i) for i in instances}) == 1
            # Chroma constructor called exactly once
            assert mock_chroma.call_count == 1
        reset_store()

    @patch("src.vectorstore.chroma_store.Chroma")
    @patch("src.vectorstore.chroma_store._get_embeddings")
    def test_concurrent_get_vectorstore_in_server_mode(self, mock_embed, mock_chroma):
        """Server mode is the production path, so its locking matters too.

        It also must not reconnect per call: the client is never recycled
        on the refresh interval because the server is authoritative.
        """
        from src.vectorstore.chroma_store import get_vectorstore, reset_store

        mock_chroma.return_value = MagicMock()

        with (
            patch("src.vectorstore.chroma_store.settings") as s,
            patch("chromadb.HttpClient", return_value=MagicMock()) as http,
        ):
            s.chroma_mode = "server"
            s.chroma_host = "chroma"
            s.chroma_port = 8000
            s.chroma_ssl = False
            s.chroma_auth_token = ""
            s.chroma_collection = "test_docs"
            s.chroma_refresh_interval = 0  # would recycle in embedded mode
            reset_store()

            instances = _run_concurrent(get_vectorstore)
            assert len({id(i) for i in instances}) == 1
            assert mock_chroma.call_count == 1
            assert http.call_count == 1  # one connection, not one per caller
        reset_store()
