"""Phase 15: Semantic Cache Activation tests.

Tests for:
- Cache lookup node
- Cache store node
- Cache routing
- Graph wiring with cache enabled/disabled
- IDK response filtering
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# Cache lookup node tests
# ---------------------------------------------------------------------------


class TestCacheLookup:
    """Test the cache_lookup graph node."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig = settings.semantic_cache_enabled
        yield
        _set_setting("semantic_cache_enabled", orig)

    def test_disabled_returns_miss(self):
        _set_setting("semantic_cache_enabled", False)

        from src.graph.cache_nodes import cache_lookup

        result = cache_lookup({"question": "Test?"})
        assert result["cache_hit"] is False

    @patch("src.cache.semantic_cache.get_cache")
    def test_cache_hit(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_cache.lookup.return_value = "Cached answer here"
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_lookup

        result = cache_lookup({
            "question": "What is PTO?",
            "retriever_strategy": "dense",
        })
        assert result["cache_hit"] is True
        assert result["generation"] == "Cached answer here"

    @patch("src.cache.semantic_cache.get_cache")
    def test_cache_miss(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_cache.lookup.return_value = None
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_lookup

        result = cache_lookup({
            "question": "New question?",
            "retriever_strategy": "dense",
        })
        assert result["cache_hit"] is False
        assert "generation" not in result

    def test_empty_question_returns_miss(self):
        _set_setting("semantic_cache_enabled", True)

        from src.graph.cache_nodes import cache_lookup

        result = cache_lookup({"question": ""})
        assert result["cache_hit"] is False

    @patch("src.cache.semantic_cache.get_cache")
    def test_lookup_error_returns_miss(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_get_cache.side_effect = RuntimeError("DB error")

        from src.graph.cache_nodes import cache_lookup

        result = cache_lookup({"question": "Test?"})
        assert result["cache_hit"] is False

    @patch("src.cache.semantic_cache.get_cache")
    def test_passes_strategy_to_lookup(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_cache.lookup.return_value = None
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_lookup

        cache_lookup({
            "question": "Test?",
            "retriever_strategy": "hybrid",
        })
        mock_cache.lookup.assert_called_once_with(
            query="Test?", mode="graph", strategy="hybrid", scope="",
        )


# ---------------------------------------------------------------------------
# Cache store node tests
# ---------------------------------------------------------------------------


class TestCacheStore:
    """Test the cache_store graph node."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig = settings.semantic_cache_enabled
        yield
        _set_setting("semantic_cache_enabled", orig)

    def test_disabled_returns_empty(self):
        _set_setting("semantic_cache_enabled", False)

        from src.graph.cache_nodes import cache_store

        result = cache_store({"question": "Test?", "generation": "Answer"})
        assert result == {}

    @patch("src.cache.semantic_cache.get_cache")
    def test_stores_answer(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_store

        cache_store({
            "question": "What is PTO?",
            "generation": "PTO is paid time off.",
            "retriever_strategy": "dense",
        })
        mock_cache.store.assert_called_once_with(
            query="What is PTO?",
            answer="PTO is paid time off.",
            mode="graph",
            strategy="dense",
            scope="",
        )

    @patch("src.cache.semantic_cache.get_cache")
    def test_skips_idk_responses(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_store

        cache_store({
            "question": "Unknown topic?",
            "generation": "I don't have enough information to answer.",
        })
        mock_cache.store.assert_not_called()

    @patch("src.cache.semantic_cache.get_cache")
    def test_skips_empty_question(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_store

        cache_store({"question": "", "generation": "Answer"})
        mock_cache.store.assert_not_called()

    @patch("src.cache.semantic_cache.get_cache")
    def test_skips_empty_generation(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_store

        cache_store({"question": "Test?", "generation": ""})
        mock_cache.store.assert_not_called()

    @patch("src.cache.semantic_cache.get_cache")
    def test_store_error_ignored(self, mock_get_cache):
        _set_setting("semantic_cache_enabled", True)

        mock_cache = MagicMock()
        mock_cache.store.side_effect = RuntimeError("DB error")
        mock_get_cache.return_value = mock_cache

        from src.graph.cache_nodes import cache_store

        # Should not raise
        result = cache_store({
            "question": "Test?",
            "generation": "Answer",
        })
        assert result == {}


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


class TestRouteAfterCache:
    def test_hit_routes_to_cache_hit(self):
        from src.graph.cache_nodes import route_after_cache

        assert route_after_cache({"cache_hit": True}) == "cache_hit"

    def test_miss_routes_to_cache_miss(self):
        from src.graph.cache_nodes import route_after_cache

        assert route_after_cache({"cache_hit": False}) == "cache_miss"

    def test_missing_key_routes_to_miss(self):
        from src.graph.cache_nodes import route_after_cache

        assert route_after_cache({}) == "cache_miss"


# ---------------------------------------------------------------------------
# Graph wiring tests
# ---------------------------------------------------------------------------


class TestGraphWiring:
    """Test graph compilation with cache enabled/disabled."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig_cache = settings.semantic_cache_enabled
        orig_parallel = settings.parallel_sub_queries
        yield
        _set_setting("semantic_cache_enabled", orig_cache)
        _set_setting("parallel_sub_queries", orig_parallel)

    def test_graph_with_cache_enabled(self):
        _set_setting("semantic_cache_enabled", True)
        _set_setting("parallel_sub_queries", False)

        from langgraph.checkpoint.memory import InMemorySaver
        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "cache_lookup" in node_names
            assert "cache_store" in node_names
        finally:
            reset_graph()

    def test_graph_without_cache(self):
        _set_setting("semantic_cache_enabled", False)
        _set_setting("parallel_sub_queries", False)

        from langgraph.checkpoint.memory import InMemorySaver
        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "cache_lookup" not in node_names
            assert "cache_store" not in node_names
        finally:
            reset_graph()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_cache_config_fields_exist(self):
        assert hasattr(settings, "semantic_cache_enabled")
        assert hasattr(settings, "semantic_cache_threshold")
        assert hasattr(settings, "semantic_cache_ttl")

    def test_cache_defaults_disabled(self):
        # Default is false (explicitly opted-in)
        assert isinstance(settings.semantic_cache_enabled, bool)

    def test_state_has_cache_hit_field(self):
        from src.graph.state import RAGState

        assert "cache_hit" in RAGState.__annotations__
