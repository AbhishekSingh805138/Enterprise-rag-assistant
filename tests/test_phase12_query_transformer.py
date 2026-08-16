"""Phase 12: Query Transformer Pipeline tests.

Tests for:
- Entity extraction (regex fallback)
- Query transformation pipeline
- Query transform graph node
- Config flag bypass
- Integration with intent detection
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# Entity extraction tests (regex fallback)
# ---------------------------------------------------------------------------


class TestRegexEntityExtraction:
    """Test regex-based entity extraction fallback."""

    def test_extracts_departments(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("What is the engineering onboarding process?")
        names = [e.name.lower() for e in entities]
        assert "engineering" in names
        types = {e.name.lower(): e.type for e in entities}
        assert types["engineering"] == "department"

    def test_extracts_multiple_departments(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("Compare HR and finance policies")
        dept_names = [e.name.lower() for e in entities if e.type == "department"]
        assert "hr" in dept_names
        assert "finance" in dept_names

    def test_extracts_documents(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("Where is the employee handbook?")
        names = [e.name.lower() for e in entities]
        assert "handbook" in names

    def test_extracts_roles(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("What do contractors need to sign?")
        names = [e.name.lower() for e in entities]
        assert "contractor" in [n.rstrip("s") for n in names] or "contractors" in names

    def test_extracts_dates(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("What changed in Q3 2024?")
        names = [e.name for e in entities]
        assert "Q3" in names
        assert "2024" in names

    def test_deduplicates_entities(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("HR said HR policy is in the HR handbook")
        hr_entities = [e for e in entities if e.name.lower() == "hr" and e.type == "department"]
        assert len(hr_entities) == 1  # should not duplicate

    def test_empty_query(self):
        from src.retrieval.entity_extractor import _regex_extract

        assert _regex_extract("") == []

    def test_no_entities(self):
        from src.retrieval.entity_extractor import _regex_extract

        entities = _regex_extract("What is the meaning of life?")
        # May or may not find entities — just shouldn't crash
        assert isinstance(entities, list)


class TestExtractEntities:
    """Test the full extract_entities function with LLM fallback."""

    def test_empty_query_returns_empty(self):
        from src.retrieval.entity_extractor import extract_entities

        assert extract_entities("") == []
        assert extract_entities("   ") == []

    @patch("src.retrieval.entity_extractor.get_breaker")
    def test_falls_back_to_regex_on_llm_failure(self, mock_breaker):
        """Should use regex when LLM fails."""
        mock_cb = MagicMock()
        mock_breaker.return_value = mock_cb
        mock_cb.call.side_effect = RuntimeError("LLM down")

        from src.retrieval.entity_extractor import extract_entities

        entities = extract_entities("What is the engineering handbook?")
        # Should still find entities via regex
        names = [e.name.lower() for e in entities]
        assert "engineering" in names or "handbook" in names


# ---------------------------------------------------------------------------
# Query transformation pipeline tests
# ---------------------------------------------------------------------------


class TestTransformQuery:
    """Test the query transformation pipeline."""

    @patch("src.retrieval.query_transformer.extract_entities")
    @patch("src.retrieval.query_transformer._intent_rewrite")
    def test_basic_transform(self, mock_rewrite, mock_extract):
        """Basic transform: normalize + no rewrite + no entities."""
        mock_rewrite.return_value = ""
        mock_extract.return_value = []

        from src.retrieval.query_transformer import transform_query

        result = transform_query("What is the  PTO  policy?")
        assert result.original == "What is the  PTO  policy?"
        assert "paid time off" in result.normalized  # acronym expanded
        assert result.expanded == result.normalized  # no rewrite, no extra entities
        assert result.entities == []
        assert result.rewritten == ""

    @patch("src.retrieval.query_transformer.extract_entities")
    @patch("src.retrieval.query_transformer._intent_rewrite")
    def test_transform_with_entities(self, mock_rewrite, mock_extract):
        """Entities not in base query should be appended."""
        from src.retrieval.entity_extractor import Entity

        mock_rewrite.return_value = ""
        mock_extract.return_value = [
            Entity(name="Engineering", type="department"),
            Entity(name="handbook", type="document"),
        ]

        from src.retrieval.query_transformer import transform_query

        # "handbook" is already in the query, but "Engineering" is not
        result = transform_query("Where is the handbook?")
        assert "Engineering" in result.expanded
        assert len(result.entities) == 2

    @patch("src.retrieval.query_transformer.extract_entities")
    @patch("src.retrieval.query_transformer._intent_rewrite")
    def test_transform_with_rewrite(self, mock_rewrite, mock_extract):
        """Intent-aware rewrite should be used as the base."""
        mock_rewrite.return_value = "What are the steps for the engineering onboarding process"
        mock_extract.return_value = []

        from src.retrieval.query_transformer import transform_query

        result = transform_query("How does engineering onboard?", intent="procedural")
        assert result.rewritten == mock_rewrite.return_value
        assert result.expanded == mock_rewrite.return_value

    @patch("src.retrieval.query_transformer.extract_entities")
    @patch("src.retrieval.query_transformer._intent_rewrite")
    def test_empty_query(self, mock_rewrite, mock_extract):
        """Empty query should pass through."""
        from src.retrieval.query_transformer import transform_query

        result = transform_query("")
        assert result.original == ""
        assert result.expanded == ""
        mock_rewrite.assert_not_called()
        mock_extract.assert_not_called()

    @patch("src.retrieval.query_transformer.extract_entities")
    @patch("src.retrieval.query_transformer._intent_rewrite")
    def test_informational_no_rewrite(self, mock_rewrite, mock_extract):
        """Informational intent should not trigger rewrite."""
        mock_rewrite.return_value = ""
        mock_extract.return_value = []

        from src.retrieval.query_transformer import transform_query

        result = transform_query("What is the PTO policy?", intent="informational")
        # _intent_rewrite is called but returns "" for informational
        assert result.rewritten == ""

    @patch("src.retrieval.query_transformer.extract_entities")
    @patch("src.retrieval.query_transformer._intent_rewrite")
    def test_entities_already_in_query(self, mock_rewrite, mock_extract):
        """Entities already in the query should not be duplicated."""
        from src.retrieval.entity_extractor import Entity

        mock_rewrite.return_value = ""
        mock_extract.return_value = [
            Entity(name="engineering", type="department"),
        ]

        from src.retrieval.query_transformer import transform_query

        result = transform_query("What is the engineering policy?")
        # "engineering" is already in the normalized query
        assert result.expanded == result.normalized


# ---------------------------------------------------------------------------
# Intent rewrite tests
# ---------------------------------------------------------------------------


class TestIntentRewrite:
    """Test intent-aware query rewriting."""

    def test_no_rewrite_for_informational(self):
        from src.retrieval.query_transformer import _intent_rewrite

        result = _intent_rewrite("What is PTO?", "informational")
        assert result == ""

    def test_no_rewrite_for_factual(self):
        from src.retrieval.query_transformer import _intent_rewrite

        result = _intent_rewrite("When was the last report?", "factual")
        assert result == ""

    def test_comparative_has_prompt(self):
        """Comparative intent should have a rewrite prompt defined."""
        from src.retrieval.query_transformer import _REWRITE_PROMPTS

        assert "comparative" in _REWRITE_PROMPTS

    def test_procedural_has_prompt(self):
        from src.retrieval.query_transformer import _REWRITE_PROMPTS

        assert "procedural" in _REWRITE_PROMPTS

    def test_analytical_has_prompt(self):
        from src.retrieval.query_transformer import _REWRITE_PROMPTS

        assert "analytical" in _REWRITE_PROMPTS

    def test_multi_hop_has_prompt(self):
        from src.retrieval.query_transformer import _REWRITE_PROMPTS

        assert "multi_hop" in _REWRITE_PROMPTS


# ---------------------------------------------------------------------------
# Graph node tests
# ---------------------------------------------------------------------------


class TestQueryTransformNode:
    """Test the query_transform graph node."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig = settings.query_transform_enabled
        yield
        _set_setting("query_transform_enabled", orig)

    @patch("src.retrieval.query_transformer.transform_query")
    def test_node_returns_transformed_query(self, mock_transform):
        """Node should return transformed_query and extracted_entities."""
        from src.retrieval.query_transformer import TransformedQuery

        _set_setting("query_transform_enabled", True)

        mock_transform.return_value = TransformedQuery(
            original="What is the PTO policy?",
            normalized="What is the PTO (paid time off) policy",
            expanded="What is the PTO (paid time off) policy",
            entities=[],
            rewritten="",
        )

        from src.retrieval.query_transformer import query_transform_node

        state = {"question": "What is the PTO policy?", "intent": "informational"}
        result = query_transform_node(state)

        assert "transformed_query" in result
        assert "extracted_entities" in result
        assert isinstance(result["extracted_entities"], list)

    def test_node_disabled_returns_empty(self):
        """When disabled, node should return empty dict."""
        _set_setting("query_transform_enabled", False)

        from src.retrieval.query_transformer import query_transform_node

        state = {"question": "Test?", "intent": "informational"}
        result = query_transform_node(state)
        assert result == {}

    @patch("src.retrieval.query_transformer.transform_query")
    def test_node_uses_intent_from_state(self, mock_transform):
        """Node should pass intent from state to transform_query."""
        from src.retrieval.query_transformer import TransformedQuery

        _set_setting("query_transform_enabled", True)

        mock_transform.return_value = TransformedQuery(
            original="q", normalized="q", expanded="q",
        )

        from src.retrieval.query_transformer import query_transform_node

        state = {"question": "Compare X and Y", "intent": "comparative"}
        query_transform_node(state)

        mock_transform.assert_called_once_with("Compare X and Y", "comparative")

    @patch("src.retrieval.query_transformer.transform_query")
    def test_node_defaults_to_informational(self, mock_transform):
        """Node should default to informational if no intent in state."""
        from src.retrieval.query_transformer import TransformedQuery

        _set_setting("query_transform_enabled", True)

        mock_transform.return_value = TransformedQuery(
            original="q", normalized="q", expanded="q",
        )

        from src.retrieval.query_transformer import query_transform_node

        state = {"question": "Test query"}
        query_transform_node(state)

        mock_transform.assert_called_once_with("Test query", "informational")


# ---------------------------------------------------------------------------
# Graph wiring tests
# ---------------------------------------------------------------------------


class TestGraphWiring:
    """Test that query_transform is wired into the graph correctly."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig = settings.query_transform_enabled
        orig_parallel = settings.parallel_sub_queries
        yield
        _set_setting("query_transform_enabled", orig)
        _set_setting("parallel_sub_queries", orig_parallel)

    def test_graph_includes_query_transform_when_enabled(self):
        _set_setting("query_transform_enabled", True)
        _set_setting("parallel_sub_queries", False)

        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "query_transform" in node_names
        finally:
            reset_graph()

    def test_graph_excludes_query_transform_when_disabled(self):
        _set_setting("query_transform_enabled", False)
        _set_setting("parallel_sub_queries", False)

        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "query_transform" not in node_names
        finally:
            reset_graph()


# ---------------------------------------------------------------------------
# Retrieve node integration tests
# ---------------------------------------------------------------------------


class TestRetrieveUsesTransformedQuery:
    """Test that retrieve node uses transformed_query from state."""

    @patch("src.graph.nodes.get_breaker")
    @patch("src.graph.nodes.get_retriever")
    def test_uses_transformed_query(self, mock_get_retriever, mock_breaker):
        """Retrieve should use transformed_query when present in state."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        mock_cb = MagicMock()
        mock_breaker.return_value = mock_cb
        mock_cb.call.side_effect = lambda fn: fn()

        from src.graph.nodes import retrieve

        state = {
            "question": "Original question",
            "transformed_query": "Expanded transformed query",
            "retriever_strategy": "dense",
        }
        retrieve(state)

        # The retriever should be invoked with the transformed query
        mock_retriever.invoke.assert_called_once()
        call_arg = mock_retriever.invoke.call_args[0][0]
        assert call_arg == "Expanded transformed query"

    @patch("src.graph.nodes.get_breaker")
    @patch("src.graph.nodes.get_retriever")
    def test_falls_back_to_normalizer(self, mock_get_retriever, mock_breaker):
        """Retrieve should normalize inline when no transformed_query."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        mock_cb = MagicMock()
        mock_breaker.return_value = mock_cb
        mock_cb.call.side_effect = lambda fn: fn()

        from src.graph.nodes import retrieve

        state = {
            "question": "What is PTO?",
            "retriever_strategy": "dense",
        }
        retrieve(state)

        mock_retriever.invoke.assert_called_once()
        call_arg = mock_retriever.invoke.call_args[0][0]
        # Should contain the acronym expansion from normalizer
        assert "paid time off" in call_arg.lower()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_config_field_exists(self):
        assert hasattr(settings, "query_transform_enabled")

    def test_defaults_to_enabled(self):
        assert settings.query_transform_enabled is True
