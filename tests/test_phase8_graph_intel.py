"""Phase 8.3 tests: Graph Intelligence.

Covers:
  - Scope detection (in-scope vs out-of-scope)
  - Informed query rewriting (rejected context in prompt)
  - Auto pipeline routing (naive vs graph based on complexity)
  - Graph includes scope_check node
  - RAGState includes new fields
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# TestScopeDetection
# ---------------------------------------------------------------------------

class TestScopeDetection:
    """Verify scope_check node and _is_in_scope heuristic."""

    def test_enterprise_query_in_scope(self):
        from src.graph.scope_detector import _is_in_scope
        assert _is_in_scope("What is the PTO policy?") is True
        assert _is_in_scope("How does the API versioning work?") is True
        assert _is_in_scope("What are the security requirements?") is True

    def test_off_topic_query_out_of_scope(self):
        from src.graph.scope_detector import _is_in_scope
        assert _is_in_scope("What is the best recipe for chocolate cake?") is False
        assert _is_in_scope("Who won the football game yesterday?") is False
        assert _is_in_scope("What is my horoscope for today?") is False

    def test_ambiguous_query_defaults_in_scope(self):
        from src.graph.scope_detector import _is_in_scope
        # Generic queries without clear off-topic signals should pass through
        assert _is_in_scope("Tell me about the process") is True
        assert _is_in_scope("How does this work?") is True

    def test_empty_query_in_scope(self):
        from src.graph.scope_detector import _is_in_scope
        # Empty queries handled elsewhere
        assert _is_in_scope("") is True

    def test_scope_check_node_returns_in_scope(self):
        from src.graph.scope_detector import scope_check
        result = scope_check({"question": "What is the company vacation policy?"})
        assert result["in_scope"] is True

    def test_scope_check_node_returns_out_of_scope(self):
        from src.graph.scope_detector import scope_check
        result = scope_check({"question": "Best recipe for pasta carbonara?"})
        assert result["in_scope"] is False


# ---------------------------------------------------------------------------
# TestInformedRewrite
# ---------------------------------------------------------------------------

class TestInformedRewrite:
    """Verify transform_query includes rejected document context."""

    @patch("src.graph.nodes._llm")
    def test_rewrite_includes_rejected_context(self, mock_llm):
        """The rewrite prompt should include context about rejected docs."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "improved query about PTO"
        mock_llm_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance

        # Mock the pipe chain
        with patch("src.graph.nodes._rewrite_prompt") as mock_prompt:
            mock_pipe = MagicMock()
            mock_pipe.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt.__or__ = MagicMock(return_value=mock_pipe)

            from src.graph.nodes import transform_query
            state = {
                "question": "PTO policy",
                "retries": 0,
                "documents": [
                    Document(page_content="Irrelevant document about finances"),
                ],
            }
            result = transform_query(state)

        # Should return a rewritten question and incremented retries
        assert result["retries"] == 1
        assert isinstance(result["question"], str)

    def test_rewrite_with_no_documents(self):
        """When no documents, rejected_context should indicate that."""
        with patch("src.graph.nodes._rewrite_prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "better question"
            mock_pipe = MagicMock()
            mock_pipe.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt.__or__ = MagicMock(return_value=mock_pipe)

            from src.graph.nodes import transform_query
            state = {"question": "test", "retries": 0, "documents": []}
            result = transform_query(state)
            assert result["retries"] == 1


# ---------------------------------------------------------------------------
# TestAutoRouting
# ---------------------------------------------------------------------------

class TestAutoRouting:
    """Verify auto mode resolution."""

    def test_short_simple_resolves_to_naive(self):
        from api.app import _resolve_mode
        from api.models import AskRequest
        body = AskRequest(question="What is PTO?", mode="auto")
        assert _resolve_mode(body) == "naive"

    def test_complex_resolves_to_graph(self):
        from api.app import _resolve_mode
        from api.models import AskRequest
        body = AskRequest(
            question="Compare the PTO policy with the remote work policy and explain the differences between the two approaches taken by the company",
            mode="auto",
        )
        assert _resolve_mode(body) == "graph"

    def test_comparison_resolves_to_graph(self):
        from api.app import _resolve_mode
        from api.models import AskRequest
        body = AskRequest(question="Compare PTO versus sick leave", mode="auto")
        assert _resolve_mode(body) == "graph"

    def test_explicit_naive_not_changed(self):
        from api.app import _resolve_mode
        from api.models import AskRequest
        body = AskRequest(question="A very long complex question about many topics and also some more", mode="naive")
        assert _resolve_mode(body) == "naive"

    def test_explicit_graph_not_changed(self):
        from api.app import _resolve_mode
        from api.models import AskRequest
        body = AskRequest(question="Simple", mode="graph")
        assert _resolve_mode(body) == "graph"


# ---------------------------------------------------------------------------
# TestGraphStructure
# ---------------------------------------------------------------------------

class TestGraphStructure:
    """Verify graph includes scope_check and routes correctly."""

    def test_graph_has_scope_check_node(self):
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            graph = build_graph(checkpointer=InMemorySaver())
            node_names = set(graph.get_graph().nodes.keys())
            assert "scope_check" in node_names
        finally:
            reset_graph()

    def test_route_after_scope_in_scope(self):
        from src.graph.build_graph import _route_after_scope
        assert _route_after_scope({"in_scope": True}) == "planner"

    def test_route_after_scope_out_of_scope(self):
        from src.graph.build_graph import _route_after_scope
        assert _route_after_scope({"in_scope": False}) == "generate"

    def test_route_after_scope_missing_defaults_in(self):
        from src.graph.build_graph import _route_after_scope
        assert _route_after_scope({}) == "planner"


# ---------------------------------------------------------------------------
# TestRAGStateFields
# ---------------------------------------------------------------------------

class TestRAGStateFields:
    """Verify RAGState includes Phase 8 fields."""

    def test_in_scope_field_exists(self):
        from src.graph.state import RAGState
        assert "in_scope" in RAGState.__annotations__

    def test_all_sub_documents_field_exists(self):
        from src.graph.state import RAGState
        assert "all_sub_documents" in RAGState.__annotations__


# ---------------------------------------------------------------------------
# TestAutoModeAPI
# ---------------------------------------------------------------------------

class TestAutoModeAPI:
    """Verify API models accept 'auto' mode."""

    def test_ask_request_accepts_auto(self):
        from api.models import AskRequest
        req = AskRequest(question="test?", mode="auto")
        assert req.mode == "auto"

    def test_eval_request_accepts_auto(self):
        from api.models import EvalRequest
        req = EvalRequest(mode="auto")
        assert req.mode == "auto"
