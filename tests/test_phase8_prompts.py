"""Phase 8.6 tests: Prompt Engineering & Tool Integration.

Covers:
  - Less conservative generation prompt (partial answers)
  - Chain-of-thought feature flag
  - Few-shot examples in planner prompt
  - Tool router (calculator, data_lookup)
  - Tools disabled by default
  - Naive RAG prompt alignment
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# TestPartialAnswerPrompt
# ---------------------------------------------------------------------------

class TestPartialAnswerPrompt:
    """Verify the generation prompt allows partial answers."""

    def test_gen_prompt_mentions_partial(self):
        """The gen prompt should instruct to provide partial answers."""
        from src.graph.nodes import _GEN_SYSTEM_BASE
        assert "partial information" in _GEN_SYSTEM_BASE.lower()

    def test_gen_prompt_only_idk_when_completely_irrelevant(self):
        """The gen prompt should only say IDK if context is completely irrelevant."""
        from src.graph.nodes import _GEN_SYSTEM_BASE
        assert "completely irrelevant" in _GEN_SYSTEM_BASE.lower()

    def test_gen_prompt_requires_citations(self):
        from src.graph.nodes import _GEN_SYSTEM_BASE
        assert "cite" in _GEN_SYSTEM_BASE.lower()

    def test_gen_prompt_no_inventing(self):
        from src.graph.nodes import _GEN_SYSTEM_BASE
        assert "invent" in _GEN_SYSTEM_BASE.lower()


# ---------------------------------------------------------------------------
# TestNaiveRAGPromptAlignment
# ---------------------------------------------------------------------------

class TestNaiveRAGPromptAlignment:
    """Verify naive RAG prompt is aligned with graph gen prompt."""

    def test_naive_prompt_mentions_partial(self):
        from src.rag.naive_rag import SYSTEM_PROMPT
        assert "partial information" in SYSTEM_PROMPT.lower()

    def test_naive_prompt_only_idk_when_irrelevant(self):
        from src.rag.naive_rag import SYSTEM_PROMPT
        assert "completely irrelevant" in SYSTEM_PROMPT.lower()

    def test_naive_prompt_requires_citations(self):
        from src.rag.naive_rag import SYSTEM_PROMPT
        assert "cite" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# TestChainOfThought
# ---------------------------------------------------------------------------

class TestChainOfThought:
    """Verify chain-of-thought feature flag."""

    def test_cot_addendum_content(self):
        """The CoT addendum should instruct reasoning in a thinking block."""
        from src.graph.nodes import _GEN_COT_ADDENDUM
        assert "thinking" in _GEN_COT_ADDENDUM.lower()
        assert "reason" in _GEN_COT_ADDENDUM.lower()

    def test_cot_disabled_by_default(self):
        """With default settings, CoT should not be in the prompt."""
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.chain_of_thought is False

    def test_cot_enabled_builds_prompt_with_thinking(self):
        """When CoT is enabled, _build_gen_prompt includes the addendum."""
        from src.graph.nodes import _GEN_COT_ADDENDUM, _GEN_SYSTEM_BASE

        with patch("src.graph.nodes.settings") as mock_settings:
            mock_settings.chain_of_thought = True
            from src.graph.nodes import _build_gen_prompt
            prompt = _build_gen_prompt()
            # The system message should include the CoT addendum
            messages = prompt.format_messages(question="test", context="test")
            system_msg = messages[0].content
            assert "thinking" in system_msg.lower()

    def test_cot_disabled_builds_prompt_without_thinking(self):
        """When CoT is disabled, _build_gen_prompt omits the addendum."""
        with patch("src.graph.nodes.settings") as mock_settings:
            mock_settings.chain_of_thought = False
            from src.graph.nodes import _build_gen_prompt
            prompt = _build_gen_prompt()
            messages = prompt.format_messages(question="test", context="test")
            system_msg = messages[0].content
            assert "<thinking>" not in system_msg


# ---------------------------------------------------------------------------
# TestPlannerFewShot
# ---------------------------------------------------------------------------

class TestPlannerFewShot:
    """Verify planner prompt includes few-shot examples."""

    def test_planner_prompt_has_examples(self):
        """The planner prompt should include worked examples."""
        from src.graph.planner import _planner_prompt
        messages = _planner_prompt.format_messages(question="test")
        system_msg = messages[0].content
        assert "examples" in system_msg.lower()

    def test_planner_prompt_has_simple_example(self):
        """The planner prompt should show a simple question example."""
        from src.graph.planner import _planner_prompt
        messages = _planner_prompt.format_messages(question="test")
        system_msg = messages[0].content
        assert "is_multi_part=false" in system_msg

    def test_planner_prompt_has_comparison_example(self):
        """The planner prompt should show a comparison question example."""
        from src.graph.planner import _planner_prompt
        messages = _planner_prompt.format_messages(question="test")
        system_msg = messages[0].content
        assert "compare" in system_msg.lower()

    def test_planner_prompt_has_multi_dept_example(self):
        """The planner prompt should show a multi-department example."""
        from src.graph.planner import _planner_prompt
        messages = _planner_prompt.format_messages(question="test")
        system_msg = messages[0].content
        assert "security" in system_msg.lower() and "legal" in system_msg.lower()


# ---------------------------------------------------------------------------
# TestToolRouter
# ---------------------------------------------------------------------------

class TestToolRouter:
    """Verify tool routing detects and dispatches to tools."""

    def test_math_query_detected(self):
        from src.graph.tool_node import _needs_calculator
        assert _needs_calculator("What is 47 * 3?") is True
        assert _needs_calculator("Calculate the total cost") is True
        assert _needs_calculator("What is the PTO policy?") is False

    def test_lookup_query_detected(self):
        from src.graph.tool_node import _needs_lookup
        assert _needs_lookup("Look up HR policies") is True
        assert _needs_lookup("Find in the legal documents") is True
        assert _needs_lookup("What is the PTO policy?") is False

    def test_expression_extraction(self):
        from src.graph.tool_node import _extract_expression
        assert _extract_expression("What is 47 * 3?") == "47 * 3"
        assert _extract_expression("Calculate 100 + 200 - 50") == "100 + 200 - 50"
        assert _extract_expression("No math here") is None

    def test_department_extraction(self):
        from src.graph.tool_node import _extract_department
        assert _extract_department("Look up HR policies") == "hr"
        assert _extract_department("Search legal documents") == "legal"
        assert _extract_department("Find something") is None

    @patch("src.graph.tool_node.settings")
    def test_tool_router_disabled_returns_empty(self, mock_settings):
        """When ENABLE_TOOLS is False, tool_router returns empty results."""
        mock_settings.enable_tools = False
        from src.graph.tool_node import tool_router
        result = tool_router({"question": "Calculate 2 + 2", "tool_results": []})
        assert result["tool_results"] == []

    @patch("src.graph.tool_node.settings")
    def test_tool_router_calculator(self, mock_settings):
        """When tools enabled and math detected, calculator is invoked."""
        mock_settings.enable_tools = True

        with patch("src.tools.calculator.calculator") as mock_calc:
            mock_calc.invoke.return_value = "141"
            from src.graph.tool_node import tool_router
            result = tool_router({"question": "What is 47 * 3?", "tool_results": []})
            assert len(result["tool_results"]) == 1
            assert "141" in result["tool_results"][0]

    @patch("src.graph.tool_node.settings")
    def test_tool_router_lookup(self, mock_settings):
        """When tools enabled and lookup detected, data_lookup is invoked."""
        mock_settings.enable_tools = True

        with patch("src.tools.data_lookup.data_lookup") as mock_lookup:
            mock_lookup.invoke.return_value = "Policy content here"
            from src.graph.tool_node import tool_router
            result = tool_router({
                "question": "Look up HR policy on remote work",
                "tool_results": [],
            })
            assert len(result["tool_results"]) == 1
            assert "Policy content" in result["tool_results"][0]


# ---------------------------------------------------------------------------
# TestToolsDisabledDefault
# ---------------------------------------------------------------------------

class TestToolsDisabledDefault:
    """Verify tools are disabled by default."""

    def test_enable_tools_default_false(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.enable_tools is False

    def test_graph_has_no_tool_router_by_default(self):
        """With default settings, tool_router is not in the graph."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            with patch("src.graph.build_graph.settings") as mock_settings:
                mock_settings.enable_tools = False
                mock_settings.checkpoint_dir = "checkpoints"
                graph = build_graph(checkpointer=InMemorySaver())
                node_names = set(graph.get_graph().nodes.keys())
                assert "tool_router" not in node_names
        finally:
            reset_graph()

    def test_graph_has_tool_router_when_enabled(self):
        """With ENABLE_TOOLS=true, tool_router is in the graph."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph.build_graph import build_graph, reset_graph

        reset_graph()
        try:
            with patch("src.graph.build_graph.settings") as mock_settings:
                mock_settings.enable_tools = True
                mock_settings.checkpoint_dir = "checkpoints"
                graph = build_graph(checkpointer=InMemorySaver())
                node_names = set(graph.get_graph().nodes.keys())
                assert "tool_router" in node_names
        finally:
            reset_graph()


# ---------------------------------------------------------------------------
# TestGenerateIncludesToolResults
# ---------------------------------------------------------------------------

class TestGenerateIncludesToolResults:
    """Verify generate node includes tool results in context."""

    def test_generate_with_tool_results(self):
        """When tool_results are present, they should be included in context."""
        from src.graph.nodes import generate

        doc = Document(page_content="Test content", metadata={"filename": "test.md"})
        state = {
            "question": "What is 2 + 2?",
            "documents": [doc],
            "tool_results": ["Calculator(2 + 2) = 4"],
        }

        with patch("src.graph.nodes._gen_prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "The answer is 4 (test.md)"
            mock_prompt.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            result = generate(state)
            # Check that the invoke was called with context containing tool results
            call_args = mock_chain.invoke.call_args[0][0]
            assert "[tool]" in call_args["context"]
            assert "Calculator" in call_args["context"]

    def test_generate_without_tool_results(self):
        """When no tool_results, context should not have [tool] markers."""
        from src.graph.nodes import generate

        doc = Document(page_content="Test content", metadata={"filename": "test.md"})
        state = {
            "question": "What is PTO?",
            "documents": [doc],
        }

        with patch("src.graph.nodes._gen_prompt") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "PTO is paid time off (test.md)"
            mock_prompt.__or__ = MagicMock(
                return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))
            )
            result = generate(state)
            call_args = mock_chain.invoke.call_args[0][0]
            assert "[tool]" not in call_args["context"]
