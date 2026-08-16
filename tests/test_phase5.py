"""Phase 5 tests: planner, synthesizer, web search, tools, routing.

Covers:
  - Planner: simple vs multi-part classification, decomposition, error handling
  - Synthesizer: combines sub-answers, handles edge cases
  - Process sub-query: retrieves + generates per sub-question, index advancement
  - Web search: Tavily integration (mocked), graceful degradation without API key
  - Calculator tool: arithmetic, functions, error cases
  - Data lookup tool: department filtering
  - Router functions: correct routing for simple vs multi-part
  - Graph integration: simple and multi-part paths through the full graph
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# === Planner tests ===

class TestPlanner:
    def test_simple_question_passthrough(self):
        """Simple question should set is_multi_part=False with original as sole sub-question."""
        from src.graph.planner import PlanResult, _planner_prompt, planner

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = PlanResult(
            is_multi_part=False,
            sub_questions=["What is the remote work policy?"],
        )
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with patch("src.graph.planner._planner_prompt", mock_prompt):
            result = planner({"question": "What is the remote work policy?"})

        assert result["is_multi_part"] is False
        assert result["sub_questions"] == ["What is the remote work policy?"]
        assert result["sub_answers"] == []
        assert result["current_sub_idx"] == 0
        assert result["original_question"] == "What is the remote work policy?"

    def test_multi_part_decomposition(self):
        """Multi-part question should decompose into sub-questions."""
        from src.graph.planner import PlanResult, planner

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = PlanResult(
            is_multi_part=True,
            sub_questions=[
                "What is the data retention policy for customer records?",
                "What is the data retention policy for employee records?",
            ],
        )
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with patch("src.graph.planner._planner_prompt", mock_prompt):
            result = planner({
                "question": "Compare data retention for customer vs employee records"
            })

        assert result["is_multi_part"] is True
        assert len(result["sub_questions"]) == 2
        assert result["current_sub_idx"] == 0

    def test_planner_error_falls_back_to_simple(self):
        """If planner LLM fails, treat as simple question."""
        from src.graph.planner import planner

        mock_prompt = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("LLM down")
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with patch("src.graph.planner._planner_prompt", mock_prompt):
            result = planner({"question": "What is the policy?"})

        assert result["is_multi_part"] is False
        assert result["sub_questions"] == ["What is the policy?"]

    def test_planner_clamps_sub_questions(self):
        """Planner should cap sub-questions at MAX_SUB_QUESTIONS."""
        from src.graph.planner import MAX_SUB_QUESTIONS, PlanResult, planner

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = PlanResult(
            is_multi_part=True,
            sub_questions=[f"Sub-question {i}" for i in range(10)],
        )
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with patch("src.graph.planner._planner_prompt", mock_prompt):
            result = planner({"question": "Very complex question"})

        assert len(result["sub_questions"]) == MAX_SUB_QUESTIONS

    def test_planner_empty_sub_questions_uses_original(self):
        """If LLM returns empty sub_questions, fall back to original question."""
        from src.graph.planner import PlanResult, planner

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = PlanResult(
            is_multi_part=False,
            sub_questions=[],
        )
        mock_prompt = MagicMock()
        mock_prompt.__or__ = MagicMock(return_value=mock_chain)

        with patch("src.graph.planner._planner_prompt", mock_prompt):
            result = planner({"question": "What is the policy?"})

        assert result["sub_questions"] == ["What is the policy?"]


# === PlanResult model tests ===

class TestPlanResult:
    def test_simple_plan(self):
        from src.graph.planner import PlanResult
        p = PlanResult(is_multi_part=False, sub_questions=["question"])
        assert p.is_multi_part is False
        assert len(p.sub_questions) == 1

    def test_multi_part_plan(self):
        from src.graph.planner import PlanResult
        p = PlanResult(is_multi_part=True, sub_questions=["a", "b", "c"])
        assert p.is_multi_part is True
        assert len(p.sub_questions) == 3


# === Synthesizer tests ===

class TestSynthesize:
    def test_synthesize_combines_sub_answers(self):
        """Synthesizer should combine sub-answers into a generation."""
        from src.graph.planner import synthesize

        with patch("src.graph.planner._llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm_fn.return_value = mock_llm
            # Patch StrOutputParser to return a chain that returns a string
            with patch("src.graph.planner.StrOutputParser") as mock_parser_cls:
                mock_final_chain = MagicMock()
                mock_final_chain.invoke.return_value = "Combined answer. (handbook.md) (policy.md)"
                # _synthesize_prompt | _llm() returns intermediate, | StrOutputParser() returns final
                mock_parser_cls.return_value = MagicMock()
                with patch("src.graph.planner._synthesize_prompt") as mock_prompt:
                    mock_intermediate = MagicMock()
                    mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)
                    mock_intermediate.__or__ = MagicMock(return_value=mock_final_chain)

                    result = synthesize({
                        "original_question": "Compare X and Y",
                        "sub_questions": ["What is X?", "What is Y?"],
                        "sub_answers": ["X is foo. (handbook.md)", "Y is bar. (policy.md)"],
                    })

        assert "generation" in result
        assert result["generation"] == "Combined answer. (handbook.md) (policy.md)"

    def test_synthesize_no_sub_answers(self):
        """Synthesizer should return IDK when no sub-answers exist."""
        from src.graph.planner import synthesize
        result = synthesize({
            "original_question": "test",
            "sub_questions": [],
            "sub_answers": [],
        })
        assert "don't have enough information" in result["generation"].lower()

    def test_synthesize_error_falls_back_to_concatenation(self):
        """If synthesis LLM fails, concatenate sub-answers."""
        from src.graph.planner import synthesize

        with patch("src.graph.planner._synthesize_prompt") as mock_prompt:
            mock_intermediate = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)
            mock_final_chain = MagicMock()
            mock_final_chain.invoke.side_effect = RuntimeError("LLM down")
            mock_intermediate.__or__ = MagicMock(return_value=mock_final_chain)

            result = synthesize({
                "original_question": "Compare X and Y",
                "sub_questions": ["What is X?", "What is Y?"],
                "sub_answers": ["X is foo.", "Y is bar."],
            })

        assert "X is foo." in result["generation"]
        assert "Y is bar." in result["generation"]


# === Process sub-query tests ===

class TestProcessSubQuery:
    def test_processes_current_sub_question(self):
        """Should retrieve and generate for the current sub-question index."""
        from src.graph.planner import process_sub_query

        fake_docs = [Document(page_content="Relevant content.", metadata={"filename": "test.md"})]

        # Mock the LCEL chain: _gen_prompt | _llm() | StrOutputParser()
        with (
            patch("src.retrieval.get_retriever") as mock_get_ret,
            patch("src.graph.nodes._gen_prompt") as mock_prompt,
        ):
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = fake_docs
            mock_get_ret.return_value = mock_retriever

            mock_intermediate = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)
            mock_final = MagicMock()
            mock_final.invoke.return_value = "Answer to sub-question. (test.md)"
            mock_intermediate.__or__ = MagicMock(return_value=mock_final)

            result = process_sub_query({
                "sub_questions": ["What is X?", "What is Y?"],
                "sub_answers": [],
                "current_sub_idx": 0,
                "retriever_strategy": "dense",
            })

        assert result["current_sub_idx"] == 1
        assert len(result["sub_answers"]) == 1
        assert result["documents"] == fake_docs

    def test_increments_index(self):
        """Should advance current_sub_idx."""
        from src.graph.planner import process_sub_query

        with (
            patch("src.retrieval.get_retriever") as mock_get_ret,
            patch("src.graph.nodes._gen_prompt") as mock_prompt,
        ):
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [
                Document(page_content="doc", metadata={"filename": "f.md"})
            ]
            mock_get_ret.return_value = mock_retriever

            mock_intermediate = MagicMock()
            mock_prompt.__or__ = MagicMock(return_value=mock_intermediate)
            mock_final = MagicMock()
            mock_final.invoke.return_value = "Answer."
            mock_intermediate.__or__ = MagicMock(return_value=mock_final)

            result = process_sub_query({
                "sub_questions": ["Q1", "Q2", "Q3"],
                "sub_answers": ["A1"],
                "current_sub_idx": 1,
                "retriever_strategy": "dense",
            })

        assert result["current_sub_idx"] == 2
        assert len(result["sub_answers"]) == 2

    def test_no_docs_returns_idk(self):
        """Should return IDK answer when retrieval returns no documents."""
        from src.graph.planner import process_sub_query

        with patch("src.retrieval.get_retriever") as mock_get_ret:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []
            mock_get_ret.return_value = mock_retriever

            result = process_sub_query({
                "sub_questions": ["What is Z?"],
                "sub_answers": [],
                "current_sub_idx": 0,
                "retriever_strategy": "dense",
            })

        assert "don't have enough information" in result["sub_answers"][0].lower()

    def test_out_of_bounds_index(self):
        """Should handle index >= len(sub_questions) gracefully."""
        from src.graph.planner import process_sub_query

        result = process_sub_query({
            "sub_questions": ["Q1"],
            "sub_answers": ["A1"],
            "current_sub_idx": 5,
        })
        assert result["current_sub_idx"] == 5  # unchanged


# === Router function tests ===

class TestRouterFunctions:
    def test_route_after_plan_simple(self):
        from src.graph.planner import route_after_plan
        assert route_after_plan({"is_multi_part": False}) == "retrieve"

    def test_route_after_plan_multi_part(self):
        from src.graph.planner import route_after_plan
        assert route_after_plan({"is_multi_part": True}) == "process_sub_query"

    def test_route_after_plan_missing_key(self):
        from src.graph.planner import route_after_plan
        assert route_after_plan({}) == "retrieve"

    def test_has_more_sub_queries_yes(self):
        from src.graph.planner import has_more_sub_queries
        result = has_more_sub_queries({
            "current_sub_idx": 1,
            "sub_questions": ["a", "b", "c"],
        })
        assert result == "process_sub_query"

    def test_has_more_sub_queries_no(self):
        from src.graph.planner import has_more_sub_queries
        result = has_more_sub_queries({
            "current_sub_idx": 3,
            "sub_questions": ["a", "b", "c"],
        })
        assert result == "synthesize"

    def test_has_more_sub_queries_empty(self):
        from src.graph.planner import has_more_sub_queries
        assert has_more_sub_queries({}) == "synthesize"


# === Web search tests ===

class TestWebSearch:
    def test_web_search_without_api_key(self):
        """Web search should degrade gracefully without TAVILY_API_KEY."""
        from src.graph.nodes import web_search

        with patch("src.graph.nodes.settings") as mock_settings:
            mock_settings.tavily_api_key = ""
            result = web_search({"question": "test", "documents": []})

        assert result["web_fallback_used"] is True
        assert len(result["documents"]) == 1
        assert "unavailable" in result["documents"][0].page_content.lower()

    def test_web_search_with_tavily(self):
        """Web search should return Documents from Tavily results."""
        from src.graph.nodes import web_search

        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = {
            "results": [
                {"content": "Result 1 content", "url": "https://example.com/1", "title": "Title 1"},
                {"content": "Result 2 content", "url": "https://example.com/2", "title": "Title 2"},
            ]
        }

        with (
            patch("src.graph.nodes.settings") as mock_settings,
            patch("tavily.TavilyClient", return_value=mock_client_instance),
        ):
            mock_settings.tavily_api_key = "tvly-test-key"
            existing = [Document(page_content="existing doc", metadata={})]
            result = web_search({"question": "test query", "documents": existing})

        assert result["web_fallback_used"] is True
        assert len(result["documents"]) == 3  # 1 existing + 2 web
        assert result["documents"][1].page_content == "Result 1 content"
        assert result["documents"][1].metadata["source"] == "https://example.com/1"

    def test_web_search_error_handling(self):
        """Web search should handle Tavily errors gracefully."""
        from src.graph.nodes import web_search

        with (
            patch("src.graph.nodes.settings") as mock_settings,
            patch("tavily.TavilyClient") as mock_client_cls,
        ):
            mock_settings.tavily_api_key = "tvly-test-key"
            mock_client_cls.return_value.search.side_effect = RuntimeError("API error")
            result = web_search({"question": "test", "documents": []})

        assert result["web_fallback_used"] is True
        assert "failed" in result["documents"][0].page_content.lower()


# === Calculator tool tests ===

class TestCalculator:
    def test_basic_arithmetic(self):
        from src.tools.calculator import calculator
        assert calculator.invoke({"expression": "2 + 3"}) == "5"
        assert calculator.invoke({"expression": "10 - 4"}) == "6"
        assert calculator.invoke({"expression": "3 * 7"}) == "21"
        assert calculator.invoke({"expression": "15 / 4"}) == "3.75"

    def test_parentheses(self):
        from src.tools.calculator import calculator
        assert calculator.invoke({"expression": "(2 + 3) * 4"}) == "20"

    def test_powers(self):
        from src.tools.calculator import calculator
        assert calculator.invoke({"expression": "2 ** 10"}) == "1024"

    def test_math_functions(self):
        from src.tools.calculator import calculator
        assert calculator.invoke({"expression": "sqrt(144)"}) == "12"
        assert calculator.invoke({"expression": "abs(-5)"}) == "5"

    def test_constants(self):
        from src.tools.calculator import calculator
        result = calculator.invoke({"expression": "pi"})
        assert result.startswith("3.14159")

    def test_division_by_zero(self):
        from src.tools.calculator import calculator
        result = calculator.invoke({"expression": "10 / 0"})
        assert "Error" in result

    def test_invalid_expression(self):
        from src.tools.calculator import calculator
        result = calculator.invoke({"expression": "import os"})
        assert "Error" in result

    def test_exponent_limit(self):
        from src.tools.calculator import calculator
        result = calculator.invoke({"expression": "2 ** 200"})
        assert "Error" in result

    def test_percentage_calculation(self):
        from src.tools.calculator import calculator
        assert calculator.invoke({"expression": "47.3 * 0.12"}) == "5.676"


# === Data lookup tool tests ===

class TestDataLookup:
    def test_valid_department_filter(self):
        from src.tools.data_lookup import data_lookup

        fake_docs = [
            Document(
                page_content="HR policy content",
                metadata={"filename": "handbook.md", "department": "hr"},
            )
        ]
        with patch("src.retrieval.get_retriever") as mock_get_ret:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = fake_docs
            mock_get_ret.return_value = mock_retriever

            result = data_lookup.invoke({"query": "leave policy", "department": "hr"})

        assert "handbook.md" in result
        assert "HR policy content" in result
        mock_get_ret.assert_called_once_with(strategy="dense", k=4, filter={"department": "hr"})

    def test_invalid_department(self):
        from src.tools.data_lookup import data_lookup
        result = data_lookup.invoke({"query": "test", "department": "nonexistent"})
        assert "Unknown department" in result

    def test_no_department_filter(self):
        from src.tools.data_lookup import data_lookup

        with patch("src.retrieval.get_retriever") as mock_get_ret:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []
            mock_get_ret.return_value = mock_retriever

            result = data_lookup.invoke({"query": "test"})

        mock_get_ret.assert_called_once_with(strategy="dense", k=4, filter=None)
        assert "No documents found" in result

    def test_general_department_no_filter(self):
        """'general' department should not add a metadata filter."""
        from src.tools.data_lookup import data_lookup

        with patch("src.retrieval.get_retriever") as mock_get_ret:
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = []
            mock_get_ret.return_value = mock_retriever

            data_lookup.invoke({"query": "test", "department": "general"})

        mock_get_ret.assert_called_once_with(strategy="dense", k=4, filter=None)


# === Graph integration tests (simple and multi-part paths) ===

class TestGraphIntegration:
    def test_simple_path_through_planner(self):
        """Simple question should go through planner → retrieve → ... → critic."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph import build_graph as bg

        def fake_planner(state):
            return {
                "is_multi_part": False,
                "sub_questions": [state["question"]],
                "sub_answers": [],
                "current_sub_idx": 0,
                "original_question": state["question"],
            }

        def fake_retrieve(state):
            return {
                "documents": [Document(page_content="doc", metadata={"filename": "f.md"})],
                "retries": 0,
            }

        def fake_grade(state):
            return {"relevant": True}

        def fake_generate(state):
            return {"generation": "Simple answer. (f.md)"}

        def fake_critic(state):
            return {"critic_passed": True, "claims_removed": 0}

        bg.reset_graph()
        try:
            # Patch in build_graph module namespace (where add_node captures references)
            with (
                patch("src.graph.build_graph.planner", side_effect=fake_planner),
                patch("src.graph.build_graph.retrieve", side_effect=fake_retrieve),
                patch("src.graph.build_graph.grade_documents", side_effect=fake_grade),
                patch("src.graph.build_graph.generate", side_effect=fake_generate),
                patch("src.graph.build_graph.critic", side_effect=fake_critic),
            ):
                graph = bg.build_graph(checkpointer=InMemorySaver())
                result = graph.invoke(
                    {"question": "What is the policy?", "retries": 0},
                    {"configurable": {"thread_id": "test-simple"}},
                )
            assert result["generation"] == "Simple answer. (f.md)"
            assert result["is_multi_part"] is False
        finally:
            bg.reset_graph()

    def test_multi_part_path_through_planner(self):
        """Multi-part question should go planner → sub-query loop → synthesize → critic."""
        from langgraph.checkpoint.memory import InMemorySaver

        from src.graph import build_graph as bg

        def fake_planner(state):
            return {
                "is_multi_part": True,
                "sub_questions": ["What is X?", "What is Y?"],
                "sub_answers": [],
                "current_sub_idx": 0,
                "original_question": state["question"],
            }

        def fake_process_sub_query(state):
            idx = state.get("current_sub_idx", 0)
            sub_answers = list(state.get("sub_answers", []))
            sub_answers.append(f"Answer to sub-question {idx+1}. (doc{idx+1}.md)")
            return {
                "sub_answers": sub_answers,
                "current_sub_idx": idx + 1,
                "documents": [Document(page_content="doc", metadata={"filename": f"doc{idx+1}.md"})],
            }

        def fake_synthesize(state):
            return {
                "generation": "Synthesized: X is A, Y is B. (doc1.md) (doc2.md)",
                "question": state.get("original_question", ""),
            }

        def fake_critic(state):
            return {"critic_passed": True, "claims_removed": 0}

        bg.reset_graph()
        try:
            # Patch in build_graph module namespace
            with (
                patch("src.graph.build_graph.planner", side_effect=fake_planner),
                patch("src.graph.build_graph.process_sub_query", side_effect=fake_process_sub_query),
                patch("src.graph.build_graph.synthesize", side_effect=fake_synthesize),
                patch("src.graph.build_graph.critic", side_effect=fake_critic),
            ):
                graph = bg.build_graph(checkpointer=InMemorySaver())
                result = graph.invoke(
                    {"question": "Compare X and Y", "retries": 0},
                    {"configurable": {"thread_id": "test-multi"}},
                )
            assert "Synthesized" in result["generation"]
            assert result["is_multi_part"] is True
            assert len(result["sub_answers"]) == 2
        finally:
            bg.reset_graph()
