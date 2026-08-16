"""Unified query analysis tests.

The unified analyzer replaces the intent_detect -> query_transform -> planner
chain (three serial LLM calls) with a single structured-output call. Tests:
- QueryAnalysis Pydantic model
- Zero-LLM fallback (_fallback_analysis)
- analyze_query node: empty query, success path, invalid intent, circuit
  open, generic error
- Graph wiring: analyze_query replaces the three nodes when UNIFIED_ANALYSIS
  is on, and is absent (classic planner used) when off
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# QueryAnalysis model
# ---------------------------------------------------------------------------


class TestQueryAnalysisModel:
    def test_valid_creation(self):
        from src.graph.analyzer import QueryAnalysis
        r = QueryAnalysis(
            intent="comparative",
            confidence=0.9,
            entities=["HR", "engineering"],
            search_query="compare HR and engineering onboarding",
            is_multi_part=True,
            sub_questions=["What is HR onboarding?", "What is engineering onboarding?"],
        )
        assert r.intent == "comparative"
        assert r.is_multi_part is True
        assert len(r.sub_questions) == 2

    def test_confidence_bounds(self):
        from src.graph.analyzer import QueryAnalysis
        with pytest.raises(Exception):
            QueryAnalysis(
                intent="factual", confidence=1.5, search_query="x", is_multi_part=False
            )

    def test_defaults(self):
        from src.graph.analyzer import QueryAnalysis
        r = QueryAnalysis(
            intent="factual", confidence=0.5, search_query="x", is_multi_part=False
        )
        assert r.entities == []
        assert r.sub_questions == []


# ---------------------------------------------------------------------------
# Zero-LLM fallback
# ---------------------------------------------------------------------------


class TestFallbackAnalysis:
    def test_returns_expected_keys(self):
        from src.graph.analyzer import _fallback_analysis
        out = _fallback_analysis("Compare X and Y", "compare x and y")
        for key in (
            "intent",
            "intent_confidence",
            "extracted_entities",
            "transformed_query",
            "is_multi_part",
            "sub_questions",
        ):
            assert key in out
        # Fallback never decomposes
        assert out["is_multi_part"] is False
        assert out["sub_questions"] == ["Compare X and Y"]
        assert out["transformed_query"] == "compare x and y"

    def test_uses_heuristic_intent(self):
        from src.graph.analyzer import _fallback_analysis
        out = _fallback_analysis("Compare the engineering and HR onboarding", "...")
        assert out["intent"] == "comparative"


# ---------------------------------------------------------------------------
# analyze_query node
# ---------------------------------------------------------------------------


class TestAnalyzeQueryNode:
    def test_empty_question_returns_defaults(self):
        from src.graph.analyzer import analyze_query
        out = analyze_query({"question": "   "})
        assert out["is_multi_part"] is False
        assert out["sub_questions"] == ["   "]
        assert out["intent"] == "informational"
        assert out["intent_confidence"] == 0.0
        assert out["sub_answers"] == []
        assert out["current_sub_idx"] == 0

    def test_success_multi_part(self):
        """Mocked LLM returns a multi-part analysis; entities fold into query."""
        from src.graph.analyzer import QueryAnalysis, analyze_query

        analysis = QueryAnalysis(
            intent="comparative",
            confidence=0.88,
            entities=["Net 30", "SLA"],
            search_query="compare vendor payment terms and SLA",
            is_multi_part=True,
            sub_questions=[
                "What are the vendor payment terms?",
                "What are the SLA requirements?",
            ],
        )
        with patch("src.graph.analyzer.get_llm", return_value=MagicMock()), patch(
            "src.graph.analyzer.get_breaker"
        ) as mock_cb:
            mock_cb.return_value.call.return_value = analysis
            out = analyze_query({"question": "Vendor payment terms and SLA?"})

        assert out["intent"] == "comparative"
        assert out["intent_confidence"] == 0.88
        assert out["is_multi_part"] is True
        assert len(out["sub_questions"]) == 2
        assert out["original_question"] == "Vendor payment terms and SLA?"
        # entities not already in the search query are appended
        assert "Net 30" in out["transformed_query"]
        assert "SLA" in out["transformed_query"]

    def test_success_simple_single_sub_question(self):
        from src.graph.analyzer import QueryAnalysis, analyze_query

        analysis = QueryAnalysis(
            intent="factual",
            confidence=0.9,
            entities=[],
            search_query="remote work policy",
            is_multi_part=True,  # LLM says multi but only one sub → coerced simple
            sub_questions=["What is the remote work policy?"],
        )
        with patch("src.graph.analyzer.get_llm", return_value=MagicMock()), patch(
            "src.graph.analyzer.get_breaker"
        ) as mock_cb:
            mock_cb.return_value.call.return_value = analysis
            out = analyze_query({"question": "remote work?"})

        # is_multi_part requires >1 sub-question
        assert out["is_multi_part"] is False
        assert out["sub_questions"] == ["What is the remote work policy?"]

    def test_invalid_intent_falls_back_to_heuristic(self):
        from src.graph.analyzer import QueryAnalysis, analyze_query

        analysis = QueryAnalysis(
            intent="bogus_intent",
            confidence=0.99,
            entities=[],
            search_query="how do I file expenses",
            is_multi_part=False,
            sub_questions=["How do I file expenses?"],
        )
        with patch("src.graph.analyzer.get_llm", return_value=MagicMock()), patch(
            "src.graph.analyzer.get_breaker"
        ) as mock_cb:
            mock_cb.return_value.call.return_value = analysis
            out = analyze_query({"question": "How do I file expenses?"})

        from src.graph.intent_detector import VALID_INTENTS
        assert out["intent"] in VALID_INTENTS
        assert out["intent"] == "procedural"  # heuristic on "how do I"

    def test_circuit_open_falls_back(self):
        from src.graph.analyzer import analyze_query
        from src.resilience.circuit_breaker import CircuitBreakerOpen

        with patch("src.graph.analyzer.get_llm", return_value=MagicMock()), patch(
            "src.graph.analyzer.get_breaker"
        ) as mock_cb:
            mock_cb.return_value.call.side_effect = CircuitBreakerOpen("llm", 30.0)
            out = analyze_query({"question": "Compare engineering and HR onboarding"})

        assert out["intent"] == "comparative"  # heuristic
        assert out["is_multi_part"] is False
        assert out["sub_questions"] == ["Compare engineering and HR onboarding"]

    def test_generic_error_falls_back(self):
        from src.graph.analyzer import analyze_query

        with patch("src.graph.analyzer.get_llm", return_value=MagicMock()), patch(
            "src.graph.analyzer.get_breaker"
        ) as mock_cb:
            mock_cb.return_value.call.side_effect = RuntimeError("boom")
            out = analyze_query({"question": "What is the PTO policy?"})

        assert out["intent"] in {
            "informational", "comparative", "procedural",
            "analytical", "multi_hop", "factual",
        }
        assert out["sub_questions"] == ["What is the PTO policy?"]


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


class TestGraphWiring:
    @pytest.fixture(autouse=True)
    def restore_flag(self):
        orig = settings.unified_analysis
        import src.graph.build_graph as bg
        yield
        _set_setting("unified_analysis", orig)
        bg.reset_graph()

    def test_unified_on_replaces_three_nodes(self):
        import src.graph.build_graph as bg
        _set_setting("unified_analysis", True)
        bg.reset_graph()
        nodes = set(bg.build_graph().get_graph().nodes)
        assert "analyze_query" in nodes
        assert "planner" not in nodes
        assert "intent_detect" not in nodes
        assert "query_transform" not in nodes

    def test_unified_off_uses_classic_planner(self):
        import src.graph.build_graph as bg
        _set_setting("unified_analysis", False)
        bg.reset_graph()
        nodes = set(bg.build_graph().get_graph().nodes)
        assert "analyze_query" not in nodes
        assert "planner" in nodes
