"""Phase 11: Intent Detection tests.

Tests for:
- Heuristic intent classification (regex-based fallback)
- IntentResult Pydantic model
- Graph node behavior (enabled/disabled)
- Edge cases (empty query, unknown intent)
"""
from __future__ import annotations

import pytest

from config import settings


def _set_setting(name: str, value):
    object.__setattr__(settings, name, value)


# ---------------------------------------------------------------------------
# Heuristic intent tests
# ---------------------------------------------------------------------------


class TestHeuristicIntent:
    """Test the regex-based fallback classifier."""

    def test_comparative_keywords(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, conf = _heuristic_intent("Compare the engineering and HR onboarding")
        assert intent == "comparative"
        assert conf >= 0.5

    def test_comparative_versus(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("PTO policy vs sick leave policy")
        assert intent == "comparative"

    def test_procedural_how_to(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("How do I submit an expense report?")
        assert intent == "procedural"

    def test_procedural_steps(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("What are the steps for onboarding?")
        assert intent == "procedural"

    def test_factual_when(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("When was the last quarterly report?")
        assert intent == "factual"

    def test_factual_how_many(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("How many PTO days do employees get?")
        assert intent == "factual"

    def test_analytical_implications(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("What are the implications of the new security policy?")
        assert intent == "analytical"

    def test_analytical_why(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("Why does the company require NDAs?")
        assert intent == "analytical"

    def test_multi_hop_across(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("How does the security policy affect engineering practices?")
        assert intent == "multi_hop"

    def test_default_informational(self):
        from src.graph.intent_detector import _heuristic_intent
        intent, _ = _heuristic_intent("What is the remote work policy?")
        assert intent == "informational"

    def test_confidence_range(self):
        from src.graph.intent_detector import _heuristic_intent
        _, conf = _heuristic_intent("anything")
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# IntentResult model tests
# ---------------------------------------------------------------------------


class TestIntentResultModel:
    """Test the Pydantic model."""

    def test_valid_creation(self):
        from src.graph.intent_detector import IntentResult
        r = IntentResult(intent="comparative", confidence=0.85)
        assert r.intent == "comparative"
        assert r.confidence == 0.85

    def test_confidence_bounds(self):
        from src.graph.intent_detector import IntentResult
        with pytest.raises(Exception):
            IntentResult(intent="factual", confidence=1.5)
        with pytest.raises(Exception):
            IntentResult(intent="factual", confidence=-0.1)


# ---------------------------------------------------------------------------
# Graph node tests
# ---------------------------------------------------------------------------


class TestIntentDetectNode:
    """Test the intent_detect graph node."""

    @pytest.fixture(autouse=True)
    def save_restore(self):
        orig = settings.intent_detection_enabled
        yield
        _set_setting("intent_detection_enabled", orig)

    def test_disabled_returns_default(self):
        _set_setting("intent_detection_enabled", False)
        from src.graph.intent_detector import intent_detect
        result = intent_detect({"question": "Compare X and Y"})
        assert result["intent"] == "informational"
        assert result["intent_confidence"] == 0.0

    def test_empty_question_returns_default(self):
        _set_setting("intent_detection_enabled", True)
        from src.graph.intent_detector import intent_detect
        result = intent_detect({"question": ""})
        assert result["intent"] == "informational"
        assert result["intent_confidence"] == 0.0

    def test_falls_back_to_heuristic_on_error(self):
        """When the LLM call fails, should fall back to heuristic."""
        _set_setting("intent_detection_enabled", True)
        from unittest.mock import patch

        from src.graph.intent_detector import intent_detect

        # Mock the circuit breaker to raise an exception
        with patch("src.graph.intent_detector.get_breaker") as mock_cb:
            mock_cb.return_value.call.side_effect = Exception("LLM down")
            result = intent_detect({"question": "Compare engineering and HR"})
            assert result["intent"] == "comparative"
            assert result["intent_confidence"] > 0.0

    def test_returns_valid_intent_keys(self):
        _set_setting("intent_detection_enabled", True)
        from unittest.mock import patch

        from src.graph.intent_detector import VALID_INTENTS, intent_detect

        with patch("src.graph.intent_detector.get_breaker") as mock_cb:
            mock_cb.return_value.call.side_effect = Exception("LLM down")
            result = intent_detect({"question": "How do I file expenses?"})
            assert result["intent"] in VALID_INTENTS


# ---------------------------------------------------------------------------
# Valid intents set
# ---------------------------------------------------------------------------


class TestValidIntents:
    def test_all_six_intents_present(self):
        from src.graph.intent_detector import VALID_INTENTS
        assert len(VALID_INTENTS) == 6
        assert "informational" in VALID_INTENTS
        assert "comparative" in VALID_INTENTS
        assert "procedural" in VALID_INTENTS
        assert "analytical" in VALID_INTENTS
        assert "multi_hop" in VALID_INTENTS
        assert "factual" in VALID_INTENTS
