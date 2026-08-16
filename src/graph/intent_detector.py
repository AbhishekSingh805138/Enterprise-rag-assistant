"""Intent detection node for the CRAG graph.

Classifies query intent into categories that inform downstream routing,
retrieval strategy selection, and prompt template choice.

Intent types:
- informational: General knowledge lookup ("What is the PTO policy?")
- comparative: Compare/contrast across topics ("Compare engineering and HR onboarding")
- procedural: Step-by-step instructions ("How do I submit an expense report?")
- analytical: Requires reasoning/analysis ("What are the implications of policy X?")
- multi_hop: Requires connecting multiple pieces of info ("What policies apply to contractors in security?")
- factual: Simple fact lookup ("What is the NDA clause for vendors?")
"""
from __future__ import annotations

import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from src.prompts import register
from src.graph.tracing import traced
from src.llm_pool import get_llm
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker

logger = logging.getLogger(__name__)

VALID_INTENTS = frozenset({
    "informational", "comparative", "procedural",
    "analytical", "multi_hop", "factual",
})


class IntentResult(BaseModel):
    """Structured output from the intent classifier."""
    intent: str = Field(
        description=(
            "The classified intent. One of: informational, comparative, "
            "procedural, analytical, multi_hop, factual."
        )
    )
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )


_intent_prompt = register(
    "intent_detection",
    "v1",

    [
        (
            "system",
            "You are a query intent classifier for an enterprise knowledge assistant. "
            "Classify the user's question into exactly one of these categories:\n\n"
            "- **informational**: General knowledge lookup about a single topic\n"
            "- **comparative**: Comparing or contrasting two or more topics\n"
            "- **procedural**: Asking for step-by-step instructions or a process\n"
            "- **analytical**: Requires reasoning, analysis, or interpretation\n"
            "- **multi_hop**: Requires connecting information from multiple sources\n"
            "- **factual**: Simple, direct fact lookup (dates, numbers, names)\n\n"
            "Also provide a confidence score (0.0 to 1.0).\n\n"
            "Examples:\n"
            '  "What is the PTO policy?" → informational, 0.95\n'
            '  "Compare engineering and HR onboarding" → comparative, 0.9\n'
            '  "How do I submit an expense report?" → procedural, 0.95\n'
            '  "What are the implications of the new security policy?" → analytical, 0.85\n'
            '  "What policies apply to contractors in the security department?" → multi_hop, 0.8\n'
            '  "When was the last quarterly report published?" → factual, 0.9\n',
        ),
        ("human", "{question}"),
    ]
)


# --- Heuristic fallback (no LLM needed) ---

_COMPARATIVE_PATTERNS = re.compile(
    r"\b(compare|contrast|difference|versus|vs\.?|differ|similarities|"
    r"how does .+ differ|which is better)\b",
    re.IGNORECASE,
)
_PROCEDURAL_PATTERNS = re.compile(
    r"\b(how (do|can|should|to)|steps|process|procedure|guide|instructions|"
    r"walk me through|tutorial)\b",
    re.IGNORECASE,
)
_FACTUAL_PATTERNS = re.compile(
    r"\b(when (was|is|did)|what (is|are) the (date|number|name|amount)|"
    r"how (many|much)|who (is|was))\b",
    re.IGNORECASE,
)
_ANALYTICAL_PATTERNS = re.compile(
    r"\b(implications|impact|analysis|evaluate|assess|why (does|did|is|are)|"
    r"what (would|could) happen|consequence|risk)\b",
    re.IGNORECASE,
)
_MULTI_HOP_PATTERNS = re.compile(
    r"\b(relate|connection|affect|apply to .+ in|across .+ and|"
    r"how does .+ impact .+)\b",
    re.IGNORECASE,
)


def _heuristic_intent(question: str) -> tuple[str, float]:
    """Fast, zero-LLM intent classification using regex patterns."""
    if _COMPARATIVE_PATTERNS.search(question):
        return "comparative", 0.7
    if _PROCEDURAL_PATTERNS.search(question):
        return "procedural", 0.7
    if _ANALYTICAL_PATTERNS.search(question):
        return "analytical", 0.6
    if _MULTI_HOP_PATTERNS.search(question):
        return "multi_hop", 0.6
    if _FACTUAL_PATTERNS.search(question):
        return "factual", 0.7
    return "informational", 0.5


# --- Graph node ---

@traced
def intent_detect(state: dict) -> dict:
    """Classify the intent of the user's question."""
    if not settings.intent_detection_enabled:
        return {"intent": "informational", "intent_confidence": 0.0}

    question = state.get("question", "")
    if not question.strip():
        return {"intent": "informational", "intent_confidence": 0.0}

    # Try LLM-based classification first
    try:
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        classifier = _intent_prompt | get_llm().with_structured_output(IntentResult)
        result: IntentResult = cb.call(classifier.invoke, {"question": question})

        intent = result.intent.lower().strip()
        if intent not in VALID_INTENTS:
            logger.warning("LLM returned invalid intent '%s', falling back to heuristic", intent)
            intent, confidence = _heuristic_intent(question)
        else:
            confidence = result.confidence

        logger.info("Intent detected: %s (confidence=%.2f) for: %s", intent, confidence, question[:80])
        return {"intent": intent, "intent_confidence": confidence}

    except CircuitBreakerOpen as exc:
        logger.warning("LLM circuit open during intent detection: %s — using heuristic", exc)
        intent, confidence = _heuristic_intent(question)
        return {"intent": intent, "intent_confidence": confidence}

    except Exception:
        logger.debug("Intent detection LLM failed — falling back to heuristic", exc_info=True)
        intent, confidence = _heuristic_intent(question)
        return {"intent": intent, "intent_confidence": confidence}
