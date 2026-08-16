"""Unified pre-retrieval query analysis — one LLM call instead of three.

Replaces the intent_detect -> query_transform -> planner chain (three
serial LLM round-trips) with a single structured-output call that:
  1. Classifies intent (same six categories as intent_detector)
  2. Extracts search-relevant entities
  3. Produces a retrieval-optimized rewrite of the question
  4. Decides whether the question is multi-part and decomposes it

Gated behind UNIFIED_ANALYSIS (default: off). When the LLM is
unavailable, falls back to the same zero-LLM heuristics the individual
nodes use: regex intent classification, regex entity extraction, and
treating the question as simple (no decomposition).

Emits exactly the state keys the downstream graph expects
(route_after_plan, retrieve, process_sub_query, synthesize), so the
rest of the pipeline is unchanged.
"""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings
from src.prompts import register
from src.graph.intent_detector import VALID_INTENTS, _heuristic_intent
from src.graph.tracing import traced
from src.llm_pool import get_llm
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker
from src.retrieval.entity_extractor import _regex_extract
from src.retrieval.normalizer import normalize_query

logger = logging.getLogger(__name__)


class QueryAnalysis(BaseModel):
    """Structured output of the unified analysis call."""

    intent: str = Field(
        description=(
            "The classified intent. One of: informational, comparative, "
            "procedural, analytical, multi_hop, factual."
        )
    )
    confidence: float = Field(
        description="Confidence in the intent classification, 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Named entities relevant for searching a corporate knowledge "
            "base: policy names, departments, documents, roles, dates, "
            "tools. Only entities explicitly mentioned in the question."
        ),
    )
    search_query: str = Field(
        description=(
            "The question rewritten for semantic search: acronyms expanded, "
            "key concepts explicit, comparison targets or process names "
            "spelled out. Keep it concise — one sentence."
        )
    )
    is_multi_part: bool = Field(
        description=(
            "True if the question contains multiple distinct sub-questions "
            "or asks to compare/contrast across topics."
        )
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description=(
            "For multi-part questions: 2-5 focused, independently answerable "
            "sub-questions preserving the original intent. For simple "
            "questions: a list with just the original question."
        ),
    )


_analysis_prompt = register(
    "unified_analysis",
    "v1",

    [
        (
            "system",
            "You are the query analyzer for an enterprise knowledge assistant. "
            "In a single pass, produce ALL of the following for the user's question:\n\n"
            "1. INTENT — classify into exactly one category:\n"
            "   - informational: general knowledge lookup about a single topic\n"
            "   - comparative: comparing or contrasting two or more topics\n"
            "   - procedural: step-by-step instructions or a process\n"
            "   - analytical: requires reasoning, analysis, or interpretation\n"
            "   - multi_hop: requires connecting information from multiple sources\n"
            "   - factual: simple, direct fact lookup (dates, numbers, names)\n\n"
            "2. ENTITIES — named entities useful for search: policy names, "
            "departments, documents, roles, dates, tools. Only what is "
            "explicitly mentioned; do not infer.\n\n"
            "3. SEARCH QUERY — rewrite the question for semantic retrieval: "
            "expand acronyms, make key concepts explicit. One concise sentence.\n\n"
            "4. DECOMPOSITION — a question is MULTI-PART if it asks about two "
            "or more distinct topics, spans multiple departments, or asks to "
            "compare across sources. If multi-part, decompose into 2-5 "
            "self-contained sub-questions, each independently answerable and "
            "preserving the original intent. Do NOT add topics not mentioned. "
            "If simple, set is_multi_part=false and sub_questions=[the original "
            "question].\n\n"
            "Examples:\n"
            '  "What is the PTO policy?"\n'
            "  -> intent=informational, is_multi_part=false,\n"
            '     sub_questions=["What is the PTO policy?"]\n\n'
            '  "Compare the engineering and HR onboarding processes"\n'
            "  -> intent=comparative, is_multi_part=true,\n"
            '     sub_questions=["What is the engineering department onboarding process?",\n'
            '                    "What is the HR department onboarding process?"]',
        ),
        ("human", "{question}"),
    ]
)


def _fallback_analysis(question: str, normalized: str) -> dict:
    """Zero-LLM analysis mirroring the individual nodes' fallbacks."""
    intent, confidence = _heuristic_intent(question)
    entities = [e.name for e in _regex_extract(question)]
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "extracted_entities": entities,
        "transformed_query": normalized,
        "is_multi_part": False,
        "sub_questions": [question],
    }


@traced
def analyze_query(state: dict) -> dict:
    """Classify intent, extract entities, rewrite, and decompose — one LLM call."""
    question = state.get("question", "")
    base = {
        "original_question": question,
        "sub_answers": [],
        "current_sub_idx": 0,
    }

    if not question.strip():
        return {
            **base,
            "intent": "informational",
            "intent_confidence": 0.0,
            "extracted_entities": [],
            "transformed_query": "",
            "is_multi_part": False,
            "sub_questions": [question],
        }

    normalized = normalize_query(question)

    try:
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        chain = _analysis_prompt | get_llm().with_structured_output(QueryAnalysis)
        result: QueryAnalysis = cb.call(chain.invoke, {"question": question})

        intent = result.intent.lower().strip()
        if intent in VALID_INTENTS:
            confidence = result.confidence
        else:
            logger.warning("Analyzer returned invalid intent %r — using heuristic", intent)
            intent, confidence = _heuristic_intent(question)

        entities = [e.strip() for e in result.entities if e and e.strip()]

        # Build the retrieval query: LLM rewrite (or normalized question),
        # expanded with entities not already present — same expansion the
        # query_transformer applied.
        base_query = result.search_query.strip() or normalized
        extra_terms = [e for e in entities if e.lower() not in base_query.lower()]
        transformed = f"{base_query} ({', '.join(extra_terms)})" if extra_terms else base_query

        subs = [s.strip() for s in result.sub_questions if s and s.strip()]
        subs = subs[: settings.max_sub_questions] or [question]
        is_multi = result.is_multi_part and len(subs) > 1

        logger.info(
            "Query analysis: intent=%s (%.2f), %d entities, multi_part=%s (%d subs)",
            intent, confidence, len(entities), is_multi, len(subs),
        )
        return {
            **base,
            "intent": intent,
            "intent_confidence": confidence,
            "extracted_entities": entities,
            "transformed_query": transformed,
            "is_multi_part": is_multi,
            "sub_questions": subs,
        }

    except CircuitBreakerOpen as exc:
        logger.warning("LLM circuit open during query analysis: %s — using heuristics", exc)
        return {**base, **_fallback_analysis(question, normalized)}
    except Exception:
        logger.exception("Query analysis failed — falling back to heuristics")
        return {**base, **_fallback_analysis(question, normalized)}
