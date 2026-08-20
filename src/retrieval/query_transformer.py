"""Unified pre-retrieval query transformation pipeline.

Chains normalization, entity extraction, and intent-aware rewriting
into a single transform step. The transformed query replaces inline
normalization in the retrieve node.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser

from config import settings
from src.graph.tracing import traced
from src.llm_pool import get_llm
from src.prompts import register
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker
from src.retrieval.entity_extractor import Entity, extract_entities
from src.retrieval.normalizer import normalize_query

logger = logging.getLogger(__name__)


@dataclass
class TransformedQuery:
    """Result of the query transformation pipeline."""
    original: str
    normalized: str
    expanded: str  # final query used for retrieval
    entities: list[Entity] = field(default_factory=list)
    rewritten: str = ""  # intent-aware rewrite (empty if not applied)


# Intent-aware rewriting prompts
_REWRITE_PROMPTS: dict[str, str] = {
    "comparative": (
        "The user is asking a comparison question. Rewrite the query to "
        "separately identify the items being compared, so retrieval can "
        "find relevant documents for each. Keep the query concise.\n\n"
        "Original: {query}\nRewritten:"
    ),
    "procedural": (
        "The user is asking for step-by-step instructions. Rewrite the query "
        "to focus on the process or procedure name, adding synonyms like "
        "'steps', 'process', 'how to', 'guide'.\n\n"
        "Original: {query}\nRewritten:"
    ),
    "analytical": (
        "The user is asking an analytical question. Rewrite the query to "
        "capture the key concepts and relationships being analyzed. Add "
        "relevant context terms.\n\n"
        "Original: {query}\nRewritten:"
    ),
    "multi_hop": (
        "The user's question requires connecting multiple pieces of information. "
        "Rewrite the query to include all key concepts and their relationships, "
        "making it easier for retrieval to find connecting documents.\n\n"
        "Original: {query}\nRewritten:"
    ),
}


def _intent_rewrite(query: str, intent: str) -> str:
    """Rewrite a query based on its detected intent. Returns empty string if no rewrite needed."""
    prompt_template = _REWRITE_PROMPTS.get(intent)
    if not prompt_template:
        return ""

    try:
        prompt = register(
            "query_transform",
            "v1",
            [
            ("system", prompt_template),
        ])
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        chain = prompt | get_llm(temperature=0.3) | StrOutputParser()
        rewritten = cb.call(chain.invoke, {"query": query})
        logger.debug("Intent rewrite (%s): %s → %s", intent, query[:60], rewritten[:60])
        return rewritten.strip()
    except (CircuitBreakerOpen, Exception):
        logger.debug("Intent rewrite failed for intent=%s", intent, exc_info=True)
        return ""


def transform_query(query: str, intent: str = "informational") -> TransformedQuery:
    """Run the full query transformation pipeline.

    Steps:
    1. Normalize (acronym expansion, whitespace cleanup)
    2. Extract entities (LLM or regex fallback)
    3. Intent-aware rewrite (for comparative, procedural, analytical, multi_hop)
    4. Build final expanded query
    """
    if not query.strip():
        return TransformedQuery(original=query, normalized=query, expanded=query)

    # Step 1: Normalize
    normalized = normalize_query(query)

    # Step 2: Extract entities
    entities = extract_entities(query)
    entity_names = [e.name for e in entities]

    # Step 3: Intent-aware rewrite
    rewritten = _intent_rewrite(normalized, intent)

    # Step 4: Build expanded query
    # Use rewritten if available, otherwise normalized
    base = rewritten if rewritten else normalized

    # Append entity context if entities were found that aren't already in the base
    extra_terms = [
        name for name in entity_names
        if name.lower() not in base.lower()
    ]
    if extra_terms:
        expanded = f"{base} ({', '.join(extra_terms)})"
    else:
        expanded = base

    return TransformedQuery(
        original=query,
        normalized=normalized,
        expanded=expanded,
        entities=entities,
        rewritten=rewritten,
    )


# --- Graph node ---------------------------------------------------------------

@traced
def query_transform_node(state: dict) -> dict:
    """Graph node: transform the query before retrieval."""
    if not settings.query_transform_enabled:
        return {}

    question = state.get("question", "")
    intent = state.get("intent", "informational")

    result = transform_query(question, intent)
    logger.info(
        "Query transformed: %d entities, rewritten=%s, expanded=%s",
        len(result.entities), bool(result.rewritten), result.expanded[:100],
    )

    return {
        "transformed_query": result.expanded,
        "extracted_entities": [e.name for e in result.entities],
    }
