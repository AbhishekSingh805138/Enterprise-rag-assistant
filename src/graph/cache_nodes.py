"""Semantic cache nodes for the CRAG graph.

- cache_lookup: checks if a similar query was recently answered (short-circuit)
- cache_store: saves the answer after critic verification
"""
from __future__ import annotations

import json
import logging

from config import settings
from src.graph.tracing import traced

logger = logging.getLogger(__name__)


def _cache_scope(state: dict) -> str:
    """Serialize the retrieval filter into a stable cache scope key.

    Queries with different metadata filters (e.g. department) must never
    share cache entries — that would bypass access boundaries.
    """
    filter_dict = state.get("filter") or {}
    if not filter_dict:
        return ""
    return json.dumps(dict(sorted(filter_dict.items())), sort_keys=True)


@traced
def cache_lookup(state: dict) -> dict:
    """Check semantic cache for a previous answer to a similar query."""
    if not settings.semantic_cache_enabled:
        return {"cache_hit": False}

    question = state.get("question", "")
    strategy = state.get("retriever_strategy", "dense")

    if not question.strip():
        return {"cache_hit": False}

    # Skip the cache for follow-up turns: the same question text can mean
    # something different depending on conversation history.
    if state.get("memory_context"):
        logger.debug("Cache skipped — conversation context present")
        return {"cache_hit": False}

    try:
        from src.cache.semantic_cache import get_cache

        cache = get_cache()
        cached_answer = cache.lookup(
            query=question,
            mode="graph",
            strategy=strategy,
            scope=_cache_scope(state),
        )

        if cached_answer is not None:
            logger.info("Cache HIT for: %s", question[:80])
            return {
                "cache_hit": True,
                "generation": cached_answer,
            }
    except Exception:
        logger.debug("Cache lookup failed", exc_info=True)

    return {"cache_hit": False}


@traced
def cache_store(state: dict) -> dict:
    """Store the answer in the semantic cache after generation."""
    if not settings.semantic_cache_enabled:
        return {}

    # Store under the user's original question (query rewrites may have
    # replaced state["question"]) so lookup and store keys match.
    question = state.get("original_question") or state.get("question", "")
    generation = state.get("generation", "")
    strategy = state.get("retriever_strategy", "dense")

    if not question.strip() or not generation.strip():
        return {}

    # Answers shaped by conversation history are not reusable across sessions.
    if state.get("memory_context"):
        logger.debug("Cache store skipped — conversation context present")
        return {}

    # Don't cache IDK responses
    idk_phrases = ["don't have enough information", "cannot answer", "cannot process"]
    if any(phrase in generation.lower() for phrase in idk_phrases):
        logger.debug("Skipping cache for IDK response")
        return {}

    try:
        from src.cache.semantic_cache import get_cache

        cache = get_cache()
        cache.store(
            query=question,
            answer=generation,
            mode="graph",
            strategy=strategy,
            scope=_cache_scope(state),
        )
        logger.debug("Cached answer for: %s", question[:80])
    except Exception:
        logger.debug("Cache store failed", exc_info=True)

    return {}


def route_after_cache(state: dict) -> str:
    """Route based on cache lookup: hit -> end, miss -> continue."""
    if state.get("cache_hit", False):
        return "cache_hit"
    return "cache_miss"
