"""Retriever factory — single entry point for all retrieval strategies.

Usage:
    from src.retrieval import get_retriever

    retriever = get_retriever("dense")         # Phase 1 baseline
    retriever = get_retriever("hybrid")        # BM25 + dense + RRF
    retriever = get_retriever("multi_query")   # LLM query expansion
    retriever = get_retriever("rerank")        # dense + LLM reranking

All returned objects satisfy the LangChain BaseRetriever interface
(i.e., have an .invoke(query) -> list[Document] method).
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry pattern — maps strategy name to a lazy builder function.
# Each builder is imported on demand to avoid circular imports.
# ---------------------------------------------------------------------------

def _build_dense(k=None, filter=None):
    from src.vectorstore.chroma_store import get_retriever as _dense
    return _dense(k=k, filter=filter)


def _build_hybrid(k=None, filter=None):
    from src.retrieval.hybrid import build_hybrid_retriever
    return build_hybrid_retriever(k=k, filter=filter)


def _build_multi_query(k=None, filter=None):
    from src.retrieval.multi_query import build_multi_query_retriever
    return build_multi_query_retriever(k=k, filter=filter)


def _build_rerank(k=None, filter=None):
    from src.retrieval.rerank import build_rerank_retriever
    return build_rerank_retriever(k=k, filter=filter)


def _build_hybrid_rerank(k=None, filter=None):
    from src.retrieval.composed import build_composed_retriever
    return build_composed_retriever(k=k, filter=filter, first_stage="hybrid")


def _build_cross_rerank(k=None, filter=None):
    from src.retrieval.cross_encoder_rerank import build_cross_encoder_retriever
    return build_cross_encoder_retriever(k=k, filter=filter)


def _build_hybrid_cross_rerank(k=None, filter=None):
    from src.retrieval.composed import build_composed_retriever
    return build_composed_retriever(k=k, filter=filter, first_stage="hybrid", reranker_type="cross_encoder")


def _build_knowledge_graph(k=None, filter=None):
    from src.knowledge_graph.retriever import build_kg_retriever
    return build_kg_retriever(k=k or 4)


_REGISTRY: dict[str, Callable] = {
    "dense": _build_dense,
    "hybrid": _build_hybrid,
    "multi_query": _build_multi_query,
    "rerank": _build_rerank,
    "hybrid_rerank": _build_hybrid_rerank,
    "cross_rerank": _build_cross_rerank,
    "hybrid_cross_rerank": _build_hybrid_cross_rerank,
    "knowledge_graph": _build_knowledge_graph,
}

STRATEGIES = tuple(_REGISTRY.keys())

# Intent-based routing for the "auto" strategy:
#   factual        — precise single-fact lookup: dense similarity suffices
#   comparative /
#   analytical     — answer quality depends on ranking: full hybrid + rerank
#   everything else — hybrid for keyword+semantic recall
_INTENT_STRATEGY: dict[str, str] = {
    "factual": "dense",
    "informational": "hybrid",
    "procedural": "hybrid",
    "multi_hop": "hybrid",
    "comparative": "hybrid_cross_rerank",
    "analytical": "hybrid_cross_rerank",
}
_AUTO_DEFAULT = "hybrid"


def resolve_strategy(strategy: str, intent: str | None = None) -> str:
    """Resolve the "auto" strategy to a concrete one based on query intent.

    Concrete strategy names pass through unchanged.
    """
    if strategy != "auto":
        return strategy
    return _INTENT_STRATEGY.get((intent or "").lower(), _AUTO_DEFAULT)


def register_strategy(name: str, builder: Callable) -> None:
    """Register a custom retrieval strategy at runtime."""
    _REGISTRY[name] = builder
    global STRATEGIES
    STRATEGIES = tuple(_REGISTRY.keys())


def get_retriever(
    strategy: str = "dense",
    k: int | None = None,
    filter: dict | None = None,
    intent: str | None = None,
) -> BaseRetriever:
    """Return a retriever for the given strategy.

    Args:
        strategy: One of STRATEGIES, or "auto" to route by intent.
        k: Number of final documents to return.
        filter: Metadata filter dict (passed to dense retrievers).
        intent: Query intent used to resolve the "auto" strategy.

    Returns:
        A LangChain BaseRetriever.
    """
    strategy = resolve_strategy(strategy, intent)
    builder = _REGISTRY.get(strategy)
    if builder is None:
        raise ValueError(
            f"Unknown retrieval strategy {strategy!r}. "
            f"Choose from: {', '.join(STRATEGIES)}"
        )
    return builder(k=k, filter=filter)
