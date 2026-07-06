"""Phase 3: Advanced retrieval strategies.

Provides a factory that returns a LangChain-compatible retriever for any
configured strategy. All strategies share the same interface so they plug
directly into the naive RAG chain and CRAG graph nodes.

Strategies:
    dense       — ChromaDB cosine similarity (Phase 1 baseline)
    hybrid      — BM25 sparse + dense + Reciprocal Rank Fusion
    multi_query — LLM-generated query variants with union retrieval
    rerank      — Dense retrieval + LLM cross-encoder reranking
"""
from src.retrieval.factory import get_retriever, resolve_strategy, STRATEGIES
from src.retrieval.normalizer import normalize_query
from src.retrieval.dept_detector import detect_department

__all__ = ["get_retriever", "resolve_strategy", "STRATEGIES", "normalize_query", "detect_department"]
