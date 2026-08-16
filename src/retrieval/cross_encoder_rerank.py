"""Cross-encoder reranking retriever using sentence-transformers.

Replaces the expensive LLM-based reranking (one API call per document)
with a dedicated cross-encoder model that runs locally. Batch scoring
makes this significantly faster and cheaper than LLM reranking while
producing more calibrated relevance scores.

Falls back to LLM-based reranking if sentence-transformers is not
installed, allowing graceful degradation.
"""
from __future__ import annotations

import logging
import threading

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from config import settings

logger = logging.getLogger(__name__)

# Module-level cached model (lazy-loaded, thread-safe)
_cross_encoder = None
_model_lock = threading.Lock()


def _get_cross_encoder():
    """Lazy-load the cross-encoder model (singleton)."""
    global _cross_encoder
    with _model_lock:
        if _cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                model_name = settings.cross_encoder_model
                device = settings.cross_encoder_device
                _cross_encoder = CrossEncoder(model_name, device=device)
                logger.info("Cross-encoder model loaded: %s (device=%s)", model_name, device)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for cross-encoder reranking. "
                    "Install with: pip install sentence-transformers>=3.0"
                ) from e
        return _cross_encoder


def reset_cross_encoder() -> None:
    """Reset the cached model (for testing)."""
    global _cross_encoder
    with _model_lock:
        _cross_encoder = None


class CrossEncoderRetriever(BaseRetriever):
    """Retrieves candidates from dense store, then reranks with a cross-encoder model."""

    k: int = Field(default=4, description="Number of final documents to return")
    fetch_k: int = Field(
        default_factory=lambda: settings.rerank_fetch_k,
        description="Candidates to retrieve before reranking",
    )
    filter: dict | None = Field(default=None, description="Metadata filter for dense retriever")

    _model: object = PrivateAttr(default=None)

    def _get_candidates(self, query: str) -> list[Document]:
        from src.vectorstore.chroma_store import get_retriever as _dense
        return _dense(k=self.fetch_k, filter=self.filter).invoke(query)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve candidates, rerank with cross-encoder, return top-k."""
        logger.info("Cross-encoder rerank: %s", query[:120])

        candidates = self._get_candidates(query)
        if not candidates:
            return []

        model = _get_cross_encoder()

        # Build query-document pairs for batch scoring
        pairs = [(query, doc.page_content) for doc in candidates]

        # Batch score all candidates at once
        batch_size = settings.cross_encoder_batch_size
        scores = model.predict(pairs, batch_size=batch_size)

        # Pair scores with documents and sort descending
        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        result = [doc for doc, _ in scored[:self.k]]
        top_scores = [f"{float(s):.3f}" for _, s in scored[:self.k]]
        logger.info(
            "Cross-encoder: %d candidates -> top-%d (scores: %s)",
            len(candidates), len(result), top_scores,
        )
        return result


def build_cross_encoder_retriever(
    k: int | None = None,
    filter: dict | None = None,
) -> CrossEncoderRetriever:
    """Build and return a cross-encoder reranking retriever."""
    return CrossEncoderRetriever(
        k=k or settings.top_k,
        fetch_k=min(15, max(12, (k or settings.top_k) * 3)),
        filter=filter,
    )
