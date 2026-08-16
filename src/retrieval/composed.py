"""Composed retriever: chains a first-stage retriever with a reranker.

This enables strategy combinations like hybrid + rerank, where the first
stage maximizes recall (hybrid BM25+dense) and the second stage maximizes
precision (LLM reranking).
"""
from __future__ import annotations

import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr

from config import settings
from src.llm_pool import get_llm

logger = logging.getLogger(__name__)


class ComposedRetriever(BaseRetriever):
    """Chains a first-stage retriever with reranking (LLM or cross-encoder)."""

    k: int = Field(default=4, description="Number of final documents to return")
    first_stage_strategy: str = Field(default="hybrid", description="First stage retriever strategy")
    filter: dict | None = Field(default=None, description="Metadata filter for retriever")
    reranker_type: str = Field(default="llm", description="Reranker backend: 'llm' or 'cross_encoder'")

    _first_stage: BaseRetriever | None = PrivateAttr(default=None)

    def _get_first_stage(self) -> BaseRetriever:
        """Lazily build the first-stage retriever."""
        if self._first_stage is None:
            from src.retrieval.factory import get_retriever
            # Fetch more candidates than final k for reranking
            fetch_k = min(15, self.k * 3)
            self._first_stage = get_retriever(
                strategy=self.first_stage_strategy,
                k=fetch_k,
                filter=self.filter,
            )
        return self._first_stage

    def _rerank_with_llm(self, query: str, candidates: list[Document]) -> list[Document]:
        """Rerank candidates using LLM scoring."""
        from src.retrieval.rerank import _RERANK_SYSTEM, RelevanceScore

        llm = get_llm(temperature=0)

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _score(doc: Document) -> tuple[Document, int]:
            try:
                scorer = llm.with_structured_output(RelevanceScore)
                result: RelevanceScore = scorer.invoke(
                    [
                        {"role": "system", "content": _RERANK_SYSTEM},
                        {
                            "role": "human",
                            "content": (
                                f"Question: {query}\n\n"
                                f"Document (source: {doc.metadata.get('filename', '?')}):\n"
                                f"{doc.page_content}"
                            ),
                        },
                    ]
                )
                return (doc, result.score)
            except Exception:
                return (doc, 5)

        scored: list[tuple[Document, int]] = []
        max_workers = min(settings.rerank_max_workers, len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_score, doc): doc for doc in candidates}
            for future in as_completed(futures):
                scored.append(future.result())

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:self.k]]

    def _rerank_with_cross_encoder(self, query: str, candidates: list[Document]) -> list[Document]:
        """Rerank candidates using a cross-encoder model."""
        from src.retrieval.cross_encoder_rerank import _get_cross_encoder

        model = _get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = model.predict(pairs, batch_size=settings.cross_encoder_batch_size)

        scored = list(zip(candidates, scores, strict=True))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        return [doc for doc, _ in scored[:self.k]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve via first stage, then rerank."""
        logger.info("Composed retrieval (%s + %s rerank): %s", self.first_stage_strategy, self.reranker_type, query[:120])

        # First stage: broad recall
        candidates = self._get_first_stage().invoke(query)
        if not candidates:
            return []

        # Second stage: reranking
        if self.reranker_type == "cross_encoder":
            result = self._rerank_with_cross_encoder(query, candidates)
        else:
            result = self._rerank_with_llm(query, candidates)

        logger.info(
            "Composed: %d candidates -> top-%d reranked (%s)",
            len(candidates), len(result), self.reranker_type,
        )
        return result


def build_composed_retriever(
    k: int | None = None,
    filter: dict | None = None,
    first_stage: str = "hybrid",
    reranker_type: str = "llm",
) -> ComposedRetriever:
    """Build and return a composed (first-stage + rerank) retriever."""
    return ComposedRetriever(
        k=k or settings.top_k,
        first_stage_strategy=first_stage,
        filter=filter,
        reranker_type=reranker_type,
    )
