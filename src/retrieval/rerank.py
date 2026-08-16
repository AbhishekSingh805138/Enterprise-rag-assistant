"""Reranking retriever: dense retrieval + LLM-based cross-encoder reranking.

Embedding-based retrieval is fast but coarse — cosine similarity doesn't
capture nuanced relevance. Reranking retrieves a larger candidate set
(fetch_k) from the dense store, then uses the LLM as a cross-encoder to
score each candidate's relevance to the query. The top-k by relevance
score are returned.

This is particularly effective for improving **context precision** and
**faithfulness** — the generation model sees only the most relevant chunks,
reducing hallucination from noisy context.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from config import settings
from src.llm_pool import get_llm
from src.prompts import register

logger = logging.getLogger(__name__)


class RelevanceScore(BaseModel):
    """Structured score from the LLM reranker."""
    score: int = Field(
        description="Relevance score from 0 (irrelevant) to 10 (perfectly relevant)"
    )
    reasoning: str = Field(
        description="One-sentence explanation of the score"
    )


_RERANK_SYSTEM = (
    "You are a relevance assessor. Given a question and a document chunk, "
    "rate how relevant the document is for answering the question.\n\n"
    "Score 0-10:\n"
    "  0-2: Completely irrelevant or off-topic\n"
    "  3-4: Tangentially related but wouldn't help answer\n"
    "  5-6: Somewhat relevant, contains partial information\n"
    "  7-8: Relevant, contains key information for the answer\n"
    "  9-10: Highly relevant, directly answers the question\n\n"
    "Be strict — only score 7+ if the document clearly helps answer the question."
)

# Registered even though it is passed as a raw system message rather than
# a template: a prompt left out of the inventory is a prompt whose change
# cannot be correlated with a change in retrieval quality.
register("rerank_relevance", "v1", [("system", _RERANK_SYSTEM)])


class RerankRetriever(BaseRetriever):
    """Retrieves candidates from dense store, then reranks with LLM scoring."""

    k: int = Field(default=4, description="Number of final documents to return")
    fetch_k: int = Field(default_factory=lambda: settings.rerank_fetch_k, description="Candidates to retrieve before reranking")
    filter: dict | None = Field(default=None, description="Metadata filter for dense retriever")

    def _get_candidates(self, query: str) -> list[Document]:
        from src.vectorstore.chroma_store import get_retriever as _dense
        return _dense(k=self.fetch_k, filter=self.filter).invoke(query)

    def _score_document(
        self,
        llm: Runnable,
        question: str,
        doc: Document,
    ) -> tuple[Document, int]:
        """Score a single document's relevance to the question."""
        try:
            scorer = llm.with_structured_output(RelevanceScore)
            result: RelevanceScore = scorer.invoke(
                [
                    {"role": "system", "content": _RERANK_SYSTEM},
                    {
                        "role": "human",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Document (source: {doc.metadata.get('filename', '?')}):\n"
                            f"{doc.page_content}"
                        ),
                    },
                ]
            )
            return (doc, result.score)
        except Exception:
            logger.debug("Rerank scoring failed for a chunk — assigning score 5")
            return (doc, 5)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve candidates, rerank with LLM, return top-k."""
        logger.info("Rerank retrieval: %s", query[:120])

        candidates = self._get_candidates(query)
        if not candidates:
            return []

        llm = get_llm(temperature=0)

        # Score documents in parallel using thread pool
        max_workers = min(settings.rerank_max_workers, len(candidates))
        scored: list[tuple[Document, int]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._score_document, llm, query, doc): doc
                for doc in candidates
            }
            for future in as_completed(futures):
                scored.append(future.result())
        scored.sort(key=lambda x: x[1], reverse=True)

        result = [doc for doc, _score in scored[: self.k]]
        logger.info(
            "Rerank: %d candidates → top-%d (scores: %s)",
            len(candidates),
            len(result),
            [s for _, s in scored[: self.k]],
        )
        return result


def build_rerank_retriever(
    k: int | None = None,
    filter: dict | None = None,
) -> RerankRetriever:
    """Build and return a reranking retriever."""
    return RerankRetriever(
        k=k or settings.top_k,
        fetch_k=min(15, max(12, (k or settings.top_k) * 3)),
        filter=filter,
    )
