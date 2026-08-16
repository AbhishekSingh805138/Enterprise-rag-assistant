"""Multi-query expansion retriever.

A single user question may miss relevant documents because of vocabulary
mismatch. This retriever asks the LLM to generate N alternative phrasings
of the question, runs each through the dense retriever, then deduplicates
and returns the union. This improves recall at a modest LLM cost.

Example: "What's the password policy?"
  → "What are the minimum password requirements?"
  → "How often must employees change their passwords?"
  → "What is the company's credential security policy?"
Each variant may surface documents the original query missed.
"""
from __future__ import annotations

import hashlib
import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from config import settings
from src.prompts import register
from src.llm_pool import get_llm

logger = logging.getLogger(__name__)

NUM_VARIANTS = settings.num_query_variants

_EXPAND_PROMPT = register(
    "multi_query_expand",
    "v1",

    [
        (
            "system",
            "You are an expert at reformulating search queries for an enterprise "
            "document retrieval system. Given the user's original question, generate "
            "exactly {n} alternative phrasings that might surface different relevant "
            "documents. Each variant should approach the question from a different "
            "angle — different keywords, synonyms, or specificity levels.\n\n"
            "Return ONLY the {n} variants, one per line, no numbering or bullets.",
        ),
        ("human", "{question}"),
    ]
)


def _generate_variants(question: str, n: int = NUM_VARIANTS) -> list[str]:
    """Use the LLM to generate alternative query phrasings."""
    # Pooled, and covered by provider failover — a direct ChatOpenAI here
    # would keep this path pinned to one vendor.
    llm = get_llm(temperature=0.7)
    chain = _EXPAND_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "n": n})
    variants = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    logger.info("Generated %d query variants for: %s", len(variants), question[:80])
    return variants


class MultiQueryRetriever(BaseRetriever):
    """Expands a query into N variants and retrieves the union of results."""

    k: int = Field(default=4, description="Number of final documents to return")
    per_query_k: int = Field(default=4, description="Documents to retrieve per variant")
    filter: dict | None = Field(default=None, description="Metadata filter for dense retriever")

    def _get_dense_results(self, query: str) -> list[Document]:
        from src.vectorstore.chroma_store import get_retriever as _dense
        return _dense(k=self.per_query_k, filter=self.filter).invoke(query)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve via multi-query expansion + deduplication."""
        logger.info("Multi-query retrieval: %s", query[:120])

        # Generate variants
        try:
            variants = _generate_variants(query)
        except Exception:
            logger.exception("Query expansion failed — falling back to single query")
            variants = []

        # Include the original query
        all_queries = [query] + variants

        # Retrieve for each query
        seen_keys: set[str] = set()
        unique_docs: list[Document] = []

        for q in all_queries:
            docs = self._get_dense_results(q)
            for doc in docs:
                key = hashlib.md5(doc.page_content.encode()).hexdigest()
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_docs.append(doc)

        # Return top-k (ordered by first appearance — original query docs first)
        result = unique_docs[: self.k]
        logger.info(
            "Multi-query: %d queries → %d unique docs → returning %d",
            len(all_queries), len(unique_docs), len(result),
        )
        return result


def build_multi_query_retriever(
    k: int | None = None,
    filter: dict | None = None,
) -> MultiQueryRetriever:
    """Build and return a multi-query expansion retriever."""
    return MultiQueryRetriever(
        k=k or settings.top_k,
        filter=filter,
    )
