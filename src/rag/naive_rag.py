"""Phase 1: naive RAG baseline.

load -> chunk -> embed -> retrieve top-k -> stuff into prompt -> generate.
This is intentionally simple. It exists to establish a baseline you will
measure (Phase 2) and beat (Phase 3+). Built with LCEL so the pipe operator
makes the data flow explicit.
"""
from __future__ import annotations

import logging
import time
import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer the question using ONLY "
    "the context below. Follow these rules:\n\n"
    "1. If the context contains partial information, provide what you can "
    "and note what specific aspects are not covered in the documents.\n"
    "2. Only say you cannot answer if the context is completely irrelevant "
    "to the question. In that case respond with: "
    '"I don\'t have enough information in the available documents to answer this '
    'question."\n'
    "3. Do NOT invent facts or numbers not present in the context.\n"
    "4. Cite the source filename in parentheses after each claim — "
    "e.g. (handbook.md).\n"
    "5. If the answer draws from multiple sources, cite each one.\n\n"
    "Context:\n{context}"
)

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('filename', d.metadata.get('source', '?'))}]\n{d.page_content}"
        for d in docs
    )


def build_naive_rag_chain(
    k: int | None = None,
    filter: dict | None = None,
    retriever_strategy: str = "dense",
) -> Runnable:
    """Compose the LCEL chain: {context, question} -> prompt -> llm -> str."""
    from src.retrieval import get_retriever
    from src.llm_pool import get_llm
    retriever = get_retriever(strategy=retriever_strategy, k=k, filter=filter)
    llm = get_llm(temperature=0)

    context_and_question = RunnableParallel(
        context=retriever | _format_docs,
        question=RunnablePassthrough(),
    )

    return context_and_question | _prompt | llm | StrOutputParser()


def answer(
    question: str,
    k: int | None = None,
    filter: dict | None = None,
    retriever_strategy: str = "dense",
) -> str:
    """Answer a question using naive RAG. Returns the LLM response string."""
    if not question or not question.strip():
        return "Please provide a question."

    logger.info("Naive RAG query (retriever=%s): %s", retriever_strategy, question[:120])

    from src.observability.cost_callback import CostCallbackHandler

    handler = CostCallbackHandler()
    tid = uuid.uuid4().hex[:12]

    try:
        chain = build_naive_rag_chain(
            k=k, filter=filter, retriever_strategy=retriever_strategy,
        )
        start = time.perf_counter()
        result = chain.invoke(question, config={"callbacks": [handler]})
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("Naive RAG response length: %d chars", len(result))

        # Record metrics — never let metrics failures break the query
        try:
            from src.observability.cost_callback import is_idk_response

            metrics = handler.flush(
                thread_id=tid,
                question=question,
                latency_ms=latency_ms,
                retriever_strategy=retriever_strategy,
                mode="naive",
                is_idk=is_idk_response(result),
            )
            from src.observability.metrics_store import get_store
            get_store().record(metrics)
            logger.info(
                "Query metrics: $%.5f, %d tokens, %.0fms, idk=%s",
                metrics.estimated_cost_usd, metrics.total_tokens, metrics.latency_ms,
                metrics.is_idk,
            )
        except Exception:
            logger.debug("Failed to record query metrics", exc_info=True)

        return result
    except Exception:
        logger.exception("Naive RAG failed for query: %s", question[:120])
        raise
