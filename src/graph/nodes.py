"""Node functions for the Corrective RAG graph.

Each node is a plain function: (state) -> partial state update.
Keeping nodes small and pure makes them unit-testable and easy to trace
in LangSmith.

The grader uses structured output (Pydantic) so the model returns a typed
boolean rather than free text you have to parse.
"""
from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import settings
from src.graph.tracing import traced
from src.llm_pool import get_llm
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker
from src.retrieval import get_retriever

logger = logging.getLogger(__name__)

MAX_RETRIES = settings.max_retries


def _llm(temperature: float = 0) -> ChatOpenAI:
    """Return a cached LLM instance from the pool."""
    return get_llm(temperature=temperature)


# --- Node: retrieve --------------------------------------------------------

@traced
def retrieve(state: dict) -> dict:
    """Retrieve documents for the current question."""
    question = state["question"]
    from src.retrieval.factory import resolve_strategy
    strategy = resolve_strategy(
        state.get("retriever_strategy", "dense"), state.get("intent")
    )

    # Phase 12: use transformed query if available, else normalize inline
    transformed = state.get("transformed_query", "")
    if transformed:
        normalized = transformed
    else:
        from src.retrieval.normalizer import normalize_query
        normalized = normalize_query(question)

    # Phase 8: auto-detect department filter if none provided
    filter_dict = state.get("filter")
    if not filter_dict:
        from src.retrieval.dept_detector import detect_department
        dept = detect_department(normalized)
        if dept:
            filter_dict = {"department": dept}
            logger.info("Auto-detected department: %s", dept)

    # Explicit top_k from the API request wins over adaptive heuristics
    k = state.get("top_k")
    if k is None and settings.adaptive_k:
        word_count = len(normalized.split())
        if word_count <= 5:
            k = settings.adaptive_k_min
        elif word_count >= 15:
            k = settings.adaptive_k_max
        else:
            # Linear interpolation between min and max
            ratio = (word_count - 5) / 10.0
            k = int(settings.adaptive_k_min + ratio * (settings.adaptive_k_max - settings.adaptive_k_min))

    logger.info("Retrieve node (strategy=%s) — query: %s", strategy, normalized[:120])
    try:
        cb = get_breaker(
            "retrieval",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        docs = cb.call(
            lambda: get_retriever(strategy=strategy, k=k, filter=filter_dict).invoke(normalized)
        )
        logger.info("Retrieved %d document(s)", len(docs))
    except CircuitBreakerOpen as exc:
        logger.warning("Retrieval circuit open: %s", exc)
        docs = []
    except Exception:
        logger.exception("Retrieval failed for: %s", normalized[:120])
        docs = []
    return {"documents": docs, "retries": state.get("retries", 0)}


# --- Node: grade documents -------------------------------------------------

class GradeResult(BaseModel):
    """Structured verdict from the relevance grader."""
    relevant: bool = Field(
        description="True if the documents contain enough information to answer the question."
    )


_grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether the retrieved documents are "
            "relevant and sufficient to answer the user's question.\n\n"
            "Return relevant=true ONLY if the documents clearly contain "
            "information that can be used to answer the question. "
            "If the documents are off-topic, too vague, or don't address the "
            "question, return relevant=false.",
        ),
        ("human", "Question:\n{question}\n\nDocuments:\n{context}"),
    ]
)


_per_doc_grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a grader assessing whether a SINGLE retrieved document is "
            "relevant to the user's question.\n\n"
            "Return relevant=true if the document contains information that "
            "can help answer the question. Return relevant=false if the "
            "document is off-topic or irrelevant.",
        ),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]
)


def _grade_single_doc(question: str, doc: Document) -> tuple[Document, bool]:
    """Grade a single document for relevance. Returns (doc, is_relevant)."""
    try:
        grader = _per_doc_grade_prompt | _llm().with_structured_output(GradeResult)
        verdict: GradeResult = grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        return doc, verdict.relevant
    except Exception:
        logger.debug("Per-doc grading failed for one document — keeping it")
        return doc, True  # keep on failure


@traced
def grade_documents(state: dict) -> dict:
    """Grade whether retrieved documents are relevant to the question.

    When PER_DOC_GRADING is enabled, each document is graded individually
    (optionally in parallel) and irrelevant ones are filtered out. Otherwise,
    all documents are graded as a batch.
    """
    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("No documents to grade")
        return {"relevant": False}

    # Per-document grading mode
    if settings.per_doc_grading:
        try:
            max_workers = min(settings.rerank_max_workers, len(documents))
            relevant_docs = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # copy_context propagates LangChain's callback config into
                # worker threads so token usage is still cost-tracked.
                futures = {
                    executor.submit(
                        contextvars.copy_context().run, _grade_single_doc, question, doc
                    ): doc
                    for doc in documents
                }
                for future in as_completed(futures):
                    doc, is_relevant = future.result()
                    if is_relevant:
                        relevant_docs.append(doc)

            # Preserve original document order
            original_order = {id(d): i for i, d in enumerate(documents)}
            relevant_docs.sort(key=lambda d: original_order.get(id(d), 0))

            logger.info(
                "Per-doc grading: %d/%d documents relevant",
                len(relevant_docs), len(documents),
            )
            return {
                "relevant": len(relevant_docs) > 0,
                "documents": relevant_docs,
            }
        except Exception:
            logger.exception("Per-doc grading failed — falling back to batch grading")
            # Fall through to batch grading

    # Batch grading mode (default)
    try:
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        grader = _grade_prompt | _llm().with_structured_output(GradeResult)
        context = "\n\n".join(d.page_content for d in documents)
        verdict: GradeResult = cb.call(
            grader.invoke, {"question": question, "context": context}
        )
        logger.info("Grader verdict: relevant=%s", verdict.relevant)
        return {"relevant": verdict.relevant}
    except CircuitBreakerOpen as exc:
        # LLM is down — the rewrite/retry path would also fail, so don't take it.
        logger.warning("LLM circuit open during grading: %s", exc)
        return {"relevant": False}
    except Exception:
        # A grader error is not evidence the documents are bad. Returning
        # False here would trigger the expensive rewrite→retry→web-search
        # path on a transient failure. Proceed to generation — the critic
        # still verifies claims against the sources.
        logger.exception("Grading failed — proceeding to generation (critic will verify)")
        return {"relevant": True}


# --- Node: transform query (corrective step) -------------------------------

_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriter. The previous retrieval returned documents "
            "that were NOT relevant to the user's question. Rewrite the question "
            "to target different aspects or use different keywords.\n\n"
            "Context about what was already retrieved (and rejected):\n"
            "{rejected_context}\n\n"
            "Guidelines:\n"
            "- Expand acronyms, add synonyms, and rephrase for better semantic search\n"
            "- Try to approach the topic from a different angle than before\n"
            "- Return ONLY the rewritten question, nothing else.",
        ),
        ("human", "{question}"),
    ]
)


@traced
def transform_query(state: dict) -> dict:
    """Rewrite the question for better retrieval on the next attempt."""
    original = state["question"]
    retries = state.get("retries", 0) + 1

    # Build a summary of rejected documents for informed rewriting
    documents = state.get("documents", [])
    if documents:
        rejected_context = "; ".join(
            d.page_content[:100] for d in documents[:3]
        )
    else:
        rejected_context = "(no documents were retrieved)"

    try:
        rewriter = _rewrite_prompt | _llm(temperature=0.3) | StrOutputParser()
        rewritten = rewriter.invoke(
            {"question": original, "rejected_context": rejected_context}
        )
        logger.info("Query rewritten (retry %d): %s → %s", retries, original[:80], rewritten[:80])
    except Exception:
        logger.exception("Query rewrite failed — keeping original")
        rewritten = original

    return {"question": rewritten, "retries": retries}


# --- Node: web search fallback (stub) --------------------------------------

@traced
def web_search(state: dict) -> dict:
    """Search the web via Tavily when corpus retrieval is insufficient.

    Appends web results as Documents alongside any existing retrieved docs.
    Gracefully degrades to a placeholder if TAVILY_API_KEY is not set.
    """
    question = state["question"]
    existing_docs = state.get("documents", [])

    if not settings.tavily_api_key:
        logger.warning("Web search skipped — TAVILY_API_KEY not set")
        fallback = Document(
            page_content="(web search unavailable — no API key configured)",
            metadata={"filename": "web_fallback", "source": "web"},
        )
        return {
            "documents": existing_docs + [fallback],
            "web_fallback_used": True,
        }

    try:
        from tavily import TavilyClient

        cb = get_breaker(
            "tavily",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        client = TavilyClient(api_key=settings.tavily_api_key)
        results = cb.call(client.search, query=question, max_results=3)

        web_docs = [
            Document(
                page_content=r["content"],
                metadata={
                    "source": r["url"],
                    "filename": "web_search",
                    "title": r.get("title", ""),
                },
            )
            for r in results.get("results", [])
        ]
        logger.info("Web search returned %d result(s) for: %s", len(web_docs), question[:80])
        return {
            "documents": existing_docs + web_docs,
            "web_fallback_used": True,
        }
    except CircuitBreakerOpen as exc:
        logger.warning("Tavily circuit open: %s", exc)
        fallback = Document(
            page_content="(web search temporarily unavailable — circuit breaker open)",
            metadata={"filename": "web_fallback", "source": "web"},
        )
        return {
            "documents": existing_docs + [fallback],
            "web_fallback_used": True,
        }
    except Exception:
        logger.exception("Web search failed for: %s", question[:80])
        fallback = Document(
            page_content="(web search failed — see logs for details)",
            metadata={"filename": "web_fallback", "source": "web"},
        )
        return {
            "documents": existing_docs + [fallback],
            "web_fallback_used": True,
        }


# --- Node: generate --------------------------------------------------------

_GEN_SYSTEM_BASE = (
    "You are an enterprise knowledge assistant. Answer the question "
    "using the provided context. Follow these rules:\n\n"
    "1. If the context contains partial information, provide what you can "
    "and note what specific aspects are not covered in the documents.\n"
    "2. Only say you cannot answer if the context is completely irrelevant "
    "to the question. In that case respond with: "
    '"I don\'t have enough information in the available documents to '
    'answer this question."\n'
    "3. Do NOT invent facts or numbers not present in the context.\n"
    "4. Cite the source filename in parentheses after each claim — "
    "e.g. (handbook.md).\n"
    "5. If the answer draws from multiple sources, cite each one."
)

_GEN_COT_ADDENDUM = (
    "\n\n6. Before answering, briefly reason through which parts of the "
    "context are relevant and how they connect to the question. Present "
    "your reasoning in a <thinking> block, then give the final answer."
)

_GEN_MEMORY_ADDENDUM = (
    "\n\nThe user may be referring to a previous conversation. Use the "
    "conversation history below for context about what was discussed, but "
    "still answer based on the retrieved documents.\n\n{memory_context}"
)


def _build_gen_prompt(*, has_memory: bool = False) -> ChatPromptTemplate:
    """Build generation prompt, optionally with chain-of-thought and memory."""
    system = _GEN_SYSTEM_BASE
    if settings.chain_of_thought:
        system += _GEN_COT_ADDENDUM
    if has_memory:
        system += _GEN_MEMORY_ADDENDUM
    system += "\n\nContext:\n{context}"
    return ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{question}")]
    )


_gen_prompt = _build_gen_prompt()
_gen_prompt_with_memory = _build_gen_prompt(has_memory=True)


@traced
def generate(state: dict) -> dict:
    """Generate the final cited answer from retrieved context."""
    question = state["question"]
    documents = state.get("documents", [])
    memory_context = state.get("memory_context", "")

    if not documents:
        logger.warning("Generate called with no documents")
        return {
            "generation": (
                "I don't have enough information in the available documents "
                "to answer this question."
            )
        }

    try:
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        # Use memory-aware prompt when conversation history is available
        prompt = _gen_prompt_with_memory if memory_context else _gen_prompt
        chain = prompt | _llm() | StrOutputParser()
        # Phase 14: use context builder for intelligent context construction
        from src.context.context_builder import build_context
        built = build_context(documents, query=question, memory_context=memory_context)
        context = built.text
        # Include tool results if available
        tool_results = state.get("tool_results", [])
        if tool_results:
            tool_context = "\n".join(f"[tool] {r}" for r in tool_results)
            context = f"{context}\n\n{tool_context}"
        invoke_args = {"question": question, "context": context}
        if memory_context:
            invoke_args["memory_context"] = memory_context
        answer = cb.call(chain.invoke, invoke_args)
        logger.info("Generated answer: %d chars", len(answer))
        return {"generation": answer}
    except CircuitBreakerOpen as exc:
        logger.warning("LLM circuit open during generation: %s", exc)
        return {
            "generation": "The service is temporarily unavailable. Please try again shortly."
        }
    except Exception:
        logger.exception("Generation failed for: %s", question[:120])
        return {
            "generation": "An error occurred while generating the answer. Please try again."
        }


# --- Node: critic (claim verification) ------------------------------------

class ClaimVerdict(BaseModel):
    """Structured output for claim-level verification."""
    supported_claims: list[str] = Field(
        description="List of claims that ARE supported by the source documents"
    )
    unsupported_claims: list[str] = Field(
        description="List of claims that are NOT supported by the source documents"
    )


_critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a fact-checking critic for an enterprise knowledge assistant. "
            "Your job is to verify every factual claim in the generated answer "
            "against the source documents.\n\n"
            "Rules:\n"
            "1. A claim is SUPPORTED if the source documents explicitly state "
            "or directly imply the information.\n"
            "2. A claim is UNSUPPORTED if it adds details, numbers, or facts not "
            "found in the sources — even if the claim seems reasonable.\n"
            "3. Paraphrases and reasonable inferences from source content are SUPPORTED. "
            "When in doubt, mark as SUPPORTED.\n"
            "4. General framing sentences (e.g. 'According to the documents...') "
            "are SUPPORTED — they are not factual claims.\n"
            "5. Citations like '(handbook.md)' are not claims themselves.\n"
            "6. Summaries that accurately condense source information are SUPPORTED.\n"
            "7. Be lenient: only mark a claim as UNSUPPORTED when it clearly "
            "contradicts or fabricates information beyond what's in the sources.\n\n"
            "Extract each distinct factual claim from the answer, then classify it.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Generated Answer:\n{answer}\n\n"
            "Source Documents:\n{context}",
        ),
    ]
)

_rewrite_prompt_critic = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are rewriting an answer to remove unsupported claims while "
            "preserving the structure and flow of the original answer. "
            "Remove ONLY the unsupported claims, keeping the rest of the "
            "answer intact including citations. "
            "Maintain natural transitions between remaining points. "
            "If no supported claims remain, respond with exactly: "
            '"I don\'t have enough information in the available documents '
            'to answer this question."\n\n'
            "Do NOT add any new information. Only use what is in the supported claims.",
        ),
        (
            "human",
            "Original question: {question}\n\n"
            "Original answer:\n{original_answer}\n\n"
            "Supported claims:\n{supported}\n\n"
            "Rewrite the answer keeping the supported claims and removing unsupported ones:",
        ),
    ]
)


@traced
def critic(state: dict) -> dict:
    """Verify claims in the generated answer against source documents.

    If all claims are supported, passes the answer through unchanged.
    If some claims are unsupported, rewrites the answer without them.
    """
    question = state["question"]
    generation = state.get("generation", "")
    # Use all_sub_documents (accumulated across sub-queries) if available,
    # otherwise fall back to documents from the last node
    documents = state.get("all_sub_documents", state.get("documents", []))

    # Skip critic for "I don't know" answers
    idk_phrases = ["don't have enough information", "cannot answer", "no information available"]
    if any(phrase in generation.lower() for phrase in idk_phrases):
        logger.info("Critic: skipping — answer is already an IDK response")
        return {"critic_passed": True, "claims_removed": 0}

    if not documents:
        logger.warning("Critic: no source documents to verify against")
        return {"critic_passed": True, "claims_removed": 0}

    # Critic mode gating: verification costs 1-2 extra LLM round-trips.
    mode = settings.critic_mode.lower()
    if mode == "off":
        return {"critic_passed": True, "claims_removed": 0}
    if mode == "adaptive":
        # Skip only low-risk answers: the grader vouched for the context,
        # nothing came from the web, it isn't a multi-source synthesis,
        # and the answer is short. Everything else is still verified.
        if (
            state.get("relevant")
            and not state.get("web_fallback_used")
            and not state.get("is_multi_part")
            and len(generation) <= settings.critic_skip_max_chars
        ):
            logger.info(
                "Critic: skipped (adaptive) — grader passed, %d chars", len(generation)
            )
            return {"critic_passed": True, "claims_removed": 0}

    try:
        context = "\n\n".join(
            f"[{d.metadata.get('filename', '?')}] {d.page_content}"
            for d in documents
        )

        verifier = _critic_prompt | _llm().with_structured_output(ClaimVerdict)
        verdict: ClaimVerdict = verifier.invoke(
            {"question": question, "answer": generation, "context": context}
        )

        n_supported = len(verdict.supported_claims)
        n_unsupported = len(verdict.unsupported_claims)
        logger.info(
            "Critic: %d supported, %d unsupported claims",
            n_supported, n_unsupported,
        )

        if n_unsupported == 0:
            # All claims verified — pass through
            return {"critic_passed": True, "claims_removed": 0}

        if n_supported == 0:
            # Nothing is supported — return IDK
            logger.warning("Critic: no supported claims — replacing with IDK")
            return {
                "generation": (
                    "I don't have enough information in the available documents "
                    "to answer this question."
                ),
                "critic_passed": False,
                "claims_removed": n_unsupported,
            }

        # Some claims unsupported — rewrite without them
        supported_text = "\n".join(f"- {c}" for c in verdict.supported_claims)
        rewriter = _rewrite_prompt_critic | _llm() | StrOutputParser()
        rewritten = rewriter.invoke(
            {
                "question": question,
                "original_answer": generation,
                "supported": supported_text,
            }
        )
        logger.info(
            "Critic: rewrote answer (removed %d unsupported claims, %d chars → %d chars)",
            n_unsupported, len(generation), len(rewritten),
        )
        return {
            "generation": rewritten,
            "critic_passed": False,
            "claims_removed": n_unsupported,
        }

    except Exception:
        logger.exception("Critic failed — passing answer through unchanged")
        return {"critic_passed": True, "claims_removed": 0}


# --- Router: decide what to do after grading -------------------------------

def decide_after_grade(state: dict) -> str:
    """Conditional edge. Returns the name of the next node."""
    if state.get("relevant"):
        logger.debug("Route → generate (context is relevant)")
        return "generate"
    if state.get("retries", 0) >= MAX_RETRIES:
        logger.debug("Route → web_search (retries exhausted)")
        return "web_search"
    logger.debug("Route → transform_query (context weak, retrying)")
    return "transform_query"
