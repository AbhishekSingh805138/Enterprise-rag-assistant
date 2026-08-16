"""Phase 5: Planner and synthesizer for multi-part question decomposition.

The planner classifies incoming questions as simple or multi-part. Simple
questions pass through unchanged to the existing CRAG flow. Multi-part
questions are decomposed into focused sub-queries that are each answered
independently, then the synthesizer combines the sub-answers into a
coherent, citation-preserving final response.

This addresses the legal/compliance analyst persona (US2) who asks questions
spanning multiple contracts or departments.
"""
from __future__ import annotations

import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from config import settings
from src.prompts import register
from src.graph.tracing import traced
from src.llm_pool import get_llm

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = settings.max_sub_questions


# --- Structured output models ------------------------------------------------

class PlanResult(BaseModel):
    """Structured output from the planner LLM."""
    is_multi_part: bool = Field(
        description=(
            "True if the question contains multiple distinct sub-questions "
            "or asks to compare/contrast across topics."
        )
    )
    sub_questions: list[str] = Field(
        description=(
            "List of focused sub-questions. For simple questions, return "
            "a list with just the original question. For multi-part, "
            "decompose into 2-5 self-contained sub-questions."
        )
    )


# --- Prompts ------------------------------------------------------------------

_planner_prompt = register(
    "planner_decompose",
    "v1",

    [
        (
            "system",
            "You are a question decomposition planner for an enterprise knowledge "
            "assistant. Your job is to analyze a user's question and decide whether "
            "it should be answered directly or decomposed into sub-questions.\n\n"
            "A question is MULTI-PART if it:\n"
            "  - Asks about two or more distinct topics (e.g., 'Compare X and Y')\n"
            "  - Spans multiple departments or document types\n"
            "  - Asks for a list of items from different sources\n"
            "  - Contains conjunctions like 'and', 'as well as', 'both', 'compare'\n\n"
            "A question is SIMPLE if it:\n"
            "  - Asks about one specific topic\n"
            "  - Can be answered from a single document or department\n"
            "  - Is a straightforward factual lookup\n\n"
            "Rules:\n"
            "1. For SIMPLE questions: set is_multi_part=false, sub_questions=[original question]\n"
            "2. For MULTI-PART questions: set is_multi_part=true, decompose into 2-5 "
            "focused, self-contained sub-questions\n"
            "3. Each sub-question must be independently answerable without context "
            "from the other sub-questions\n"
            "4. Preserve the original intent and specificity in each sub-question\n"
            "5. Do NOT add sub-questions about topics not mentioned in the original\n\n"
            "Examples:\n\n"
            'Q: "What is the PTO policy?"\n'
            "→ is_multi_part=false, sub_questions=[\"What is the PTO policy?\"]\n"
            "(Simple single-topic lookup)\n\n"
            'Q: "Compare the engineering and HR onboarding processes"\n'
            "→ is_multi_part=true, sub_questions=[\n"
            '    "What is the engineering department onboarding process?",\n'
            '    "What is the HR department onboarding process?"\n'
            "]\n"
            "(Comparison across two departments)\n\n"
            'Q: "What are the security incident response steps and how do they relate to the legal compliance requirements?"\n'
            "→ is_multi_part=true, sub_questions=[\n"
            '    "What are the security incident response steps?",\n'
            '    "What are the legal compliance requirements for security incidents?"\n'
            "]\n"
            "(Multi-topic spanning security and legal departments)",
        ),
        ("human", "{question}"),
    ]
)

_synthesize_prompt = register(
    "planner_synthesize",
    "v1",

    [
        (
            "system",
            "You are a synthesis assistant for an enterprise knowledge system. "
            "You are given a user's original question and individual answers to "
            "its sub-parts. Your job is to combine these into a single, coherent "
            "response.\n\n"
            "Rules:\n"
            "1. Preserve ALL source citations (e.g., '(handbook.md)') from the sub-answers\n"
            "2. Do NOT add any information not present in the sub-answers\n"
            "3. If a sub-answer says it doesn't have enough information, note that "
            "in your synthesis\n"
            "4. Organize the response logically — group related information together\n"
            "5. Use clear headings or bullet points for multi-topic answers\n"
            "6. Keep the response concise but complete",
        ),
        (
            "human",
            "Original question: {question}\n\n"
            "Sub-answers:\n{sub_answers}",
        ),
    ]
)


def _llm(temperature: float = 0) -> Runnable:
    """Return a cached LLM instance from the pool."""
    return get_llm(temperature=temperature)


# --- Node: planner -----------------------------------------------------------

@traced
def planner(state: dict) -> dict:
    """Classify question as simple or multi-part and decompose if needed."""
    question = state["question"]
    logger.info("Planner analyzing: %s", question[:120])

    try:
        chain = _planner_prompt | _llm().with_structured_output(PlanResult)
        result: PlanResult = chain.invoke({"question": question})

        # Clamp sub-questions to MAX_SUB_QUESTIONS
        subs = result.sub_questions[:MAX_SUB_QUESTIONS]
        if not subs:
            subs = [question]

        logger.info(
            "Planner: is_multi_part=%s, sub_questions=%d",
            result.is_multi_part, len(subs),
        )
        return {
            "is_multi_part": result.is_multi_part and len(subs) > 1,
            "sub_questions": subs,
            "sub_answers": [],
            "current_sub_idx": 0,
            "original_question": question,
        }
    except Exception:
        logger.exception("Planner failed — treating as simple question")
        return {
            "is_multi_part": False,
            "sub_questions": [question],
            "sub_answers": [],
            "current_sub_idx": 0,
            "original_question": question,
        }


# --- Helper: process a single sub-question ----------------------------------

def _process_single_sub_query(sub_q: str, strategy: str) -> tuple[str, list]:
    """Process one sub-question: retrieve + optional mini-CRAG + generate.

    Returns (answer_text, documents_list).
    Extracted for reuse in both sequential and parallel modes.
    """
    from src.retrieval import get_retriever

    try:
        docs = get_retriever(strategy=strategy).invoke(sub_q)
    except Exception:
        logger.exception("Retrieval failed for sub-question: %s", sub_q[:80])
        docs = []

    # Mini-CRAG: if docs seem irrelevant, rewrite and retry once
    if docs and settings.sub_query_max_retries > 0:
        try:
            from src.graph.nodes import GradeResult, _grade_prompt
            from src.graph.nodes import _llm as _nodes_llm
            grader = _grade_prompt | _nodes_llm().with_structured_output(GradeResult)
            context = "\n\n".join(d.page_content for d in docs[:3])
            verdict = grader.invoke({"question": sub_q, "context": context})
            if not verdict.relevant:
                logger.info("Sub-query docs irrelevant — rewriting and retrying once")
                from src.graph.nodes import _rewrite_prompt
                rewriter = _rewrite_prompt | _llm(temperature=0.3) | StrOutputParser()
                rewritten = rewriter.invoke({
                    "question": sub_q,
                    "rejected_context": context[:300],
                })
                try:
                    docs = get_retriever(strategy=strategy).invoke(rewritten)
                except Exception:
                    logger.debug("Sub-query retry retrieval failed")
        except Exception:
            logger.debug("Sub-query mini-CRAG failed", exc_info=True)

    # Generate answer
    if not docs:
        answer = f"I don't have enough information in the available documents to answer: {sub_q}"
    else:
        try:
            from src.context.context_builder import build_context
            from src.graph.nodes import _gen_prompt
            chain = _gen_prompt | _llm() | StrOutputParser()
            built = build_context(docs, query=sub_q)
            context = built.text
            answer = chain.invoke({"question": sub_q, "context": context})
        except Exception:
            logger.exception("Generation failed for sub-question: %s", sub_q[:80])
            answer = f"An error occurred while answering: {sub_q}"

    return answer, docs


# --- Node: process_sub_query ------------------------------------------------

@traced
def process_sub_query(state: dict) -> dict:
    """Process the current sub-question: retrieve + grade + generate.

    This is a compact node that runs retrieval and generation for a single
    sub-question, then advances the index. It does NOT use the full CRAG
    retry loop — each sub-query gets one retrieval attempt with the
    configured strategy.
    """
    sub_questions = state.get("sub_questions", [])
    idx = state.get("current_sub_idx", 0)
    sub_answers = list(state.get("sub_answers", []))
    strategy = state.get("retriever_strategy", "dense")

    if idx >= len(sub_questions):
        logger.warning("process_sub_query called with index %d >= %d sub-questions", idx, len(sub_questions))
        return {"current_sub_idx": idx}

    sub_q = sub_questions[idx]
    all_sub_docs = list(state.get("all_sub_documents", []))
    logger.info("Processing sub-question [%d/%d]: %s", idx + 1, len(sub_questions), sub_q[:100])

    sub_answer, docs = _process_single_sub_query(sub_q, strategy)

    all_sub_docs.extend(docs)
    sub_answers.append(sub_answer)
    logger.info("Sub-answer [%d/%d]: %d chars", idx + 1, len(sub_questions), len(sub_answer))

    return {
        "sub_answers": sub_answers,
        "current_sub_idx": idx + 1,
        "documents": docs,  # keep docs from last sub-query
        "all_sub_documents": all_sub_docs,  # accumulated across all sub-queries
    }


# --- Node: synthesize -------------------------------------------------------

@traced
def synthesize(state: dict) -> dict:
    """Combine sub-answers into a single coherent response with citations."""
    original_question = state.get("original_question", state.get("question", ""))
    sub_questions = state.get("sub_questions", [])
    sub_answers = state.get("sub_answers", [])

    if not sub_answers:
        logger.warning("Synthesize called with no sub-answers")
        return {
            "generation": (
                "I don't have enough information in the available documents "
                "to answer this question."
            )
        }

    # Format sub-answers for the synthesis prompt
    formatted = "\n\n".join(
        f"--- Sub-question {i+1}: {q} ---\n{a}"
        for i, (q, a) in enumerate(zip(sub_questions, sub_answers, strict=True))
    )

    try:
        chain = _synthesize_prompt | _llm() | StrOutputParser()
        synthesis = chain.invoke({
            "question": original_question,
            "sub_answers": formatted,
        })
        logger.info("Synthesized answer: %d chars from %d sub-answers", len(synthesis), len(sub_answers))
        return {"generation": synthesis, "question": original_question}
    except Exception:
        logger.exception("Synthesis failed — concatenating sub-answers")
        # Fallback: simple concatenation
        fallback = "\n\n".join(
            f"**{q}**\n{a}"
            for q, a in zip(sub_questions, sub_answers, strict=True)
        )
        return {"generation": fallback, "question": original_question}


# --- Node: process_sub_queries_parallel --------------------------------------

@traced
def process_sub_queries_parallel(state: dict) -> dict:
    """Process ALL sub-questions in parallel using a thread pool.

    Alternative to the sequential process_sub_query loop. Enabled via
    PARALLEL_SUB_QUERIES=true config flag. Processes all sub-questions
    at once and returns all answers, skipping the sequential loop.
    """
    sub_questions = state.get("sub_questions", [])
    strategy = state.get("retriever_strategy", "dense")

    if not sub_questions:
        return {"sub_answers": [], "current_sub_idx": 0, "all_sub_documents": []}

    max_workers = min(settings.sub_query_max_workers, len(sub_questions))
    logger.info("Processing %d sub-questions in parallel (workers=%d)", len(sub_questions), max_workers)

    all_sub_docs = []
    sub_answers = [""] * len(sub_questions)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # copy_context propagates LangChain's callback config into worker
        # threads so sub-query LLM calls are still cost-tracked.
        future_to_idx = {
            executor.submit(
                contextvars.copy_context().run, _process_single_sub_query, sq, strategy
            ): i
            for i, sq in enumerate(sub_questions)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                answer, docs = future.result()
                sub_answers[idx] = answer
                all_sub_docs.extend(docs)
                logger.info("Parallel sub-answer [%d/%d]: %d chars", idx + 1, len(sub_questions), len(answer))
            except Exception:
                logger.exception("Parallel sub-query %d failed", idx)
                sub_answers[idx] = f"An error occurred while answering: {sub_questions[idx]}"

    return {
        "sub_answers": sub_answers,
        "current_sub_idx": len(sub_questions),
        "all_sub_documents": all_sub_docs,
        "documents": [],  # no single-doc context after parallel
    }


# --- Router functions --------------------------------------------------------

def route_after_plan(state: dict) -> str:
    """Route simple questions to CRAG flow, multi-part to sub-query loop."""
    if state.get("is_multi_part"):
        logger.debug("Route → process_sub_query (multi-part detected)")
        return "process_sub_query"
    logger.debug("Route → retrieve (simple question)")
    return "retrieve"


def has_more_sub_queries(state: dict) -> str:
    """Check if there are more sub-questions to process."""
    idx = state.get("current_sub_idx", 0)
    total = len(state.get("sub_questions", []))
    if idx < total:
        logger.debug("Route → process_sub_query (%d/%d remaining)", total - idx, total)
        return "process_sub_query"
    logger.debug("Route → synthesize (all %d sub-questions answered)", total)
    return "synthesize"
