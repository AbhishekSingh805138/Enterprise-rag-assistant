"""Build and compile the Corrective RAG (CRAG) graph.

Flow (Phase 5 — with planner):

    START -> planner -> (conditional: route_after_plan)
      |
      |- simple question:
      |    retrieve -> grade_documents -> (conditional)
      |                                    |- relevant -> generate -> critic -> END
      |                                    |- retry     -> transform_query -> retrieve
      |                                    |- exhausted -> web_search -> generate -> critic -> END
      |
      |- multi-part question:
           process_sub_query -> (conditional: has_more_sub_queries?)
             |- yes -> process_sub_query (loop)
             |- no  -> synthesize -> critic -> END

The planner detects multi-part questions and decomposes them into sub-queries.
Simple questions go through the original CRAG flow unchanged. Multi-part
questions are processed sequentially, then synthesized into a single response.
The critic node verifies claims in the final answer regardless of path.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import settings
from src.graph.analyzer import analyze_query
from src.graph.cache_nodes import cache_lookup, cache_store, route_after_cache
from src.graph.guardrail_node import guardrail_check, route_after_guardrail
from src.graph.intent_detector import intent_detect
from src.graph.memory_nodes import load_memory, save_memory
from src.graph.nodes import (
    critic,
    decide_after_grade,
    generate,
    grade_documents,
    retrieve,
    transform_query,
    web_search,
)
from src.graph.planner import (
    has_more_sub_queries,
    planner,
    process_sub_queries_parallel,
    process_sub_query,
    route_after_plan,
    synthesize,
)
from src.graph.scope_detector import scope_check
from src.graph.state import RAGState
from src.graph.tool_node import tool_router
from src.retrieval.query_transformer import query_transform_node

logger = logging.getLogger(__name__)

# Module-level cached graph — compiled once, reused across calls.
_compiled_graph = None
_sqlite_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _get_sqlite_checkpointer() -> SqliteSaver:
    """Create a SQLite checkpointer for persistent graph state."""
    global _sqlite_conn
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / "graph_checkpoints.db"
    _sqlite_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(_sqlite_conn)
    logger.info("SQLite checkpointer initialized at %s", db_path)
    return saver


def _route_after_scope(state: dict) -> str:
    """Route based on scope check: in-scope -> planner, out-of-scope -> generate."""
    if state.get("in_scope", True):
        return "planner"
    return "generate"


def build_graph(checkpointer=None):
    """Build and compile the CRAG StateGraph with Phase 5 planner."""
    builder = StateGraph(RAGState)

    # Phase 17: guardrails (first node in pipeline)
    if settings.guardrails_enabled:
        builder.add_node("guardrail_check", guardrail_check)

    # Phase 10: conversation memory
    if settings.memory_enabled:
        builder.add_node("load_memory", load_memory)
        builder.add_node("save_memory", save_memory)

    # Phase 15: semantic cache
    if settings.semantic_cache_enabled:
        builder.add_node("cache_lookup", cache_lookup)
        builder.add_node("cache_store", cache_store)

    # Phase 8: scope detection (before planner)
    builder.add_node("scope_check", scope_check)

    # Unified analysis (opt-in): one structured LLM call replaces
    # intent_detect -> query_transform -> planner. When enabled, those three
    # nodes are not added; analyze_query takes the planner's routing role.
    _unified = settings.unified_analysis

    # Phase 11: intent detection (after scope check, before planner)
    if settings.intent_detection_enabled and not _unified:
        builder.add_node("intent_detect", intent_detect)

    # Phase 12: query transformation (after intent detect, before planner)
    if settings.query_transform_enabled and not _unified:
        builder.add_node("query_transform", query_transform_node)

    # Phase 8: tool routing (optional, gated by ENABLE_TOOLS)
    if settings.enable_tools:
        builder.add_node("tool_router", tool_router)

    # Phase 5: planner and multi-part processing nodes. The planner-role node
    # is analyze_query when unified analysis is on, else the classic planner.
    _planner_node = "analyze_query" if _unified else "planner"
    if _unified:
        builder.add_node("analyze_query", analyze_query)
    else:
        builder.add_node("planner", planner)
    if settings.parallel_sub_queries:
        builder.add_node("process_sub_queries_parallel", process_sub_queries_parallel)
    else:
        builder.add_node("process_sub_query", process_sub_query)
    builder.add_node("synthesize", synthesize)

    # Existing CRAG nodes
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("transform_query", transform_query)
    builder.add_node("web_search", web_search)
    builder.add_node("generate", generate)
    builder.add_node("critic", critic)

    # Entry: [guardrail] → [memory load] → [cache_lookup] → scope check → ...
    # Determine the entry chain
    if settings.memory_enabled:
        _after_guardrail = "load_memory"
    else:
        _after_guardrail = "cache_lookup" if settings.semantic_cache_enabled else "scope_check"

    if settings.guardrails_enabled:
        builder.add_edge(START, "guardrail_check")
        # Blocked path goes to end (through save_memory if enabled)
        _blocked_target = "save_memory" if settings.memory_enabled else END
        builder.add_conditional_edges(
            "guardrail_check",
            route_after_guardrail,
            {
                "continue": _after_guardrail,
                "blocked": _blocked_target,
            },
        )
    else:
        builder.add_edge(START, _after_guardrail)

    # Wire load_memory → [cache_lookup] → scope_check
    if settings.memory_enabled:
        _after_memory = "cache_lookup" if settings.semantic_cache_enabled else "scope_check"
        builder.add_edge("load_memory", _after_memory)

    # Wire cache_lookup → scope_check (miss) or end (hit)
    if settings.semantic_cache_enabled:
        # On cache hit, go to save_memory (to save the turn) or END
        _cache_hit_target = "save_memory" if settings.memory_enabled else END
        builder.add_conditional_edges(
            "cache_lookup",
            route_after_cache,
            {
                "cache_miss": "scope_check",
                "cache_hit": _cache_hit_target,
            },
        )
    # Determine the node that follows scope_check (in-scope path)
    # Chain: scope_check → [intent_detect] → [query_transform] → [tool_router] → planner
    # Build the chain from right to left to find the first enabled node
    _chain: list[str] = [_planner_node]
    if settings.enable_tools:
        _chain.insert(0, "tool_router")
    if settings.query_transform_enabled and not _unified:
        _chain.insert(0, "query_transform")
    if settings.intent_detection_enabled and not _unified:
        _chain.insert(0, "intent_detect")

    # First node in the chain is what scope_check routes to
    _scope_next = _chain[0]

    builder.add_conditional_edges(
        "scope_check",
        _route_after_scope,
        {
            "planner": _scope_next,
            "generate": "generate",
        },
    )

    # Wire sequential edges for enabled nodes in the chain
    for i in range(len(_chain) - 1):
        builder.add_edge(_chain[i], _chain[i + 1])
    # Route multi-part questions to parallel or sequential processing
    _multi_part_target = "process_sub_queries_parallel" if settings.parallel_sub_queries else "process_sub_query"
    builder.add_conditional_edges(
        _planner_node,
        route_after_plan,
        {
            "retrieve": "retrieve",                       # simple question → existing CRAG
            "process_sub_query": _multi_part_target,      # multi-part → parallel or sequential
        },
    )

    # Simple path: existing CRAG flow (unchanged)
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        decide_after_grade,
        {
            "generate": "generate",
            "transform_query": "transform_query",
            "web_search": "web_search",
        },
    )
    builder.add_edge("transform_query", "retrieve")  # the corrective loop
    builder.add_edge("web_search", "generate")
    builder.add_edge("generate", "critic")  # critic verifies claims before output

    # Multi-part path: parallel or sequential → synthesize
    if settings.parallel_sub_queries:
        builder.add_edge("process_sub_queries_parallel", "synthesize")
    else:
        builder.add_conditional_edges(
            "process_sub_query",
            has_more_sub_queries,
            {
                "process_sub_query": "process_sub_query",  # loop back
                "synthesize": "synthesize",                 # all done
            },
        )
    builder.add_edge("synthesize", "critic")  # critic checks synthesized answer

    # Both paths end after critic: [cache_store] → [save_memory] → END
    if settings.semantic_cache_enabled:
        builder.add_edge("critic", "cache_store")
        if settings.memory_enabled:
            builder.add_edge("cache_store", "save_memory")
            builder.add_edge("save_memory", END)
        else:
            builder.add_edge("cache_store", END)
    elif settings.memory_enabled:
        builder.add_edge("critic", "save_memory")
        builder.add_edge("save_memory", END)
    else:
        builder.add_edge("critic", END)

    if checkpointer is None:
        # Request-scoped invocations use a fresh thread_id every time and
        # never resume, so durable checkpoints are pure write overhead —
        # full state (including Documents) would be serialized to SQLite at
        # every super-step. Default to in-memory; opt into SQLite via
        # GRAPH_CHECKPOINTER=sqlite for flows that genuinely resume threads.
        if settings.graph_checkpointer.lower() == "sqlite":
            checkpointer = _get_sqlite_checkpointer()
        else:
            checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def get_graph(checkpointer=None):
    """Return the cached compiled graph (singleton). Thread-safe."""
    global _compiled_graph
    with _lock:
        if _compiled_graph is None:
            _compiled_graph = build_graph(checkpointer=checkpointer)
            logger.info("CRAG graph compiled and cached")
        return _compiled_graph


def reset_graph() -> None:
    """Reset the cached graph and close the SQLite connection."""
    global _compiled_graph, _sqlite_conn
    with _lock:
        _compiled_graph = None
        if _sqlite_conn is not None:
            try:
                _sqlite_conn.close()
            except Exception:
                pass
            _sqlite_conn = None


def ask(
    question: str,
    thread_id: str | None = None,
    retriever_strategy: str = "dense",
    session_id: str | None = None,
) -> str:
    """Run a question through the CRAG graph. Returns the answer string."""
    if not question or not question.strip():
        return "Please provide a question."

    tid = thread_id or uuid.uuid4().hex[:12]
    sid = session_id or tid
    logger.info("CRAG query (thread=%s, session=%s, retriever=%s): %s", tid, sid, retriever_strategy, question[:120])

    from src.observability.cost_callback import CostCallbackHandler

    handler = CostCallbackHandler()
    graph = get_graph()
    config = {"configurable": {"thread_id": tid}, "callbacks": [handler]}

    try:
        from src.graph.tracing import start_run_capture

        start_run_capture()  # scope node latencies to this run
        start = time.perf_counter()
        result = graph.invoke(
            {"question": question, "retries": 0, "retriever_strategy": retriever_strategy, "session_id": sid},
            config,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        answer = result.get("generation", "No answer was generated.")
        logger.info("CRAG answer (thread=%s): %d chars", tid, len(answer))

        # Record metrics — never let metrics failures break the query
        try:
            from src.graph.tracing import get_last_run_latencies
            from src.observability.cost_callback import is_idk_response

            node_lats = get_last_run_latencies()
            grader_rejected = 1 if not result.get("relevant", True) else 0
            metrics = handler.flush(
                thread_id=tid,
                question=question,
                latency_ms=latency_ms,
                retriever_strategy=retriever_strategy,
                mode="graph",
                is_idk=is_idk_response(answer),
                grader_rejected=grader_rejected,
                node_latencies=node_lats,
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

        return answer
    except Exception:
        logger.exception("CRAG graph failed for: %s", question[:120])
        raise
