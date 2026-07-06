"""Per-node tracing for the CRAG graph.

Wraps graph node functions to log timing, input/output sizes, and metadata.
This lightweight decorator is always-on. LangSmith integration (Phase 6)
works automatically via LangChain callbacks when LANGSMITH_TRACING=true.

Phase 8 additions:
  - Module-level metric accumulator for per-node latency percentiles
  - get_node_metrics() / reset_node_metrics() for observability

Concurrency-safety additions:
  - Global accumulator is bounded (deque) so a long-running server does
    not leak memory.
  - start_run_capture() opens a per-request capture via a ContextVar so
    get_last_run_latencies() returns THIS request's node timings even
    when other requests run concurrently. Without an active capture it
    falls back to the most recent global value per node.

Usage:
    from src.graph.tracing import traced

    @traced
    def retrieve(state: dict) -> dict:
        ...
"""
from __future__ import annotations

import functools
import logging
import statistics
import threading
import time
from collections import deque
from contextvars import ContextVar
from typing import Callable

logger = logging.getLogger(__name__)

# Keep at most this many samples per node for percentile stats.
_MAX_SAMPLES = 1000

# Module-level accumulator: node_name -> bounded deque of elapsed_ms values
_node_metrics: dict[str, deque[float]] = {}
_lock = threading.Lock()

# Per-request capture. The dict is created by start_run_capture() and
# mutated in place by traced() — threads that inherit a copy of the
# context (e.g. via asyncio.to_thread) share the same dict object.
_run_capture: ContextVar[dict[str, float] | None] = ContextVar(
    "node_run_capture", default=None
)


def start_run_capture() -> None:
    """Begin capturing node latencies for the current request/context.

    Call immediately before graph.invoke()/graph.stream() in the same
    thread or task; get_last_run_latencies() afterwards returns only the
    latencies recorded since this call.
    """
    _run_capture.set({})


def get_node_metrics() -> dict[str, dict]:
    """Return per-node latency statistics.

    Returns a dict like:
        {"retrieve": {"count": 10, "p50": 120.0, "p95": 340.0, "p99": 500.0}, ...}
    """
    with _lock:
        snapshot = {name: list(timings) for name, timings in _node_metrics.items()}

    result = {}
    for name, timings in snapshot.items():
        n = len(timings)
        if n == 0:
            continue
        sorted_t = sorted(timings)
        result[name] = {
            "count": n,
            "p50": round(sorted_t[n // 2], 1),
            "p95": round(sorted_t[int(n * 0.95)] if n >= 20 else sorted_t[-1], 1),
            "p99": round(sorted_t[int(n * 0.99)] if n >= 100 else sorted_t[-1], 1),
            "mean": round(statistics.mean(timings), 1),
            "last": round(sorted_t[-1], 1),
        }
    return result


def get_last_run_latencies() -> dict[str, float]:
    """Return the current request's per-node latencies.

    If a capture was started via start_run_capture(), returns exactly the
    nodes recorded in this request. Otherwise falls back to the most
    recent global value per node (single-threaded/testing use).
    """
    captured = _run_capture.get()
    if captured is not None:
        return {name: round(ms, 1) for name, ms in captured.items()}
    with _lock:
        return {
            name: round(timings[-1], 1)
            for name, timings in _node_metrics.items()
            if timings
        }


def reset_node_metrics() -> None:
    """Clear all accumulated metrics (for testing)."""
    with _lock:
        _node_metrics.clear()


def traced(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Decorator that adds timing and size logging to a graph node function."""

    @functools.wraps(fn)
    def wrapper(state: dict) -> dict:
        node_name = fn.__name__
        start = time.perf_counter()

        # Log input summary
        question = state.get("question", "")[:80]
        n_docs_in = len(state.get("documents", []))
        logger.debug(
            "TRACE [%s] START — question=%r, docs_in=%d, retries=%d",
            node_name, question, n_docs_in, state.get("retries", 0),
        )

        result = fn(state)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Record global metric (thread-safe, bounded)
        with _lock:
            _node_metrics.setdefault(
                node_name, deque(maxlen=_MAX_SAMPLES)
            ).append(elapsed_ms)

        # Record into the per-request capture if one is active
        captured = _run_capture.get()
        if captured is not None:
            captured[node_name] = elapsed_ms

        # Log output summary
        n_docs_out = len(result.get("documents", []))
        gen_len = len(result.get("generation", ""))
        extra = ""
        if "relevant" in result:
            extra += f", relevant={result['relevant']}"
        if "critic_passed" in result:
            extra += f", critic_passed={result['critic_passed']}"
        if result.get("claims_removed", 0) > 0:
            extra += f", claims_removed={result['claims_removed']}"

        logger.info(
            "TRACE [%s] %.0fms — docs_out=%d, gen_len=%d%s",
            node_name, elapsed_ms, n_docs_out, gen_len, extra,
        )
        return result

    return wrapper
