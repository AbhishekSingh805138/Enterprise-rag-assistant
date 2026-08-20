"""Per-query cost and token tracking via LangChain callback.

Hooks into every ChatOpenAI call to capture token usage and compute cost.
Works with or without LangSmith — the callback fires regardless.

Note: OpenAIEmbeddings calls do NOT trigger on_llm_end. Embedding cost is
negligible (~$0.0001/query at text-embedding-3-small rates) and is excluded.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Pricing per 1K tokens: (input_cost, output_cost)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o-mini-2024-07-18": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-2024-08-06": (0.0025, 0.01),
    "text-embedding-3-small": (0.00002, 0.00002),
}

# IDK phrases used to detect "I don't know" answers
IDK_PHRASES = [
    "don't have enough information",
    "cannot answer",
    "no information available",
    "not enough information",
    "unable to answer",
    "no relevant information",
]


@dataclass
class QueryMetrics:
    """Snapshot of cost/latency for a single query."""

    thread_id: str
    question_preview: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    retriever_strategy: str
    mode: str  # "naive" or "graph"
    # Phase 8 additions
    is_idk: bool = False
    grader_rejected: int = 0
    node_latencies: dict | None = None


def is_idk_response(text: str) -> bool:
    """Check if a response is an 'I don't know' answer."""
    lower = text.lower()
    return any(phrase in lower for phrase in IDK_PHRASES)


# Models we've already warned about (avoid log spam)
_warned_unknown_models: set[str] = set()


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute estimated cost in USD for a single LLM call."""
    costs = MODEL_COSTS.get(model)
    if costs is None:
        # Silently returning $0 would blind cost monitoring when the model
        # is changed — warn once per unknown model name.
        if model and model not in _warned_unknown_models:
            _warned_unknown_models.add(model)
            logger.warning(
                "Model %r not in MODEL_COSTS — its cost will be recorded as $0. "
                "Add its pricing to src/observability/cost_callback.py.", model,
            )
        return 0.0
    input_cost, output_cost = costs
    return (prompt_tokens * input_cost + completion_tokens * output_cost) / 1000


class CostCallbackHandler(BaseCallbackHandler):
    """Accumulates token usage across all LLM calls in a single query."""

    always_verbose = True  # fire even when LangChain verbose=False

    def __init__(self) -> None:
        super().__init__()
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_cost: float = 0.0
        # Parallel sub-queries / per-doc grading fire callbacks from
        # multiple threads — guard the accumulators.
        self._counter_lock = threading.Lock()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Extract token usage from the LLM response."""
        prompt_tok = 0
        completion_tok = 0
        model = ""

        # Path 1: usage_metadata on the AIMessage (langchain-openai >= 1.0)
        if response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg is not None:
                        usage = getattr(msg, "usage_metadata", None)
                        if usage and isinstance(usage, dict):
                            prompt_tok += usage.get("input_tokens", 0)
                            completion_tok += usage.get("output_tokens", 0)
                        model = msg.response_metadata.get("model_name", "") if hasattr(msg, "response_metadata") and msg.response_metadata else model

        # Path 2: fallback to llm_output (older LangChain versions)
        if prompt_tok == 0 and completion_tok == 0 and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
            prompt_tok = token_usage.get("prompt_tokens", 0)
            completion_tok = token_usage.get("completion_tokens", 0)
            model = response.llm_output.get("model_name", model)

        call_cost = compute_cost(model, prompt_tok, completion_tok)
        with self._counter_lock:
            self._prompt_tokens += prompt_tok
            self._completion_tokens += completion_tok
            self._total_cost += call_cost

        # Report to the per-query ceiling so downstream stages can see the
        # spend as it happens, rather than only in the metrics row written
        # after the money is gone.
        from src.observability.cost_guard import record_spend

        record_spend(call_cost)

    def flush(
        self,
        thread_id: str,
        question: str,
        latency_ms: float,
        retriever_strategy: str,
        mode: str,
        is_idk: bool = False,
        grader_rejected: int = 0,
        node_latencies: dict | None = None,
    ) -> QueryMetrics:
        """Snapshot current totals, reset counters, and return metrics."""
        with self._counter_lock:
            prompt_tokens = self._prompt_tokens
            completion_tokens = self._completion_tokens
            total_cost = self._total_cost
            # Reset for next query
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_cost = 0.0

        return QueryMetrics(
            thread_id=thread_id,
            question_preview=question[:200],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost_usd=total_cost,
            latency_ms=latency_ms,
            retriever_strategy=retriever_strategy,
            mode=mode,
            is_idk=is_idk,
            grader_rejected=grader_rejected,
            node_latencies=node_latencies,
        )
