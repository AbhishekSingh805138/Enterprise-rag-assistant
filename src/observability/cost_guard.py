"""Per-query and per-day spend ceilings, actually enforced.

``COST_BUDGET_PER_QUERY`` was read into a module constant and used only
to put an asterisk next to expensive rows in a CLI report. Nothing could
reject or truncate anything, so a multi-part question could decompose
into sub-questions, each retrieving, grading, generating and criticising,
with no ceiling at all — and no daily cap behind that.

Two ceilings, deliberately different in character:

**Per query** degrades rather than denies. When the budget is spent the
pipeline stops *elaborating* — no further sub-questions, no query
expansion, no LLM reranking, no critic pass — and answers with what it
has. Refusing outright would turn a cost control into an availability
problem, and the caller would rather have a good-enough answer than an
error.

**Per day** denies. It exists precisely to stop runaway spend, so past
the cap the API rejects with 503 and an operator has to intervene.
Disabled by default: turning it on is a deployment decision.
"""
from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryBudget:
    """Spend tracker for one query."""

    limit: float
    spent: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Stages that were skipped because the budget ran out, for metrics.
    skipped: list[str] = field(default_factory=list)

    def add(self, cost: float) -> None:
        with self._lock:
            self.spent += cost

    @property
    def exceeded(self) -> bool:
        # A limit of zero or less disables the ceiling rather than
        # blocking every query, which is what a naive `spent >= limit`
        # would do and would look like a total outage.
        if self.limit <= 0:
            return False
        return self.spent >= self.limit

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.spent) if self.limit > 0 else float("inf")

    def allow(self, stage: str) -> bool:
        """Whether *stage* may run. Records the skip when it may not."""
        if not self.exceeded:
            return True
        with self._lock:
            if stage not in self.skipped:
                self.skipped.append(stage)
                logger.warning(
                    "Cost budget spent ($%.5f of $%.5f) — skipping %s",
                    self.spent, self.limit, stage,
                )
        return False


_budget: ContextVar[QueryBudget | None] = ContextVar("query_budget", default=None)


def start_query_budget(limit: float | None = None) -> QueryBudget:
    """Begin tracking spend for the current query. Returns the budget."""
    budget = QueryBudget(limit=settings.cost_budget_per_query if limit is None else limit)
    _budget.set(budget)
    return budget


def get_query_budget() -> QueryBudget | None:
    """The active budget, or None outside a tracked query."""
    return _budget.get()


def clear_query_budget() -> None:
    _budget.set(None)


def record_spend(cost: float) -> None:
    """Add *cost* to the active budget, if there is one."""
    budget = _budget.get()
    if budget is not None and cost:
        budget.add(cost)


def allow(stage: str) -> bool:
    """Whether an optional, LLM-consuming *stage* may run.

    Returns True when there is no budget in scope, so anything running
    outside a request (scripts, ingestion, tests) is unaffected.
    """
    budget = _budget.get()
    if budget is None:
        return True
    return budget.allow(stage)


def remaining_sub_questions(planned: int) -> int:
    """How many sub-questions the remaining budget can pay for.

    Decomposition is where cost multiplies: each sub-question runs a full
    retrieve/grade/generate cycle. Truncating the plan is far cheaper than
    discovering the overrun after the fact.
    """
    budget = _budget.get()
    if budget is None or budget.limit <= 0 or planned <= 1:
        return planned
    # Charge the first sub-question to what has already been spent, then
    # assume later ones cost about the same.
    per_sub = budget.spent if budget.spent > 0 else budget.limit / max(planned, 1)
    if per_sub <= 0:
        return planned
    affordable = max(1, int(budget.remaining / per_sub))
    if affordable < planned:
        logger.warning(
            "Cost budget allows %d of %d planned sub-questions "
            "($%.5f remaining of $%.5f)",
            affordable, planned, budget.remaining, budget.limit,
        )
    return min(planned, affordable)


# ---------------------------------------------------------------------------
# Daily cap
# ---------------------------------------------------------------------------

_daily_cache: tuple[float, float] = (0.0, 0.0)  # (checked_at, spend)
_daily_lock = threading.Lock()
DAILY_CACHE_TTL_S = 30.0


def daily_spend(force: bool = False) -> float:
    """Total recorded spend for the current UTC day.

    Cached briefly: this is consulted on every request, and a SQL
    aggregate per request would put the cost control itself on the hot
    path. Thirty seconds of drift cannot meaningfully overshoot a daily
    cap.
    """
    global _daily_cache
    now = time.monotonic()
    with _daily_lock:
        checked_at, cached = _daily_cache
        if not force and checked_at and (now - checked_at) < DAILY_CACHE_TTL_S:
            return cached

    try:
        from src.observability.metrics_store import get_store

        spend = get_store().spend_today()
    except Exception as e:
        # Never let the cost check be the reason a query fails.
        logger.debug("Could not read daily spend: %s", e)
        return 0.0

    with _daily_lock:
        _daily_cache = (now, spend)
    return spend


def reset_daily_cache() -> None:
    global _daily_cache
    with _daily_lock:
        _daily_cache = (0.0, 0.0)


def daily_cap_exceeded() -> bool:
    """Whether today's spend has reached ``COST_DAILY_CAP_USD``."""
    cap = settings.cost_daily_cap_usd
    if cap <= 0:
        return False
    spend = daily_spend()
    if spend >= cap:
        logger.error(
            "DAILY COST CAP REACHED: $%.4f of $%.4f — rejecting queries "
            "until the next UTC day or until COST_DAILY_CAP_USD is raised.",
            spend, cap,
        )
        return True
    return False
