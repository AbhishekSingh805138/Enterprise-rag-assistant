"""SQLite-backed persistence for per-query cost and latency metrics.

Stores metrics in the same database as the LangGraph checkpointer but uses
a separate connection. The query_metrics table is created idempotently on
first access.

Phase 8 additions:
  - IDK rate tracking (is_idk column)
  - Grader rejection tracking (grader_rejected column)
  - Latency percentiles (p50/p95/p99)
  - Cost alerting (cost_alert_check)
  - MetricsStoreProtocol for backend abstraction
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from config import settings
from src.observability.cost_callback import QueryMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — any backend must implement these methods
# ---------------------------------------------------------------------------

@runtime_checkable
class MetricsStoreProtocol(Protocol):
    """Protocol for metrics storage backends.

    Enables future PostgreSQL/Redis/etc. implementations without changing
    the calling code.
    """

    def record(self, m: QueryMetrics) -> None: ...
    def query_recent(self, n: int = 20) -> list[dict]: ...
    def summary(self, n: int | None = None) -> dict: ...
    def idk_rate(self, n: int | None = None) -> float: ...
    def grader_rejection_rate(self, n: int | None = None) -> float: ...
    def latency_percentiles(self, n: int | None = None) -> dict: ...
    def cost_alert_check(self, threshold: float | None = None) -> list[dict]: ...
    def close(self) -> None: ...

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS query_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    thread_id   TEXT    NOT NULL,
    question    TEXT    NOT NULL,
    mode        TEXT    NOT NULL,
    retriever   TEXT    NOT NULL,
    prompt_tok  INTEGER NOT NULL DEFAULT 0,
    compl_tok   INTEGER NOT NULL DEFAULT 0,
    total_tok   INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL    NOT NULL DEFAULT 0.0,
    latency_ms  REAL    NOT NULL DEFAULT 0.0
);
"""

# Phase 8: schema migration for new columns
_MIGRATIONS = [
    "ALTER TABLE query_metrics ADD COLUMN is_idk INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE query_metrics ADD COLUMN grader_rejected INTEGER NOT NULL DEFAULT 0",
    # Which prompt set produced this answer. Without it a drop in answer
    # quality can be observed but not attributed to the prompt change
    # that caused it — the measurement and the cause live in different
    # systems, and correlating them afterwards is guesswork.
    "ALTER TABLE query_metrics ADD COLUMN prompt_version TEXT NOT NULL DEFAULT ''",
]

COST_BUDGET = settings.cost_budget_per_query


class MetricsStore:
    """Thin wrapper around a SQLite table for query metrics."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply schema migrations (idempotent — skips if columns exist)."""
        for sql in _MIGRATIONS:
            try:
                self._conn.execute(sql)
                self._conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    logger.debug("Migration skipped: %s", e)

    def record(self, m: QueryMetrics) -> None:
        """Insert a single query's metrics."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO query_metrics "
                "(ts, thread_id, question, mode, retriever, prompt_tok, compl_tok, "
                "total_tok, cost_usd, latency_ms, is_idk, grader_rejected, prompt_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    m.thread_id,
                    m.question_preview,
                    m.mode,
                    m.retriever_strategy,
                    m.prompt_tokens,
                    m.completion_tokens,
                    m.total_tokens,
                    m.estimated_cost_usd,
                    m.latency_ms,
                    1 if m.is_idk else 0,
                    m.grader_rejected,
                ),
            )
            self._conn.commit()

        # Cost alerting (outside lock — logging doesn't need DB protection)
        if m.estimated_cost_usd > settings.cost_alert_threshold:
            logger.warning(
                "COST ALERT: query cost $%.5f exceeds threshold $%.3f — %s",
                m.estimated_cost_usd,
                settings.cost_alert_threshold,
                m.question_preview[:80],
            )

    def query_recent(self, n: int = 20) -> list[dict]:
        """Return the last *n* query metrics, newest first."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM query_metrics ORDER BY id DESC LIMIT ?", (n,)
            )
            return [dict(row) for row in cur.fetchall()]

    def summary(self, n: int | None = None) -> dict:
        """Aggregate statistics over the last *n* queries (or all if None)."""
        with self._lock:
            if n is not None:
                sql = (
                    "SELECT COUNT(*) AS cnt, "
                    "COALESCE(SUM(cost_usd), 0) AS total_cost, "
                    "COALESCE(AVG(cost_usd), 0) AS avg_cost, "
                    "COALESCE(AVG(latency_ms), 0) AS avg_latency, "
                    "COALESCE(SUM(total_tok), 0) AS total_tokens, "
                    "COALESCE(SUM(CASE WHEN cost_usd > ? THEN 1 ELSE 0 END), 0) AS over_budget "
                    "FROM (SELECT * FROM query_metrics ORDER BY id DESC LIMIT ?)"
                )
                row = self._conn.execute(sql, (COST_BUDGET, n)).fetchone()
            else:
                sql = (
                    "SELECT COUNT(*) AS cnt, "
                    "COALESCE(SUM(cost_usd), 0) AS total_cost, "
                    "COALESCE(AVG(cost_usd), 0) AS avg_cost, "
                    "COALESCE(AVG(latency_ms), 0) AS avg_latency, "
                    "COALESCE(SUM(total_tok), 0) AS total_tokens, "
                    "COALESCE(SUM(CASE WHEN cost_usd > ? THEN 1 ELSE 0 END), 0) AS over_budget "
                    "FROM query_metrics"
                )
                row = self._conn.execute(sql, (COST_BUDGET,)).fetchone()
            return dict(row)

    def idk_rate(self, n: int | None = None) -> float:
        """Return the fraction of queries that resulted in IDK responses."""
        with self._lock:
            if n is not None:
                sql = (
                    "SELECT COALESCE(AVG(is_idk * 1.0), 0) AS rate "
                    "FROM (SELECT is_idk FROM query_metrics ORDER BY id DESC LIMIT ?)"
                )
                row = self._conn.execute(sql, (n,)).fetchone()
            else:
                sql = "SELECT COALESCE(AVG(is_idk * 1.0), 0) AS rate FROM query_metrics"
                row = self._conn.execute(sql).fetchone()
            return float(row["rate"]) if row else 0.0

    def grader_rejection_rate(self, n: int | None = None) -> float:
        """Return the fraction of queries where the grader rejected docs."""
        with self._lock:
            if n is not None:
                sql = (
                    "SELECT COALESCE(AVG(CASE WHEN grader_rejected > 0 THEN 1.0 ELSE 0.0 END), 0) AS rate "
                    "FROM (SELECT grader_rejected FROM query_metrics ORDER BY id DESC LIMIT ?)"
                )
                row = self._conn.execute(sql, (n,)).fetchone()
            else:
                sql = (
                    "SELECT COALESCE(AVG(CASE WHEN grader_rejected > 0 THEN 1.0 ELSE 0.0 END), 0) AS rate "
                    "FROM query_metrics"
                )
                row = self._conn.execute(sql).fetchone()
            return float(row["rate"]) if row else 0.0

    def latency_percentiles(self, n: int | None = None) -> dict:
        """Return p50, p95, p99 latency in milliseconds."""
        with self._lock:
            if n is not None:
                sql = "SELECT latency_ms FROM query_metrics ORDER BY id DESC LIMIT ?"
                rows = self._conn.execute(sql, (n,)).fetchall()
            else:
                sql = "SELECT latency_ms FROM query_metrics"
                rows = self._conn.execute(sql).fetchall()

        if not rows:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = sorted(r["latency_ms"] for r in rows)
        count = len(latencies)
        return {
            "p50": round(latencies[count // 2], 1),
            "p95": round(latencies[int(count * 0.95)] if count >= 20 else latencies[-1], 1),
            "p99": round(latencies[int(count * 0.99)] if count >= 100 else latencies[-1], 1),
        }

    def cost_alert_check(self, threshold: float | None = None) -> list[dict]:
        """Return recent queries that exceeded the cost threshold."""
        t = threshold or settings.cost_alert_threshold
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts, thread_id, question, cost_usd, mode, retriever "
                "FROM query_metrics WHERE cost_usd > ? ORDER BY id DESC LIMIT 20",
                (t,),
            )
            return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: MetricsStore | None = None
_store_lock = threading.Lock()


def _default_db_path() -> str:
    """Resolve the default DB path from settings."""
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / "graph_checkpoints.db")


def get_store(db_path: str | None = None) -> MetricsStore:
    """Return the singleton MetricsStore, creating it on first call. Thread-safe."""
    global _store
    with _store_lock:
        if _store is None:
            _store = MetricsStore(db_path or _default_db_path())
            logger.info("MetricsStore initialized at %s", db_path or _default_db_path())
        return _store


def reset_store() -> None:
    """Close and discard the singleton (for testing)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
            _store = None
