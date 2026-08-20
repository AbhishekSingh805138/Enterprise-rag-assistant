"""Prometheus exposition for the metrics the system already collects.

Everything here was already being measured — cost, latency, IDK rate,
queue depth, dead-letter counts — and all of it landed in SQLite readable
only by a CLI. That is a record, not monitoring: nothing was graphed and,
most consequentially, nothing paged anyone. A non-zero dead-letter queue
degraded the health check, but only if a person went and looked.

Written by hand rather than with ``prometheus_client`` on purpose. The
values are aggregates over a SQLite table, not in-process counters, so
the client library's registry would add a second source of truth that
could disagree with the CLI and the health check. Rendering the same
queries the rest of the system uses keeps one set of numbers.

Scrape cost is bounded by a short cache: Prometheus scrapes every 15s by
default and several aggregates are full-table scans.
"""
from __future__ import annotations

import logging
import threading
import time

from config import settings

logger = logging.getLogger(__name__)

CACHE_TTL_S = 10.0
# Recent-window size for rate metrics. Rates over all time never recover
# from a bad hour, so an alert on them fires forever or not at all.
RECENT_WINDOW = 200

_cache: tuple[float, str] = (0.0, "")
_lock = threading.Lock()


def _line(name: str, value: float | int, labels: str = "") -> str:
    return f"{name}{labels} {value}"


def _metric(
    name: str, kind: str, help_text: str, samples: list[tuple[str, float | int]]
) -> list[str]:
    out = [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
    out.extend(_line(name, value, labels) for labels, value in samples)
    return out


def _query_metrics() -> list[str]:
    """Cost, latency and answer-quality series from the metrics store."""
    from src.observability.metrics_store import get_store

    store = get_store()
    summary = store.summary()
    recent = store.summary(RECENT_WINDOW)
    percentiles = store.latency_percentiles(RECENT_WINDOW)

    lines: list[str] = []
    lines += _metric(
        "rag_queries_total", "counter",
        "Queries answered since the metrics store was created.",
        [("", int(summary.get("cnt", 0)))],
    )
    lines += _metric(
        "rag_cost_usd_total", "counter",
        "Cumulative estimated LLM spend in USD.",
        [("", round(float(summary.get("total_cost", 0.0)), 6))],
    )
    lines += _metric(
        "rag_cost_usd_today", "gauge",
        "Estimated LLM spend for the current UTC day. Alert against COST_DAILY_CAP_USD.",
        [("", round(store.spend_today(), 6))],
    )
    lines += _metric(
        "rag_cost_daily_cap_usd", "gauge",
        "Configured daily spend cap; 0 means no cap. Exported so an alert "
        "can fire before the cap starts returning 503s.",
        [("", float(settings.cost_daily_cap_usd))],
    )
    lines += _metric(
        "rag_query_cost_usd_avg", "gauge",
        f"Mean cost per query over the last {RECENT_WINDOW} queries.",
        [("", round(float(recent.get("avg_cost", 0.0)), 6))],
    )
    lines += _metric(
        "rag_queries_over_budget_total", "counter",
        "Queries whose cost exceeded COST_BUDGET_PER_QUERY.",
        [("", int(summary.get("over_budget", 0)))],
    )
    lines += _metric(
        "rag_query_latency_ms", "gauge",
        f"Query latency percentiles over the last {RECENT_WINDOW} queries.",
        [
            ('{quantile="0.5"}', round(percentiles.get("p50", 0.0), 2)),
            ('{quantile="0.95"}', round(percentiles.get("p95", 0.0), 2)),
            ('{quantile="0.99"}', round(percentiles.get("p99", 0.0), 2)),
        ],
    )
    lines += _metric(
        "rag_idk_rate", "gauge",
        "Share of recent answers that were 'I don't know'. A spike usually "
        "means retrieval broke, not that the questions got harder.",
        [("", round(store.idk_rate(RECENT_WINDOW), 4))],
    )
    lines += _metric(
        "rag_grader_rejection_rate", "gauge",
        "Share of recent queries where the relevance grader rejected the context.",
        [("", round(store.grader_rejection_rate(RECENT_WINDOW), 4))],
    )
    return lines


def _ingestion_metrics() -> list[str]:
    """Queue depth, dead letters and document lifecycle counts."""
    from src.events.bus import get_event_bus
    from src.ingestion.registry import get_registry

    lines: list[str] = []

    bus = get_event_bus()
    backlog = int(bus.depth(settings.kafka_topic_ingestion) or 0)
    dlq = int(bus.depth(settings.kafka_topic_dlq) or 0)
    lines += _metric(
        "rag_ingestion_queue_depth", "gauge",
        "Events awaiting indexing. Rising steadily means workers cannot keep up.",
        [("", backlog)],
    )
    lines += _metric(
        "rag_ingestion_dlq_depth", "gauge",
        "Dead-lettered events. Any non-zero value needs a human: these "
        "documents will never be indexed without a redrive.",
        [("", dlq)],
    )
    lines += _metric(
        "rag_ingestion_queue_limit", "gauge",
        "Queue depth at which uploads are refused (INGEST_MAX_QUEUE_DEPTH).",
        [("", int(settings.ingest_max_queue_depth))],
    )

    stats = get_registry().stats()
    lines += _metric(
        "rag_documents", "gauge",
        "Registered documents by ingestion status.",
        [
            (f'{{status="{status.lower()}"}}', int(count))
            for status, count in sorted(stats.items())
            if status != "TOTAL"
        ],
    )
    return lines


def _build_id_metrics() -> list[str]:
    """Build and prompt identity, so a regression can be attributed."""
    from src.prompts import prompt_fingerprint

    return _metric(
        "rag_build_info", "gauge",
        "Always 1. Labels identify the prompt set and model that served "
        "these metrics, so a quality change can be tied to a deploy.",
        [
            (
                f'{{prompt_set="{prompt_fingerprint()}",llm_model="{settings.llm_model}",embedding_model="{settings.embedding_model}"}}',
                1,
            )
        ],
    )


_COLLECTORS = (
    ("query", _query_metrics),
    ("ingestion", _ingestion_metrics),
    ("build", _build_id_metrics),
)


def render(force: bool = False) -> str:
    """Render the exposition body, cached for CACHE_TTL_S.

    A collector that fails is skipped rather than failing the scrape. A
    broken metric should cost visibility of that metric, not of every
    other one — and a scrape endpoint that 500s takes the monitoring
    down at exactly the moment it is most needed.
    """
    global _cache
    now = time.monotonic()
    with _lock:
        rendered_at, body = _cache
        if not force and rendered_at and (now - rendered_at) < CACHE_TTL_S:
            return body

    lines: list[str] = []
    failed: list[str] = []
    for name, collector in _COLLECTORS:
        try:
            lines.extend(collector())
        except Exception as e:
            failed.append(name)
            logger.warning("Prometheus collector %r failed: %s", name, e)

    lines += _metric(
        "rag_metrics_collector_errors", "gauge",
        "Collectors that failed on the most recent scrape. Non-zero means "
        "some series below are missing, not that they are zero.",
        [("", len(failed))],
    )

    body = "\n".join(lines) + "\n"
    with _lock:
        _cache = (now, body)
    return body


def reset_cache() -> None:
    global _cache
    with _lock:
        _cache = (0.0, "")
