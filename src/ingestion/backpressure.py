"""Refuse uploads the ingestion pipeline cannot keep up with.

Uploads were accepted regardless of queue depth. With workers stalled the
queue and the object store grew without bound, and — worse than the disk
cost — every caller kept receiving ``202 Accepted`` for work that would
not happen for hours. A 202 is a promise; issuing it when the backlog is
already hours deep makes the API dishonest rather than merely slow.

The check is deliberately cheap and deliberately fail-open. Queue depth
is polled at most once per ``BACKPRESSURE_CACHE_TTL_S``, because a count
per upload puts the queue on the request path; and if depth cannot be
read at all, uploads proceed. A monitoring failure should not become an
ingestion outage.
"""
from __future__ import annotations

import logging
import threading
import time

from config import settings

logger = logging.getLogger(__name__)

CACHE_TTL_S = 5.0

_cache: tuple[float, int] = (0.0, 0)
_lock = threading.Lock()


class QueueSaturated(RuntimeError):
    """The ingestion backlog is at or past its limit."""

    def __init__(self, depth: int, limit: int):
        self.depth = depth
        self.limit = limit
        super().__init__(
            f"Ingestion queue is saturated ({depth} events pending, limit {limit}). "
            f"Retry once the backlog drains."
        )


def queue_depth(force: bool = False) -> int:
    """Pending events on the ingestion topic, cached briefly.

    Returns 0 when the depth cannot be determined, so an unavailable
    metric never blocks uploads.
    """
    global _cache
    now = time.monotonic()
    with _lock:
        checked_at, cached = _cache
        if not force and checked_at and (now - checked_at) < CACHE_TTL_S:
            return cached

    try:
        from src.events.bus import get_event_bus

        bus = get_event_bus()
        depth = bus.depth(settings.kafka_topic_ingestion)
        depth = int(depth) if depth is not None else 0
    except Exception as e:
        logger.debug("Could not read ingestion queue depth: %s", e)
        return 0

    with _lock:
        _cache = (now, depth)
    return depth


def reset_cache() -> None:
    global _cache
    with _lock:
        _cache = (0.0, 0)


def check_capacity() -> None:
    """Raise :class:`QueueSaturated` if the backlog is at its limit."""
    limit = settings.ingest_max_queue_depth
    if limit <= 0:
        return  # disabled
    depth = queue_depth()
    if depth >= limit:
        logger.warning(
            "Rejecting upload: ingestion queue depth %d has reached the limit %d",
            depth, limit,
        )
        raise QueueSaturated(depth, limit)


def retry_after_seconds(depth: int, limit: int) -> int:
    """A Retry-After the client can act on, from the current backlog.

    Estimated from the configured drain rate rather than a fixed
    constant: telling every caller to come back in 60 seconds when the
    queue is four hours deep just moves the stampede.
    """
    rate = max(settings.ingest_drain_rate_per_minute, 1e-6)
    minutes = max(0, depth - limit + 1) / rate
    return max(5, min(int(minutes * 60) + 5, 3600))
