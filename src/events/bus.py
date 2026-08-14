"""Event envelope and the bus protocol the ingestion pipeline speaks.

The upload API publishes; the ingestion worker consumes. Neither knows
whether the transport is Kafka or the SQLite queue — that choice is
config (``EVENT_BUS``), which keeps the local developer loop identical to
production behaviour including retries and dead-lettering.

Delivery is at-least-once in both backends. That is safe here because the
worker's first action is an idempotency check against the document
registry, so a redelivered event is acknowledged rather than re-indexed.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from config import settings

logger = logging.getLogger(__name__)

# Event type published when a document has been durably stored and is
# awaiting indexing.
DOCUMENT_UPLOADED = "document.uploaded"


class EventBusError(RuntimeError):
    """Raised when publishing or consuming fails."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    """A message on the bus.

    *attempt* is carried in the envelope rather than tracked broker-side so
    the retry budget survives redelivery across any transport, and so a
    consumer can tell a first delivery from a third one.
    """

    event_type: str
    document_id: str
    payload: dict = field(default_factory=dict)
    attempt: int = 1
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    published_at: str = field(default_factory=_utc_now_iso)
    request_id: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "document_id": self.document_id,
                "payload": self.payload,
                "attempt": self.attempt,
                "published_at": self.published_at,
                "request_id": self.request_id,
            }
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> "Event":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise EventBusError(f"Malformed event payload: {e}") from e
        if not isinstance(data, dict):
            raise EventBusError("Event payload must be a JSON object")
        try:
            return cls(
                event_type=data["event_type"],
                document_id=data["document_id"],
                payload=data.get("payload") or {},
                attempt=int(data.get("attempt", 1)),
                event_id=data.get("event_id") or uuid.uuid4().hex,
                published_at=data.get("published_at") or _utc_now_iso(),
                request_id=data.get("request_id") or "",
            )
        except (KeyError, TypeError, ValueError) as e:
            raise EventBusError(f"Event is missing required fields: {e}") from e

    def next_attempt(self) -> "Event":
        """Return a copy of this event marked as one delivery further along."""
        return Event(
            event_type=self.event_type,
            document_id=self.document_id,
            payload=dict(self.payload),
            attempt=self.attempt + 1,
            event_id=self.event_id,  # stable across retries for tracing
            published_at=_utc_now_iso(),
            request_id=self.request_id,
        )


@runtime_checkable
class Delivery(Protocol):
    """A consumed event plus the handles to settle it."""

    event: Event

    def ack(self) -> None:
        """Mark the event as handled; it will not be redelivered."""
        ...

    def nack(self, delay_s: float = 0.0) -> None:
        """Return the event to the queue, visible again after *delay_s*."""
        ...


@runtime_checkable
class EventBus(Protocol):
    """Transport for ingestion events."""

    def publish(self, topic: str, event: Event, delay_s: float = 0.0) -> None:
        """Append *event* to *topic*, visible after *delay_s* seconds.

        The SQLite backend honours the delay natively; Kafka approximates
        it (see :mod:`src.events.kafka_bus`).
        """
        ...

    def poll(
        self, topic: str, group: str, max_messages: int = 1, timeout_s: float = 1.0
    ) -> list[Delivery]:
        """Claim up to *max_messages* pending events from *topic*."""
        ...

    def close(self) -> None:
        """Release transport resources."""
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_bus: EventBus | None = None
_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Return the configured event bus (singleton, thread-safe)."""
    global _bus
    with _lock:
        if _bus is None:
            backend = settings.event_bus.lower()
            if backend == "kafka":
                from src.events.kafka_bus import KafkaEventBus

                _bus = KafkaEventBus()
            elif backend == "sqlite":
                from src.events.sqlite_bus import SQLiteEventBus

                _bus = SQLiteEventBus()
            else:
                raise EventBusError(
                    f"Unknown EVENT_BUS {settings.event_bus!r}. Choose from: sqlite, kafka"
                )
        return _bus


def reset_event_bus() -> None:
    """Close and drop the cached bus (tests, config reload, worker shutdown)."""
    global _bus
    with _lock:
        if _bus is not None:
            try:
                _bus.close()
            except Exception:
                logger.debug("Error closing event bus", exc_info=True)
        _bus = None
