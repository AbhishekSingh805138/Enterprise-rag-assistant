"""Ingestion worker — consumes upload events and indexes documents.

Implements the consumer lane of the ingestion architecture, including the
failure path the diagram calls out: up to ``INGEST_MAX_ATTEMPTS`` tries,
then the event goes to the dead-letter queue instead of cycling forever.

Per event::

    consume -> already indexed? -> yes: ack and move on
                                -> no : download, parse, chunk, embed, store
                                        -> mark processed -> ack
                                        -> failure -> retry or dead-letter

Two classes of failure are treated differently, because retrying them
blindly is the difference between a self-healing pipeline and one that
burns its budget on work that can never succeed:

  permanent  a corrupt file, or an object missing from storage. Retrying
             produces the identical error, so it is dead-lettered on the
             first occurrence.
  transient  embedding API rate limits, vector store hiccups, network
             blips. Retried with exponential backoff.

Every event is acknowledged exactly once. Retries work by publishing a new
event carrying ``attempt + 1`` and then acking the original, so the retry
budget survives redelivery on any transport.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

from config import settings
from src.events.bus import Delivery, Event, EventBus, EventBusError, get_event_bus
from src.ingestion.pipeline import DocumentParseError, process_document
from src.ingestion.registry import (
    STATUS_DEAD_LETTER,
    STATUS_PROCESSED,
    DocumentRegistry,
    get_registry,
)
from src.observability.request_context import reset_request_id, set_request_id
from src.storage.object_store import ObjectNotFoundError, ObjectStore, get_object_store

logger = logging.getLogger(__name__)

# Outcomes, returned by handle() so callers and tests can assert on the
# path taken rather than inferring it from side effects.
OUTCOME_PROCESSED = "processed"
OUTCOME_SKIPPED = "skipped"
OUTCOME_RETRIED = "retried"
OUTCOME_DEAD_LETTERED = "dead_lettered"
OUTCOME_UNKNOWN_DOCUMENT = "unknown_document"

# Failures that will never succeed on a retry.
_PERMANENT_ERRORS = (DocumentParseError, ObjectNotFoundError)


@dataclass
class WorkerStats:
    """Running counters, surfaced in logs and the deep health check."""

    polled: int = 0
    processed: int = 0
    skipped: int = 0
    retried: int = 0
    dead_lettered: int = 0
    unknown: int = 0
    chunks_indexed: int = 0
    started_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "polled": self.polled,
            "processed": self.processed,
            "skipped": self.skipped,
            "retried": self.retried,
            "dead_lettered": self.dead_lettered,
            "unknown": self.unknown,
            "chunks_indexed": self.chunks_indexed,
            "uptime_s": round(time.time() - self.started_at, 1),
        }


class IngestionWorker:
    """Polls the ingestion topic and indexes documents.

    Dependencies are injected so the worker can be driven in tests against
    an in-process queue and a temp-directory object store, exercising the
    real retry and dead-letter logic without a broker.
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        registry: DocumentRegistry | None = None,
        object_store: ObjectStore | None = None,
        topic: str | None = None,
        dlq_topic: str | None = None,
        group: str | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self._bus = bus or get_event_bus()
        self._registry = registry or get_registry()
        self._store = object_store or get_object_store()
        self._topic = topic or settings.kafka_topic_ingestion
        self._dlq_topic = dlq_topic or settings.kafka_topic_dlq
        self._group = group or settings.kafka_consumer_group
        self._max_attempts = max_attempts or settings.ingest_max_attempts
        self.stats = WorkerStats()

    # -- single event ------------------------------------------------------

    def handle(self, delivery: Delivery) -> str:
        """Process one delivery to completion. Always settles the delivery."""
        event = delivery.event
        token = set_request_id(event.request_id or None)
        try:
            return self._handle_inner(delivery, event)
        finally:
            reset_request_id(token)

    def _handle_inner(self, delivery: Delivery, event: Event) -> str:
        record = self._registry.get(event.document_id)
        if record is None:
            # The registry row is written before the event is published, so
            # this means the registry was reset or the event is foreign.
            # There is nothing to index and nothing to retry.
            logger.error(
                "Event %s references unknown document %s — acking",
                event.event_id, event.document_id,
            )
            delivery.ack()
            self.stats.unknown += 1
            return OUTCOME_UNKNOWN_DOCUMENT

        # --- "Already Indexed?" ---
        if not self._registry.claim_for_processing(event.document_id):
            current = self._registry.get(event.document_id)
            status = current.status if current else "UNKNOWN"
            if status == STATUS_PROCESSED:
                logger.info(
                    "Document %s already indexed (%d chunks) — ignoring event %s",
                    event.document_id,
                    current.chunks_indexed if current else 0,
                    event.event_id,
                )
            elif status == STATUS_DEAD_LETTER:
                logger.warning(
                    "Document %s is dead-lettered — ignoring event %s",
                    event.document_id, event.event_id,
                )
            else:
                logger.info(
                    "Document %s is held by another worker (status=%s) — ignoring event %s",
                    event.document_id, status, event.event_id,
                )
            delivery.ack()
            self.stats.skipped += 1
            return OUTCOME_SKIPPED

        # Re-read so the attempt count reflects this claim.
        record = self._registry.get(event.document_id) or record

        try:
            result = process_document(record, object_store=self._store)
        except _PERMANENT_ERRORS as e:
            return self._dead_letter(delivery, event, f"{type(e).__name__}: {e}", permanent=True)
        except Exception as e:  # transient by assumption — retry with backoff
            return self._retry_or_dead_letter(delivery, event, f"{type(e).__name__}: {e}")

        # --- "Mark Document Processed" ---
        self._registry.mark_processed(event.document_id, result.chunks_added)
        delivery.ack()
        self.stats.processed += 1
        self.stats.chunks_indexed += result.chunks_added
        logger.info(
            "Indexed document %s (%s): %d chunk(s) added",
            event.document_id, record.filename, result.chunks_added,
        )
        return OUTCOME_PROCESSED

    def _retry_or_dead_letter(self, delivery: Delivery, event: Event, error: str) -> str:
        if event.attempt >= self._max_attempts:
            return self._dead_letter(delivery, event, error, permanent=False)

        backoff = settings.ingest_retry_backoff_s * (2 ** (event.attempt - 1))
        self._registry.mark_failed(event.document_id, error)
        retry_event = event.next_attempt()
        try:
            self._bus.publish(self._topic, retry_event, delay_s=backoff)
        except EventBusError:
            # Could not re-enqueue. Leave the original unacked so the queue
            # redelivers it — losing the event here would silently drop the
            # document, which is worse than a duplicate delivery.
            logger.exception(
                "Could not schedule retry for %s — leaving event for redelivery",
                event.document_id,
            )
            delivery.nack(delay_s=backoff)
            self.stats.retried += 1
            return OUTCOME_RETRIED

        delivery.ack()
        self.stats.retried += 1
        logger.warning(
            "Document %s failed (attempt %d/%d), retrying in %.1fs: %s",
            event.document_id, event.attempt, self._max_attempts, backoff, error,
        )
        return OUTCOME_RETRIED

    def _dead_letter(
        self, delivery: Delivery, event: Event, error: str, *, permanent: bool
    ) -> str:
        reason = "unrecoverable" if permanent else f"{self._max_attempts} attempts exhausted"
        self._registry.mark_dead_letter(event.document_id, error)
        dlq_event = Event(
            event_type=f"{event.event_type}.dlq",
            document_id=event.document_id,
            payload={**event.payload, "error": error, "reason": reason},
            attempt=event.attempt,
            event_id=event.event_id,
            request_id=event.request_id,
        )
        try:
            self._bus.publish(self._dlq_topic, dlq_event)
        except EventBusError:
            # The registry already records DEAD_LETTER, so the document is
            # not lost even when the DLQ publish fails.
            logger.exception("Failed to publish %s to the DLQ", event.document_id)

        delivery.ack()
        self.stats.dead_lettered += 1
        logger.error(
            "Document %s dead-lettered (%s): %s", event.document_id, reason, error
        )
        return OUTCOME_DEAD_LETTERED

    # -- loop --------------------------------------------------------------

    def poll_once(self, timeout_s: float | None = None) -> list[str]:
        """Claim and handle one batch. Returns the outcome of each event."""
        deliveries = self._bus.poll(
            self._topic,
            self._group,
            max_messages=max(1, settings.worker_batch_size),
            timeout_s=timeout_s if timeout_s is not None else settings.worker_poll_interval_s,
        )
        self.stats.polled += len(deliveries)
        outcomes = []
        for delivery in deliveries:
            try:
                outcomes.append(self.handle(delivery))
            except Exception:
                # handle() is meant to settle every delivery itself; if it
                # raised before doing so, release the event rather than
                # holding it invisible until the timeout lapses.
                logger.exception("Unhandled error while processing a delivery")
                try:
                    delivery.nack(delay_s=settings.ingest_retry_backoff_s)
                except Exception:
                    logger.debug("Could not nack after unhandled error", exc_info=True)
        return outcomes

    def run(self, stop_event: threading.Event | None = None) -> WorkerStats:
        """Consume until *stop_event* is set. Blocks the calling thread."""
        stop = stop_event or threading.Event()
        logger.info(
            "Ingestion worker started (topic=%s group=%s dlq=%s max_attempts=%d)",
            self._topic, self._group, self._dlq_topic, self._max_attempts,
        )
        while not stop.is_set():
            try:
                outcomes = self.poll_once()
            except EventBusError:
                logger.exception("Event bus error — backing off before retrying")
                stop.wait(max(1.0, settings.worker_poll_interval_s))
                continue
            except Exception:
                logger.exception("Unexpected worker error — continuing")
                stop.wait(max(1.0, settings.worker_poll_interval_s))
                continue

            if not outcomes:
                # Idle: wait on the stop event rather than sleeping, so
                # shutdown is immediate instead of up to one interval late.
                stop.wait(settings.worker_poll_interval_s)

        logger.info("Ingestion worker stopped: %s", self.stats.to_dict())
        return self.stats
