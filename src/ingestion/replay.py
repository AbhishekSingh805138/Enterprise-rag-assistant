"""Recover dead-lettered documents without asking the user to upload again.

A dead-letter queue that cannot be drained is a graveyard: the document was
accepted from a user, the bytes are still in object storage, the registry
knows exactly what happened — and yet the only way back into the pipeline
was for someone to re-upload the same file.

Two entry points:

    replay_document()  one document, by id — used by POST /documents/{id}/retry
    redrive_dlq()      consume the dead-letter topic and requeue everything

Both reset the retry budget and publish a *fresh* event (attempt 1), so a
replayed document gets the full retry policy again rather than resuming
one attempt from exhaustion.

Replay is safe to run repeatedly. A document that already succeeded is
skipped rather than re-embedded, and the worker's own idempotency check is
the second line of defence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config import settings
from src.events.bus import DOCUMENT_UPLOADED, Event, EventBus, get_event_bus
from src.ingestion.registry import (
    STATUS_DEAD_LETTER,
    STATUS_FAILED,
    STATUS_PROCESSED,
    DocumentRegistry,
    get_registry,
)
from src.storage.object_store import ObjectStore, get_object_store

logger = logging.getLogger(__name__)

# Outcomes, named so operators can tell "nothing to do" from "cannot fix".
OUTCOME_REQUEUED = "requeued"
OUTCOME_ALREADY_PROCESSED = "already_processed"
OUTCOME_NOT_REPLAYABLE = "not_replayable"
OUTCOME_MISSING_OBJECT = "missing_object"
OUTCOME_UNKNOWN_DOCUMENT = "unknown_document"
OUTCOME_PUBLISH_FAILED = "publish_failed"

# States a replay may act on. PENDING/PROCESSING are live already.
_REPLAYABLE = (STATUS_FAILED, STATUS_DEAD_LETTER)


@dataclass
class ReplayResult:
    """What happened to one replay attempt."""

    document_id: str
    outcome: str
    detail: str = ""
    filename: str = ""

    @property
    def requeued(self) -> bool:
        return self.outcome == OUTCOME_REQUEUED

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "outcome": self.outcome,
            "detail": self.detail,
            "requeued": self.requeued,
        }


def replay_document(
    document_id: str,
    *,
    registry: DocumentRegistry | None = None,
    bus: EventBus | None = None,
    store: ObjectStore | None = None,
    topic: str | None = None,
    force: bool = False,
) -> ReplayResult:
    """Reset a failed document and publish a fresh ingestion event.

    *force* also replays documents in states outside FAILED/DEAD_LETTER,
    for the case where an operator knows a PROCESSING row is abandoned.
    """
    registry = registry or get_registry()
    bus = bus or get_event_bus()
    store = store or get_object_store()
    topic = topic or settings.kafka_topic_ingestion

    record = registry.get(document_id)
    if record is None:
        return ReplayResult(document_id, OUTCOME_UNKNOWN_DOCUMENT,
                            "No registry entry for this id")

    if record.status == STATUS_PROCESSED and not force:
        return ReplayResult(document_id, OUTCOME_ALREADY_PROCESSED,
                            f"Already indexed ({record.chunks_indexed} chunks)",
                            record.filename)

    if record.status not in _REPLAYABLE and not force:
        return ReplayResult(document_id, OUTCOME_NOT_REPLAYABLE,
                            f"Status is {record.status}; nothing to replay",
                            record.filename)

    # The whole point of replay is that the bytes outlived the failure. If
    # they did not, the document genuinely needs re-uploading and saying so
    # is more useful than queueing work that will fail again.
    if not record.storage_key or not store.exists(record.storage_key):
        return ReplayResult(document_id, OUTCOME_MISSING_OBJECT,
                            "Stored object is gone; the file must be uploaded again",
                            record.filename)

    if not registry.reset_for_retry(document_id) and not force:
        # Lost a race with another replay or a re-upload.
        current = registry.get(document_id)
        return ReplayResult(document_id, OUTCOME_NOT_REPLAYABLE,
                            f"Could not reset (status is now {current.status if current else '?'})",
                            record.filename)

    try:
        bus.publish(
            topic,
            Event(
                event_type=DOCUMENT_UPLOADED,
                document_id=document_id,
                payload={
                    "storage_key": record.storage_key,
                    "filename": record.filename,
                    "department": record.department,
                    "checksum": record.checksum,
                    "replayed": True,
                },
                request_id=record.request_id,
            ),
        )
    except Exception as e:
        registry.mark_failed(document_id, f"Replay publish failed: {e}")
        logger.exception("Could not publish replay event for %s", document_id)
        return ReplayResult(document_id, OUTCOME_PUBLISH_FAILED, str(e), record.filename)

    logger.info("Replayed document %s (%s)", document_id, record.filename)
    return ReplayResult(document_id, OUTCOME_REQUEUED,
                        "Reset and requeued for indexing", record.filename)


def redrive_dlq(
    limit: int = 50,
    *,
    registry: DocumentRegistry | None = None,
    bus: EventBus | None = None,
    store: ObjectStore | None = None,
    topic: str | None = None,
    dlq_topic: str | None = None,
    group: str = "dlq-redrive",
    dry_run: bool = False,
) -> list[ReplayResult]:
    """Consume the dead-letter topic and requeue each document.

    Consuming (rather than reading the registry) is what actually drains
    the DLQ, so its depth returns to zero and the health check stops
    reporting a backlog of abandoned documents.

    Events are acknowledged even when the document turns out not to be
    replayable — leaving them would mean redriving the same unfixable
    document forever. The registry keeps the record either way.
    """
    bus = bus or get_event_bus()
    dlq_topic = dlq_topic or settings.kafka_topic_dlq
    results: list[ReplayResult] = []

    if dry_run:
        # Claim every event first and release them only at the end. Nacking
        # as we go would make each event immediately visible again and the
        # loop would re-poll the same message until it hit *limit*.
        registry = registry or get_registry()
        claimed = []
        while len(claimed) < limit:
            batch = bus.poll(dlq_topic, group, max_messages=1, timeout_s=0.5)
            if not batch:
                break
            claimed.extend(batch)
        try:
            for delivery in claimed:
                record = registry.get(delivery.event.document_id)
                results.append(ReplayResult(
                    delivery.event.document_id, "would_replay",
                    f"status={record.status if record else 'unknown'}",
                    record.filename if record else "",
                ))
        finally:
            for delivery in claimed:
                delivery.nack()
        return results

    while len(results) < limit:
        deliveries = bus.poll(dlq_topic, group, max_messages=1, timeout_s=0.5)
        if not deliveries:
            break

        for delivery in deliveries:
            result = replay_document(
                delivery.event.document_id,
                registry=registry, bus=bus, store=store, topic=topic,
            )
            results.append(result)
            delivery.ack()

    logger.info(
        "DLQ redrive complete: %d event(s), %d requeued",
        len(results), sum(1 for r in results if r.requeued),
    )
    return results
