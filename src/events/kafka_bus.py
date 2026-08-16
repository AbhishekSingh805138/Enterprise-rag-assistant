"""Kafka transport for ingestion events (``EVENT_BUS=kafka``).

Chosen for production because it gives what the SQLite queue cannot:
consumer groups that spread partitions across many worker pods, retention
that survives the workers entirely, and replay from an offset.

Offsets are committed manually, only after the worker has settled the
event. A worker that dies mid-processing therefore leaves the offset
uncommitted and the event is redelivered to whoever takes over the
partition — the same crash-recovery guarantee the SQLite backend gets from
its visibility timeout.

Requires ``kafka-python``; it is an optional dependency, so the import
failure carries an actionable message rather than an ImportError traceback.
"""
from __future__ import annotations

import logging
import threading
import time

from config import settings
from src.events.bus import Event, EventBusError

logger = logging.getLogger(__name__)

# Upper bound on how long publish() will block to approximate a delayed
# send. Kafka has no native per-message delay.
_MAX_INLINE_DELAY_S = 30.0


def _require_kafka():
    try:
        import kafka

        return kafka
    except ImportError as e:  # pragma: no cover - depends on env
        raise EventBusError(
            "EVENT_BUS=kafka requires kafka-python. Install it with: "
            "pip install kafka-python"
        ) from e


class KafkaDelivery:
    """A consumed record; ack commits its offset, nack leaves it uncommitted."""

    def __init__(self, bus: KafkaEventBus, record, event: Event) -> None:
        self._bus = bus
        self._record = record
        self.event = event
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def ack(self) -> None:
        if self._settled:
            return
        self._bus._commit(self._record)
        self._settled = True

    def nack(self, delay_s: float = 0.0) -> None:
        """Leave the offset uncommitted so the event is redelivered.

        *delay_s* is ignored: redelivery timing is Kafka's to decide (on
        rebalance or restart). The worker's retry path does not rely on
        this — it republishes an incremented event and acks the original.
        """
        if self._settled:
            return
        self._bus._seek_back(self._record)
        self._settled = True


class KafkaEventBus:
    """Kafka-backed event bus. Producer is eager; consumers are per-group."""

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        kafka = _require_kafka()
        self._servers = [
            s.strip()
            for s in (bootstrap_servers or settings.kafka_bootstrap_servers).split(",")
            if s.strip()
        ]
        self._kafka = kafka
        self._lock = threading.Lock()
        self._consumers: dict[tuple[str, str], object] = {}

        try:
            self._producer = kafka.KafkaProducer(
                bootstrap_servers=self._servers,
                value_serializer=lambda v: v.encode("utf-8"),
                key_serializer=lambda v: v.encode("utf-8") if v else None,
                acks="all",          # durability over latency for ingestion
                retries=3,
                linger_ms=5,
            )
        except Exception as e:
            raise EventBusError(
                f"Could not connect to Kafka at {self._servers}: {e}"
            ) from e
        logger.info("Kafka event bus connected to %s", self._servers)

    # -- producer ----------------------------------------------------------

    def publish(self, topic: str, event: Event, delay_s: float = 0.0) -> None:
        """Produce *event* to *topic*, keyed by document id.

        Keying on the document id puts every event for one document on the
        same partition, so retries stay ordered behind the original.
        """
        if not topic:
            raise EventBusError("topic is required")
        if delay_s > 0:
            # No native delayed delivery in Kafka. For the small backoffs the
            # retry path uses this is adequate; a high-volume deployment
            # should instead use tiered retry topics consumed on a timer.
            time.sleep(min(delay_s, _MAX_INLINE_DELAY_S))
        try:
            future = self._producer.send(topic, key=event.document_id, value=event.to_json())
            future.get(timeout=10)
        except Exception as e:
            raise EventBusError(f"Failed to publish to {topic!r}: {e}") from e
        logger.debug(
            "Published %s to %s (event_id=%s attempt=%d)",
            event.event_type, topic, event.event_id, event.attempt,
        )

    # -- consumer ----------------------------------------------------------

    def _get_consumer(self, topic: str, group: str):
        key = (topic, group)
        with self._lock:
            consumer = self._consumers.get(key)
            if consumer is None:
                try:
                    consumer = self._kafka.KafkaConsumer(
                        topic,
                        bootstrap_servers=self._servers,
                        group_id=group,
                        # Offsets are committed by ack(), never automatically —
                        # otherwise a crash could skip an unprocessed document.
                        enable_auto_commit=False,
                        auto_offset_reset="earliest",
                        value_deserializer=lambda v: v.decode("utf-8"),
                        max_poll_records=10,
                    )
                except Exception as e:
                    raise EventBusError(
                        f"Could not subscribe to {topic!r} as group {group!r}: {e}"
                    ) from e
                self._consumers[key] = consumer
            return consumer

    def poll(
        self, topic: str, group: str, max_messages: int = 1, timeout_s: float = 1.0
    ) -> list[KafkaDelivery]:
        consumer = self._get_consumer(topic, group)
        try:
            batches = consumer.poll(
                timeout_ms=int(timeout_s * 1000), max_records=max(1, max_messages)
            )
        except Exception as e:
            raise EventBusError(f"Failed to poll {topic!r}: {e}") from e

        deliveries: list[KafkaDelivery] = []
        for records in batches.values():
            for record in records:
                try:
                    event = Event.from_json(record.value)
                except EventBusError:
                    # Poison payload: commit past it so it cannot block the
                    # partition, and record it loudly.
                    logger.error(
                        "Discarding malformed event at %s[%d]@%d",
                        record.topic, record.partition, record.offset,
                    )
                    self._commit(record)
                    continue
                deliveries.append(KafkaDelivery(self, record, event))
        return deliveries

    def _commit(self, record) -> None:
        try:
            from kafka import TopicPartition
            from kafka.structs import OffsetAndMetadata

            consumer = None
            with self._lock:
                for (topic, _group), candidate in self._consumers.items():
                    if topic == record.topic:
                        consumer = candidate
                        break
            if consumer is None:
                logger.warning("No consumer to commit offset for topic=%s", record.topic)
                return
            tp = TopicPartition(record.topic, record.partition)
            consumer.commit({tp: OffsetAndMetadata(record.offset + 1, None)})
        except Exception:
            logger.exception(
                "Failed to commit offset %s[%d]@%d",
                record.topic, record.partition, record.offset,
            )

    def _seek_back(self, record) -> None:
        """Rewind to this record so the next poll redelivers it."""
        try:
            from kafka import TopicPartition

            with self._lock:
                consumers = list(self._consumers.items())
            for (topic, _group), consumer in consumers:
                if topic == record.topic:
                    consumer.seek(
                        TopicPartition(record.topic, record.partition), record.offset
                    )
                    return
        except Exception:
            logger.debug("Could not rewind consumer; relying on rebalance", exc_info=True)

    # -- introspection -----------------------------------------------------

    def depth(self, topic: str) -> int:
        """Best-effort lag for *topic* (end offsets minus committed position)."""
        try:
            consumer = self._get_consumer(topic, settings.kafka_consumer_group)
            partitions = consumer.partitions_for_topic(topic)
            if not partitions:
                return 0
            from kafka import TopicPartition

            tps = [TopicPartition(topic, p) for p in partitions]
            end_offsets = consumer.end_offsets(tps)
            total = 0
            for tp in tps:
                committed = consumer.committed(tp) or 0
                total += max(0, end_offsets.get(tp, 0) - committed)
            return total
        except Exception:
            logger.debug("Could not compute depth for topic=%s", topic, exc_info=True)
            return -1

    def close(self) -> None:
        with self._lock:
            consumers = list(self._consumers.values())
            self._consumers.clear()
        for consumer in consumers:
            try:
                consumer.close()
            except Exception:
                logger.debug("Error closing Kafka consumer", exc_info=True)
        try:
            self._producer.flush(timeout=5)
            self._producer.close(timeout=5)
        except Exception:
            logger.debug("Error closing Kafka producer", exc_info=True)
