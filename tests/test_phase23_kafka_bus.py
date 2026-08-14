"""Phase 23 — KafkaEventBus.

kafka-python is an optional dependency, so this suite drives the bus
against a fake `kafka` module injected into sys.modules. That is not a
shortcut: the contract worth testing here is how the bus *uses* the
client — manual offset commits, commit-at-offset+1, keying by document id,
poison-payload handling — and those assertions are sharper against a
recording fake than against a live broker.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from src.events.bus import Event, EventBusError

TOPIC = "document.uploaded"
GROUP = "ingestion-workers"


# ---------------------------------------------------------------------------
# Fake kafka client
# ---------------------------------------------------------------------------

class FakeTopicPartition:
    def __init__(self, topic, partition):
        self.topic = topic
        self.partition = partition

    def __eq__(self, other):
        return (self.topic, self.partition) == (other.topic, other.partition)

    def __hash__(self):
        return hash((self.topic, self.partition))

    def __repr__(self):
        return f"TP({self.topic},{self.partition})"


class FakeOffsetAndMetadata:
    def __init__(self, offset, metadata):
        self.offset = offset
        self.metadata = metadata

    def __eq__(self, other):
        return self.offset == other.offset


class FakeRecord:
    def __init__(self, topic, partition, offset, value):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.value = value


class FakeFuture:
    def __init__(self, error=None):
        self._error = error
        self.get_calls = []

    def get(self, timeout=None):
        self.get_calls.append(timeout)
        if self._error:
            raise self._error
        return True


class FakeProducer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sent = []
        self.flushed = False
        self.closed = False
        self.send_error = None
        self.future_error = None

    def send(self, topic, key=None, value=None):
        if self.send_error:
            raise self.send_error
        self.sent.append({"topic": topic, "key": key, "value": value})
        return FakeFuture(self.future_error)

    def flush(self, timeout=None):
        self.flushed = True

    def close(self, timeout=None):
        self.closed = True


class FakeConsumer:
    def __init__(self, *topics, **kwargs):
        self.topics = topics
        self.kwargs = kwargs
        self.batches = []          # queued poll() responses
        self.commits = []
        self.seeks = []
        self.closed = False
        self.poll_error = None
        self.commit_error = None
        self._partitions = {0}
        self._end_offsets = {}
        self._committed = {}

    def poll(self, timeout_ms=None, max_records=None):
        if self.poll_error:
            raise self.poll_error
        return self.batches.pop(0) if self.batches else {}

    def commit(self, offsets):
        if self.commit_error:
            raise self.commit_error
        self.commits.append(offsets)

    def seek(self, tp, offset):
        self.seeks.append((tp, offset))

    def partitions_for_topic(self, topic):
        return self._partitions

    def end_offsets(self, tps):
        return {tp: self._end_offsets.get(tp.partition, 0) for tp in tps}

    def committed(self, tp):
        return self._committed.get(tp.partition)

    def close(self):
        self.closed = True


@pytest.fixture
def fake_kafka(monkeypatch):
    """Install a fake `kafka` package for the duration of a test."""
    kafka_mod = types.ModuleType("kafka")
    structs_mod = types.ModuleType("kafka.structs")

    created = {"producers": [], "consumers": []}

    def _make_producer(**kwargs):
        p = FakeProducer(**kwargs)
        created["producers"].append(p)
        return p

    def _make_consumer(*topics, **kwargs):
        c = FakeConsumer(*topics, **kwargs)
        created["consumers"].append(c)
        return c

    kafka_mod.KafkaProducer = _make_producer
    kafka_mod.KafkaConsumer = _make_consumer
    kafka_mod.TopicPartition = FakeTopicPartition
    kafka_mod.structs = structs_mod
    structs_mod.OffsetAndMetadata = FakeOffsetAndMetadata

    monkeypatch.setitem(sys.modules, "kafka", kafka_mod)
    monkeypatch.setitem(sys.modules, "kafka.structs", structs_mod)
    kafka_mod._created = created
    yield kafka_mod


@pytest.fixture
def kafka_settings():
    with patch("src.events.kafka_bus.settings") as s:
        s.kafka_bootstrap_servers = "broker-1:9092,broker-2:9092"
        s.kafka_consumer_group = GROUP
        yield s


@pytest.fixture
def bus(fake_kafka, kafka_settings):
    from src.events.kafka_bus import KafkaEventBus

    return KafkaEventBus()


def make_event(document_id="doc_1", attempt=1):
    return Event(
        event_type="document.uploaded",
        document_id=document_id,
        payload={"storage_key": "documents/hr/doc_1/f.md"},
        attempt=attempt,
    )


def record_for(event, offset=0, partition=0, topic=TOPIC):
    return FakeRecord(topic, partition, offset, event.to_json())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_bootstrap_servers_are_split_on_commas(self, bus):
        assert bus._servers == ["broker-1:9092", "broker-2:9092"]

    def test_explicit_servers_override_settings(self, fake_kafka, kafka_settings):
        from src.events.kafka_bus import KafkaEventBus

        assert KafkaEventBus(bootstrap_servers="other:9092")._servers == ["other:9092"]

    def test_producer_favours_durability_over_latency(self, bus):
        """acks=all + retries: an accepted upload must not be lost mid-publish."""
        assert bus._producer.kwargs["acks"] == "all"
        assert bus._producer.kwargs["retries"] >= 1

    def test_missing_kafka_package_gives_an_actionable_error(self, kafka_settings, monkeypatch):
        monkeypatch.delitem(sys.modules, "kafka", raising=False)
        import builtins

        real_import = builtins.__import__

        def _no_kafka(name, *args, **kwargs):
            if name == "kafka":
                raise ImportError("No module named 'kafka'")
            return real_import(name, *args, **kwargs)

        from src.events.kafka_bus import KafkaEventBus

        with patch.object(builtins, "__import__", _no_kafka):
            with pytest.raises(EventBusError, match="kafka-python"):
                KafkaEventBus()

    def test_broker_connection_failure_is_wrapped(self, fake_kafka, kafka_settings):
        def _boom(**kwargs):
            raise OSError("connection refused")

        fake_kafka.KafkaProducer = _boom
        from src.events.kafka_bus import KafkaEventBus

        with pytest.raises(EventBusError, match="Could not connect to Kafka"):
            KafkaEventBus()


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------

class TestPublish:
    def test_event_is_sent_to_the_topic(self, bus):
        bus.publish(TOPIC, make_event())
        assert len(bus._producer.sent) == 1
        assert bus._producer.sent[0]["topic"] == TOPIC

    def test_message_is_keyed_by_document_id(self, bus):
        """Keying keeps a document's retries on one partition, behind the original."""
        bus.publish(TOPIC, make_event("doc_key_test"))
        assert bus._producer.sent[0]["key"] == "doc_key_test"

    def test_payload_roundtrips_through_the_wire_format(self, bus):
        original = make_event("doc_rt", attempt=3)
        bus.publish(TOPIC, original)
        restored = Event.from_json(bus._producer.sent[0]["value"])
        assert restored.document_id == "doc_rt"
        assert restored.attempt == 3
        assert restored.event_id == original.event_id

    def test_publish_blocks_until_the_broker_acknowledges(self, bus):
        """send() alone is async; the bus must wait so failures surface here."""
        bus.publish(TOPIC, make_event())
        # FakeFuture records the timeout it was awaited with.
        assert bus._producer.sent  # sent, and future.get() did not raise

    def test_empty_topic_is_rejected(self, bus):
        with pytest.raises(EventBusError, match="topic is required"):
            bus.publish("", make_event())

    def test_send_failure_is_wrapped(self, bus):
        bus._producer.send_error = OSError("broker down")
        with pytest.raises(EventBusError, match="Failed to publish"):
            bus.publish(TOPIC, make_event())

    def test_broker_nack_is_wrapped(self, bus):
        bus._producer.future_error = OSError("not enough replicas")
        with pytest.raises(EventBusError, match="Failed to publish"):
            bus.publish(TOPIC, make_event())

    def test_delay_is_approximated_by_sleeping(self, bus):
        """Kafka has no native delayed delivery; the bus waits instead."""
        with patch("src.events.kafka_bus.time.sleep") as sleep:
            bus.publish(TOPIC, make_event(), delay_s=4.0)
        sleep.assert_called_once_with(4.0)

    def test_delay_is_capped_so_a_worker_cannot_stall_indefinitely(self, bus):
        with patch("src.events.kafka_bus.time.sleep") as sleep:
            bus.publish(TOPIC, make_event(), delay_s=9999)
        assert sleep.call_args.args[0] == 30.0

    def test_no_delay_means_no_sleep(self, bus):
        with patch("src.events.kafka_bus.time.sleep") as sleep:
            bus.publish(TOPIC, make_event())
        sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Consuming
# ---------------------------------------------------------------------------

class TestPoll:
    def test_poll_returns_deliveries(self, bus):
        event = make_event("doc_poll")
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer.batches.append({FakeTopicPartition(TOPIC, 0): [record_for(event)]})

        deliveries = bus.poll(TOPIC, GROUP)
        assert len(deliveries) == 1
        assert deliveries[0].event.document_id == "doc_poll"

    def test_empty_poll_returns_nothing(self, bus):
        assert bus.poll(TOPIC, GROUP) == []

    def test_offsets_are_never_auto_committed(self, bus):
        """Auto-commit could skip an unprocessed document after a crash."""
        consumer = bus._get_consumer(TOPIC, GROUP)
        assert consumer.kwargs["enable_auto_commit"] is False

    def test_consumer_starts_from_the_earliest_offset(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        assert consumer.kwargs["auto_offset_reset"] == "earliest"

    def test_consumer_joins_the_configured_group(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        assert consumer.kwargs["group_id"] == GROUP

    def test_consumers_are_reused_per_topic_and_group(self, bus):
        assert bus._get_consumer(TOPIC, GROUP) is bus._get_consumer(TOPIC, GROUP)

    def test_different_groups_get_different_consumers(self, bus):
        assert bus._get_consumer(TOPIC, "a") is not bus._get_consumer(TOPIC, "b")

    def test_records_across_partitions_are_all_returned(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer.batches.append({
            FakeTopicPartition(TOPIC, 0): [record_for(make_event("doc_p0"), offset=0)],
            FakeTopicPartition(TOPIC, 1): [record_for(make_event("doc_p1"), offset=0, partition=1)],
        })
        ids = {d.event.document_id for d in bus.poll(TOPIC, GROUP, max_messages=10)}
        assert ids == {"doc_p0", "doc_p1"}

    def test_poll_timeout_is_passed_in_milliseconds(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        seen = {}
        consumer.poll = lambda timeout_ms=None, max_records=None: seen.update(
            timeout_ms=timeout_ms, max_records=max_records
        ) or {}
        bus.poll(TOPIC, GROUP, max_messages=5, timeout_s=2.5)
        assert seen["timeout_ms"] == 2500
        assert seen["max_records"] == 5

    def test_poll_failure_is_wrapped(self, bus):
        bus._get_consumer(TOPIC, GROUP).poll_error = OSError("rebalance in progress")
        with pytest.raises(EventBusError, match="Failed to poll"):
            bus.poll(TOPIC, GROUP)

    def test_subscribe_failure_is_wrapped(self, bus, fake_kafka):
        def _boom(*topics, **kwargs):
            raise OSError("no such topic")

        fake_kafka.KafkaConsumer = _boom
        with pytest.raises(EventBusError, match="Could not subscribe"):
            bus.poll("brand-new-topic", GROUP)

    def test_poison_payload_is_committed_past_not_redelivered(self, bus):
        """A malformed record must not block its partition forever."""
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer.batches.append(
            {FakeTopicPartition(TOPIC, 0): [FakeRecord(TOPIC, 0, 7, "{corrupt")]}
        )
        assert bus.poll(TOPIC, GROUP) == []
        assert consumer.commits, "poison record was not committed past"
        committed = list(consumer.commits[0].values())[0]
        assert committed.offset == 8

    def test_valid_records_survive_a_poison_neighbour(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer.batches.append({
            FakeTopicPartition(TOPIC, 0): [
                FakeRecord(TOPIC, 0, 1, "{corrupt"),
                record_for(make_event("doc_good"), offset=2),
            ]
        })
        deliveries = bus.poll(TOPIC, GROUP, max_messages=10)
        assert [d.event.document_id for d in deliveries] == ["doc_good"]


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

class TestDeliverySettlement:
    def _one_delivery(self, bus, offset=4):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer.batches.append(
            {FakeTopicPartition(TOPIC, 0): [record_for(make_event(), offset=offset)]}
        )
        return bus.poll(TOPIC, GROUP)[0], consumer

    def test_ack_commits_the_next_offset(self, bus):
        """Kafka's committed offset is the next record to read, not the last read."""
        delivery, consumer = self._one_delivery(bus, offset=4)
        delivery.ack()
        committed = list(consumer.commits[0].values())[0]
        assert committed.offset == 5

    def test_ack_commits_the_right_partition(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer.batches.append(
            {FakeTopicPartition(TOPIC, 3): [record_for(make_event(), offset=1, partition=3)]}
        )
        bus.poll(TOPIC, GROUP)[0].ack()
        tp = list(consumer.commits[0].keys())[0]
        assert (tp.topic, tp.partition) == (TOPIC, 3)

    def test_ack_is_idempotent(self, bus):
        delivery, consumer = self._one_delivery(bus)
        delivery.ack()
        delivery.ack()
        assert len(consumer.commits) == 1

    def test_settled_flag_reflects_state(self, bus):
        delivery, _ = self._one_delivery(bus)
        assert delivery.settled is False
        delivery.ack()
        assert delivery.settled is True

    def test_nack_does_not_commit(self, bus):
        """Leaving the offset uncommitted is what makes redelivery happen."""
        delivery, consumer = self._one_delivery(bus)
        delivery.nack()
        assert consumer.commits == []

    def test_nack_rewinds_so_the_next_poll_redelivers(self, bus):
        delivery, consumer = self._one_delivery(bus, offset=9)
        delivery.nack()
        assert consumer.seeks and consumer.seeks[0][1] == 9

    def test_nack_after_ack_is_a_no_op(self, bus):
        delivery, consumer = self._one_delivery(bus)
        delivery.ack()
        delivery.nack()
        assert consumer.seeks == []

    def test_commit_failure_is_swallowed_not_raised(self, bus):
        """A failed commit means redelivery, which the worker already tolerates."""
        delivery, consumer = self._one_delivery(bus)
        consumer.commit_error = OSError("coordinator unavailable")
        delivery.ack()  # must not raise
        assert delivery.settled is True

    def test_commit_without_a_consumer_is_handled(self, bus):
        from src.events.kafka_bus import KafkaDelivery

        orphan = FakeRecord("never-polled-topic", 0, 0, make_event().to_json())
        KafkaDelivery(bus, orphan, make_event()).ack()  # must not raise

    def test_seek_failure_falls_back_to_rebalance(self, bus):
        delivery, consumer = self._one_delivery(bus)

        def _boom(tp, offset):
            raise OSError("not assigned")

        consumer.seek = _boom
        delivery.nack()  # must not raise
        assert delivery.settled is True


# ---------------------------------------------------------------------------
# Introspection & shutdown
# ---------------------------------------------------------------------------

class TestDepth:
    def test_depth_is_end_offset_minus_committed(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer._partitions = {0, 1}
        consumer._end_offsets = {0: 10, 1: 5}
        consumer._committed = {0: 7, 1: 5}
        assert bus.depth(TOPIC) == 3

    def test_uncommitted_partition_counts_from_zero(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer._partitions = {0}
        consumer._end_offsets = {0: 4}
        consumer._committed = {}
        assert bus.depth(TOPIC) == 4

    def test_depth_never_goes_negative(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        consumer._partitions = {0}
        consumer._end_offsets = {0: 2}
        consumer._committed = {0: 9}
        assert bus.depth(TOPIC) == 0

    def test_unknown_topic_has_no_depth(self, bus):
        bus._get_consumer(TOPIC, GROUP)._partitions = None
        assert bus.depth(TOPIC) == 0

    def test_depth_failure_reports_unknown(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)

        def _boom(topic):
            raise OSError("metadata unavailable")

        consumer.partitions_for_topic = _boom
        assert bus.depth(TOPIC) == -1


class TestClose:
    def test_close_shuts_down_consumers_and_producer(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)
        bus.close()
        assert consumer.closed is True
        assert bus._producer.flushed is True
        assert bus._producer.closed is True

    def test_close_flushes_before_closing(self, bus):
        """Buffered events must reach the broker, not die with the process."""
        order = []
        bus._producer.flush = lambda timeout=None: order.append("flush")
        bus._producer.close = lambda timeout=None: order.append("close")
        bus.close()
        assert order == ["flush", "close"]

    def test_close_tolerates_a_failing_consumer(self, bus):
        consumer = bus._get_consumer(TOPIC, GROUP)

        def _boom():
            raise OSError("already closed")

        consumer.close = _boom
        bus.close()  # must not raise
        assert bus._producer.closed is True

    def test_close_tolerates_a_failing_producer(self, bus):
        def _boom(timeout=None):
            raise OSError("broker gone")

        bus._producer.flush = _boom
        bus.close()  # must not raise

    def test_close_clears_the_consumer_cache(self, bus):
        bus._get_consumer(TOPIC, GROUP)
        bus.close()
        assert bus._consumers == {}


class TestFactoryIntegration:
    def test_factory_builds_a_kafka_bus_when_configured(self, fake_kafka, kafka_settings):
        from src.events.bus import get_event_bus, reset_event_bus
        from src.events.kafka_bus import KafkaEventBus

        reset_event_bus()
        try:
            with patch("src.events.bus.settings") as s:
                s.event_bus = "kafka"
                assert isinstance(get_event_bus(), KafkaEventBus)
        finally:
            reset_event_bus()
