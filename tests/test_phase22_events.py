"""Phase 22 — event envelope and the durable SQLite queue.

The queue backs the retry and dead-letter paths, so the guarantees tested
here (at-least-once delivery, visibility timeout, delayed redelivery, no
double-claim) are what the worker's correctness rests on.
"""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from src.events.bus import Event, EventBusError, reset_event_bus
from src.events.sqlite_bus import SQLiteEventBus

TOPIC = "document.uploaded"
GROUP = "ingestion-workers"


@pytest.fixture
def bus(tmp_path):
    b = SQLiteEventBus(db_path=tmp_path / "events.db")
    yield b
    b.close()


@pytest.fixture(autouse=True)
def _visibility_timeout():
    """Keep the visibility timeout generous unless a test overrides it."""
    with patch("src.events.sqlite_bus.settings") as s:
        s.ingest_visibility_timeout_s = 300
        yield s


def make_event(document_id: str = "doc_1", attempt: int = 1) -> Event:
    return Event(
        event_type="document.uploaded",
        document_id=document_id,
        payload={"storage_key": "documents/hr/doc_1/f.txt"},
        attempt=attempt,
    )


class TestEventEnvelope:
    def test_json_roundtrip_preserves_every_field(self):
        original = Event(
            event_type="document.uploaded",
            document_id="doc_abc",
            payload={"storage_key": "k", "department": "hr"},
            attempt=2,
            request_id="req123",
        )
        restored = Event.from_json(original.to_json())
        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.document_id == original.document_id
        assert restored.payload == original.payload
        assert restored.attempt == 2
        assert restored.request_id == "req123"

    def test_next_attempt_increments_but_keeps_event_id(self):
        """A stable event id is what lets a retry be traced to its original."""
        first = make_event(attempt=1)
        second = first.next_attempt()
        assert second.attempt == 2
        assert second.event_id == first.event_id
        assert second.document_id == first.document_id
        assert first.attempt == 1  # original untouched

    def test_next_attempt_copies_payload(self):
        first = make_event()
        second = first.next_attempt()
        second.payload["mutated"] = True
        assert "mutated" not in first.payload

    def test_malformed_json_raises_event_bus_error(self):
        with pytest.raises(EventBusError):
            Event.from_json("{not json")

    def test_non_object_payload_raises(self):
        with pytest.raises(EventBusError):
            Event.from_json("[1, 2, 3]")

    def test_missing_required_field_raises(self):
        with pytest.raises(EventBusError):
            Event.from_json('{"event_type": "x"}')


class TestPublishAndPoll:
    def test_published_event_is_delivered(self, bus):
        bus.publish(TOPIC, make_event("doc_a"))
        deliveries = bus.poll(TOPIC, GROUP)
        assert len(deliveries) == 1
        assert deliveries[0].event.document_id == "doc_a"

    def test_poll_on_empty_topic_returns_nothing(self, bus):
        assert bus.poll(TOPIC, GROUP) == []

    def test_events_are_delivered_in_publish_order(self, bus):
        for i in range(3):
            bus.publish(TOPIC, make_event(f"doc_{i}"))
        ids = [d.event.document_id for d in bus.poll(TOPIC, GROUP, max_messages=3)]
        assert ids == ["doc_0", "doc_1", "doc_2"]

    def test_topics_are_isolated(self, bus):
        bus.publish(TOPIC, make_event("doc_main"))
        bus.publish("document.uploaded.dlq", make_event("doc_dead"))
        assert len(bus.poll(TOPIC, GROUP)) == 1
        assert len(bus.poll("document.uploaded.dlq", GROUP)) == 1

    def test_publish_requires_a_topic(self, bus):
        with pytest.raises(EventBusError):
            bus.publish("", make_event())

    def test_delayed_event_is_not_immediately_visible(self, bus):
        bus.publish(TOPIC, make_event(), delay_s=60)
        assert bus.poll(TOPIC, GROUP) == []


class TestDeliverySettlement:
    def test_claimed_event_is_invisible_to_a_second_poll(self, bus):
        bus.publish(TOPIC, make_event())
        first = bus.poll(TOPIC, GROUP)
        assert len(first) == 1
        # Still in flight, so nobody else may claim it.
        assert bus.poll(TOPIC, GROUP) == []

    def test_ack_removes_the_event_permanently(self, bus):
        bus.publish(TOPIC, make_event())
        bus.poll(TOPIC, GROUP)[0].ack()
        assert bus.depth(TOPIC) == 0

    def test_nack_returns_the_event_for_redelivery(self, bus):
        bus.publish(TOPIC, make_event())
        bus.poll(TOPIC, GROUP)[0].nack()
        redelivered = bus.poll(TOPIC, GROUP)
        assert len(redelivered) == 1

    def test_nack_with_delay_defers_redelivery(self, bus):
        bus.publish(TOPIC, make_event())
        bus.poll(TOPIC, GROUP)[0].nack(delay_s=60)
        assert bus.poll(TOPIC, GROUP) == []

    def test_settling_twice_is_a_no_op(self, bus):
        bus.publish(TOPIC, make_event())
        delivery = bus.poll(TOPIC, GROUP)[0]
        delivery.ack()
        delivery.nack()  # must not resurrect the acked event
        assert bus.depth(TOPIC) == 0

    def test_expired_visibility_timeout_redelivers(self, bus, _visibility_timeout):
        """A worker that dies mid-processing must not lose the document."""
        _visibility_timeout.ingest_visibility_timeout_s = 0  # expire instantly
        bus.publish(TOPIC, make_event())
        claimed = bus.poll(TOPIC, GROUP)
        assert len(claimed) == 1
        # Never acked — simulates the worker being killed.
        reclaimed = bus.poll(TOPIC, GROUP)
        assert len(reclaimed) == 1
        assert reclaimed[0].event.document_id == claimed[0].event.document_id


class TestConcurrency:
    def test_two_pollers_never_claim_the_same_event(self, bus):
        for i in range(20):
            bus.publish(TOPIC, make_event(f"doc_{i}"))

        claimed: list[str] = []
        lock = threading.Lock()

        def drain():
            while True:
                batch = bus.poll(TOPIC, GROUP, max_messages=3)
                if not batch:
                    return
                with lock:
                    claimed.extend(d.event.document_id for d in batch)
                for d in batch:
                    d.ack()

        threads = [threading.Thread(target=drain) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert sorted(claimed) == sorted(f"doc_{i}" for i in range(20))
        assert len(claimed) == len(set(claimed))  # no double delivery


class TestIntrospection:
    def test_depth_counts_unsettled_events(self, bus):
        assert bus.depth(TOPIC) == 0
        bus.publish(TOPIC, make_event("a"))
        bus.publish(TOPIC, make_event("b"))
        assert bus.depth(TOPIC) == 2

    def test_peek_does_not_claim(self, bus):
        bus.publish(TOPIC, make_event("doc_peek"))
        peeked = bus.peek(TOPIC)
        assert [e.document_id for e in peeked] == ["doc_peek"]
        assert len(bus.poll(TOPIC, GROUP)) == 1  # still claimable

    def test_purge_clears_a_topic(self, bus):
        bus.publish(TOPIC, make_event())
        bus.publish("other", make_event())
        bus.purge(TOPIC)
        assert bus.depth(TOPIC) == 0
        assert bus.depth("other") == 1

    def test_malformed_row_is_discarded_not_redelivered_forever(self, bus):
        """A poison payload must not block the queue."""
        bus.publish(TOPIC, make_event())
        with bus._lock:
            bus._conn.execute("UPDATE events SET body = ?", ("{corrupt",))
            bus._conn.commit()
        assert bus.poll(TOPIC, GROUP) == []
        assert bus.depth(TOPIC) == 0


class TestDurability:
    def test_events_survive_reopening_the_database(self, tmp_path):
        path = tmp_path / "durable.db"
        first = SQLiteEventBus(db_path=path)
        first.publish(TOPIC, make_event("doc_durable"))
        first.close()

        second = SQLiteEventBus(db_path=path)
        try:
            delivered = second.poll(TOPIC, GROUP)
            assert len(delivered) == 1
            assert delivered[0].event.document_id == "doc_durable"
        finally:
            second.close()


class TestFactory:
    def teardown_method(self):
        reset_event_bus()

    def test_unknown_backend_raises(self):
        reset_event_bus()
        from src.events.bus import get_event_bus

        with patch("src.events.bus.settings") as s:
            s.event_bus = "rabbitmq"
            with pytest.raises(EventBusError, match="Unknown EVENT_BUS"):
                get_event_bus()

    def test_sqlite_backend_is_the_default(self, tmp_path):
        reset_event_bus()
        from src.events.bus import get_event_bus

        with (
            patch("src.events.bus.settings") as s,
            patch("src.events.sqlite_bus.settings") as s2,
        ):
            s.event_bus = "sqlite"
            s2.event_bus_path = str(tmp_path / "bus.db")
            assert isinstance(get_event_bus(), SQLiteEventBus)
