"""Integration tests against real Kafka, MinIO and ChromaDB.

Everything else in the suite fakes these backends, which proves the call
contracts but not that they work. These tests run the production code
paths — KafkaEventBus, S3ObjectStore, the Chroma server client and the
whole upload -> queue -> worker -> vector store pipeline — against real
services, and are the only place the shipped production configuration is
actually exercised.

Marked `integration` and excluded from the default run (see pytest.ini):

    docker compose -f docker-compose.test.yml up -d
    pytest -m integration
    docker compose -f docker-compose.test.yml down -v
"""
from __future__ import annotations

import time
import uuid

import pytest

pytestmark = pytest.mark.integration

KAFKA_BOOTSTRAP = "localhost:19092"
S3_ENDPOINT = "http://localhost:19000"
S3_BUCKET = "rag-test-documents"
CHROMA_HOST, CHROMA_PORT = "localhost", 18001


def unique(prefix: str) -> str:
    """Topics/keys must not collide across runs — Kafka retains messages."""
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------------

@pytest.fixture
def kafka_bus(kafka_settings):
    from src.events.kafka_bus import KafkaEventBus

    bus = KafkaEventBus(bootstrap_servers=KAFKA_BOOTSTRAP)
    yield bus
    bus.close()


@pytest.fixture
def kafka_settings(monkeypatch):
    from unittest.mock import patch

    with patch("src.events.kafka_bus.settings") as s:
        s.kafka_bootstrap_servers = KAFKA_BOOTSTRAP
        s.kafka_consumer_group = "integration-tests"
        yield s


def make_event(document_id="doc_int_1", attempt=1):
    from src.events.bus import Event

    return Event(
        event_type="document.uploaded",
        document_id=document_id,
        payload={"storage_key": "documents/hr/doc/x.md"},
        attempt=attempt,
    )


def poll_until(bus, topic, group, timeout_s=30, want=1):
    """Poll until *want* deliveries arrive or the deadline passes.

    A fresh consumer group has to join and be assigned partitions, so the
    first poll after subscribing routinely returns nothing.

    Transient EventBusErrors are retried rather than failed on, mirroring
    what IngestionWorker.run() already does — the client surfaces occasional
    socket-level errors ("Invalid file descriptor: -1" on Windows) while a
    connection is being established, and the production loop treats those
    as retryable rather than fatal.
    """
    from src.events.bus import EventBusError

    deadline = time.time() + timeout_s
    collected = []
    while time.time() < deadline and len(collected) < want:
        try:
            collected.extend(bus.poll(topic, group, max_messages=want, timeout_s=2.0))
        except EventBusError:
            time.sleep(0.5)
    return collected


class TestKafkaRoundTrip:
    def test_published_event_is_consumed(self, kafka_bus):
        topic, group = unique("t-roundtrip"), unique("g")
        kafka_bus.publish(topic, make_event("doc_rt"))

        got = poll_until(kafka_bus, topic, group)
        assert len(got) == 1
        assert got[0].event.document_id == "doc_rt"

    def test_payload_survives_the_wire(self, kafka_bus):
        topic, group = unique("t-payload"), unique("g")
        original = make_event("doc_payload", attempt=3)
        kafka_bus.publish(topic, original)

        got = poll_until(kafka_bus, topic, group)[0].event
        assert got.event_id == original.event_id
        assert got.attempt == 3
        assert got.payload == original.payload

    def test_ack_commits_so_a_new_consumer_does_not_redeliver(self, kafka_bus):
        """The offset contract the worker depends on, against a real broker."""
        topic, group = unique("t-ack"), unique("g")
        kafka_bus.publish(topic, make_event("doc_ack"))

        first = poll_until(kafka_bus, topic, group)
        assert len(first) == 1
        first[0].ack()
        kafka_bus.close()

        from src.events.kafka_bus import KafkaEventBus

        fresh = KafkaEventBus(bootstrap_servers=KAFKA_BOOTSTRAP)
        try:
            # Same group, new consumer: the committed offset must be honoured.
            assert poll_until(fresh, topic, group, timeout_s=10) == []
        finally:
            fresh.close()

    def test_uncommitted_event_is_redelivered_to_the_group(self, kafka_bus):
        """A worker that dies without acking must not lose the document."""
        topic, group = unique("t-nack"), unique("g")
        kafka_bus.publish(topic, make_event("doc_nack"))

        first = poll_until(kafka_bus, topic, group)
        assert len(first) == 1
        # Never acked — simulates the worker being killed mid-processing.
        kafka_bus.close()

        from src.events.kafka_bus import KafkaEventBus

        fresh = KafkaEventBus(bootstrap_servers=KAFKA_BOOTSTRAP)
        try:
            redelivered = poll_until(fresh, topic, group, timeout_s=30)
            assert len(redelivered) == 1
            assert redelivered[0].event.document_id == "doc_nack"
        finally:
            fresh.close()

    def test_ordering_is_preserved_per_document(self, kafka_bus):
        """Keying by document id keeps a retry behind its original."""
        topic, group = unique("t-order"), unique("g")
        for i in range(5):
            kafka_bus.publish(topic, make_event("doc_same", attempt=i + 1))

        got = poll_until(kafka_bus, topic, group, want=5, timeout_s=30)
        assert [d.event.attempt for d in got] == [1, 2, 3, 4, 5]

    def test_separate_groups_each_receive_the_event(self, kafka_bus):
        topic = unique("t-groups")
        kafka_bus.publish(topic, make_event("doc_fanout"))
        for group in (unique("g1"), unique("g2")):
            assert len(poll_until(kafka_bus, topic, group)) == 1

    def test_depth_reports_lag(self, kafka_bus, kafka_settings):
        topic = unique("t-depth")
        kafka_settings.kafka_consumer_group = unique("g-depth")
        for i in range(3):
            kafka_bus.publish(topic, make_event(f"doc_{i}"))
        assert kafka_bus.depth(topic) == 3

    def test_malformed_payload_does_not_block_the_partition(self, kafka_bus):
        """Poison message must be committed past, not retried forever."""
        topic, group = unique("t-poison"), unique("g")
        kafka_bus._producer.send(topic, key="k", value="{not json").get(timeout=10)
        kafka_bus.publish(topic, make_event("doc_after_poison"))

        got = poll_until(kafka_bus, topic, group, timeout_s=30)
        assert [d.event.document_id for d in got] == ["doc_after_poison"]


# ---------------------------------------------------------------------------
# S3 / MinIO
# ---------------------------------------------------------------------------

@pytest.fixture
def s3_store():
    from unittest.mock import patch

    from src.storage.object_store import S3ObjectStore

    with patch("src.storage.object_store.settings") as s:
        s.s3_bucket = S3_BUCKET
        s.s3_region = "us-east-1"
        s.s3_endpoint_url = S3_ENDPOINT
        s.s3_prefix = "documents"
        import os

        os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
        yield S3ObjectStore()


class TestS3ObjectStore:
    def test_put_then_get_roundtrip(self, s3_store):
        key = f"documents/hr/{unique('doc')}/handbook.md"
        s3_store.put(key, b"# Handbook\nPTO is 20 days.\n", "text/markdown")
        assert s3_store.get(key) == b"# Handbook\nPTO is 20 days.\n"

    def test_binary_content_is_not_corrupted(self, s3_store):
        key = f"documents/hr/{unique('doc')}/report.pdf"
        payload = bytes(range(256)) * 40
        s3_store.put(key, payload, "application/pdf")
        assert s3_store.get(key) == payload

    def test_exists_reflects_reality(self, s3_store):
        key = f"documents/hr/{unique('doc')}/x.md"
        assert s3_store.exists(key) is False
        s3_store.put(key, b"x")
        assert s3_store.exists(key) is True

    def test_missing_key_raises_not_found(self, s3_store):
        """Mapped to a permanent failure by the worker — must be distinct."""
        from src.storage.object_store import ObjectNotFoundError

        with pytest.raises(ObjectNotFoundError):
            s3_store.get(f"documents/none/{unique('missing')}/x.md")

    def test_delete_then_absent(self, s3_store):
        key = f"documents/hr/{unique('doc')}/gone.md"
        s3_store.put(key, b"x")
        s3_store.delete(key)
        assert s3_store.exists(key) is False

    def test_overwrite_replaces_content(self, s3_store):
        key = f"documents/hr/{unique('doc')}/v.md"
        s3_store.put(key, b"first")
        s3_store.put(key, b"second")
        assert s3_store.get(key) == b"second"

    def test_uri_is_s3_scheme(self, s3_store):
        assert s3_store.uri("a/b.md") == f"s3://{S3_BUCKET}/a/b.md"

    def test_bad_bucket_surfaces_as_store_error(self):
        from unittest.mock import patch

        from src.storage.object_store import ObjectStoreError, S3ObjectStore

        with patch("src.storage.object_store.settings") as s:
            s.s3_bucket = "no-such-bucket-here"
            s.s3_region = "us-east-1"
            s.s3_endpoint_url = S3_ENDPOINT
            store = S3ObjectStore()
            with pytest.raises(ObjectStoreError):
                store.put("k.md", b"x")


# ---------------------------------------------------------------------------
# Full pipeline on the production backends
# ---------------------------------------------------------------------------

class TestPipelineOnProductionBackends:
    def test_upload_through_kafka_and_s3_reaches_the_vector_store(self, tmp_path):
        """The whole architecture on the backends production actually uses."""
        from unittest.mock import patch

        from src.events.kafka_bus import KafkaEventBus
        from src.ingestion.pipeline import process_document
        from src.ingestion.registry import DocumentRegistry, compute_checksum
        from src.ingestion.worker import OUTCOME_PROCESSED, IngestionWorker
        from src.storage.object_store import S3ObjectStore

        topic, dlq, group = unique("t-e2e"), unique("t-e2e-dlq"), unique("g-e2e")
        content = b"# Travel Policy\nThe international meal allowance is $110.\n"

        with (
            patch("src.storage.object_store.settings") as ss,
            patch("src.events.kafka_bus.settings") as ks,
            patch("src.ingestion.registry.settings") as rs,
            patch("src.ingestion.worker.settings") as ws,
        ):
            ss.s3_bucket, ss.s3_region = S3_BUCKET, "us-east-1"
            ss.s3_endpoint_url, ss.s3_prefix = S3_ENDPOINT, "documents"
            ks.kafka_bootstrap_servers = KAFKA_BOOTSTRAP
            ks.kafka_consumer_group = group
            rs.ingest_visibility_timeout_s = 300
            ws.ingest_max_attempts, ws.ingest_retry_backoff_s = 3, 0.0
            ws.worker_batch_size, ws.worker_poll_interval_s = 1, 0.1

            store = S3ObjectStore()
            bus = KafkaEventBus(bootstrap_servers=KAFKA_BOOTSTRAP)
            registry = DocumentRegistry(db_path=tmp_path / "documents.db")
            try:
                # --- API side: register, store to S3, publish to Kafka ---
                record, created = registry.register_upload(
                    checksum=compute_checksum(content), filename="travel.md",
                    department="finance", size_bytes=len(content),
                )
                assert created
                key = f"documents/finance/{record.document_id}/travel.md"
                store.put(key, content, "text/markdown")
                registry.attach_storage(record.document_id, key, store.uri(key))

                from src.events.bus import Event

                bus.publish(topic, Event(
                    event_type="document.uploaded",
                    document_id=record.document_id,
                    payload={"storage_key": key},
                ))

                # --- Worker side: consume, download from S3, index ---
                worker = IngestionWorker(
                    bus=bus, registry=registry, object_store=store,
                    topic=topic, dlq_topic=dlq, group=group, max_attempts=3,
                )
                with patch("src.vectorstore.chroma_store.add_chunks", return_value=2) as add:
                    outcomes = []
                    deadline = time.time() + 45
                    while not outcomes and time.time() < deadline:
                        outcomes = worker.poll_once(timeout_s=2.0)

                assert outcomes == [OUTCOME_PROCESSED]
                assert registry.get(record.document_id).status == "PROCESSED"

                # The bytes really made the round trip through S3 and were parsed.
                chunks = add.call_args.args[0]
                assert "international meal allowance is $110" in chunks[0].page_content
                assert chunks[0].metadata["source"] == "uploads/finance/travel.md"
            finally:
                bus.close()
                registry.close()

    def test_chroma_server_write_is_visible_to_a_second_client(self):
        """The cross-process visibility fix, against a real Chroma server."""
        import chromadb

        collection = unique("coll").replace("-", "_")
        writer = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        col = writer.get_or_create_collection(collection)
        col.upsert(
            ids=["a1"], documents=["badge replacement costs $50"],
            embeddings=[[0.25] * 8],
        )

        # A completely separate client, as the API is to the worker.
        reader = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        found = reader.get_collection(collection).get(ids=["a1"])
        assert found["documents"] == ["badge replacement costs $50"]
        writer.delete_collection(collection)
