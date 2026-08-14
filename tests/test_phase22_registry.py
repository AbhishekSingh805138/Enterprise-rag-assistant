"""Phase 22 — document registry (idempotency + lifecycle).

The registry answers both decision points in the ingestion architecture,
so these tests focus on the two races that would break it: two uploads of
the same file, and two workers claiming the same document.
"""
from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from src.ingestion.registry import (
    STATUS_DEAD_LETTER,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSED,
    STATUS_PROCESSING,
    DocumentRegistry,
    compute_checksum,
    derive_document_id,
    get_registry,
    reset_registry,
)


@pytest.fixture
def registry(tmp_path):
    r = DocumentRegistry(db_path=tmp_path / "documents.db")
    yield r
    r.close()


@pytest.fixture(autouse=True)
def _visibility_timeout():
    with patch("src.ingestion.registry.settings") as s:
        s.ingest_visibility_timeout_s = 300
        yield s


def register(registry, *, content=b"hello", filename="f.txt", department="hr"):
    return registry.register_upload(
        checksum=compute_checksum(content),
        filename=filename,
        department=department,
        content_type="text/plain",
        size_bytes=len(content),
    )


class TestDocumentIdentity:
    def test_checksum_is_content_addressed(self):
        assert compute_checksum(b"abc") == compute_checksum(b"abc")
        assert compute_checksum(b"abc") != compute_checksum(b"abd")

    def test_same_content_and_department_yields_the_same_id(self):
        checksum = compute_checksum(b"payload")
        assert derive_document_id(checksum, "hr") == derive_document_id(checksum, "hr")

    def test_department_is_part_of_identity(self):
        """Department drives access level, so it must not be collapsed."""
        checksum = compute_checksum(b"payload")
        assert derive_document_id(checksum, "hr") != derive_document_id(checksum, "legal")

    def test_department_matching_is_case_insensitive(self):
        checksum = compute_checksum(b"payload")
        assert derive_document_id(checksum, "HR") == derive_document_id(checksum, "hr")

    def test_id_has_a_readable_prefix(self):
        assert derive_document_id(compute_checksum(b"x"), "hr").startswith("doc_")


class TestRegisterUpload:
    def test_first_upload_is_created(self, registry):
        record, created = register(registry)
        assert created is True
        assert record.status == STATUS_PENDING
        assert record.attempts == 0

    def test_identical_reupload_is_not_created(self, registry):
        first, _ = register(registry)
        second, created = register(registry)
        assert created is False
        assert second.document_id == first.document_id

    def test_same_bytes_in_another_department_is_a_new_document(self, registry):
        first, _ = register(registry, department="hr")
        second, created = register(registry, department="legal")
        assert created is True
        assert second.document_id != first.document_id

    def test_different_content_is_a_new_document(self, registry):
        first, _ = register(registry, content=b"one")
        second, created = register(registry, content=b"two")
        assert created is True
        assert second.document_id != first.document_id

    def test_metadata_is_persisted(self, registry):
        record, _ = register(registry, content=b"12345", filename="report.pdf")
        stored = registry.get(record.document_id)
        assert stored.filename == "report.pdf"
        assert stored.size_bytes == 5
        assert stored.department == "hr"

    def test_concurrent_registration_creates_exactly_one(self, registry):
        """Two simultaneous uploads of one file must not both be 'created'."""
        results: list[bool] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def upload():
            barrier.wait()
            _, created = register(registry)
            with lock:
                results.append(created)

        threads = [threading.Thread(target=upload) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert sum(results) == 1
        assert len(results) == 8


class TestClaimForProcessing:
    def test_pending_document_can_be_claimed(self, registry):
        record, _ = register(registry)
        assert registry.claim_for_processing(record.document_id) is True
        assert registry.get(record.document_id).status == STATUS_PROCESSING

    def test_claim_increments_attempts(self, registry):
        record, _ = register(registry)
        registry.claim_for_processing(record.document_id)
        assert registry.get(record.document_id).attempts == 1

    def test_processed_document_cannot_be_reclaimed(self, registry):
        """This is the 'Already Indexed?' yes-branch."""
        record, _ = register(registry)
        registry.claim_for_processing(record.document_id)
        registry.mark_processed(record.document_id, chunks_indexed=7)
        assert registry.claim_for_processing(record.document_id) is False

    def test_failed_document_can_be_reclaimed(self, registry):
        record, _ = register(registry)
        registry.claim_for_processing(record.document_id)
        registry.mark_failed(record.document_id, "boom")
        assert registry.claim_for_processing(record.document_id) is True
        assert registry.get(record.document_id).attempts == 2

    def test_dead_lettered_document_cannot_be_reclaimed(self, registry):
        record, _ = register(registry)
        registry.claim_for_processing(record.document_id)
        registry.mark_dead_letter(record.document_id, "gave up")
        assert registry.claim_for_processing(record.document_id) is False

    def test_in_flight_document_cannot_be_double_claimed(self, registry):
        record, _ = register(registry)
        assert registry.claim_for_processing(record.document_id) is True
        assert registry.claim_for_processing(record.document_id) is False

    def test_stale_processing_document_is_reclaimable(self, registry):
        """A worker killed mid-document must not strand it in PROCESSING."""
        record, _ = register(registry)
        # Claim it with a backdated timestamp: the document looks like it
        # was picked up long ago by a worker that never came back.
        with patch(
            "src.ingestion.registry._utc_now",
            return_value="2000-01-01T00:00:00+00:00",
        ):
            assert registry.claim_for_processing(record.document_id) is True

        # Refused while the claim is still within its visibility window...
        century = 86400 * 365 * 100
        assert registry.claim_for_processing(record.document_id, stale_after_s=century) is False
        # ...and picked back up once that window has lapsed.
        assert registry.claim_for_processing(record.document_id, stale_after_s=1) is True

    def test_unknown_document_cannot_be_claimed(self, registry):
        assert registry.claim_for_processing("doc_does_not_exist") is False

    def test_only_one_of_many_workers_wins_the_claim(self, registry):
        record, _ = register(registry)
        wins: list[bool] = []
        lock = threading.Lock()
        barrier = threading.Barrier(8)

        def claim():
            barrier.wait()
            won = registry.claim_for_processing(record.document_id)
            with lock:
                wins.append(won)

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert sum(wins) == 1


class TestLifecycleTransitions:
    def test_mark_processed_records_chunk_count(self, registry):
        record, _ = register(registry)
        registry.mark_processed(record.document_id, chunks_indexed=12)
        stored = registry.get(record.document_id)
        assert stored.status == STATUS_PROCESSED
        assert stored.chunks_indexed == 12
        assert stored.is_terminal is True

    def test_mark_processed_clears_a_previous_error(self, registry):
        record, _ = register(registry)
        registry.mark_failed(record.document_id, "transient blip")
        registry.mark_processed(record.document_id, chunks_indexed=3)
        assert registry.get(record.document_id).error == ""

    def test_mark_failed_records_the_error(self, registry):
        record, _ = register(registry)
        registry.mark_failed(record.document_id, "RateLimitError: slow down")
        stored = registry.get(record.document_id)
        assert stored.status == STATUS_FAILED
        assert "RateLimitError" in stored.error
        assert stored.is_terminal is False

    def test_long_errors_are_truncated(self, registry):
        record, _ = register(registry)
        registry.mark_failed(record.document_id, "x" * 5000)
        assert len(registry.get(record.document_id).error) <= 2000

    def test_reset_for_retry_revives_a_dead_letter(self, registry):
        record, _ = register(registry)
        registry.claim_for_processing(record.document_id)
        registry.mark_dead_letter(record.document_id, "gave up")
        assert registry.reset_for_retry(record.document_id) is True
        stored = registry.get(record.document_id)
        assert stored.status == STATUS_PENDING
        assert stored.attempts == 0
        assert stored.error == ""

    def test_reset_for_retry_leaves_processed_documents_alone(self, registry):
        record, _ = register(registry)
        registry.mark_processed(record.document_id, chunks_indexed=1)
        assert registry.reset_for_retry(record.document_id) is False
        assert registry.get(record.document_id).status == STATUS_PROCESSED

    def test_attach_storage_records_location(self, registry):
        record, _ = register(registry)
        registry.attach_storage(record.document_id, "documents/hr/x/f.txt", "s3://b/k")
        stored = registry.get(record.document_id)
        assert stored.storage_key == "documents/hr/x/f.txt"
        assert stored.storage_uri == "s3://b/k"


class TestQueries:
    def test_get_unknown_returns_none(self, registry):
        assert registry.get("doc_nope") is None

    def test_find_by_content(self, registry):
        record, _ = register(registry, content=b"findable")
        found = registry.find_by_content(compute_checksum(b"findable"), "hr")
        assert found.document_id == record.document_id

    def test_find_by_content_is_department_scoped(self, registry):
        register(registry, content=b"findable", department="hr")
        assert registry.find_by_content(compute_checksum(b"findable"), "legal") is None

    def test_list_filters_by_status(self, registry):
        a, _ = register(registry, content=b"a")
        register(registry, content=b"b")
        registry.mark_processed(a.document_id, 1)
        processed = registry.list_documents(status=STATUS_PROCESSED)
        assert [r.document_id for r in processed] == [a.document_id]

    def test_list_filters_by_department(self, registry):
        register(registry, content=b"a", department="hr")
        register(registry, content=b"b", department="legal")
        assert len(registry.list_documents(department="legal")) == 1

    def test_list_respects_limit(self, registry):
        for i in range(5):
            register(registry, content=f"doc{i}".encode())
        assert len(registry.list_documents(limit=2)) == 2

    def test_stats_zero_fills_every_status(self, registry):
        stats = registry.stats()
        assert stats["TOTAL"] == 0
        assert stats[STATUS_PROCESSED] == 0
        assert stats[STATUS_DEAD_LETTER] == 0

    def test_stats_counts_by_status(self, registry):
        a, _ = register(registry, content=b"a")
        b, _ = register(registry, content=b"b")
        register(registry, content=b"c")
        registry.mark_processed(a.document_id, 1)
        registry.mark_dead_letter(b.document_id, "err")
        stats = registry.stats()
        assert stats["TOTAL"] == 3
        assert stats[STATUS_PROCESSED] == 1
        assert stats[STATUS_DEAD_LETTER] == 1
        assert stats[STATUS_PENDING] == 1


class TestDurability:
    def test_registry_survives_reopening(self, tmp_path):
        path = tmp_path / "documents.db"
        first = DocumentRegistry(db_path=path)
        record, _ = first.register_upload(
            checksum=compute_checksum(b"persist"), filename="f.txt", department="hr"
        )
        first.close()

        second = DocumentRegistry(db_path=path)
        try:
            assert second.get(record.document_id) is not None
        finally:
            second.close()


class TestSingleton:
    def teardown_method(self):
        reset_registry()

    def test_get_registry_is_a_singleton(self, tmp_path):
        reset_registry()
        with patch("src.ingestion.registry.settings") as s:
            s.document_registry_path = str(tmp_path / "reg.db")
            assert get_registry() is get_registry()
