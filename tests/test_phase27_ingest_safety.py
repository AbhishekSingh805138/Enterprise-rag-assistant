"""Phase 27 — P1-11: bounds on parsing untrusted files, and PII at ingest.

The worker parses PDFs and DOCX files a stranger uploaded, in-process,
with libraries that have a long history of memory-safety bugs, and with
nothing bounding what one document may consume. Separately, document text
was embedded verbatim: the answer-time filter redacts the response, but
the vectors — and every backup of them — held the original.
"""
from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.ingestion.safety import (
    DocumentTooLarge,
    ParseTimeout,
    apply_pii_policy,
    enforce_extracted_size,
    parse_with_timeout,
)
from src.security.pii import ANSWER_CATEGORIES, PATTERNS, redact, scan

SSN = "123-45-6789"
CARD = "4111 1111 1111 1111"
PHONE = "+1 (555) 867-5309"
EMAIL = "payroll@example.com"


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

class TestPiiPatterns:
    @pytest.mark.parametrize(
        "category,sample",
        [("ssn", SSN), ("credit_card", CARD), ("phone", PHONE), ("email", EMAIL)],
    )
    def test_each_category_is_detected(self, category, sample):
        assert category in scan(f"contact {sample} today")

    def test_clean_text_finds_nothing(self):
        assert scan("The remote work policy allows three days per week.") == {}

    def test_scan_does_not_modify_the_text(self):
        text = f"call {PHONE}"
        scan(text)
        assert text == f"call {PHONE}"

    def test_redact_replaces_with_a_placeholder(self):
        cleaned, hits = redact(f"SSN is {SSN}")
        assert SSN not in cleaned
        assert "[SSN_REDACTED]" in cleaned
        assert hits == 1

    def test_multiple_categories_are_all_redacted(self):
        cleaned, hits = redact(f"{SSN} / {CARD} / {EMAIL}")
        assert SSN not in cleaned and CARD not in cleaned and EMAIL not in cleaned
        assert hits >= 3

    def test_counts_are_reported_per_category(self):
        found = scan(f"{EMAIL} and other@example.org")
        assert found["email"] == 2

    def test_a_card_is_not_reported_only_as_a_phone_number(self):
        """A 16-digit group with separators satisfies both patterns."""
        assert "credit_card" in scan(CARD)


class TestAnswerVsIngestCategories:
    def test_the_answer_filter_leaves_email_alone(self):
        """"Contact facilities@company.com" is the useful part of the answer."""
        assert "email" not in ANSWER_CATEGORIES
        cleaned, _ = redact(f"Email {EMAIL} for access.", ANSWER_CATEGORIES)
        assert EMAIL in cleaned

    def test_the_answer_filter_still_redacts_the_dangerous_ones(self):
        cleaned, _ = redact(f"{SSN} {CARD} {PHONE}", ANSWER_CATEGORIES)
        assert SSN not in cleaned and CARD not in cleaned and PHONE not in cleaned

    def test_ingest_redaction_covers_every_category(self):
        """Nobody reads a vector, so there is no usefulness to trade away."""
        cleaned, _ = redact(f"{SSN} {CARD} {PHONE} {EMAIL}")
        for sample in (SSN, CARD, PHONE, EMAIL):
            assert sample not in cleaned

    def test_answer_categories_are_real_categories(self):
        assert {p.name for p in PATTERNS} >= ANSWER_CATEGORIES

    def test_the_output_filter_behaviour_is_unchanged(self):
        """The refactor must not alter what answers look like."""
        from src.security.output_filter import filter_output

        with patch("src.security.output_filter.settings") as s:
            s.pii_detection_enabled = True
            out = filter_output(f"SSN {SSN}, card {CARD}, phone {PHONE}, mail {EMAIL}")
        assert "[SSN_REDACTED]" in out
        assert "[CC_REDACTED]" in out
        assert EMAIL in out  # unchanged: still passes through

    def test_the_output_filter_respects_its_flag(self):
        from src.security.output_filter import filter_output

        with patch("src.security.output_filter.settings") as s:
            s.pii_detection_enabled = False
            assert filter_output(f"SSN {SSN}") == f"SSN {SSN}"


# ---------------------------------------------------------------------------
# PII policy at ingest
# ---------------------------------------------------------------------------

class TestIngestPiiPolicy:
    def _docs(self):
        return [
            Document(page_content=f"Employee SSN {SSN}", metadata={}),
            Document(page_content="Remote work is allowed three days a week.", metadata={}),
        ]

    def _mode(self, mode):
        s = MagicMock()
        s.ingest_pii_mode = mode
        return patch("src.ingestion.safety.settings", s)

    def test_off_changes_nothing(self):
        """The default must leave indexing behaviour exactly as it was."""
        docs = self._docs()
        with self._mode("off"):
            assert apply_pii_policy(docs) == 0
        assert SSN in docs[0].page_content

    def test_warn_reports_without_changing_the_text(self, caplog):
        """Knowing the corpus holds PII is worth having on its own."""
        docs = self._docs()
        with self._mode("warn"), caplog.at_level(logging.WARNING):
            assert apply_pii_policy(docs, document_id="doc1") == 1
        assert SSN in docs[0].page_content  # indexed as-is
        assert "indexed unredacted" in caplog.text
        assert "ssn=1" in caplog.text

    def test_redact_strips_before_embedding(self):
        docs = self._docs()
        with self._mode("redact"):
            apply_pii_policy(docs, document_id="doc1")
        assert SSN not in docs[0].page_content
        assert "[SSN_REDACTED]" in docs[0].page_content

    def test_redaction_leaves_clean_documents_alone(self):
        docs = self._docs()
        before = docs[1].page_content
        with self._mode("redact"):
            apply_pii_policy(docs)
        assert docs[1].page_content == before

    def test_a_clean_corpus_logs_nothing(self, caplog):
        docs = [Document(page_content="Nothing sensitive here.", metadata={})]
        with self._mode("warn"), caplog.at_level(logging.WARNING):
            assert apply_pii_policy(docs) == 0
        assert caplog.text == ""

    def test_the_mode_is_case_insensitive(self):
        docs = self._docs()
        with self._mode("  REDACT  "):
            apply_pii_policy(docs)
        assert SSN not in docs[0].page_content

    def test_an_invalid_mode_is_rejected_at_startup(self):
        import config

        with pytest.raises(ValueError, match="INGEST_PII_MODE"):
            config.Settings(openai_api_key="sk-test", ingest_pii_mode="strip").validate()

    def test_the_shipped_default_is_off(self):
        """Redaction changes what is retrievable; that is the owner's call."""
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(r'ingest_pii_mode:.*"INGEST_PII_MODE",\s*"off"', source)


# ---------------------------------------------------------------------------
# Resource bounds
# ---------------------------------------------------------------------------

class TestExtractedSizeCeiling:
    def _limit(self, chars):
        s = MagicMock()
        s.ingest_max_extracted_chars = chars
        return patch("src.ingestion.safety.settings", s)

    def test_a_normal_document_passes(self):
        with self._limit(1000):
            enforce_extracted_size([Document(page_content="x" * 500, metadata={})])

    def test_an_expanding_document_is_rejected(self):
        """A 2 MB upload that becomes gigabytes of text is the classic bomb."""
        with self._limit(1000):
            with pytest.raises(DocumentTooLarge, match="decompression bomb"):
                enforce_extracted_size([Document(page_content="x" * 5000, metadata={})])

    def test_the_limit_applies_to_the_whole_document_not_each_page(self):
        """Per-page checks miss a file with a million small pages."""
        pages = [Document(page_content="x" * 100, metadata={}) for _ in range(20)]
        with self._limit(1000):
            with pytest.raises(DocumentTooLarge):
                enforce_extracted_size(pages)

    def test_a_zero_limit_disables_the_ceiling(self):
        with self._limit(0):
            enforce_extracted_size([Document(page_content="x" * 10_000_000, metadata={})])

    def test_empty_content_does_not_crash(self):
        with self._limit(1000):
            enforce_extracted_size([Document(page_content="", metadata={})])


class TestParseTimeout:
    def _timeout(self, seconds):
        s = MagicMock()
        s.ingest_parse_timeout_s = seconds
        return patch("src.ingestion.safety.settings", s)

    def test_a_fast_parse_returns_its_result(self):
        with self._timeout(5):
            assert parse_with_timeout(lambda: "parsed") == "parsed"

    def test_arguments_are_forwarded(self):
        with self._timeout(5):
            assert parse_with_timeout(lambda a, b=0: a + b, 1, b=2) == 3

    def test_a_hanging_parse_is_abandoned(self):
        """A pathological file must not stall the queue behind it."""
        with self._timeout(0.2):
            with pytest.raises(ParseTimeout, match="abandoned"):
                parse_with_timeout(lambda: time.sleep(5))

    def test_the_worker_does_not_wait_for_the_stuck_thread(self):
        """The deadline's real guarantee: the queue keeps moving."""
        with self._timeout(0.2):
            started = time.monotonic()
            with pytest.raises(ParseTimeout):
                parse_with_timeout(lambda: time.sleep(10))
            assert time.monotonic() - started < 3

    def test_a_zero_timeout_disables_the_deadline(self):
        with self._timeout(0):
            assert parse_with_timeout(lambda: "ok") == "ok"

    def test_a_parser_error_propagates_unchanged(self):
        """The deadline must not mask a real parse failure."""
        with self._timeout(5):
            with pytest.raises(ValueError, match="corrupt"):
                parse_with_timeout(lambda: (_ for _ in ()).throw(ValueError("corrupt")))


# ---------------------------------------------------------------------------
# Wiring into the pipeline
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def _record(self):
        record = MagicMock()
        record.storage_key = "documents/general/policy.txt"
        record.filename = "policy.txt"
        record.department = "general"
        record.document_id = "doc1"
        record.checksum = "abc"
        return record

    def _store(self):
        store = MagicMock()
        store.get.return_value = b"hello world"
        return store

    def test_a_timeout_becomes_a_permanent_failure(self):
        """Retrying a pathological file spends the budget on the same result."""
        from src.ingestion.pipeline import DocumentParseError, process_document

        with patch(
            "src.ingestion.pipeline.parse_with_timeout", side_effect=ParseTimeout("too slow")
        ):
            with pytest.raises(DocumentParseError, match="too slow"):
                process_document(self._record(), object_store=self._store())

    def test_an_oversized_document_becomes_a_permanent_failure(self):
        from src.ingestion.pipeline import DocumentParseError, process_document

        with (
            patch("src.ingestion.pipeline.parse_with_timeout", return_value=[MagicMock()]),
            patch(
                "src.ingestion.pipeline.enforce_extracted_size",
                side_effect=DocumentTooLarge("too big"),
            ),
        ):
            with pytest.raises(DocumentParseError, match="too big"):
                process_document(self._record(), object_store=self._store())

    def test_a_permanent_failure_is_dead_lettered_not_retried(self):
        """DocumentParseError is in the worker's permanent-error set."""
        from src.ingestion.pipeline import DocumentParseError
        from src.ingestion.worker import _PERMANENT_ERRORS

        assert DocumentParseError in _PERMANENT_ERRORS

    def test_pii_is_screened_before_anything_is_embedded(self):
        """Order matters: after embedding, the vectors already hold it."""
        import inspect

        import src.ingestion.pipeline as pipeline

        source = inspect.getsource(pipeline.process_document)
        assert source.index("apply_pii_policy(") < source.index("add_chunks(")

    def test_the_ceiling_is_checked_before_chunking(self):
        import inspect

        import src.ingestion.pipeline as pipeline

        source = inspect.getsource(pipeline.process_document)
        assert source.index("enforce_extracted_size(") < source.index("chunk_documents(")


class TestContainerHardening:
    """Code bounds what a document consumes; the container bounds the blast."""

    @pytest.fixture(scope="class")
    def compose(self):
        from pathlib import Path

        import yaml

        path = Path(__file__).resolve().parent.parent / "docker-compose.prod.yml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("service", ["worker", "api"])
    def test_the_parsing_containers_are_read_only(self, compose, service):
        assert compose["services"][service]["read_only"] is True

    @pytest.mark.parametrize("service", ["worker", "api"])
    def test_capabilities_are_dropped(self, compose, service):
        assert compose["services"][service]["cap_drop"] == ["ALL"]

    @pytest.mark.parametrize("service", ["worker", "api"])
    def test_privilege_escalation_is_blocked(self, compose, service):
        assert "no-new-privileges:true" in compose["services"][service]["security_opt"]

    @pytest.mark.parametrize("service", ["worker", "api"])
    def test_the_temp_directory_forbids_execution(self, compose, service):
        """Parsing materialises the upload there; it must not be runnable."""
        tmpfs = " ".join(compose["services"][service]["tmpfs"])
        assert "/tmp" in tmpfs
        assert "noexec" in tmpfs
        assert "nosuid" in tmpfs
