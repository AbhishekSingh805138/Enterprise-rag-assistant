"""Phase 23 — ChromaDB wrapper (src/vectorstore/chroma_store.py).

The behaviour worth pinning down is deduplication: content-hash IDs are
what stop a re-ingest from doubling the corpus, and they are also what
makes the same file indexed through the sync and async upload paths land
as one document rather than two.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import src.vectorstore.chroma_store as cs


@pytest.fixture(autouse=True)
def _reset_singletons():
    cs.reset_store()
    yield
    cs.reset_store()


@pytest.fixture
def store_settings():
    with patch("src.vectorstore.chroma_store.settings") as s:
        s.chroma_collection = "test_docs"
        s.chroma_dir = "./test_chroma"
        s.embedding_model = "text-embedding-3-small"
        s.openai_api_key = "sk-test"
        s.chroma_refresh_interval = 300
        s.top_k = 4
        s.document_ttl_days = 0
        yield s


@pytest.fixture
def fake_chroma(store_settings):
    """Patch the Chroma class and hand back the instance it returns."""
    instance = MagicMock()
    instance.get.return_value = {"ids": []}
    instance._collection.count.return_value = 7
    with (
        patch("src.vectorstore.chroma_store.Chroma", return_value=instance) as klass,
        patch("src.vectorstore.chroma_store.OpenAIEmbeddings") as emb,
    ):
        yield instance, klass, emb


def chunk(text="hello", **meta):
    base = {"filename": "handbook.md", "department": "hr", "start_index": 0}
    base.update(meta)
    return Document(page_content=text, metadata=base)


# ---------------------------------------------------------------------------
# Singletons & lifecycle
# ---------------------------------------------------------------------------

class TestVectorstoreSingleton:
    def test_store_is_opened_once(self, fake_chroma):
        _instance, klass, _emb = fake_chroma
        assert cs.get_vectorstore() is cs.get_vectorstore()
        klass.assert_called_once()

    def test_store_is_opened_with_the_configured_collection(self, fake_chroma):
        _instance, klass, _emb = fake_chroma
        cs.get_vectorstore()
        assert klass.call_args.kwargs["collection_name"] == "test_docs"
        assert klass.call_args.kwargs["persist_directory"] == "./test_chroma"

    def test_embeddings_are_created_once(self, fake_chroma):
        _instance, _klass, emb = fake_chroma
        cs.get_vectorstore()
        cs.get_vectorstore()
        emb.assert_called_once()

    def test_refresh_reopens_on_next_access(self, fake_chroma):
        _instance, klass, _emb = fake_chroma
        cs.get_vectorstore()
        cs.refresh_store()
        cs.get_vectorstore()
        assert klass.call_count == 2

    def test_stale_connection_is_refreshed_after_the_interval(self, fake_chroma, store_settings):
        _instance, klass, _emb = fake_chroma
        store_settings.chroma_refresh_interval = 0  # everything is immediately stale
        cs.get_vectorstore()
        cs.get_vectorstore()
        assert klass.call_count == 2

    def test_reset_clears_embeddings_too(self, fake_chroma):
        _instance, _klass, emb = fake_chroma
        cs.get_vectorstore()
        cs.reset_store()
        cs.get_vectorstore()
        assert emb.call_count == 2


# ---------------------------------------------------------------------------
# Content-hash identity
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_same_content_and_origin_hash_identically(self):
        a = cs._content_hash("text", {"filename": "f.md", "department": "hr"})
        b = cs._content_hash("text", {"filename": "f.md", "department": "hr"})
        assert a == b

    def test_different_content_hashes_differently(self):
        meta = {"filename": "f.md", "department": "hr"}
        assert cs._content_hash("one", meta) != cs._content_hash("two", meta)

    def test_department_is_part_of_identity(self):
        a = cs._content_hash("text", {"filename": "f.md", "department": "hr"})
        b = cs._content_hash("text", {"filename": "f.md", "department": "legal"})
        assert a != b

    def test_start_index_separates_chunks_of_one_file(self):
        a = cs._content_hash("text", {"filename": "f.md", "start_index": 0})
        b = cs._content_hash("text", {"filename": "f.md", "start_index": 500})
        assert a != b

    def test_falls_back_to_source_when_filename_is_absent(self):
        """CLI ingestion has no filename metadata; identity must still be stable."""
        a = cs._content_hash("text", {"source": "/data/hr/f.md"})
        b = cs._content_hash("text", {"source": "/data/hr/f.md"})
        c = cs._content_hash("text", {"source": "/data/legal/f.md"})
        assert a == b and a != c

    def test_hash_is_hex_sha256(self):
        h = cs._content_hash("text", {"filename": "f.md"})
        assert len(h) == 64
        int(h, 16)  # raises if not hex


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    def test_empty_input_is_a_no_op(self, fake_chroma):
        assert cs.add_chunks([]) == 0

    def test_new_chunks_are_added(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {"ids": []}
        assert cs.add_chunks([chunk("a"), chunk("b")]) == 2
        instance.add_documents.assert_called_once()

    def test_documents_are_added_with_deterministic_ids(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {"ids": []}
        cs.add_chunks([chunk("a")])
        ids = instance.add_documents.call_args.kwargs["ids"]
        assert ids == [cs._content_hash("a", chunk("a").metadata)]

    def test_existing_chunks_are_skipped(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        existing = cs._content_hash("a", chunk("a").metadata)
        instance.get.return_value = {"ids": [existing]}
        assert cs.add_chunks([chunk("a")]) == 0
        instance.add_documents.assert_not_called()

    def test_partial_overlap_adds_only_the_new_ones(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        existing = cs._content_hash("a", chunk("a").metadata)
        instance.get.return_value = {"ids": [existing]}
        assert cs.add_chunks([chunk("a"), chunk("b")]) == 1

    def test_dedup_check_failure_still_attempts_the_add(self, fake_chroma):
        """A failed existence probe must not silently drop the ingest."""
        instance, _klass, _emb = fake_chroma
        instance.get.side_effect = RuntimeError("collection locked")
        assert cs.add_chunks([chunk("a")]) == 1
        instance.add_documents.assert_called_once()

    def test_bm25_cache_is_invalidated_when_the_corpus_changes(self, fake_chroma):
        """Sparse retrieval reads a cached corpus; a stale one misses new docs."""
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {"ids": []}
        with patch("src.retrieval.hybrid.reset_bm25_cache") as reset:
            cs.add_chunks([chunk("a")])
        reset.assert_called_once()

    def test_verification_failure_does_not_raise(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.side_effect = [{"ids": []}, RuntimeError("verify failed")]
        assert cs.add_chunks([chunk("a")]) == 1


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class TestGetRetriever:
    def test_defaults_to_configured_top_k(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        cs.get_retriever()
        assert instance.as_retriever.call_args.kwargs["search_kwargs"] == {"k": 4}

    def test_explicit_k_wins(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        cs.get_retriever(k=9)
        assert instance.as_retriever.call_args.kwargs["search_kwargs"]["k"] == 9

    def test_filter_is_passed_through(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        cs.get_retriever(filter={"department": "legal"})
        kwargs = instance.as_retriever.call_args.kwargs["search_kwargs"]
        assert kwargs["filter"] == {"department": "legal"}

    def test_no_filter_key_when_none_given(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        cs.get_retriever()
        assert "filter" not in instance.as_retriever.call_args.kwargs["search_kwargs"]


# ---------------------------------------------------------------------------
# TTL / staleness
# ---------------------------------------------------------------------------

class TestStaleDocuments:
    def _dated(self, days_ago):
        return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()

    def test_ttl_disabled_returns_nothing(self, fake_chroma, store_settings):
        store_settings.document_ttl_days = 0
        assert cs.get_stale_documents() == []

    def test_documents_older_than_the_cutoff_are_stale(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {
            "ids": ["old", "new"],
            "metadatas": [
                {"ingested_at": self._dated(40)},
                {"ingested_at": self._dated(1)},
            ],
        }
        assert cs.get_stale_documents(max_age_days=30) == ["old"]

    def test_documents_without_a_timestamp_are_left_alone(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {"ids": ["a"], "metadatas": [{}]}
        assert cs.get_stale_documents(max_age_days=30) == []

    def test_unparseable_timestamps_are_skipped_not_fatal(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {
            "ids": ["bad", "old"],
            "metadatas": [{"ingested_at": "not-a-date"}, {"ingested_at": self._dated(40)}],
        }
        assert cs.get_stale_documents(max_age_days=30) == ["old"]

    def test_empty_collection_is_handled(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.return_value = {"ids": []}
        assert cs.get_stale_documents(max_age_days=30) == []

    def test_query_failure_returns_empty(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.get.side_effect = RuntimeError("collection gone")
        assert cs.get_stale_documents(max_age_days=30) == []

    def test_delete_removes_stale_ids(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        with patch.object(cs, "get_stale_documents", return_value=["a", "b"]):
            assert cs.delete_stale_documents(30) == 2
        instance.delete.assert_called_once_with(ids=["a", "b"])

    def test_delete_with_nothing_stale_is_a_no_op(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        with patch.object(cs, "get_stale_documents", return_value=[]):
            assert cs.delete_stale_documents(30) == 0
        instance.delete.assert_not_called()

    def test_delete_invalidates_the_bm25_cache(self, fake_chroma):
        with (
            patch.object(cs, "get_stale_documents", return_value=["a"]),
            patch("src.retrieval.hybrid.reset_bm25_cache") as reset,
        ):
            cs.delete_stale_documents(30)
        reset.assert_called_once()

    def test_delete_failure_reports_zero(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance.delete.side_effect = RuntimeError("write locked")
        with patch.object(cs, "get_stale_documents", return_value=["a"]):
            assert cs.delete_stale_documents(30) == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestCollectionStats:
    def test_reports_count_and_location(self, fake_chroma):
        stats = cs.collection_stats()
        assert stats == {
            "collection": "test_docs",
            "persist_directory": "./test_chroma",
            "document_count": 7,
        }

    def test_count_failure_reports_minus_one(self, fake_chroma):
        instance, _klass, _emb = fake_chroma
        instance._collection.count.side_effect = RuntimeError("no collection")
        assert cs.collection_stats()["document_count"] == -1
