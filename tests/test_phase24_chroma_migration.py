"""Phase 24 — embedded → server collection migration (scripts/migrate_chroma.py).

The migration moves user data, so the properties that matter are that it
copies *stored* embeddings rather than recomputing them (cost, and drift
if the embedding model has changed), that it is idempotent, and that
--dry-run is genuinely read-only.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scripts.migrate_chroma import migrate


class FakeCollection:
    """Minimal Chroma collection backed by dicts."""

    def __init__(self, rows: dict | None = None):
        self.rows = dict(rows or {})   # id -> (document, metadata, embedding)
        self.upserts: list[dict] = []

    def count(self):
        return len(self.rows)

    def get(self, limit=None, offset=0, include=None):
        items = list(self.rows.items())[offset: (offset + limit) if limit else None]
        return {
            "ids": [i for i, _ in items],
            "documents": [v[0] for _, v in items],
            "metadatas": [v[1] for _, v in items],
            "embeddings": [v[2] for _, v in items],
        }

    def upsert(self, ids=None, documents=None, metadatas=None, embeddings=None):
        self.upserts.append({
            "ids": ids, "documents": documents,
            "metadatas": metadatas, "embeddings": embeddings,
        })
        for idx, doc_id in enumerate(ids):
            self.rows[doc_id] = (
                documents[idx] if documents else None,
                metadatas[idx] if metadatas else None,
                embeddings[idx] if embeddings else None,
            )


def make_rows(n, prefix="doc"):
    return {
        f"{prefix}_{i}": (f"content {i}", {"filename": f"f{i}.md"}, [float(i)] * 4)
        for i in range(n)
    }


@pytest.fixture
def chroma(monkeypatch):
    """Patch chromadb so migrate() runs against in-memory collections."""
    source = FakeCollection(make_rows(5))
    target = FakeCollection()

    persistent = MagicMock()
    persistent.get_collection.return_value = source
    http = MagicMock()
    http.get_or_create_collection.return_value = target

    module = MagicMock()
    module.PersistentClient.return_value = persistent
    module.HttpClient.return_value = http
    monkeypatch.setitem(__import__("sys").modules, "chromadb", module)
    return module, source, target


def run(**kw):
    params = dict(
        source_path="./chroma_db", host="h", port=8001,
        collection_name="enterprise_docs",
    )
    params.update(kw)
    return migrate(**params)


class TestMigration:
    def test_all_chunks_are_copied(self, chroma):
        _mod, source, target = chroma
        summary = run()
        assert summary["copied"] == 5
        assert target.count() == 5
        assert summary["source_count"] == 5

    def test_stored_embeddings_are_reused_not_recomputed(self, chroma):
        """Re-embedding would cost money and drift if the model changed."""
        _mod, source, target = chroma
        run()
        assert target.rows["doc_0"][2] == [0.0] * 4
        assert all("embeddings" in u and u["embeddings"] for u in target.upserts)

    def test_documents_and_metadata_survive(self, chroma):
        _mod, _source, target = chroma
        run()
        assert target.rows["doc_3"][0] == "content 3"
        assert target.rows["doc_3"][1] == {"filename": "f3.md"}

    def test_migration_is_idempotent(self, chroma):
        """Chunk ids are content hashes, so a second run must add nothing."""
        _mod, _source, target = chroma
        run()
        second = run()
        assert target.count() == 5
        assert second["added"] == 0
        assert second["copied"] == 5

    def test_existing_target_content_is_preserved(self, chroma):
        _mod, _source, target = chroma
        target.rows.update(make_rows(2, prefix="other"))
        summary = run()
        assert summary["target_before"] == 2
        assert summary["target_after"] == 7
        assert summary["added"] == 5

    def test_overlapping_chunks_are_counted_as_already_present(self, chroma):
        _mod, source, target = chroma
        target.rows["doc_0"] = source.rows["doc_0"]
        summary = run()
        assert summary["copied"] == 5
        assert summary["added"] == 4  # one was already there

    def test_batching_splits_large_collections(self, chroma):
        _mod, source, target = chroma
        source.rows = make_rows(10)
        run(batch=3)
        assert [len(u["ids"]) for u in target.upserts] == [3, 3, 3, 1]
        assert target.count() == 10

    def test_empty_source_is_a_no_op(self, chroma):
        _mod, source, target = chroma
        source.rows = {}
        summary = run()
        assert summary["copied"] == 0
        assert target.upserts == []

    def test_target_collection_can_be_renamed(self, chroma):
        mod, _source, _target = chroma
        run(target_collection_name="archive")
        mod.HttpClient.return_value.get_or_create_collection.assert_called_once_with("archive")

    def test_ssl_is_passed_to_the_client(self, chroma):
        mod, _source, _target = chroma
        run(ssl=True)
        assert mod.HttpClient.call_args.kwargs["ssl"] is True

    def test_missing_source_collection_raises_clearly(self, chroma):
        mod, _source, _target = chroma
        mod.PersistentClient.return_value.get_collection.side_effect = ValueError("nope")
        with pytest.raises(RuntimeError, match="No collection"):
            run()


class TestDryRun:
    def test_dry_run_writes_nothing(self, chroma):
        _mod, _source, target = chroma
        summary = run(dry_run=True)
        assert target.upserts == []
        assert target.count() == 0
        assert summary["dry_run"] is True

    def test_dry_run_still_reports_what_would_move(self, chroma):
        summary = run(dry_run=True)
        assert summary["source_count"] == 5
        assert summary["copied"] == 0


class TestCli:
    def test_exit_code_zero_on_success(self, chroma):
        from scripts.migrate_chroma import main

        with patch("config.setup_logging", MagicMock()):
            assert main(["--host", "h", "--port", "8001"]) == 0

    def test_failure_reports_nonzero(self, chroma):
        from scripts.migrate_chroma import main

        mod, _source, _target = chroma
        mod.PersistentClient.return_value.get_collection.side_effect = ValueError("gone")
        with patch("config.setup_logging", MagicMock()):
            assert main(["--host", "h"]) == 1

    def test_dry_run_flag_is_wired(self, chroma, capsys):
        from scripts.migrate_chroma import main

        _mod, _source, target = chroma
        with patch("config.setup_logging", MagicMock()):
            main(["--dry-run"])
        assert target.upserts == []
        assert "Dry run" in capsys.readouterr().out
