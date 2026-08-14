"""Phase 22 — object storage abstraction (src/storage).

Covers the local backend's behaviour and the path-traversal guard that
protects it, plus the factory's backend selection.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.storage.object_store import (
    LocalObjectStore,
    ObjectNotFoundError,
    ObjectStoreError,
    build_storage_key,
    get_object_store,
    reset_object_store,
)


@pytest.fixture
def store(tmp_path) -> LocalObjectStore:
    return LocalObjectStore(root=tmp_path / "objects")


class TestLocalObjectStore:
    def test_put_then_get_roundtrip(self, store):
        store.put("docs/a/file.txt", b"hello world", "text/plain")
        assert store.get("docs/a/file.txt") == b"hello world"

    def test_exists_reflects_presence(self, store):
        assert store.exists("docs/missing.txt") is False
        store.put("docs/present.txt", b"x")
        assert store.exists("docs/present.txt") is True

    def test_get_missing_key_raises_not_found(self, store):
        with pytest.raises(ObjectNotFoundError):
            store.get("nope/never.txt")

    def test_put_is_idempotent_and_overwrites(self, store):
        store.put("k.txt", b"first")
        store.put("k.txt", b"second")
        assert store.get("k.txt") == b"second"

    def test_put_creates_nested_directories(self, store):
        store.put("a/b/c/d/deep.txt", b"deep")
        assert store.get("a/b/c/d/deep.txt") == b"deep"

    def test_delete_is_forgiving(self, store):
        store.put("gone.txt", b"x")
        store.delete("gone.txt")
        store.delete("gone.txt")  # already absent — must not raise
        assert store.exists("gone.txt") is False

    def test_put_leaves_no_partial_file_behind(self, store):
        """The temp file used for the atomic rename must not survive."""
        store.put("docs/x.txt", b"payload")
        leftovers = list(store._root.rglob("*.part"))
        assert leftovers == []

    def test_uri_is_stable_without_touching_the_store(self, store):
        uri = store.uri("docs/a.txt")
        assert uri.startswith("file://")
        assert "docs/a.txt" in uri
        assert not store.exists("docs/a.txt")


class TestPathTraversalGuard:
    """Keys are partly caller-derived (filename), so escapes must be refused."""

    @pytest.mark.parametrize(
        "key",
        ["../escape.txt", "a/../../escape.txt", "/absolute.txt", ""],
    )
    def test_put_rejects_escaping_keys(self, store, key):
        with pytest.raises(ObjectStoreError):
            store.put(key, b"malicious")

    def test_traversal_does_not_write_outside_root(self, store, tmp_path):
        with pytest.raises(ObjectStoreError):
            store.put("../../pwned.txt", b"malicious")
        assert not (tmp_path / "pwned.txt").exists()

    def test_exists_returns_false_for_invalid_key(self, store):
        assert store.exists("../../etc/passwd") is False


class TestBuildStorageKey:
    def test_key_includes_department_and_document_id(self):
        with patch("src.storage.object_store.settings") as s:
            s.s3_prefix = "documents"
            key = build_storage_key("hr", "doc_abc123", "handbook.pdf")
        assert key == "documents/hr/doc_abc123/handbook.pdf"

    def test_same_filename_in_two_departments_does_not_collide(self):
        with patch("src.storage.object_store.settings") as s:
            s.s3_prefix = "documents"
            a = build_storage_key("hr", "doc_aaa", "policy.md")
            b = build_storage_key("legal", "doc_bbb", "policy.md")
        assert a != b

    def test_empty_prefix_is_omitted(self):
        with patch("src.storage.object_store.settings") as s:
            s.s3_prefix = ""
            assert build_storage_key("hr", "doc_x", "f.txt") == "hr/doc_x/f.txt"


class TestFactory:
    def teardown_method(self):
        reset_object_store()

    def test_local_backend_is_the_default(self, tmp_path):
        reset_object_store()
        with patch("src.storage.object_store.settings") as s:
            s.storage_backend = "local"
            s.storage_local_dir = str(tmp_path / "obj")
            assert isinstance(get_object_store(), LocalObjectStore)

    def test_unknown_backend_raises(self, tmp_path):
        reset_object_store()
        with patch("src.storage.object_store.settings") as s:
            s.storage_backend = "gcs"
            with pytest.raises(ObjectStoreError, match="Unknown STORAGE_BACKEND"):
                get_object_store()

    def test_factory_returns_a_singleton(self, tmp_path):
        reset_object_store()
        with patch("src.storage.object_store.settings") as s:
            s.storage_backend = "local"
            s.storage_local_dir = str(tmp_path / "obj")
            assert get_object_store() is get_object_store()

    def test_s3_backend_without_boto3_gives_actionable_error(self, tmp_path):
        reset_object_store()
        import builtins

        real_import = builtins.__import__

        def _no_boto3(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *args, **kwargs)

        with patch("src.storage.object_store.settings") as s:
            s.storage_backend = "s3"
            s.s3_bucket = "bucket"
            with patch.object(builtins, "__import__", _no_boto3):
                with pytest.raises(ObjectStoreError, match="boto3"):
                    get_object_store()
