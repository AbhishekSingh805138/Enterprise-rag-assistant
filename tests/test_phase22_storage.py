"""Phase 22 — object storage abstraction (src/storage).

Covers the local backend's behaviour and the path-traversal guard that
protects it, plus the factory's backend selection.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.storage.object_store import (
    LocalObjectStore,
    ObjectNotFoundError,
    ObjectStoreError,
    S3ObjectStore,
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


class TestS3ObjectStore:
    """S3 is the production backend, so it is tested against a fake boto3.

    boto3 is an optional dependency and a live bucket is not available in
    CI; what matters here is the call contract — bucket/key wiring, MinIO
    endpoint support, and translating S3's error codes into the store's
    own exceptions so the worker can tell "missing" from "broken".
    """

    @pytest.fixture
    def fake_boto3(self, monkeypatch):
        import sys
        import types

        module = types.ModuleType("boto3")
        client = MagicMock(name="s3_client")
        module.client = MagicMock(return_value=client)
        module._client = client
        monkeypatch.setitem(sys.modules, "boto3", module)
        return module

    @pytest.fixture
    def s3(self, fake_boto3):
        with patch("src.storage.object_store.settings") as s:
            s.s3_bucket = "rag-documents"
            s.s3_region = "eu-west-1"
            s.s3_endpoint_url = ""
            s.s3_prefix = "documents"
            yield S3ObjectStore(), fake_boto3._client

    @staticmethod
    def _client_error(code):
        err = Exception("boom")
        err.response = {"Error": {"Code": code}}
        return err

    def test_client_is_built_for_the_configured_region(self, fake_boto3):
        with patch("src.storage.object_store.settings") as s:
            s.s3_bucket = "b"
            s.s3_region = "ap-south-1"
            s.s3_endpoint_url = ""
            S3ObjectStore()
        assert fake_boto3.client.call_args.kwargs["region_name"] == "ap-south-1"

    def test_endpoint_url_enables_minio(self, fake_boto3):
        with patch("src.storage.object_store.settings") as s:
            s.s3_bucket = "b"
            s.s3_region = "us-east-1"
            s.s3_endpoint_url = "http://minio:9000"
            S3ObjectStore()
        assert fake_boto3.client.call_args.kwargs["endpoint_url"] == "http://minio:9000"

    def test_blank_endpoint_means_real_aws(self, s3, fake_boto3):
        assert fake_boto3.client.call_args.kwargs["endpoint_url"] is None

    def test_missing_bucket_is_rejected(self, fake_boto3):
        with patch("src.storage.object_store.settings") as s:
            s.s3_bucket = ""
            with pytest.raises(ObjectStoreError, match="S3_BUCKET"):
                S3ObjectStore()

    def test_put_writes_to_the_bucket(self, s3):
        store, client = s3
        store.put("documents/hr/doc_1/f.md", b"payload", "text/markdown")
        kwargs = client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "rag-documents"
        assert kwargs["Key"] == "documents/hr/doc_1/f.md"
        assert kwargs["Body"] == b"payload"
        assert kwargs["ContentType"] == "text/markdown"

    def test_put_omits_content_type_when_unknown(self, s3):
        store, client = s3
        store.put("k", b"x")
        assert "ContentType" not in client.put_object.call_args.kwargs

    def test_put_returns_the_s3_uri(self, s3):
        store, _client = s3
        assert store.put("k", b"x") == "s3://rag-documents/k"

    def test_put_failure_is_wrapped(self, s3):
        store, client = s3
        client.put_object.side_effect = OSError("network unreachable")
        with pytest.raises(ObjectStoreError, match="Failed to store"):
            store.put("k", b"x")

    def test_get_returns_the_body(self, s3):
        store, client = s3
        client.get_object.return_value = {"Body": MagicMock(read=lambda: b"stored bytes")}
        assert store.get("k") == b"stored bytes"

    def test_get_missing_key_raises_not_found(self, s3):
        """The worker treats this as permanent; a generic error would retry."""
        store, client = s3
        client.get_object.side_effect = self._client_error("NoSuchKey")
        with pytest.raises(ObjectNotFoundError):
            store.get("k")

    def test_get_other_errors_stay_retryable(self, s3):
        store, client = s3
        client.get_object.side_effect = self._client_error("ServiceUnavailable")
        with pytest.raises(ObjectStoreError) as exc:
            store.get("k")
        assert not isinstance(exc.value, ObjectNotFoundError)

    def test_exists_is_true_when_head_succeeds(self, s3):
        store, _client = s3
        assert store.exists("k") is True

    def test_exists_is_false_on_404(self, s3):
        store, client = s3
        client.head_object.side_effect = self._client_error("404")
        assert store.exists("k") is False

    def test_exists_is_false_and_logged_on_other_errors(self, s3):
        store, client = s3
        client.head_object.side_effect = self._client_error("AccessDenied")
        assert store.exists("k") is False

    def test_delete_removes_the_object(self, s3):
        store, client = s3
        store.delete("k")
        client.delete_object.assert_called_once_with(Bucket="rag-documents", Key="k")

    def test_delete_failure_is_wrapped(self, s3):
        store, client = s3
        client.delete_object.side_effect = OSError("denied")
        with pytest.raises(ObjectStoreError, match="Failed to delete"):
            store.delete("k")

    def test_uri_does_not_touch_the_bucket(self, s3):
        store, client = s3
        assert store.uri("a/b.md") == "s3://rag-documents/a/b.md"
        client.head_object.assert_not_called()

    def test_explicit_constructor_args_win_over_settings(self, fake_boto3):
        with patch("src.storage.object_store.settings") as s:
            s.s3_bucket = "from-settings"
            s.s3_region = "us-east-1"
            s.s3_endpoint_url = ""
            store = S3ObjectStore(bucket="explicit", region="sa-east-1")
        assert store.uri("k") == "s3://explicit/k"
        assert fake_boto3.client.call_args.kwargs["region_name"] == "sa-east-1"

    def test_factory_builds_an_s3_store_when_configured(self, fake_boto3):
        reset_object_store()
        try:
            with patch("src.storage.object_store.settings") as s:
                s.storage_backend = "s3"
                s.s3_bucket = "b"
                s.s3_region = "us-east-1"
                s.s3_endpoint_url = ""
                assert isinstance(get_object_store(), S3ObjectStore)
        finally:
            reset_object_store()


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
