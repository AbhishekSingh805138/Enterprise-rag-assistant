"""Durable object storage for uploaded documents.

The upload API writes the raw bytes here *before* acknowledging the client,
so the document survives a worker crash, a broker restart, or a failed
parse. Everything downstream (parse, chunk, embed) can then be replayed
from the stored object rather than asking the user to upload again — that
is what makes the retry and dead-letter paths in the architecture
meaningful.

Two backends implement the same protocol:

    local  filesystem-backed, no extra dependencies — the default, so a
           developer clone works with nothing running.
    s3     boto3; also speaks to MinIO or any S3-compatible endpoint via
           S3_ENDPOINT_URL.

Callers use :func:`get_object_store`; nothing outside this module should
care which backend is active.
"""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from config import settings

logger = logging.getLogger(__name__)


class ObjectStoreError(RuntimeError):
    """Raised when an object cannot be stored or retrieved."""


class ObjectNotFoundError(ObjectStoreError):
    """Raised when a key does not exist in the store."""


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal blob store surface used by the ingestion pipeline."""

    def put(self, key: str, data: bytes, content_type: str = "") -> str:
        """Store *data* under *key*. Returns a URI identifying the object."""
        ...

    def get(self, key: str) -> bytes:
        """Return the bytes stored under *key*.

        Raises ObjectNotFoundError when the key is absent.
        """
        ...

    def exists(self, key: str) -> bool:
        """True when *key* is present in the store."""
        ...

    def delete(self, key: str) -> None:
        """Remove *key*. Succeeds silently when the key is already gone."""
        ...

    def uri(self, key: str) -> str:
        """Return the canonical URI for *key* without touching the store."""
        ...


def build_storage_key(department: str, document_id: str, filename: str) -> str:
    """Return the canonical storage key for an uploaded document.

    Keying on the document id (a checksum-derived value) rather than the
    filename means two departments can hold same-named files without
    collision, and re-uploading identical bytes overwrites itself instead
    of accumulating copies.
    """
    prefix = settings.s3_prefix.strip("/")
    parts = [p for p in (prefix, department, document_id, filename) if p]
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------

class LocalObjectStore:
    """Filesystem-backed store. Layout mirrors the S3 key structure."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or settings.storage_local_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("Local object store rooted at %s", self._root)

    def _resolve(self, key: str) -> Path:
        """Map *key* to a path, refusing anything that escapes the root.

        The key is partly caller-influenced (filename), so a key like
        ``../../etc/passwd`` must not be able to write outside the root.
        """
        if not key or key.startswith("/") or "\x00" in key:
            raise ObjectStoreError(f"Invalid object key: {key!r}")
        target = (self._root / key).resolve()
        if target != self._root and self._root not in target.parents:
            raise ObjectStoreError(f"Object key escapes the storage root: {key!r}")
        return target

    def put(self, key: str, data: bytes, content_type: str = "") -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a sibling temp file then rename, so a reader never sees a
        # half-written object and a crash mid-write leaves no partial file.
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            tmp.write_bytes(data)
            tmp.replace(target)
        except OSError as e:
            raise ObjectStoreError(f"Failed to store {key!r}: {e}") from e
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        logger.debug("Stored %d bytes at %s", len(data), key)
        return self.uri(key)

    def get(self, key: str) -> bytes:
        target = self._resolve(key)
        if not target.is_file():
            raise ObjectNotFoundError(f"Object not found: {key!r}")
        try:
            return target.read_bytes()
        except OSError as e:
            raise ObjectStoreError(f"Failed to read {key!r}: {e}") from e

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).is_file()
        except ObjectStoreError:
            return False

    def delete(self, key: str) -> None:
        try:
            self._resolve(key).unlink(missing_ok=True)
        except ObjectStoreError:
            return
        except OSError as e:
            raise ObjectStoreError(f"Failed to delete {key!r}: {e}") from e

    def uri(self, key: str) -> str:
        return f"file://{(self._root / key).as_posix()}"

    def clear(self) -> None:
        """Remove every stored object. Intended for tests and local resets."""
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# S3 backend
# ---------------------------------------------------------------------------

class S3ObjectStore:
    """S3-backed store (boto3). Works against AWS S3 or MinIO."""

    def __init__(
        self,
        bucket: str | None = None,
        region: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3
        except ImportError as e:  # pragma: no cover - depends on env
            raise ObjectStoreError(
                "STORAGE_BACKEND=s3 requires boto3. Install it with: pip install boto3"
            ) from e

        self._bucket = bucket or settings.s3_bucket
        if not self._bucket:
            raise ObjectStoreError("STORAGE_BACKEND=s3 requires S3_BUCKET to be set.")

        endpoint = endpoint_url if endpoint_url is not None else settings.s3_endpoint_url
        self._client = boto3.client(
            "s3",
            region_name=region or settings.s3_region,
            endpoint_url=endpoint or None,
        )
        logger.info(
            "S3 object store: bucket=%s region=%s endpoint=%s",
            self._bucket, region or settings.s3_region, endpoint or "aws",
        )

    def _is_not_found(self, error: Exception) -> bool:
        code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
        return str(code) in {"404", "NoSuchKey", "NotFound"}

    def put(self, key: str, data: bytes, content_type: str = "") -> str:
        kwargs = {"Bucket": self._bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        try:
            self._client.put_object(**kwargs)
        except Exception as e:
            raise ObjectStoreError(f"Failed to store {key!r} in S3: {e}") from e
        logger.debug("Stored %d bytes at s3://%s/%s", len(data), self._bucket, key)
        return self.uri(key)

    def get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            if self._is_not_found(e):
                raise ObjectNotFoundError(f"Object not found: {key!r}") from e
            raise ObjectStoreError(f"Failed to read {key!r} from S3: {e}") from e

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as e:
            if self._is_not_found(e):
                return False
            logger.warning("S3 head_object failed for %s: %s", key, e)
            return False

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
        except Exception as e:
            raise ObjectStoreError(f"Failed to delete {key!r} from S3: {e}") from e

    def uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_store: ObjectStore | None = None
_lock = threading.Lock()


def get_object_store() -> ObjectStore:
    """Return the configured object store (singleton, thread-safe)."""
    global _store
    with _lock:
        if _store is None:
            backend = settings.storage_backend.lower()
            if backend == "s3":
                _store = S3ObjectStore()
            elif backend == "local":
                _store = LocalObjectStore()
            else:
                raise ObjectStoreError(
                    f"Unknown STORAGE_BACKEND {settings.storage_backend!r}. "
                    "Choose from: local, s3"
                )
        return _store


def reset_object_store() -> None:
    """Drop the cached store so the next call rebuilds it (tests, config reload)."""
    global _store
    with _lock:
        _store = None
