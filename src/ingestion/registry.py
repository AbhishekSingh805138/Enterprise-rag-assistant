"""Document registry — the source of truth for what has been ingested.

This table is what makes the two decision diamonds in the ingestion
architecture answerable:

  "Document Already Exists?"  -> :meth:`DocumentRegistry.register_upload`
                                 returns ``created=False`` and the caller
                                 hands back the existing id instead of
                                 storing and queueing the file again.

  "Already Indexed?"          -> :meth:`DocumentRegistry.claim_for_processing`
                                 fails when the document is already
                                 PROCESSED, so a redelivered event is
                                 acknowledged rather than re-embedded.

Identity is derived from the file's SHA-256 and its department, so the
same bytes uploaded twice produce the same document id without a lookup —
and the same bytes filed under two departments stay distinct, because
department drives both retrieval filtering and access level.

Lifecycle::

    PENDING ---claim---> PROCESSING ---success---> PROCESSED
       ^                      |
       |                      +---failure (attempts < max)---> FAILED
       |                      |
       +---re-upload/reset----+---failure (attempts >= max)--> DEAD_LETTER

Chunk-level deduplication in the vector store still applies underneath;
this registry deduplicates one level up, so a repeat upload skips the
parse and embedding cost entirely rather than paying it and discarding
the result.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


# Terminal and transient states.
STATUS_PENDING = "PENDING"
STATUS_PROCESSING = "PROCESSING"
STATUS_PROCESSED = "PROCESSED"
STATUS_FAILED = "FAILED"
STATUS_DEAD_LETTER = "DEAD_LETTER"

VALID_STATUSES = frozenset({
    STATUS_PENDING, STATUS_PROCESSING, STATUS_PROCESSED,
    STATUS_FAILED, STATUS_DEAD_LETTER,
})

# States a worker is allowed to pick up.
_CLAIMABLE = (STATUS_PENDING, STATUS_FAILED)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS documents (
    document_id    TEXT PRIMARY KEY,
    checksum       TEXT    NOT NULL,
    filename       TEXT    NOT NULL,
    department     TEXT    NOT NULL,
    content_type   TEXT    NOT NULL DEFAULT '',
    size_bytes     INTEGER NOT NULL DEFAULT 0,
    storage_key    TEXT    NOT NULL DEFAULT '',
    storage_uri    TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL DEFAULT 'PENDING',
    attempts       INTEGER NOT NULL DEFAULT 0,
    chunks_indexed INTEGER NOT NULL DEFAULT 0,
    error          TEXT    NOT NULL DEFAULT '',
    uploaded_by    TEXT    NOT NULL DEFAULT '',
    request_id     TEXT    NOT NULL DEFAULT '',
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);
"""

_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_identity ON documents(checksum, department)",
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)",
    "CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at)",
]


def compute_checksum(data: bytes) -> str:
    """SHA-256 of the raw upload — the basis of document identity."""
    return hashlib.sha256(data).hexdigest()


def derive_document_id(checksum: str, department: str) -> str:
    """Deterministic document id for (*checksum*, *department*).

    Derived rather than random so the API can answer "have I seen this
    file?" from the bytes alone, and so a retried upload cannot mint a
    second id for content that is already indexed.
    """
    digest = hashlib.sha256(f"{department.lower()}:{checksum}".encode()).hexdigest()
    return f"doc_{digest[:32]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentRecord:
    """One row of the registry."""

    document_id: str
    checksum: str
    filename: str
    department: str
    content_type: str = ""
    size_bytes: int = 0
    storage_key: str = ""
    storage_uri: str = ""
    status: str = STATUS_PENDING
    attempts: int = 0
    chunks_indexed: int = 0
    error: str = ""
    uploaded_by: str = ""
    request_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in (STATUS_PROCESSED, STATUS_DEAD_LETTER)

    def to_dict(self) -> dict:
        return asdict(self)


class DocumentRegistry:
    """SQLite-backed registry. Safe across threads and processes (WAL)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path or settings.document_registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute(_CREATE_TABLE)
            for stmt in _INDEXES:
                self._conn.execute(stmt)
            self._conn.commit()
        logger.info("Document registry ready at %s", self._db_path)

    # -- writes ------------------------------------------------------------

    def register_upload(
        self,
        *,
        checksum: str,
        filename: str,
        department: str,
        content_type: str = "",
        size_bytes: int = 0,
        storage_key: str = "",
        storage_uri: str = "",
        uploaded_by: str = "",
        request_id: str = "",
    ) -> tuple[DocumentRecord, bool]:
        """Idempotently record an upload.

        Returns ``(record, created)``. ``created`` is False when this exact
        content was already registered for this department — the caller
        should then return the existing document id rather than re-storing
        and re-queueing the file.

        The INSERT is conditional in SQL rather than a read-then-write, so
        two concurrent uploads of the same file cannot both believe they
        created it.
        """
        department = department.lower()
        document_id = derive_document_id(checksum, department)
        now = _utc_now()

        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO documents ("
                "  document_id, checksum, filename, department, content_type,"
                "  size_bytes, storage_key, storage_uri, status, attempts,"
                "  chunks_indexed, error, uploaded_by, request_id, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, '', ?, ?, ?, ?)",
                (
                    document_id, checksum, filename, department, content_type,
                    size_bytes, storage_key, storage_uri, STATUS_PENDING,
                    uploaded_by, request_id, now, now,
                ),
            )
            created = cur.rowcount == 1
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()

        record = self._to_record(row)
        logger.info(
            "Registered upload %s (%s) department=%s created=%s status=%s",
            filename, document_id, department, created, record.status,
        )
        return record, created

    def attach_storage(self, document_id: str, storage_key: str, storage_uri: str) -> None:
        """Record where the raw bytes were persisted."""
        self._update(
            document_id,
            "storage_key = ?, storage_uri = ?, updated_at = ?",
            (storage_key, storage_uri, _utc_now()),
        )

    def claim_for_processing(self, document_id: str, stale_after_s: int | None = None) -> bool:
        """Atomically move PENDING/FAILED -> PROCESSING, incrementing attempts.

        Returns False when the document is not claimable — already
        PROCESSED (the "Already Indexed?" yes-branch), currently in flight
        on another worker, or dead-lettered. The guard lives in the WHERE
        clause, so two workers racing on the same event cannot both win.

        A PROCESSING row is also claimable once it has been untouched for
        *stale_after_s*. Without that, a worker killed mid-document would
        strand it in PROCESSING forever: the queue redelivers the event
        after its visibility timeout, but every claim would be refused and
        the document would silently never be indexed.
        """
        stale_s = (
            stale_after_s if stale_after_s is not None else settings.ingest_visibility_timeout_s
        )
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=max(1, stale_s))
        ).isoformat()
        placeholders = ", ".join("?" for _ in _CLAIMABLE)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE documents SET status = ?, attempts = attempts + 1, "
                f"updated_at = ? WHERE document_id = ? AND ("
                f"  status IN ({placeholders})"
                f"  OR (status = ? AND updated_at < ?)"
                f")",
                (
                    STATUS_PROCESSING, _utc_now(), document_id, *_CLAIMABLE,
                    STATUS_PROCESSING, cutoff,
                ),
            )
            self._conn.commit()
            claimed = cur.rowcount == 1
        if not claimed:
            logger.debug("Could not claim %s — not in a claimable state", document_id)
        return claimed

    def mark_processed(self, document_id: str, chunks_indexed: int) -> None:
        """Terminal success — the "Mark Document Processed" step."""
        self._update(
            document_id,
            "status = ?, chunks_indexed = ?, error = '', updated_at = ?",
            (STATUS_PROCESSED, chunks_indexed, _utc_now()),
        )
        logger.info("Document %s processed (%d chunks indexed)", document_id, chunks_indexed)

    def mark_failed(self, document_id: str, error: str) -> None:
        """Retryable failure — the document returns to the claimable set."""
        self._update(
            document_id,
            "status = ?, error = ?, updated_at = ?",
            (STATUS_FAILED, error[:2000], _utc_now()),
        )
        logger.warning("Document %s failed: %s", document_id, error)

    def mark_dead_letter(self, document_id: str, error: str) -> None:
        """Retry budget exhausted — needs an operator, not another attempt."""
        self._update(
            document_id,
            "status = ?, error = ?, updated_at = ?",
            (STATUS_DEAD_LETTER, error[:2000], _utc_now()),
        )
        logger.error("Document %s dead-lettered after retries: %s", document_id, error)

    def reset_for_retry(self, document_id: str) -> bool:
        """Return a FAILED/DEAD_LETTER document to PENDING with a fresh budget.

        Used when a user re-uploads a document whose indexing previously
        gave up, and by the DLQ replay path.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE documents SET status = ?, attempts = 0, error = '', "
                "updated_at = ? WHERE document_id = ? AND status IN (?, ?)",
                (STATUS_PENDING, _utc_now(), document_id, STATUS_FAILED, STATUS_DEAD_LETTER),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def _update(self, document_id: str, set_clause: str, params: tuple) -> None:
        with self._lock:
            self._conn.execute(
                f"UPDATE documents SET {set_clause} WHERE document_id = ?",
                (*params, document_id),
            )
            self._conn.commit()

    # -- reads -------------------------------------------------------------

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._to_record(row) if row else None

    def find_by_content(self, checksum: str, department: str) -> DocumentRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE checksum = ? AND department = ?",
                (checksum, department.lower()),
            ).fetchone()
        return self._to_record(row) if row else None

    def list_documents(
        self, status: str | None = None, department: str | None = None, limit: int = 50
    ) -> list[DocumentRecord]:
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status.upper())
        if department:
            clauses.append("department = ?")
            params.append(department.lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ?",
                (*params, max(1, limit)),
            ).fetchall()
        return [self._to_record(r) for r in rows]

    def stats(self) -> dict[str, int]:
        """Document counts per status (all statuses present, zero-filled)."""
        counts = {s: 0 for s in sorted(VALID_STATUSES)}
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM documents GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[row["status"]] = int(row["n"])
        counts["TOTAL"] = sum(counts.values())
        return counts

    def _to_record(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            checksum=row["checksum"],
            filename=row["filename"],
            department=row["department"],
            content_type=row["content_type"],
            size_bytes=row["size_bytes"],
            storage_key=row["storage_key"],
            storage_uri=row["storage_uri"],
            status=row["status"],
            attempts=row["attempts"],
            chunks_indexed=row["chunks_indexed"],
            error=row["error"],
            uploaded_by=row["uploaded_by"],
            request_id=row["request_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.debug("Error closing document registry", exc_info=True)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: DocumentRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> DocumentRegistry:
    """Return the process-wide document registry (thread-safe)."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = DocumentRegistry()
        return _registry


def reset_registry() -> None:
    """Drop the cached registry (tests, config reload)."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            try:
                _registry.close()
            except Exception:
                logger.debug("Error closing registry", exc_info=True)
        _registry = None
