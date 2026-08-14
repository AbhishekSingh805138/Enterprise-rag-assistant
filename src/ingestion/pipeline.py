"""The indexing steps a worker runs for one document.

Mirrors the worker lane of the ingestion architecture:

    Download File from S3 -> Parse Document -> Split Into Chunks
    -> Generate Embeddings -> Store in Vector DB

Deliberately thin. Parsing, chunking and embedding are the existing
``loader`` / ``chunker`` / ``chroma_store`` functions called in the same
order with the same arguments as the synchronous upload path — moving
ingestion off the request thread should not change what ends up in the
vector store, and it does not.

Kept free of queue and registry concerns so it can be called directly
(from a test, a backfill script, or the sync upload path) without a broker.
"""
from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.registry import DocumentRecord
from src.storage.object_store import ObjectStore, get_object_store

logger = logging.getLogger(__name__)


class DocumentParseError(RuntimeError):
    """Raised when a stored object cannot be parsed into documents.

    Distinct from transient failures: retrying a malformed PDF three times
    just burns the retry budget, so the worker dead-letters these
    immediately.
    """


@dataclass
class IngestionResult:
    """Outcome of indexing one document."""

    document_id: str
    documents_loaded: int
    chunks_created: int
    chunks_added: int


def stable_source(department: str, filename: str) -> str:
    """Citation-friendly source path, stable across re-uploads.

    Must match the synchronous upload path exactly: it feeds the vector
    store's content-hash ids, so a different value here would make the same
    file indexed through the two paths deduplicate as two documents.
    """
    return f"uploads/{department}/{filename}"


def process_document(
    record: DocumentRecord,
    object_store: ObjectStore | None = None,
) -> IngestionResult:
    """Download, parse, chunk, embed and store one registered document.

    Raises ObjectNotFoundError if the stored object is gone,
    DocumentParseError if it cannot be parsed, and lets transient errors
    (embedding API, vector store) propagate so the worker can retry them.
    """
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.loader import load_path
    from src.vectorstore.chroma_store import add_chunks

    store = object_store or get_object_store()

    # --- Download File from S3 ---
    data = store.get(record.storage_key)
    logger.info(
        "Downloaded %s (%d bytes) for document %s",
        record.storage_key, len(data), record.document_id,
    )

    # Materialise under <tmp>/<department>/<filename> so the loader infers
    # the same department and access level it would for a directory ingest.
    with tempfile.TemporaryDirectory() as tmpdir:
        dept_dir = Path(tmpdir) / record.department
        dept_dir.mkdir(parents=True, exist_ok=True)
        (dept_dir / record.filename).write_bytes(data)

        # --- Parse Document ---
        try:
            docs = load_path(tmpdir)
        except FileNotFoundError as e:
            # The loader skips files it cannot read, then reports "nothing
            # found" — for a single-file ingest that means a parse failure.
            raise DocumentParseError(
                f"Could not parse {record.filename!r}: {e}"
            ) from e

        source = stable_source(record.department, record.filename)
        for doc in docs:
            doc.metadata["source"] = source
            doc.metadata["document_id"] = record.document_id
            doc.metadata["checksum"] = record.checksum

        # --- Split Into Chunks ---
        chunks = chunk_documents(docs)
        if not chunks:
            raise DocumentParseError(
                f"{record.filename!r} produced no chunks — file is empty or unreadable"
            )

        # --- Generate Embeddings + Store in Vector DB ---
        added = add_chunks(chunks)

    logger.info(
        "Indexed %s: %d document(s) -> %d chunk(s), %d new",
        record.document_id, len(docs), len(chunks), added,
    )
    return IngestionResult(
        document_id=record.document_id,
        documents_loaded=len(docs),
        chunks_created=len(chunks),
        chunks_added=added,
    )
