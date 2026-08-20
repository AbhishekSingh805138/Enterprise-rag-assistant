"""ChromaDB vector store wrapper (persistent, OpenAI embeddings).

Exposes helpers to (a) build/add to the store from chunks and (b) get a
retriever for querying. Persistence lives at settings.chroma_dir so ingestion
and querying are separate processes.

Key improvements over the skeleton:
  - Content-hash IDs prevent duplicate chunks on re-ingestion.
  - Singleton pattern avoids recreating embeddings/connections per call.
  - Error handling + logging throughout.
  - Phase 8: staleness detection (auto-refresh after interval).
  - Phase 8: document TTL (stale document detection & cleanup).
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from datetime import UTC

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config import settings
from src.llm.providers import build_embeddings, embedding_fingerprint

logger = logging.getLogger(__name__)

# Module-level singletons — avoids recreating clients on every call.
_embeddings: Embeddings | None = None
_vectorstore: Chroma | None = None
_last_refresh: float = 0.0  # monotonic timestamp of last refresh
_lock = threading.Lock()


def _get_embeddings() -> Embeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = build_embeddings(
            settings.embedding_provider, settings.embedding_model
        )
    return _embeddings


def _build_http_client():
    """Connect to a ChromaDB server (CHROMA_MODE=server).

    Every read goes to the server, so a document indexed by the ingestion
    worker is immediately visible here — which is the whole reason this
    mode exists.
    """
    import chromadb

    kwargs: dict = {
        "host": settings.chroma_host,
        "port": settings.chroma_port,
        "ssl": settings.chroma_ssl,
    }
    if settings.chroma_auth_token:
        from chromadb.config import Settings as ChromaSettings

        kwargs["settings"] = ChromaSettings(
            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
            chroma_client_auth_credentials=settings.chroma_auth_token,
        )
    return chromadb.HttpClient(**kwargs)


def get_vectorstore() -> Chroma:
    """Open (or create) the collection (singleton). Thread-safe via module lock.

    In *server* mode the client is a thin HTTP handle and is never
    recycled — the server is the single source of truth, so there is
    nothing stale to refresh.

    In *embedded* mode the periodic recycle below is retained, but note
    what it does **not** do: chromadb's PersistentClient is a
    process-level singleton keyed by path, so discarding this wrapper
    reuses the same in-memory index and will not pick up writes made by
    another process. Embedded mode is single-process only.
    """
    global _vectorstore, _last_refresh
    with _lock:
        now = time.monotonic()
        server_mode = settings.chroma_mode.lower() == "server"

        if _vectorstore is not None and not server_mode:
            elapsed = now - _last_refresh
            if elapsed > settings.chroma_refresh_interval:
                logger.info(
                    "ChromaDB refresh: %.0fs since last refresh (interval=%ds)",
                    elapsed, settings.chroma_refresh_interval,
                )
                _vectorstore = None  # force re-creation

        if _vectorstore is None:
            if server_mode:
                _vectorstore = Chroma(
                    collection_name=settings.chroma_collection,
                    embedding_function=_get_embeddings(),
                    client=_build_http_client(),
                )
                logger.info(
                    "Connected to Chroma server %s:%d (collection '%s')",
                    settings.chroma_host, settings.chroma_port,
                    settings.chroma_collection,
                )
            else:
                _vectorstore = Chroma(
                    collection_name=settings.chroma_collection,
                    embedding_function=_get_embeddings(),
                    persist_directory=settings.chroma_dir,
                )
                logger.info(
                    "Opened embedded Chroma collection '%s' at %s",
                    settings.chroma_collection, settings.chroma_dir,
                )
            _last_refresh = now
            _assert_embedding_space(_vectorstore)
        return _vectorstore


_EMBEDDING_KEY = "embedding_fingerprint"


class EmbeddingSpaceMismatch(RuntimeError):
    """The collection was built with a different embedding model."""


def _assert_embedding_space(store: Chroma) -> None:
    """Refuse to serve a collection embedded by a different model.

    Vectors from two embedding models are not comparable. Querying an
    index built by one using another does not error — it returns
    confidently ranked, semantically unrelated documents, which the RAG
    pipeline then cites as sources. There is no failure signal anywhere
    downstream, so this is checked at the only point where it is cheap.

    Stamps the fingerprint on first use so existing collections adopt it
    rather than needing a migration.
    """
    current = embedding_fingerprint()
    try:
        collection = store._collection
        metadata = collection.metadata
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        recorded = metadata.get(_EMBEDDING_KEY)

        # Only a string is a fingerprint. Anything else means the store
        # did not give us one, and guessing from it would be worse than
        # not checking at all.
        if isinstance(recorded, str) and recorded:
            if recorded != current:
                raise EmbeddingSpaceMismatch(
                    f"Collection '{settings.chroma_collection}' was indexed with "
                    f"embedding model '{recorded}' but this process is configured "
                    f"for '{current}'. Vectors from different embedding models are "
                    f"not comparable — retrieval would return plausible-looking "
                    f"but unrelated documents. Re-index the corpus, or set "
                    f"EMBEDDING_MODEL/EMBEDDING_PROVIDER back to '{recorded}'."
                )
            return

        count = collection.count()
        if not isinstance(count, int):
            return  # not a real collection handle; nothing to stamp
        collection.modify(metadata={**metadata, _EMBEDDING_KEY: current})
        if count > 0:
            # A corpus indexed before this check existed. Adopt the
            # current model as its identity rather than guessing, and say
            # so — this is the one moment a genuine mismatch can pass
            # through unnoticed.
            logger.warning(
                "Collection '%s' had no embedding fingerprint; adopting '%s'. "
                "If it was indexed with a different model, re-index it.",
                settings.chroma_collection, current,
            )
    except EmbeddingSpaceMismatch:
        raise
    except Exception as e:
        # Never let the guard itself take down retrieval.
        logger.debug("Could not verify embedding fingerprint: %s", e)


def refresh_store() -> None:
    """Force a refresh of the vectorstore connection on next access."""
    global _vectorstore, _last_refresh
    _vectorstore = None
    _last_refresh = 0.0
    logger.info("ChromaDB store marked for refresh")


def reset_store() -> None:
    """Reset singletons (useful for testing or after a full re-ingest)."""
    global _embeddings, _vectorstore, _last_refresh
    with _lock:
        _embeddings = None
        _vectorstore = None
        _last_refresh = 0.0


def _content_hash(text: str, metadata: dict) -> str:
    """Deterministic ID from content + origin so re-ingestion is idempotent.

    Uses filename + department (stable across uploads) when available,
    falling back to the full source path for CLI/ingest-based ingestion.
    """
    filename = metadata.get("filename", "")
    department = metadata.get("department", "")
    if filename:
        origin = f"{department}/{filename}"
    else:
        origin = metadata.get("source", "")
    start = str(metadata.get("start_index", ""))
    payload = f"{origin}::{start}::{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


def add_chunks(chunks: list[Document]) -> int:
    """Embed and persist chunks. Deduplicates by content hash.

    Returns the number of *new* chunks actually added.
    """
    if not chunks:
        logger.warning("add_chunks called with empty list")
        return 0

    store = get_vectorstore()

    # Build deterministic IDs to prevent duplicates.
    ids = [_content_hash(c.page_content, c.metadata) for c in chunks]

    # Check which IDs already exist to skip them.
    existing = set()
    try:
        result = store.get(ids=ids, include=[])
        existing = set(result["ids"]) if result and result.get("ids") else set()
    except Exception:
        logger.debug("Could not check existing IDs — will attempt full add")

    new_chunks = []
    new_ids = []
    for chunk, cid in zip(chunks, ids, strict=True):
        if cid not in existing:
            new_chunks.append(chunk)
            new_ids.append(cid)

    if not new_chunks:
        logger.info("All %d chunk(s) already exist — nothing to add", len(chunks))
        return 0

    store.add_documents(new_chunks, ids=new_ids)

    # Invalidate BM25 cache since corpus changed
    try:
        from src.retrieval.hybrid import reset_bm25_cache
        reset_bm25_cache()
    except ImportError:
        pass

    # Verify persistence: confirm the chunks are queryable.
    try:
        verify = store.get(ids=new_ids[:1], include=[])
        if not verify or not verify.get("ids"):
            logger.error(
                "Persistence verification FAILED — chunks may not have been stored"
            )
    except Exception:
        logger.warning("Could not verify chunk persistence", exc_info=True)

    logger.info(
        "Added %d new chunk(s) to Chroma (%d skipped as duplicates)",
        len(new_chunks), len(chunks) - len(new_chunks),
    )
    return len(new_chunks)


def get_retriever(
    k: int | None = None,
    filter: dict | None = None,
) -> VectorStoreRetriever:
    """Return a retriever with optional metadata filtering.

    Examples:
        get_retriever(filter={"department": "legal"})
        get_retriever(k=8, filter={"access_level": "internal"})
    """
    search_kwargs: dict = {"k": k or settings.top_k}
    if filter:
        search_kwargs["filter"] = filter
    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def get_stale_documents(max_age_days: int | None = None) -> list[str]:
    """Return IDs of documents older than *max_age_days*.

    Requires documents to have 'ingested_at' ISO-format metadata.
    Returns an empty list if TTL is disabled (max_age_days=0).
    """
    from datetime import datetime, timedelta

    days = max_age_days if max_age_days is not None else settings.document_ttl_days
    if days <= 0:
        return []

    cutoff = datetime.now(UTC) - timedelta(days=days)
    store = get_vectorstore()
    stale_ids: list[str] = []

    try:
        result = store.get(include=["metadatas"])
        if not result or not result.get("ids"):
            return []

        for doc_id, meta in zip(result["ids"], result["metadatas"], strict=True):
            ingested_at = meta.get("ingested_at", "")
            if not ingested_at:
                continue
            try:
                doc_ts = datetime.fromisoformat(ingested_at)
                if doc_ts < cutoff:
                    stale_ids.append(doc_id)
            except (ValueError, TypeError):
                logger.debug("Invalid ingested_at for doc %s: %s", doc_id, ingested_at)
    except Exception:
        logger.exception("Failed to check for stale documents")

    logger.info("Found %d stale document(s) older than %d days", len(stale_ids), days)
    return stale_ids


def delete_stale_documents(max_age_days: int | None = None) -> int:
    """Delete documents older than *max_age_days*. Returns count deleted."""
    stale_ids = get_stale_documents(max_age_days)
    if not stale_ids:
        return 0

    store = get_vectorstore()
    try:
        store.delete(ids=stale_ids)
        logger.info("Deleted %d stale document(s)", len(stale_ids))

        # Invalidate BM25 cache since corpus changed
        try:
            from src.retrieval.hybrid import reset_bm25_cache
            reset_bm25_cache()
        except ImportError:
            pass

        return len(stale_ids)
    except Exception:
        logger.exception("Failed to delete stale documents")
        return 0


def collection_stats() -> dict:
    """Return basic stats about the current collection.

    Reports where the vectors actually live: a directory in embedded mode,
    an endpoint in server mode. Reporting a persist_directory while
    connected to a server would point operators at a stale local copy.
    """
    store = get_vectorstore()
    try:
        count = store._collection.count()
    except Exception:
        count = -1

    server_mode = settings.chroma_mode.lower() == "server"
    scheme = "https" if settings.chroma_ssl else "http"
    return {
        "collection": settings.chroma_collection,
        "mode": "server" if server_mode else "embedded",
        "persist_directory": "" if server_mode else settings.chroma_dir,
        "endpoint": (
            f"{scheme}://{settings.chroma_host}:{settings.chroma_port}"
            if server_mode else ""
        ),
        "document_count": count,
    }
