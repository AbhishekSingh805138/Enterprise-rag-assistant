"""Semantic query cache backed by SQLite + embeddings.

Stores query-answer pairs with embedding vectors. On lookup, computes
cosine similarity between the incoming query and cached entries. Returns
a cached answer when similarity exceeds the threshold.

Gated behind SEMANTIC_CACHE_ENABLED feature flag (default: off).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_CREATE_CACHE_TABLE = """\
CREATE TABLE IF NOT EXISTS semantic_cache (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT NOT NULL,
    answer     TEXT NOT NULL,
    embedding  TEXT NOT NULL,
    mode       TEXT NOT NULL DEFAULT '',
    strategy   TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    ttl        INTEGER NOT NULL DEFAULT 3600
);
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # strict: a dimension mismatch (e.g. after an embedding model change)
    # would otherwise truncate silently and yield a plausible-looking but
    # meaningless score, serving the wrong cached answer.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """Embedding-based semantic cache for query-answer pairs."""

    def __init__(self, db_path: str, embed_fn=None) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_CACHE_TABLE)
        self._conn.commit()
        # Migration: scope column isolates cache entries by retrieval filter /
        # access scope so answers never leak across those boundaries.
        try:
            self._conn.execute(
                "ALTER TABLE semantic_cache ADD COLUMN scope TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        self._embed_fn = embed_fn
        self._lock = threading.Lock()

    def _get_embed_fn(self):
        """Lazy-load embedding function."""
        if self._embed_fn is None:
            from src.llm.providers import build_embeddings

            # Same provider as the vector store: cache entries are matched
            # by cosine similarity, so an embedder from a different model
            # would score every stored query as unrelated (or, worse,
            # spuriously similar) with no error to show for it.
            self._embed_fn = build_embeddings(
                settings.embedding_provider, settings.embedding_model
            )
        return self._embed_fn

    def lookup(
        self,
        query: str,
        threshold: float | None = None,
        mode: str = "",
        strategy: str = "",
        scope: str = "",
    ) -> str | None:
        """Look up a cached answer for a similar query.

        Only entries whose scope matches exactly are eligible — scope encodes
        the retrieval filter (e.g. department) so answers never cross
        access boundaries. Returns the cached answer if cosine similarity
        >= threshold, else None.
        """
        if not settings.semantic_cache_enabled:
            return None

        thresh = threshold or settings.semantic_cache_threshold

        try:
            embed_fn = self._get_embed_fn()
            query_embedding = embed_fn.embed_query(query)
        except Exception:
            logger.debug("Cache lookup failed: embedding error", exc_info=True)
            return None

        # Fetch all cached entries (for small cache sizes this is fine;
        # for large-scale use, switch to a vector index)
        now = datetime.now(UTC)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, query, answer, embedding, mode, strategy, created_at, ttl "
                "FROM semantic_cache WHERE scope = ?",
                (scope,),
            ).fetchall()

        best_score = 0.0
        best_answer = None

        for row in rows:
            # Check TTL
            try:
                created = datetime.fromisoformat(row["created_at"])
                age_seconds = (now - created).total_seconds()
                if age_seconds > row["ttl"]:
                    continue
            except (ValueError, TypeError):
                continue

            # Check mode/strategy filter
            if mode and row["mode"] and row["mode"] != mode:
                continue
            if strategy and row["strategy"] and row["strategy"] != strategy:
                continue

            try:
                cached_embedding = json.loads(row["embedding"])
                score = _cosine_similarity(query_embedding, cached_embedding)
                if score >= thresh and score > best_score:
                    best_score = score
                    best_answer = row["answer"]
            except (json.JSONDecodeError, TypeError):
                continue

        if best_answer is not None:
            logger.info("Cache HIT (score=%.4f) for: %s", best_score, query[:80])
        else:
            logger.debug("Cache MISS for: %s", query[:80])

        return best_answer

    def store(
        self,
        query: str,
        answer: str,
        mode: str = "",
        strategy: str = "",
        ttl: int | None = None,
        scope: str = "",
    ) -> None:
        """Store a query-answer pair in the cache."""
        if not settings.semantic_cache_enabled:
            return

        cache_ttl = ttl or settings.semantic_cache_ttl

        try:
            embed_fn = self._get_embed_fn()
            embedding = embed_fn.embed_query(query)
        except Exception:
            logger.debug("Cache store failed: embedding error", exc_info=True)
            return

        with self._lock:
            self._conn.execute(
                "INSERT INTO semantic_cache (query, answer, embedding, mode, strategy, created_at, ttl, scope) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    query,
                    answer,
                    json.dumps(embedding),
                    mode,
                    strategy,
                    datetime.now(UTC).isoformat(),
                    cache_ttl,
                    scope,
                ),
            )
            self._conn.commit()
        logger.debug("Cached answer for: %s", query[:80])

    def invalidate(self) -> int:
        """Delete all cached entries. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM semantic_cache")
            self._conn.commit()
            count = cur.rowcount
        logger.info("Invalidated %d cache entries", count)
        return count

    def cleanup_expired(self) -> int:
        """Delete expired entries. Returns count deleted."""
        with self._lock:
            # SQLite doesn't have native datetime diff, so we fetch and filter
            rows = self._conn.execute(
                "SELECT id, created_at, ttl FROM semantic_cache"
            ).fetchall()

            expired_ids = []
            now_dt = datetime.now(UTC)
            for row in rows:
                try:
                    created = datetime.fromisoformat(row["created_at"])
                    if (now_dt - created).total_seconds() > row["ttl"]:
                        expired_ids.append(row["id"])
                except (ValueError, TypeError):
                    expired_ids.append(row["id"])

            if expired_ids:
                placeholders = ",".join("?" for _ in expired_ids)
                self._conn.execute(
                    f"DELETE FROM semantic_cache WHERE id IN ({placeholders})",
                    expired_ids,
                )
                self._conn.commit()

        if expired_ids:
            logger.info("Cleaned up %d expired cache entries", len(expired_ids))
        return len(expired_ids)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total FROM semantic_cache"
            ).fetchone()
        return {"total_entries": row["total"] if row else 0}

    def close(self) -> None:
        """Close the underlying connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache: SemanticCache | None = None
_cache_lock = threading.Lock()


def _default_cache_db_path() -> str:
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / "semantic_cache.db")


def get_cache(db_path: str | None = None) -> SemanticCache:
    """Return the singleton SemanticCache. Thread-safe."""
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = SemanticCache(db_path or _default_cache_db_path())
            logger.info("SemanticCache initialized")
        return _cache


def reset_cache() -> None:
    """Close and discard the singleton (for testing)."""
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.close()
            _cache = None
