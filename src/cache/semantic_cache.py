"""Semantic query cache backed by SQLite + embeddings.

Stores query-answer pairs with embedding vectors. On lookup, computes
cosine similarity between the incoming query and cached entries. Returns
a cached answer when similarity exceeds the threshold.

Gated behind SEMANTIC_CACHE_ENABLED feature flag (default: off).
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from config import settings
from src.cache.lsh import BANDS, signature
from src.storage.sql import SqlDatabase

# Upper bound on candidates compared per lookup. LSH normally returns a
# handful; this bounds the pathological case where many entries share a
# bucket, so one unlucky query cannot turn into the scan this replaced.
_MAX_CANDIDATES = 200

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
        self._db = SqlDatabase(db_path)
        self._conn = self._db
        self._lock = self._db.lock
        self._conn.execute(_CREATE_CACHE_TABLE)
        self._conn.commit()

        # Migration: scope isolates entries by retrieval filter / access
        # scope so answers never cross those boundaries; band* are the
        # LSH buckets that keep lookup from scanning the whole table.
        #
        # Rows written before the band columns existed keep NULL and stop
        # matching. That is a cache miss, not a wrong answer — the entry
        # is recomputed and rewritten with bands, and the stale rows age
        # out on their TTL.
        migrations = [
            "ALTER TABLE semantic_cache ADD COLUMN scope TEXT NOT NULL DEFAULT ''",
            *[
                f"ALTER TABLE semantic_cache ADD COLUMN band{i} INTEGER"
                for i in range(BANDS)
            ],
            *[
                f"CREATE INDEX IF NOT EXISTS idx_cache_band{i} "
                f"ON semantic_cache (scope, band{i})"
                for i in range(BANDS)
            ],
        ]
        self._db.executescript(migrations)
        self._embed_fn = embed_fn

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

        # Retrieve only entries that hash into one of the query's bands.
        # The previous version loaded every row for the scope — each with
        # its full 1536-float embedding — and compared them in Python,
        # which made a cache hit slower than a cache miss once the table
        # grew. LSH prunes; exact cosine below still decides, so a false
        # positive costs one comparison and can never return a wrong
        # answer.
        bands = signature(query_embedding)
        now = datetime.now(UTC)
        with self._lock:
            if bands:
                # Positional comparison, not a set membership test: band
                # values are namespaced by their index, so band 2 of the
                # query can only ever equal band 2 of a stored row. That
                # makes this exactly the union of the band buckets.
                clause = " OR ".join(f"band{i} = ?" for i in range(len(bands)))
                rows = self._conn.execute(
                    "SELECT id, query, answer, embedding, mode, strategy, created_at, ttl "
                    f"FROM semantic_cache WHERE scope = ? AND ({clause}) LIMIT ?",
                    (scope, *bands, _MAX_CANDIDATES),
                ).fetchall()
            else:
                # Unusable embedding (empty or all zeros): nothing to
                # match against, and bucketing them together would make
                # every degenerate query collide.
                rows = []

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

        # Bucket on write so lookup can find it without a scan. A vector
        # that produces no signature is still stored — it just will not
        # be retrieved, which is the honest outcome for an embedding that
        # cannot be compared to anything.
        bands = signature(embedding)
        band_values = list(bands) + [None] * (BANDS - len(bands))
        band_columns = ", ".join(f"band{i}" for i in range(BANDS))
        band_placeholders = ", ".join("?" for _ in range(BANDS))

        with self._lock:
            self._conn.execute(
                "INSERT INTO semantic_cache "
                f"(query, answer, embedding, mode, strategy, created_at, ttl, scope, {band_columns}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, {band_placeholders})",
                (
                    query,
                    answer,
                    json.dumps(embedding),
                    mode,
                    strategy,
                    datetime.now(UTC).isoformat(),
                    cache_ttl,
                    scope,
                    *band_values,
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
