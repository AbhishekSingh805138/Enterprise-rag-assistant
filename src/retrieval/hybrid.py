"""Hybrid retriever: dense (ChromaDB) + sparse (BM25) fused via RRF.

Dense retrieval is great at semantic similarity but misses keyword matches.
BM25 is great at exact keyword matching but has no semantic understanding.
Reciprocal Rank Fusion combines both ranked lists into a single list that
captures the best of both worlds — particularly useful for enterprise docs
where exact terms (policy names, section numbers) matter alongside meaning.

RRF formula: score(d) = sum( 1 / (k + rank_i(d)) ) for each ranker i
where k=60 is the standard smoothing constant.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi

from config import settings
from src.security.access_control import matches_filter

logger = logging.getLogger(__name__)

RRF_K = settings.rrf_k

# Stop words for BM25 tokenization — high-frequency words that add noise
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "about", "up", "it", "its", "i",
    "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "they", "them", "their", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am",
})

@dataclass
class _CachedIndex:
    """A built BM25 index plus what it was built from.

    *source_count* is the size of the whole collection at build time (not
    of the filtered corpus), which is what makes cross-process staleness
    detectable: the ingestion worker's reset_bm25_cache() only clears its
    own process, so a long-lived API server would otherwise serve a
    sparse index that predates every uploaded document.
    """

    bm25: BM25Okapi
    docs: list[Document]
    built_at: float
    source_count: int


# Module-level BM25 cache: filter_hash -> _CachedIndex
_bm25_cache: dict[str, _CachedIndex] = {}
_bm25_lock = threading.Lock()

# Cache keys with a rebuild already running, so a burst of concurrent
# queries triggers one rebuild rather than one per request.
_rebuilding: set[str] = set()
# Single background worker: rebuilds are memory-heavy, and running several
# at once is how a refresh turns into an out-of-memory kill.
_rebuild_pool: ThreadPoolExecutor | None = None


def _submit_rebuild(fn) -> None:
    """Run *fn* off the request path, at most one rebuild at a time."""
    global _rebuild_pool
    if _rebuild_pool is None:
        _rebuild_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bm25-rebuild"
        )
    _rebuild_pool.submit(fn)


def shutdown_rebuilds(wait: bool = True) -> None:
    """Stop the background rebuild worker (tests, and clean shutdown)."""
    global _rebuild_pool
    pool, _rebuild_pool = _rebuild_pool, None
    if pool is not None:
        pool.shutdown(wait=wait)


def _tokenize(text: str) -> list[str]:
    """Regex-based tokenizer with stop-word filtering for BM25."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS]


def reset_bm25_cache() -> None:
    """Clear the BM25 cache (call after adding new documents)."""
    with _bm25_lock:
        _bm25_cache.clear()
        _rebuilding.clear()
    logger.debug("BM25 cache cleared")


def _collection_count() -> int:
    """Current document count, or -1 when it cannot be read."""
    try:
        from src.vectorstore.chroma_store import get_vectorstore

        return get_vectorstore()._collection.count()
    except Exception:
        logger.debug("Could not read collection count for BM25 staleness check", exc_info=True)
        return -1


def _is_stale(entry: _CachedIndex) -> bool:
    """Whether *entry* must be rebuilt before it is served.

    Two independent signals: the corpus size changed (catches almost
    every add or delete immediately, at the cost of one cheap count call
    per lookup), or the entry simply aged out (backstop for an add and a
    delete that happen to cancel out).
    """
    if time.monotonic() - entry.built_at > settings.bm25_cache_ttl:
        logger.info("BM25 index aged out after %ds — rebuilding", settings.bm25_cache_ttl)
        return True
    count = _collection_count()
    if count >= 0 and count != entry.source_count:
        logger.info(
            "Collection changed (%d -> %d documents) — rebuilding BM25 index",
            entry.source_count, count,
        )
        return True
    return False


class HybridRetriever(BaseRetriever):
    """Fuses dense (ChromaDB) and sparse (BM25) retrieval via RRF."""

    k: int = Field(default=4, description="Number of documents to return")
    fetch_k: int = Field(default=20, description="Candidates per ranker before fusion")
    filter: dict | None = Field(default=None, description="Metadata filter for dense retriever")

    _bm25: BM25Okapi | None = PrivateAttr(default=None)
    _corpus_docs: list[Document] = PrivateAttr(default_factory=list)

    @staticmethod
    def _filter_cache_key(filter: dict | None) -> str:
        """Produce a stable cache key from the metadata filter.

        Serialised with sorted keys because filter values are no longer
        always scalars — a department-scoped caller retrieves with
        ``{"department": {"$in": [...]}}``, and two equivalent filters
        must not land on two cache entries (or, worse, one filter reuse
        another's corpus).
        """
        if not filter:
            return "__no_filter__"
        import json

        return json.dumps(filter, sort_keys=True, default=str)

    def _schedule_rebuild(self, cache_key: str) -> None:
        """Refresh *cache_key* in the background, at most once at a time."""
        with _bm25_lock:
            if cache_key in _rebuilding:
                return
            _rebuilding.add(cache_key)

        def _run() -> None:
            try:
                # A detached retriever so the rebuild cannot mutate the
                # instance currently serving a request.
                worker = HybridRetriever(k=self.k, fetch_k=self.fetch_k, filter=self.filter)
                worker._build_index_now(cache_key)
            except Exception:
                logger.exception("Background BM25 rebuild failed for key=%s", cache_key)
            finally:
                with _bm25_lock:
                    _rebuilding.discard(cache_key)

        _submit_rebuild(_run)

    def _build_bm25_index(self) -> None:
        """Load all documents from ChromaDB and build a BM25 index.

        Uses a module-level cache keyed by filter hash so the index is
        built only once per filter combination.
        """
        cache_key = self._filter_cache_key(self.filter)

        # Check module-level cache first (thread-safe)
        with _bm25_lock:
            cached = _bm25_cache.get(cache_key)

        if cached is not None:
            if not _is_stale(cached):
                self._bm25, self._corpus_docs = cached.bm25, cached.docs
                logger.debug(
                    "BM25 cache hit for key=%s (%d docs)", cache_key, len(self._corpus_docs)
                )
                return

            # Stale, but usable. Serve it and refresh behind the request.
            #
            # Rebuilding here would mean every upload put a full corpus
            # load, re-tokenize and re-index on whichever unlucky query
            # arrived next — negligible at a hundred chunks, seconds of
            # CPU at a hundred thousand. A few seconds of staleness in
            # sparse ranking is a far smaller cost than a latency spike
            # on an arbitrary user's request.
            self._bm25, self._corpus_docs = cached.bm25, cached.docs
            self._schedule_rebuild(cache_key)
            return

        # Cold start: nothing cached, so there is nothing to serve while
        # a background build runs. This one has to be synchronous.
        self._build_index_now(cache_key)

    def _build_index_now(self, cache_key: str) -> None:
        """Load the corpus and build the index, blocking the caller."""
        from src.vectorstore.chroma_store import get_vectorstore

        store = get_vectorstore()
        collection = store._collection

        # The whole corpus is loaded into this process, per filter key, so
        # the ceiling is memory rather than time. Past the limit, degrade
        # to dense-only and say so — an OOM kill takes the replica with it
        # and gives no clue why.
        limit = settings.bm25_max_documents
        if limit > 0:
            count = _collection_count()
            if count > limit:
                logger.warning(
                    "Corpus of %d documents exceeds BM25_MAX_DOCUMENTS=%d — "
                    "serving dense-only. Move sparse retrieval server-side "
                    "(OpenSearch/Elasticsearch) to keep hybrid ranking at this scale.",
                    count, limit,
                )
                self._corpus_docs = []
                self._bm25 = None
                return

        try:
            result = collection.get(include=["documents", "metadatas"])
        except Exception:
            logger.exception("Failed to fetch documents from ChromaDB for BM25 index")
            self._corpus_docs = []
            self._bm25 = None
            return

        ids = result.get("ids", [])
        texts = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        if not texts:
            logger.warning("No documents in ChromaDB — BM25 index will be empty")
            self._corpus_docs = []
            self._bm25 = None
            return

        # Apply metadata filter if specified. matches_filter understands
        # the same operator forms Chroma does ($in, $ne, ...) — plain
        # equality here would match nothing for a department-scoped
        # caller, silently costing them sparse retrieval while dense
        # retrieval carried on working.
        self._corpus_docs = []
        for doc_id, text, meta in zip(ids, texts, metadatas, strict=True):
            if not matches_filter(meta or {}, self.filter):
                continue
            self._corpus_docs.append(
                Document(page_content=text, metadata=meta or {})
            )

        if not self._corpus_docs:
            logger.warning("No documents match filter — BM25 index empty")
            self._bm25 = None
            return

        tokenized = [_tokenize(d.page_content) for d in self._corpus_docs]
        if not any(tokenized):
            # Every document tokenized to nothing (all stop-words, single
            # characters, or punctuation). BM25Okapi divides by the size of
            # the idf table when that happens, so building here would raise
            # and take hybrid retrieval down; fall back to dense-only.
            logger.warning(
                "No indexable terms in %d document(s) — skipping BM25 index",
                len(self._corpus_docs),
            )
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built: %d documents", len(self._corpus_docs))

        # Store in module-level cache (thread-safe). source_count is the
        # whole collection, not the filtered corpus, so the staleness
        # check works the same for filtered and unfiltered indexes.
        with _bm25_lock:
            _bm25_cache[cache_key] = _CachedIndex(
                bm25=self._bm25,
                docs=self._corpus_docs,
                built_at=time.monotonic(),
                source_count=len(ids),
            )

    def _get_bm25_results(self, query: str, n: int) -> list[tuple[Document, float]]:
        """Return top-n BM25 results as (doc, score) pairs."""
        if self._bm25 is None:
            self._build_bm25_index()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self._corpus_docs[i], scores[i]) for i in ranked_indices]

    def _get_dense_results(self, query: str, n: int) -> list[Document]:
        """Return top-n dense retrieval results."""
        from src.vectorstore.chroma_store import get_retriever as _dense_retriever
        retriever = _dense_retriever(k=n, filter=self.filter)
        return retriever.invoke(query)

    def _rrf_fuse(
        self,
        dense_docs: list[Document],
        bm25_docs: list[tuple[Document, float]],
    ) -> list[Document]:
        """Reciprocal Rank Fusion over two ranked lists.

        Returns documents sorted by fused score, limited to self.k.
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        def _doc_key(doc: Document) -> str:
            return hashlib.md5(doc.page_content.encode()).hexdigest()

        # Score dense results by rank
        for rank, doc in enumerate(dense_docs, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_map[key] = doc

        # Score BM25 results by rank
        for rank, (doc, _bm25_score) in enumerate(bm25_docs, start=1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in doc_map:
                doc_map[key] = doc

        # Sort by fused score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[: self.k]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve documents using hybrid search (dense + BM25 + RRF)."""
        logger.info("Hybrid retrieval: %s", query[:120])

        dense_docs = self._get_dense_results(query, self.fetch_k)
        bm25_results = self._get_bm25_results(query, self.fetch_k)
        fused = self._rrf_fuse(dense_docs, bm25_results)

        logger.info(
            "Hybrid: %d dense + %d BM25 → %d fused",
            len(dense_docs), len(bm25_results), len(fused),
        )
        return fused


def build_hybrid_retriever(
    k: int | None = None,
    filter: dict | None = None,
) -> HybridRetriever:
    """Build and return a hybrid retriever."""
    return HybridRetriever(
        k=k or settings.top_k,
        filter=filter,
    )
