"""Phase 27 — P1-6 and the P2 cache scan.

Six stateful stores ran on single-node SQLite. WAL makes that correct for
one host and silently wrong for two: the registry is the idempotency
mechanism, so two workers with private copies each index the same
document; conversation memory is per-replica, so a follow-up question
routed elsewhere forgets the thread. Neither failure raises anything.

Separately, the semantic cache loaded every entry in scope on every
lookup and compared them in Python — which made a cache hit slower than
a cache miss once the table grew.
"""
from __future__ import annotations

import logging
import math
import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cache.lsh import BANDS, BITS_PER_BAND, expected_recall, signature
from src.storage.sql import (
    SqlDatabase,
    is_postgres,
    redact_dsn,
    to_postgres,
    warn_if_single_node,
)

# ---------------------------------------------------------------------------
# Dialect translation
# ---------------------------------------------------------------------------

class TestSqlTranslation:
    def test_placeholders_become_percent_s(self):
        assert to_postgres("SELECT * FROM t WHERE a = ? AND b = ?") == (
            "SELECT * FROM t WHERE a = %s AND b = %s"
        )

    def test_a_question_mark_inside_a_literal_is_left_alone(self):
        """Otherwise a stored question breaks its own query."""
        out = to_postgres("SELECT * FROM t WHERE q = 'why?' AND id = ?")
        assert out == "SELECT * FROM t WHERE q = 'why?' AND id = %s"

    def test_autoincrement_becomes_bigserial(self):
        assert "BIGSERIAL PRIMARY KEY" in to_postgres(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)"
        )
        assert "AUTOINCREMENT" not in to_postgres(
            "CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)"
        )

    def test_insert_or_ignore_gains_a_conflict_clause(self):
        """This is the registry's entire idempotency mechanism.

        Dropping the prefix without adding ON CONFLICT would turn a
        duplicate upload from a no-op into a unique-violation.
        """
        out = to_postgres("INSERT OR IGNORE INTO documents (a) VALUES (?)")
        assert out.startswith("INSERT INTO documents")
        assert out.endswith("ON CONFLICT DO NOTHING")

    def test_an_existing_conflict_clause_is_not_doubled(self):
        out = to_postgres(
            "INSERT OR IGNORE INTO t (a) VALUES (?) ON CONFLICT (a) DO NOTHING"
        )
        assert out.count("ON CONFLICT") == 1

    def test_a_plain_insert_gains_nothing(self):
        assert "ON CONFLICT" not in to_postgres("INSERT INTO t (a) VALUES (?)")

    def test_a_trailing_semicolon_does_not_break_the_clause(self):
        out = to_postgres("INSERT OR IGNORE INTO t (a) VALUES (?);")
        assert out.endswith("ON CONFLICT DO NOTHING")


class TestBackendSelection:
    @pytest.mark.parametrize(
        "dsn,postgres",
        [
            ("postgresql://u:p@h/db", True),
            ("postgres://u:p@h/db", True),
            ("", False),
            ("./checkpoints/documents.db", False),
        ],
    )
    def test_dsn_selects_the_backend(self, dsn, postgres):
        assert is_postgres(dsn) is postgres

    def test_sqlite_is_the_default(self):
        with patch("src.storage.sql.settings") as s:
            s.database_url = ""
            assert is_postgres() is False

    @pytest.mark.parametrize(
        "dsn,expected",
        [
            ("postgresql://rag:secret@db:5432/rag", "postgresql://***@db:5432/rag"),
            ("postgresql://db:5432/rag", "postgresql://db:5432/rag"),
        ],
    )
    def test_credentials_are_redacted_before_logging(self, dsn, expected):
        assert redact_dsn(dsn) == expected

    def test_a_missing_driver_is_reported_clearly(self):
        import builtins

        from src.storage.sql import DatabaseUnavailable

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "psycopg":
                raise ImportError("no psycopg")
            return real_import(name, *args, **kwargs)

        with patch("src.storage.sql.settings") as s:
            s.database_url = "postgresql://u:p@h/db"
            with patch.object(builtins, "__import__", blocked):
                with pytest.raises(DatabaseUnavailable, match="psycopg is not installed"):
                    SqlDatabase("unused")


class TestSingleNodeWarning:
    def test_it_warns_when_the_deployment_looks_real(self, caplog):
        """Silently unshared state looks healthy until the second replica."""
        with patch("src.storage.sql.settings") as s:
            s.database_url = ""
            s.auth_enabled = True
            s.async_ingestion = False
            with caplog.at_level(logging.WARNING):
                warn_if_single_node()
        assert "ONE host" in caplog.text

    def test_it_warns_when_a_separate_worker_is_running(self, caplog):
        with patch("src.storage.sql.settings") as s:
            s.database_url = ""
            s.auth_enabled = False
            s.async_ingestion = True
            with caplog.at_level(logging.WARNING):
                warn_if_single_node()
        assert "ONE host" in caplog.text

    def test_it_is_quiet_for_single_tenant_development(self, caplog):
        with patch("src.storage.sql.settings") as s:
            s.database_url = ""
            s.auth_enabled = False
            s.async_ingestion = False
            with caplog.at_level(logging.WARNING):
                warn_if_single_node()
        assert caplog.text == ""

    def test_it_is_quiet_when_postgres_is_configured(self, caplog):
        with patch("src.storage.sql.settings") as s:
            s.database_url = "postgresql://u:p@h/db"
            s.auth_enabled = True
            s.async_ingestion = True
            with caplog.at_level(logging.WARNING):
                warn_if_single_node()
        assert caplog.text == ""


# ---------------------------------------------------------------------------
# The stores still work on SQLite
# ---------------------------------------------------------------------------

class TestSqliteUnchanged:
    @pytest.fixture(autouse=True)
    def _sqlite(self):
        with patch("src.storage.sql.settings") as s:
            s.database_url = ""
            yield

    def test_the_registry_still_deduplicates(self, tmp_path):
        from src.ingestion.registry import DocumentRegistry

        registry = DocumentRegistry(db_path=tmp_path / "docs.db")
        try:
            first, created = registry.register_upload(
                checksum="abc", filename="a.pdf", department="hr"
            )
            second, created_again = registry.register_upload(
                checksum="abc", filename="a.pdf", department="hr"
            )
            assert created is True and created_again is False
            assert first.document_id == second.document_id
        finally:
            registry.close()

    def test_conversation_history_round_trips(self, tmp_path):
        from src.memory.conversation_store import ConversationStore

        store = ConversationStore(str(tmp_path / "conv.db"))
        store.add_message("s1", "user", "hello")
        store.add_message("s1", "assistant", "hi")
        assert len(store.get_history("s1")) == 2
        assert store.get_history("s2") == []

    def test_metrics_round_trip(self, tmp_path):
        from dataclasses import dataclass

        from src.observability.metrics_store import MetricsStore

        @dataclass
        class M:
            thread_id: str = "t"
            question_preview: str = "why?"
            mode: str = "graph"
            retriever_strategy: str = "dense"
            prompt_tokens: int = 1
            completion_tokens: int = 1
            total_tokens: int = 2
            estimated_cost_usd: float = 0.001
            latency_ms: float = 5.0
            is_idk: bool = False
            grader_rejected: int = 0

        store = MetricsStore(str(tmp_path / "m.db"))
        store.record(M())
        assert len(store.query_recent(5)) == 1
        store.close()


# ---------------------------------------------------------------------------
# The same stores on a real PostgreSQL
# ---------------------------------------------------------------------------

@pytest.fixture
def postgres_dsn():
    """DSN for the Postgres in docker-compose.test.yml, or skip."""
    import os

    dsn = os.getenv("TEST_DATABASE_URL", "postgresql://rag:ragtest@localhost:5434/rag")
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"PostgreSQL not reachable at {dsn}: {e}")
    return dsn


@pytest.mark.integration
class TestPostgresBackend:
    """SQLite passing only proves the SQL is valid for SQLite."""

    @pytest.fixture(autouse=True)
    def _postgres(self, postgres_dsn):
        with patch("src.storage.sql.settings") as s:
            s.database_url = postgres_dsn
            yield

    def _fresh(self, table):
        db = SqlDatabase("unused")
        db.execute(f"DROP TABLE IF EXISTS {table}")
        db.commit()
        db.close()

    def test_the_registry_deduplicates_on_postgres(self):
        """INSERT OR IGNORE must survive translation, or duplicates raise."""
        from src.ingestion.registry import DocumentRegistry

        self._fresh("documents")
        registry = DocumentRegistry(db_path="unused")
        try:
            _, created = registry.register_upload(
                checksum="pg-abc", filename="a.pdf", department="hr"
            )
            _, again = registry.register_upload(
                checksum="pg-abc", filename="a.pdf", department="hr"
            )
            assert created is True and again is False
        finally:
            registry.close()

    def test_the_claim_is_still_exclusive_on_postgres(self):
        """The conditional UPDATE is what makes two workers safe."""
        from src.ingestion.registry import DocumentRegistry

        self._fresh("documents")
        registry = DocumentRegistry(db_path="unused")
        try:
            record, _ = registry.register_upload(
                checksum="pg-claim", filename="a.pdf", department="hr"
            )
            assert registry.claim_for_processing(record.document_id) is True
            assert registry.claim_for_processing(record.document_id) is False
        finally:
            registry.close()

    def test_conversation_history_round_trips_on_postgres(self):
        from src.memory.conversation_store import ConversationStore

        self._fresh("conversation_history")
        store = ConversationStore("unused")
        store.add_message("pg-s1", "user", "hello")
        store.add_message("pg-s1", "assistant", "hi")
        assert len(store.get_history("pg-s1")) == 2

    def test_metrics_aggregate_on_postgres(self):
        from dataclasses import dataclass

        from src.observability.metrics_store import MetricsStore

        self._fresh("query_metrics")

        @dataclass
        class M:
            thread_id: str = "t"
            question_preview: str = "why?"  # literal question mark
            mode: str = "graph"
            retriever_strategy: str = "dense"
            prompt_tokens: int = 10
            completion_tokens: int = 5
            total_tokens: int = 15
            estimated_cost_usd: float = 0.002
            latency_ms: float = 100.0
            is_idk: bool = False
            grader_rejected: int = 0

        store = MetricsStore("unused")
        for _ in range(3):
            store.record(M())
        assert store.summary()["cnt"] == 3
        assert store.spend_today() == pytest.approx(0.006)
        store.close()

    def test_two_handles_share_one_registry(self):
        """The whole point: two workers, one source of truth."""
        from src.ingestion.registry import DocumentRegistry

        self._fresh("documents")
        worker_a = DocumentRegistry(db_path="unused")
        worker_b = DocumentRegistry(db_path="unused")
        try:
            record, created = worker_a.register_upload(
                checksum="pg-shared", filename="a.pdf", department="hr"
            )
            assert created is True
            # The second worker sees it, so it will not index it again.
            assert worker_b.get(record.document_id) is not None
            assert worker_a.claim_for_processing(record.document_id) is True
            assert worker_b.claim_for_processing(record.document_id) is False
        finally:
            worker_a.close()
            worker_b.close()


# ---------------------------------------------------------------------------
# The semantic cache is no longer a full scan
# ---------------------------------------------------------------------------

class TestLshSignature:
    def _vector(self, seed, dim=256):
        rng = random.Random(seed)
        return [rng.gauss(0, 1) for _ in range(dim)]

    def _cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    def test_a_signature_has_one_hash_per_band(self):
        assert len(signature(self._vector(1))) == BANDS

    def test_it_is_deterministic(self):
        """Per-process randomness would make one replica's entries invisible."""
        vector = self._vector(1)
        assert signature(vector) == signature(vector)

    def test_band_values_are_namespaced_by_index(self):
        """So band 2 of a query cannot match band 0 of a stored row."""
        bands = signature(self._vector(1))
        for index, value in enumerate(bands):
            assert value >> BITS_PER_BAND == index

    def test_near_vectors_share_a_band(self):
        base = self._vector(1)
        rng = random.Random(99)
        near = [x + rng.gauss(0, 0.02) for x in base]
        assert self._cosine(base, near) > 0.99
        assert set(signature(base)) & set(signature(near))

    def test_unrelated_vectors_usually_do_not(self):
        collisions = sum(
            1 for seed in range(40)
            if set(signature(self._vector(1))) & set(signature(self._vector(1000 + seed)))
        )
        assert collisions < 8, "pruning is too weak to be worth the index"

    def test_degenerate_vectors_produce_no_signature(self):
        """Bucketing them together would make every one of them collide."""
        assert signature([]) == []
        assert signature([0.0] * 16) == []

    def test_the_parameters_keep_recall_high_at_the_default_threshold(self):
        """A miss is only a recomputation, but a cache that misses is pointless."""
        assert expected_recall(0.95) > 0.85
        assert expected_recall(0.99) > 0.95

    def test_recall_falls_away_for_unrelated_vectors(self):
        assert expected_recall(0.5) < 0.25


class TestCacheUsesTheIndex:
    class _Embed:
        """Deterministic embeddings; '|' marks a near-duplicate."""

        def embed_query(self, text):
            base_rng = random.Random(hash(text.split("|")[0]) & 0xFFFFFFFF)
            vector = [base_rng.gauss(0, 1) for _ in range(128)]
            if "|" in text:
                noise = random.Random(text)
                vector = [x + noise.gauss(0, 0.03) for x in vector]
            return vector

    @pytest.fixture
    def cache(self, tmp_path):
        with (
            patch("src.storage.sql.settings") as sql_settings,
            patch("src.cache.semantic_cache.settings") as cache_settings,
        ):
            sql_settings.database_url = ""
            cache_settings.semantic_cache_enabled = True
            cache_settings.semantic_cache_threshold = 0.90
            cache_settings.semantic_cache_ttl = 3600
            from src.cache.semantic_cache import SemanticCache

            yield SemanticCache(str(tmp_path / "cache.db"), embed_fn=self._Embed())

    def test_bands_are_written_with_each_entry(self, cache):
        cache.store("what is the leave policy", "Three days.", scope="")
        with cache._lock:
            row = cache._conn.execute(
                "SELECT band0, band1 FROM semantic_cache"
            ).fetchone()
        assert row["band0"] is not None and row["band1"] is not None

    def test_an_exact_repeat_still_hits(self, cache):
        cache.store("what is the leave policy", "Three days.", scope="")
        assert cache.lookup("what is the leave policy", scope="") == "Three days."

    def test_a_paraphrase_still_hits(self, cache):
        """Pruning must not cost the cache its actual purpose."""
        cache.store("what is the leave policy", "Three days.", scope="")
        assert cache.lookup("what is the leave policy|reworded", scope="") == "Three days."

    def test_an_unrelated_query_misses(self, cache):
        cache.store("what is the leave policy", "Three days.", scope="")
        assert cache.lookup("how do I reset my password", scope="") is None

    def test_scope_still_isolates_entries(self, cache):
        """Bucketing must not become a way around the access boundary."""
        cache.store("what is the leave policy", "HR answer", scope="hr")
        assert cache.lookup("what is the leave policy", scope="legal") is None
        assert cache.lookup("what is the leave policy", scope="hr") == "HR answer"

    def test_only_a_fraction_of_the_table_is_compared(self, cache):
        """The defect: every lookup loaded and compared every entry."""
        for i in range(300):
            cache.store(f"unrelated question {i}", f"answer {i}", scope="")
        cache.store("what is the leave policy", "Three days.", scope="")

        bands = signature(self._Embed().embed_query("what is the leave policy"))
        clause = " OR ".join(f"band{i} = ?" for i in range(len(bands)))
        with cache._lock:
            total = cache._conn.execute(
                "SELECT COUNT(*) AS n FROM semantic_cache"
            ).fetchone()["n"]
            candidates = cache._conn.execute(
                f"SELECT COUNT(*) AS n FROM semantic_cache WHERE scope = ? AND ({clause})",
                ("", *bands),
            ).fetchone()["n"]

        assert total == 301
        assert candidates < total * 0.25, (
            f"{candidates} of {total} candidates — the index is not pruning"
        )
        # ...and the entry is still found among them.
        assert cache.lookup("what is the leave policy", scope="") == "Three days."

    def test_an_entry_written_before_the_index_existed_is_only_a_miss(self, cache):
        """Not a wrong answer: it is recomputed and rewritten with bands."""
        cache.store("legacy question", "old answer", scope="")
        with cache._lock:
            cache._conn.execute(
                "UPDATE semantic_cache SET "
                + ", ".join(f"band{i} = NULL" for i in range(BANDS))
            )
            cache._conn.commit()
        assert cache.lookup("legacy question", scope="") is None
