"""One SQL surface over SQLite and PostgreSQL.

Six stateful stores ran on single-node SQLite. WAL makes that correct for
one host, and completely wrong for two: the document registry is what
makes concurrent workers idempotent, and conversation memory is what lets
a user's second question reach a different API replica and still find the
first. On separate hosts with separate files, both silently stop working
— no error, just duplicated indexing and forgotten conversations.

This is a compatibility layer, not an ORM. The stores keep their hand-
written SQL, which is already simple and portable; what differs between
the two engines is small and mechanical:

* parameter style — ``?`` versus ``%s``
* autoincrementing keys — ``INTEGER PRIMARY KEY AUTOINCREMENT`` versus
  ``BIGSERIAL PRIMARY KEY``
* upsert spelling — ``INSERT OR IGNORE`` versus ``ON CONFLICT DO NOTHING``
* rows addressable by column name, which SQLite needs told and psycopg
  needs configured

Translating those four keeps one set of queries and one set of tests.
SQLite stays the default, so a single-node install needs no database.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

POSTGRES_SCHEMES = ("postgres://", "postgresql://")


class DatabaseUnavailable(RuntimeError):
    """Postgres was configured but cannot be used."""


def database_url() -> str:
    """Configured Postgres DSN, or "" for per-store SQLite files."""
    return settings.database_url.strip()


def is_postgres(dsn: str | None = None) -> bool:
    return (dsn if dsn is not None else database_url()).startswith(POSTGRES_SCHEMES)


# ---------------------------------------------------------------------------
# SQL translation
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"\?(?=(?:[^']*'[^']*')*[^']*$)")


def to_postgres(sql: str) -> str:
    """Rewrite SQLite-flavoured SQL for PostgreSQL.

    Placeholders are only substituted outside string literals, so a query
    containing a literal question mark is left intact.
    """
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "BIGSERIAL PRIMARY KEY")

    # `INSERT OR IGNORE` is the whole idempotency mechanism in the
    # registry — dropping it to a bare INSERT would turn a duplicate
    # upload into a unique-violation instead of a no-op, so the conflict
    # clause has to be added, not just the prefix removed.
    ignore_conflict = "INSERT OR IGNORE INTO" in sql
    sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
    sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    sql = re.sub(r"\bAUTOINCREMENT\b", "", sql)
    sql = _PLACEHOLDER.sub("%s", sql)
    if ignore_conflict and "ON CONFLICT" not in sql.upper():
        sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    return sql


def translate(sql: str, postgres: bool) -> str:
    return to_postgres(sql) if postgres else sql


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

class SqlDatabase:
    """A connection plus the dialect knowledge to use it.

    Deliberately holds a single connection guarded by a lock, matching
    what the stores already did with SQLite. Pooling belongs at the
    process edge (pgbouncer) rather than inside each store, and adding it
    here would change the concurrency behaviour the stores were written
    and tested against.
    """

    def __init__(self, sqlite_path: str, dsn: str | None = None) -> None:
        self._dsn = database_url() if dsn is None else dsn
        self.postgres = is_postgres(self._dsn)
        self._lock = threading.Lock()
        self._conn = self._connect(sqlite_path)

    def _connect(self, sqlite_path: str):
        if not self.postgres:
            conn = sqlite3.connect(sqlite_path, check_same_thread=False, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            return conn

        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as e:
            raise DatabaseUnavailable(
                "DATABASE_URL points at PostgreSQL but psycopg is not installed "
                "(pip install 'psycopg[binary]')."
            ) from e

        try:
            conn = psycopg.connect(self._dsn, row_factory=dict_row, autocommit=False)
        except Exception as e:
            raise DatabaseUnavailable(
                f"Could not connect to PostgreSQL at {redact_dsn(self._dsn)}: {e}"
            ) from e
        logger.info("Connected to PostgreSQL at %s", redact_dsn(self._dsn))
        return conn

    # -- statements -------------------------------------------------------
    def execute(self, sql: str, params: tuple | list = ()) -> Any:
        cursor = self._conn.execute(translate(sql, self.postgres), tuple(params))
        return cursor

    def executescript(self, statements: list[str]) -> None:
        """Run DDL, tolerating statements that a fresh schema already has."""
        for statement in statements:
            try:
                self.execute(statement)
                self.commit()
            except Exception as e:
                self.rollback()
                logger.debug("DDL skipped (%s): %s", type(e).__name__, e)

    def fetchone(self, sql: str, params: tuple | list = ()) -> dict | None:
        row = self.execute(sql, params).fetchone()
        return self.row_to_dict(row)

    def fetchall(self, sql: str, params: tuple | list = ()) -> list[dict]:
        rows = self.execute(sql, params).fetchall()
        return [self.row_to_dict(r) for r in rows if r is not None]

    @staticmethod
    def row_to_dict(row) -> dict | None:
        if row is None:
            return None
        return dict(row)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        try:
            self._conn.rollback()
        except Exception:
            logger.debug("Rollback failed", exc_info=True)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            logger.debug("Close failed", exc_info=True)

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    @property
    def raw(self):
        """The underlying DB-API connection, for store-specific needs."""
        return self._conn


def redact_dsn(dsn: str) -> str:
    """Strip credentials from a DSN before it reaches the logs."""
    if "@" not in dsn:
        return dsn
    scheme, _, rest = dsn.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


def warn_if_single_node() -> None:
    """Say plainly when shared state is not actually shared.

    Every store falling back to a local file looks identical to a working
    deployment right up until a second replica exists, at which point
    documents index twice and conversations lose their history.
    """
    if is_postgres():
        return
    if settings.auth_enabled or settings.async_ingestion:
        logger.warning(
            "Stateful stores are on local SQLite files. Correct for ONE host: "
            "a second API replica or a worker on another machine gets its own "
            "registry and its own conversation history. Set DATABASE_URL to a "
            "PostgreSQL DSN before scaling out."
        )
