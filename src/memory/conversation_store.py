"""SQLite-backed conversation history store.

Stores (session_id, role, content, timestamp) tuples for multi-turn
conversations. Follows the singleton-with-lock pattern used throughout
the codebase (metrics_store.py, semantic_cache.py).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS conversation_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_conv_session
ON conversation_history (session_id, id);
"""


class ConversationStore:
    """Thread-safe SQLite store for conversation history."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()
        self._lock = threading.Lock()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversation_history (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, datetime.now(UTC).isoformat()),
            )
            self._conn.commit()

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Return conversation history for a session, oldest first.

        Each dict has keys: role, content, created_at.
        When limit is given, returns only the most recent N messages.
        """
        max_turns = limit or settings.memory_max_turns
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM conversation_history "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, max_turns),
            ).fetchall()
        # Reverse so oldest is first (chronological order)
        return [dict(r) for r in reversed(rows)]

    def clear_session(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM conversation_history WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            return cur.rowcount

    def session_count(self) -> int:
        """Return the number of distinct sessions."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT session_id) AS cnt FROM conversation_history"
            ).fetchone()
            return row["cnt"] if row else 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: ConversationStore | None = None
_store_lock = threading.Lock()


def _default_db_path() -> str:
    checkpoint_dir = Path(settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / "conversations.db")


def get_conversation_store(db_path: str | None = None) -> ConversationStore:
    """Return the singleton ConversationStore. Thread-safe."""
    global _store
    with _store_lock:
        if _store is None:
            _store = ConversationStore(db_path or _default_db_path())
            logger.info("ConversationStore initialized")
        return _store


def reset_conversation_store() -> None:
    """Close and discard the singleton (for testing)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
            _store = None
