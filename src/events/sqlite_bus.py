"""Durable SQLite-backed event queue — the default transport.

Kafka is the right answer at scale, but requiring a broker to be running
before anyone can upload a file makes the project hard to run and hard to
test. This backend gives the same delivery semantics the worker depends on
— durability across restarts, at-least-once delivery, visibility timeouts
so a crashed worker's in-flight event is redelivered, and delayed
redelivery for backoff — in a single file with no services to start.

Swap ``EVENT_BUS=kafka`` to scale out; the worker code does not change.

Concurrency: the API and the worker are separate processes sharing one
database file, so the connection runs in WAL mode with a busy timeout.
Claiming is a conditional ``UPDATE`` guarded by a rowcount check, which
SQLite serialises — two workers cannot claim the same event.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from config import settings
from src.events.bus import Event, EventBusError, IncompatibleSchemaError

# How long to hide an event this worker is too old to read, so it does not
# spin on it while the rest of the rollout completes.
_SCHEMA_RETRY_DELAY_S = 60.0

logger = logging.getLogger(__name__)

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    topic          TEXT    NOT NULL,
    event_id       TEXT    NOT NULL,
    body           TEXT    NOT NULL,
    status         TEXT    NOT NULL DEFAULT 'pending',
    available_at   REAL    NOT NULL DEFAULT 0,
    locked_until   REAL    NOT NULL DEFAULT 0,
    consumer_group TEXT    NOT NULL DEFAULT '',
    created_at     REAL    NOT NULL DEFAULT 0
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_claim ON events(topic, status, available_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id)",
]


class SQLiteDelivery:
    """A claimed event, settled via ack/nack against the owning bus."""

    def __init__(self, bus: SQLiteEventBus, row_id: int, event: Event) -> None:
        self._bus = bus
        self._row_id = row_id
        self.event = event
        self._settled = False

    @property
    def settled(self) -> bool:
        return self._settled

    def ack(self) -> None:
        if self._settled:
            return
        self._bus._delete(self._row_id)
        self._settled = True

    def nack(self, delay_s: float = 0.0) -> None:
        if self._settled:
            return
        self._bus._release(self._row_id, delay_s)
        self._settled = True


class SQLiteEventBus:
    """Durable queue with visibility timeouts, backed by one SQLite file."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path or settings.event_bus_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets the API publish while the worker polls, in separate
            # processes, without either blocking on the other's transaction.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute(_CREATE_TABLE)
            for stmt in _INDEXES:
                self._conn.execute(stmt)
            self._conn.commit()
        logger.info("SQLite event bus ready at %s", self._db_path)

    # -- producer ----------------------------------------------------------

    def publish(self, topic: str, event: Event, delay_s: float = 0.0) -> None:
        """Append *event* to *topic*, visible after *delay_s* seconds."""
        if not topic:
            raise EventBusError("topic is required")
        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO events (topic, event_id, body, status, available_at, "
                    "locked_until, consumer_group, created_at) "
                    "VALUES (?, ?, ?, 'pending', ?, 0, '', ?)",
                    (topic, event.event_id, event.to_json(), now + max(0.0, delay_s), now),
                )
                self._conn.commit()
        except sqlite3.Error as e:
            raise EventBusError(f"Failed to publish to {topic!r}: {e}") from e
        logger.debug(
            "Published %s to %s (event_id=%s attempt=%d)",
            event.event_type, topic, event.event_id, event.attempt,
        )

    # -- consumer ----------------------------------------------------------

    def poll(
        self, topic: str, group: str, max_messages: int = 1, timeout_s: float = 1.0
    ) -> list[SQLiteDelivery]:
        """Claim up to *max_messages* events, making them invisible to peers.

        Returns immediately with whatever is available; *timeout_s* is
        accepted for protocol compatibility with the Kafka backend, which
        blocks. The worker owns its own sleep between empty polls.
        """
        self._reclaim_expired(topic)
        now = time.time()
        visibility = settings.ingest_visibility_timeout_s
        claimed: list[SQLiteDelivery] = []

        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT id, body FROM events "
                    "WHERE topic = ? AND status = 'pending' AND available_at <= ? "
                    "ORDER BY id LIMIT ?",
                    (topic, now, max(1, max_messages)),
                ).fetchall()

                for row in rows:
                    # Conditional update: if a peer claimed this row between
                    # the SELECT and here, rowcount is 0 and we skip it.
                    cur = self._conn.execute(
                        "UPDATE events SET status = 'inflight', locked_until = ?, "
                        "consumer_group = ? WHERE id = ? AND status = 'pending'",
                        (now + visibility, group, row["id"]),
                    )
                    if cur.rowcount != 1:
                        continue
                    try:
                        event = Event.from_json(row["body"])
                    except IncompatibleSchemaError as e:
                        # A newer publisher during a rolling deploy. The
                        # event is valid — this worker is simply too old
                        # to read it — so it must NOT be discarded like a
                        # malformed one. Release it and let an upgraded
                        # worker pick it up.
                        logger.error(
                            "Leaving event row id=%s on topic=%s for an upgraded "
                            "worker: %s", row["id"], topic, e,
                        )
                        self._conn.execute(
                            "UPDATE events SET status = 'pending', available_at = ?, "
                            "locked_until = 0 WHERE id = ?",
                            (now + _SCHEMA_RETRY_DELAY_S, row["id"]),
                        )
                        continue
                    except EventBusError:
                        # Undecodable payload will never succeed; drop it here
                        # rather than let it cycle through the retry budget.
                        logger.error(
                            "Discarding malformed event row id=%s on topic=%s",
                            row["id"], topic,
                        )
                        self._conn.execute("DELETE FROM events WHERE id = ?", (row["id"],))
                        continue
                    claimed.append(SQLiteDelivery(self, row["id"], event))
                self._conn.commit()
        except sqlite3.Error as e:
            raise EventBusError(f"Failed to poll {topic!r}: {e}") from e

        return claimed

    def _reclaim_expired(self, topic: str) -> None:
        """Return events whose visibility timeout lapsed back to pending.

        This is the crash-recovery path: a worker that dies mid-processing
        never acks, so its event becomes visible again after the timeout.
        """
        now = time.time()
        try:
            with self._lock:
                cur = self._conn.execute(
                    "UPDATE events SET status = 'pending', locked_until = 0 "
                    "WHERE topic = ? AND status = 'inflight' AND locked_until < ?",
                    (topic, now),
                )
                self._conn.commit()
            if cur.rowcount:
                logger.warning(
                    "Reclaimed %d abandoned in-flight event(s) on topic=%s",
                    cur.rowcount, topic,
                )
        except sqlite3.Error:
            logger.debug("Failed to reclaim expired events", exc_info=True)

    # -- settlement --------------------------------------------------------

    def _delete(self, row_id: int) -> None:
        try:
            with self._lock:
                self._conn.execute("DELETE FROM events WHERE id = ?", (row_id,))
                self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to ack event row id=%s", row_id)

    def _release(self, row_id: int, delay_s: float) -> None:
        now = time.time()
        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE events SET status = 'pending', available_at = ?, "
                    "locked_until = 0 WHERE id = ?",
                    (now + max(0.0, delay_s), row_id),
                )
                self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to nack event row id=%s", row_id)

    # -- introspection -----------------------------------------------------

    def depth(self, topic: str) -> int:
        """Number of events on *topic* not yet acknowledged.

        Used for the DLQ gauge in the health check — a non-zero dead-letter
        depth is the signal that documents need operator attention.
        """
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM events WHERE topic = ?", (topic,)
                ).fetchone()
            return int(row["n"]) if row else 0
        except sqlite3.Error:
            logger.debug("Failed to read depth for topic=%s", topic, exc_info=True)
            return -1

    def peek(self, topic: str, limit: int = 20) -> list[Event]:
        """Return events on *topic* without claiming them (DLQ inspection)."""
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT body FROM events WHERE topic = ? ORDER BY id LIMIT ?",
                    (topic, limit),
                ).fetchall()
        except sqlite3.Error:
            logger.debug("Failed to peek topic=%s", topic, exc_info=True)
            return []

        events = []
        for row in rows:
            try:
                events.append(Event.from_json(row["body"]))
            except EventBusError:
                continue
        return events

    def purge(self, topic: str | None = None) -> int:
        """Delete events (all, or just *topic*). Returns rows removed."""
        try:
            with self._lock:
                if topic:
                    cur = self._conn.execute("DELETE FROM events WHERE topic = ?", (topic,))
                else:
                    cur = self._conn.execute("DELETE FROM events")
                self._conn.commit()
                return cur.rowcount
        except sqlite3.Error:
            logger.exception("Failed to purge events")
            return 0

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                logger.debug("Error closing event bus connection", exc_info=True)
