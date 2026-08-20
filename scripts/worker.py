"""Ingestion worker entrypoint.

Run alongside the API to index uploaded documents off the request path:

    python -m scripts.worker

Scale by running more copies. With ``EVENT_BUS=kafka`` they form a consumer
group and Kafka splits the partitions between them; with the SQLite bus
they compete for rows in one queue file, so they must share
``EVENT_BUS_PATH`` and stay on one host.

Shuts down cleanly on SIGINT/SIGTERM: the loop stops claiming new events
and the in-flight document is allowed to finish, so ``docker stop`` never
tears an index write in half.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from config import settings, setup_logging

logger = logging.getLogger("ingestion.worker")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise RAG ingestion worker")
    parser.add_argument(
        "--topic", default=None,
        help=f"Topic to consume (default: {settings.kafka_topic_ingestion})",
    )
    parser.add_argument(
        "--group", default=None,
        help=f"Consumer group (default: {settings.kafka_consumer_group})",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help=f"Attempts before dead-lettering (default: {settings.ingest_max_attempts})",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Drain the currently queued events and exit (useful in CI).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()
    settings.validate()

    # Named separately from the API so spans on both sides of the queue
    # are attributable to the process that produced them.
    from src.observability import tracing_otel as otel

    otel.setup_tracing("rag-worker")

    from src.ingestion.worker import IngestionWorker

    worker = IngestionWorker(
        topic=args.topic,
        group=args.group,
        max_attempts=args.max_attempts,
    )

    if args.once:
        outcomes: list[str] = []
        while True:
            batch = worker.poll_once(timeout_s=0.1)
            if not batch:
                break
            outcomes.extend(batch)
        logger.info("Drained %d event(s): %s", len(outcomes), worker.stats.to_dict())
        return 0

    stop = threading.Event()

    def _shutdown(signum, _frame):
        logger.info("Received signal %s — finishing current document then exiting", signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):
            # Not on the main thread, or unsupported on this platform.
            logger.debug("Could not install handler for signal %s", sig)

    try:
        worker.run(stop_event=stop)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        from src.events.bus import reset_event_bus

        reset_event_bus()
    return 0


if __name__ == "__main__":
    sys.exit(main())
