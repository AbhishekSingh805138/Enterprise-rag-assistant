"""Drain the dead-letter queue and requeue documents for indexing.

Dead-lettered documents still have their bytes in object storage, so they
can be retried without asking anyone to upload again.

Usage:
    python -m scripts.replay_dlq --dry-run           # preview, drains nothing
    python -m scripts.replay_dlq                     # redrive the whole DLQ
    python -m scripts.replay_dlq --limit 10
    python -m scripts.replay_dlq --document-id doc_abc123
    python -m scripts.replay_dlq --list              # show what is stuck

Requires a worker to be running, or replayed documents simply sit in
PENDING.
"""
from __future__ import annotations

import argparse
import sys

from config import setup_logging


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay dead-lettered documents.")
    parser.add_argument("--document-id", default=None,
                        help="Replay a single document instead of draining the DLQ")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum events to redrive (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be replayed; drains nothing")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="List dead-lettered documents and exit")
    parser.add_argument("--force", action="store_true",
                        help="Replay regardless of current status (use with care)")
    return parser.parse_args(argv)


def _print_results(results) -> None:
    if not results:
        print("  Nothing to replay — the dead-letter queue is empty.")
        return
    width = max(len(r.filename or r.document_id) for r in results)
    for r in results:
        label = r.filename or r.document_id
        print(f"  {label:<{width}}  {r.outcome:<18} {r.detail}")
    requeued = sum(1 for r in results if r.requeued)
    print(f"\n  {len(results)} event(s) processed, {requeued} requeued.")
    if requeued:
        print("  A worker must be running for these to be indexed:")
        print("    python -m scripts.worker")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging()

    from src.ingestion.registry import STATUS_DEAD_LETTER, get_registry
    from src.ingestion.replay import redrive_dlq, replay_document

    if args.list_only:
        stuck = get_registry().list_documents(status=STATUS_DEAD_LETTER, limit=args.limit)
        if not stuck:
            print("  No dead-lettered documents.")
            return 0
        print(f"  {len(stuck)} dead-lettered document(s):\n")
        for r in stuck:
            print(f"  {r.document_id}  {r.filename:<28} dept={r.department:<12} "
                  f"attempts={r.attempts}")
            if r.error:
                print(f"      last error: {r.error[:110]}")
        return 0

    try:
        if args.document_id:
            result = replay_document(args.document_id, force=args.force)
            _print_results([result])
            return 0 if result.requeued else 1

        results = redrive_dlq(limit=args.limit, dry_run=args.dry_run)
        _print_results(results)
        if args.dry_run:
            print("  Dry run — the dead-letter queue was not drained.")
        return 0
    except Exception as e:
        print(f"Replay failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
