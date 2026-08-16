"""CLI script to clean up stale documents from the vector store.

Usage:
    python -m scripts.cleanup_stale --max-age-days 30
    python -m scripts.cleanup_stale --max-age-days 30 --dry-run
"""
from __future__ import annotations

import argparse

from config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up stale documents from ChromaDB")
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=0,
        help="Delete documents older than this many days (0 = use DOCUMENT_TTL_DAYS from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report stale documents without deleting them",
    )
    args = parser.parse_args()

    setup_logging()

    from src.vectorstore.chroma_store import delete_stale_documents, get_stale_documents

    max_age = args.max_age_days if args.max_age_days > 0 else None

    if args.dry_run:
        stale_ids = get_stale_documents(max_age)
        if stale_ids:
            print(f"Found {len(stale_ids)} stale document(s) that would be deleted:")
            for doc_id in stale_ids[:20]:
                print(f"  - {doc_id}")
            if len(stale_ids) > 20:
                print(f"  ... and {len(stale_ids) - 20} more")
        else:
            print("No stale documents found.")
        return

    deleted = delete_stale_documents(max_age)
    if deleted:
        print(f"Deleted {deleted} stale document(s).")
    else:
        print("No stale documents to delete.")


if __name__ == "__main__":
    main()
