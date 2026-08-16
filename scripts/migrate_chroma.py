"""Copy an embedded ChromaDB collection into a ChromaDB server.

Adopting CHROMA_MODE=server leaves any corpus previously indexed into the
local ./chroma_db directory behind. This moves it across.

Stored embeddings are copied verbatim rather than recomputed, so the
migration costs nothing in API spend and the vectors are bit-identical —
re-embedding would also drift if the embedding model had changed since
the documents were first indexed.

Chunk ids are content hashes, so the copy is idempotent: running it twice,
or against a target that already holds some of the same documents, adds
each chunk exactly once.

Usage:
    python -m scripts.migrate_chroma --dry-run
    python -m scripts.migrate_chroma
    python -m scripts.migrate_chroma --source ./chroma_db \
        --host 127.0.0.1 --port 8001 --collection enterprise_docs
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("migrate_chroma")

# Chunks per round trip. Large enough to keep the transfer quick, small
# enough that one request stays well inside the server's payload limit
# (1536 floats per embedding adds up fast).
DEFAULT_BATCH = 200


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from config import settings

    parser = argparse.ArgumentParser(
        description="Migrate an embedded Chroma collection to a Chroma server."
    )
    parser.add_argument("--source", default=settings.chroma_dir,
                        help="Embedded persistence directory to read from")
    parser.add_argument("--host", default=settings.chroma_host, help="Target server host")
    parser.add_argument("--port", type=int, default=settings.chroma_port, help="Target server port")
    parser.add_argument("--ssl", action="store_true", help="Use https for the target")
    parser.add_argument("--collection", default=settings.chroma_collection,
                        help="Collection name (same on both ends)")
    parser.add_argument("--target-collection", default=None,
                        help="Override the collection name on the target")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help=f"Chunks per request (default: {DEFAULT_BATCH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be copied, write nothing")
    return parser.parse_args(argv)


def _fetch_page(collection, limit: int, offset: int) -> dict:
    """Read one page of chunks, embeddings included."""
    return collection.get(
        limit=limit, offset=offset, include=["documents", "metadatas", "embeddings"]
    )


def migrate(
    source_path: str,
    host: str,
    port: int,
    collection_name: str,
    target_collection_name: str | None = None,
    batch: int = DEFAULT_BATCH,
    ssl: bool = False,
    dry_run: bool = False,
) -> dict:
    """Copy every chunk from the embedded collection to the server.

    Returns a summary dict with source/target counts and how many chunks
    were newly added.
    """
    import chromadb

    target_name = target_collection_name or collection_name

    source_client = chromadb.PersistentClient(path=source_path)
    try:
        source = source_client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(
            f"No collection {collection_name!r} in {source_path!r}: {e}"
        ) from e

    total = source.count()
    target_client = chromadb.HttpClient(host=host, port=port, ssl=ssl)
    target = target_client.get_or_create_collection(target_name)
    before = target.count()

    logger.info(
        "Migrating %d chunk(s) from %s -> %s:%d/%s (target holds %d)",
        total, source_path, host, port, target_name, before,
    )

    if dry_run:
        logger.info("Dry run — nothing written")
        return {
            "source_count": total, "target_before": before, "target_after": before,
            "copied": 0, "added": 0, "dry_run": True,
        }

    copied = 0
    offset = 0
    while offset < total:
        page = _fetch_page(source, batch, offset)
        ids = page.get("ids") or []
        if not ids:
            break

        embeddings = page.get("embeddings")
        payload = {
            "ids": ids,
            "documents": page.get("documents"),
            "metadatas": page.get("metadatas"),
        }
        if embeddings is not None and len(embeddings):
            # Copy the stored vectors; without this the server would try to
            # embed the documents itself and fail (it has no embedding fn).
            payload["embeddings"] = embeddings

        # upsert, not add: re-running must not error on chunks already there.
        target.upsert(**payload)
        copied += len(ids)
        offset += len(ids)
        logger.info("  copied %d/%d", copied, total)

    after = target.count()
    summary = {
        "source_count": total,
        "target_before": before,
        "target_after": after,
        "copied": copied,
        "added": after - before,
        "dry_run": False,
    }
    logger.info("Migration complete: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    from config import setup_logging

    setup_logging()
    args = _parse_args(argv)

    try:
        summary = migrate(
            source_path=args.source,
            host=args.host,
            port=args.port,
            collection_name=args.collection,
            target_collection_name=args.target_collection,
            batch=args.batch,
            ssl=args.ssl,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"Migration failed: {e}", file=sys.stderr)
        return 1

    print()
    print(f"  source ({args.source}):        {summary['source_count']} chunk(s)")
    print(f"  target before:                 {summary['target_before']} chunk(s)")
    if summary["dry_run"]:
        print(f"  would copy:                    {summary['source_count']} chunk(s)")
        print("\n  Dry run — nothing was written.")
        return 0
    print(f"  copied:                        {summary['copied']} chunk(s)")
    print(f"  target after:                  {summary['target_after']} chunk(s)")
    print(f"  newly added:                   {summary['added']} "
          f"({summary['copied'] - summary['added']} already present)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
