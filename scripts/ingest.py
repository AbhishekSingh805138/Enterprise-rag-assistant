"""Ingest documents into ChromaDB.

Usage:
    python -m scripts.ingest                          # default: ./data/sample_docs
    python -m scripts.ingest ./data/sample_docs
    python -m scripts.ingest ./some_file.pdf
"""
from __future__ import annotations

import argparse
import logging
import sys

from config import settings, setup_logging
from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import load_path
from src.vectorstore.chroma_store import add_chunks, collection_stats

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the ChromaDB vector store.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="./data/sample_docs",
        help="File or directory to ingest (default: ./data/sample_docs)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help=f"Override chunk size (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=None,
        help=f"Override chunk overlap (default: {settings.chunk_overlap})",
    )
    args = parser.parse_args()

    setup_logging()
    settings.validate()

    try:
        print(f"Loading from {args.path} ...")
        docs = load_path(args.path)
        print(f"  Loaded {len(docs)} document(s)")

        chunks = chunk_documents(
            docs,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"  Split into {len(chunks)} chunk(s)")

        added = add_chunks(chunks)

        # Report where the chunks actually went: naming the local directory
        # while connected to a Chroma server would send someone inspecting
        # the result to a stale copy.
        stats = collection_stats()
        destination = stats.get("endpoint") or stats.get("persist_directory") or "?"
        print(f"  Persisted {added} new chunk(s) to {destination}")
        print(f"  Collection total: {stats['document_count']} chunk(s)")

    except FileNotFoundError as e:
        logger.error("No documents found: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Ingestion failed")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
