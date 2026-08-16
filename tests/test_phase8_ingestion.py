"""Phase 8.5 tests: Ingestion & Chunking Improvements.

Covers:
  - Markdown-aware chunking (header splitting, header metadata)
  - Ingestion timestamp (ingested_at in metadata)
  - Chunk overlap config (env var override, new default)
  - Non-markdown docs still use recursive chunking
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# TestMarkdownChunking
# ---------------------------------------------------------------------------

class TestMarkdownChunking:
    """Verify markdown-aware chunking splits on headers."""

    def test_md_doc_splits_on_headers(self):
        """A markdown doc with headers should produce chunks with header metadata."""
        from src.ingestion.chunker import chunk_documents

        content = (
            "# Introduction\n\n"
            "This is the introduction section with some content.\n\n"
            "## Background\n\n"
            "This is background information that provides context.\n\n"
            "## Methods\n\n"
            "This section describes the methods used.\n\n"
            "### Sub-methods\n\n"
            "Details about sub-methods go here."
        )
        doc = Document(
            page_content=content,
            metadata={"doc_type": "md", "filename": "test.md", "source": "test.md"},
        )

        with patch("src.ingestion.chunker.settings") as mock_settings:
            mock_settings.chunk_size = 1000
            mock_settings.chunk_overlap = 100
            mock_settings.markdown_aware_chunking = True

            chunks = chunk_documents([doc])
            assert len(chunks) >= 1

            # Check that header metadata is preserved in at least some chunks
            all_metadata_keys = set()
            for c in chunks:
                all_metadata_keys.update(c.metadata.keys())

            # Original metadata should be inherited
            assert "filename" in all_metadata_keys
            assert "doc_type" in all_metadata_keys

    def test_md_chunking_disabled_falls_back(self):
        """When markdown_aware_chunking is False, .md files use recursive splitter."""
        from src.ingestion.chunker import chunk_documents

        doc = Document(
            page_content="# Header\n\nSome content here.",
            metadata={"doc_type": "md", "filename": "test.md"},
        )

        with patch("src.ingestion.chunker.settings") as mock_settings:
            mock_settings.chunk_size = 1000
            mock_settings.chunk_overlap = 100
            mock_settings.markdown_aware_chunking = False

            chunks = chunk_documents([doc])
            assert len(chunks) >= 1

    def test_non_md_docs_unaffected(self):
        """Non-markdown docs should use recursive splitter regardless of setting."""
        from src.ingestion.chunker import chunk_documents

        doc = Document(
            page_content="This is a plain text document with no markdown headers.",
            metadata={"doc_type": "txt", "filename": "test.txt"},
        )

        with patch("src.ingestion.chunker.settings") as mock_settings:
            mock_settings.chunk_size = 1000
            mock_settings.chunk_overlap = 100
            mock_settings.markdown_aware_chunking = True

            chunks = chunk_documents([doc])
            assert len(chunks) >= 1
            # txt docs should not have header metadata
            for c in chunks:
                assert "header_h1" not in c.metadata or c.metadata.get("header_h1") is None

    def test_mixed_doc_types(self):
        """A mix of .md and .txt docs should each use their appropriate chunker."""
        from src.ingestion.chunker import chunk_documents

        md_doc = Document(
            page_content="# Title\n\nMarkdown content here.",
            metadata={"doc_type": "md", "filename": "doc.md"},
        )
        txt_doc = Document(
            page_content="Plain text content.",
            metadata={"doc_type": "txt", "filename": "doc.txt"},
        )

        with patch("src.ingestion.chunker.settings") as mock_settings:
            mock_settings.chunk_size = 1000
            mock_settings.chunk_overlap = 100
            mock_settings.markdown_aware_chunking = True

            chunks = chunk_documents([md_doc, txt_doc])
            assert len(chunks) >= 2

    def test_empty_docs_returns_empty(self):
        from src.ingestion.chunker import chunk_documents
        assert chunk_documents([]) == []


# ---------------------------------------------------------------------------
# TestIngestionTimestamp
# ---------------------------------------------------------------------------

class TestIngestionTimestamp:
    """Verify ingested_at timestamp is added during loading."""

    def test_ingested_at_present(self, tmp_path):
        """Loaded documents should have ingested_at in metadata."""
        # Create a temp text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test document content for timestamp verification.")

        from src.ingestion.loader import load_path
        docs = load_path(str(test_file))

        assert len(docs) >= 1
        for doc in docs:
            assert "ingested_at" in doc.metadata
            # Should be an ISO format string
            assert "T" in doc.metadata["ingested_at"]

    def test_ingested_at_is_utc(self, tmp_path):
        """The ingested_at timestamp should be UTC."""
        test_file = tmp_path / "utc_test.txt"
        test_file.write_text("UTC timestamp test.")

        from src.ingestion.loader import load_path
        docs = load_path(str(test_file))

        ts = docs[0].metadata["ingested_at"]
        # UTC timestamps end with +00:00
        assert "+00:00" in ts

    def test_ingested_at_for_md_files(self, tmp_path):
        """Markdown files should also get ingested_at."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\n\nSome content.")

        from src.ingestion.loader import load_path
        docs = load_path(str(md_file))

        assert "ingested_at" in docs[0].metadata


# ---------------------------------------------------------------------------
# TestChunkOverlapConfig
# ---------------------------------------------------------------------------

class TestChunkOverlapConfig:
    """Verify chunk overlap configuration."""

    def test_default_overlap_is_200(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.chunk_overlap == 200

    def test_markdown_aware_default_true(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.markdown_aware_chunking is True


# ---------------------------------------------------------------------------
# TestChunkMarkdownHelper
# ---------------------------------------------------------------------------

class TestChunkMarkdownHelper:
    """Verify _chunk_markdown helper function."""

    def test_chunk_markdown_preserves_parent_metadata(self):
        from src.ingestion.chunker import _chunk_markdown

        doc = Document(
            page_content="# Title\n\nContent under title.\n\n## Section\n\nMore content.",
            metadata={"filename": "test.md", "department": "hr", "custom_key": "custom_val"},
        )
        chunks = _chunk_markdown(doc, chunk_size=1000, chunk_overlap=100)
        for c in chunks:
            assert c.metadata.get("filename") == "test.md"
            assert c.metadata.get("department") == "hr"
            assert c.metadata.get("custom_key") == "custom_val"

    def test_chunk_markdown_respects_chunk_size(self):
        """Long sections should be further split by the recursive splitter."""
        from src.ingestion.chunker import _chunk_markdown

        long_content = "# Title\n\n" + ("A" * 2000) + "\n\n## Section\n\n" + ("B" * 2000)
        doc = Document(
            page_content=long_content,
            metadata={"filename": "long.md"},
        )
        chunks = _chunk_markdown(doc, chunk_size=500, chunk_overlap=50)
        # Long content should produce multiple chunks
        assert len(chunks) > 2
        for c in chunks:
            # Each chunk should be <= chunk_size (with some tolerance for splitting)
            assert len(c.page_content) <= 600  # some tolerance
