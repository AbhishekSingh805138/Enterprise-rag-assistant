"""Intelligent context construction for LLM generation.

Builds an optimal context string from retrieved documents with:
- Content-hash deduplication (removes duplicate chunks)
- Source grouping (chunks from the same file grouped together)
- Token budget management (stays within max_tokens)
- Proportional allocation across sources
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ~ 4 characters (for English text with tiktoken)
_CHARS_PER_TOKEN = 4


@dataclass
class BuiltContext:
    """Result of context building."""
    text: str
    token_count: int
    sources_used: list[str]
    docs_included: int
    docs_truncated: int


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return len(text) // _CHARS_PER_TOKEN


def _content_hash(text: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_context(
    docs: list[Document],
    query: str = "",
    max_tokens: int | None = None,
    memory_context: str = "",
) -> BuiltContext:
    """Build an optimized context string from documents.

    Args:
        docs: Retrieved documents to build context from.
        query: The user's query (reserved for future relevance scoring).
        max_tokens: Maximum token budget for the context. Defaults to
            settings.context_max_tokens.
        memory_context: Conversation memory to account for in token budget.

    Returns:
        BuiltContext with the assembled text and metadata.
    """
    if max_tokens is None:
        max_tokens = settings.context_max_tokens

    if not docs:
        return BuiltContext(text="", token_count=0, sources_used=[], docs_included=0, docs_truncated=0)

    # Step 1: Deduplicate by content hash
    seen_hashes: set[str] = set()
    unique_docs: list[Document] = []
    for doc in docs:
        h = _content_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)

    dedup_removed = len(docs) - len(unique_docs)
    if dedup_removed > 0:
        logger.debug("Deduplication removed %d duplicate chunks", dedup_removed)

    # Step 2: Group by source file
    source_groups: dict[str, list[Document]] = {}
    for doc in unique_docs:
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        source_groups.setdefault(source, []).append(doc)

    # Step 3: Reserve token budget for memory context
    memory_tokens = _estimate_tokens(memory_context) if memory_context else 0
    available_tokens = max(0, max_tokens - memory_tokens)

    # Step 4: Proportional allocation across sources
    total_chars = sum(len(d.page_content) for d in unique_docs)
    if total_chars == 0:
        return BuiltContext(text="", token_count=0, sources_used=[], docs_included=0, docs_truncated=0)

    # Build context respecting token budget
    context_parts: list[str] = []
    sources_used: list[str] = []
    docs_included = 0
    docs_truncated = 0
    tokens_used = 0

    for source, group_docs in source_groups.items():
        # Proportional token budget for this source
        group_chars = sum(len(d.page_content) for d in group_docs)
        source_token_budget = int(available_tokens * (group_chars / total_chars))

        source_tokens_used = 0
        source_included = False

        for doc in group_docs:
            chunk_text = f"[{source}] {doc.page_content}"
            chunk_tokens = _estimate_tokens(chunk_text)

            if tokens_used + chunk_tokens > available_tokens:
                # Over total budget — truncate if we have room for at least some
                remaining = available_tokens - tokens_used
                if remaining > 50:  # at least ~200 chars worth
                    truncated_chars = remaining * _CHARS_PER_TOKEN
                    chunk_text = chunk_text[:truncated_chars] + "..."
                    context_parts.append(chunk_text)
                    tokens_used += remaining
                    docs_included += 1
                    docs_truncated += 1
                    source_included = True
                else:
                    docs_truncated += 1
                break
            elif source_tokens_used + chunk_tokens > source_token_budget and source_included:
                # Over source budget but might still fit in total
                # Only skip if we already have content from this source
                docs_truncated += 1
                continue

            context_parts.append(chunk_text)
            tokens_used += chunk_tokens
            source_tokens_used += chunk_tokens
            docs_included += 1
            source_included = True

        if source_included:
            sources_used.append(source)

    text = "\n\n".join(context_parts)
    final_tokens = _estimate_tokens(text)

    logger.info(
        "Context built: %d docs included, %d truncated, %d sources, ~%d tokens",
        docs_included, docs_truncated, len(sources_used), final_tokens,
    )

    return BuiltContext(
        text=text,
        token_count=final_tokens,
        sources_used=sources_used,
        docs_included=docs_included,
        docs_truncated=docs_truncated,
    )
