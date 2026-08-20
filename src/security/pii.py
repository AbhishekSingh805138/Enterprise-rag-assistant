"""PII patterns, shared by the answer filter and the ingestion scanner.

Redaction previously existed only at answer time. That protects the
response body and nothing else: the document text is embedded verbatim,
so the vector store holds raw SSNs and card numbers, and anyone with
access to the Chroma volume, a backup of it, or the ``/documents``
surface reads them unredacted. The answer-time filter is the last line,
not the only one.

Patterns live here so the two callers cannot drift apart — a category
detected at ingest but not redacted in answers (or the reverse) is the
kind of gap that only shows up in an incident.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PiiPattern:
    name: str
    pattern: re.Pattern[str]
    placeholder: str


# Ordered: credit cards are matched before phone numbers, because a
# 16-digit group written with separators also satisfies the phone pattern.
PATTERNS: tuple[PiiPattern, ...] = (
    PiiPattern("ssn", re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"), "[SSN_REDACTED]"),
    PiiPattern("credit_card", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CC_REDACTED]"),
    PiiPattern(
        "phone",
        re.compile(r"\b(?:\+\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b"),
        "[PHONE_REDACTED]",
    ),
    PiiPattern(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[EMAIL_REDACTED]",
    ),
)

# The answer filter deliberately does NOT redact email addresses.
#
# Enterprise answers are routinely "contact facilities@company.com" — an
# answer that redacts the one actionable detail is worse than useless.
# Ingested text is a different judgement: nobody reads a vector, so
# there is no usefulness to trade away.
ANSWER_CATEGORIES = frozenset({"ssn", "credit_card", "phone"})


def _selected(categories: frozenset[str] | None) -> tuple[PiiPattern, ...]:
    if categories is None:
        return PATTERNS
    return tuple(p for p in PATTERNS if p.name in categories)


def scan(text: str, categories: frozenset[str] | None = None) -> dict[str, int]:
    """Count PII matches per category, without modifying *text*.

    Used by the ingest scanner's ``warn`` mode: knowing the corpus
    contains PII is worth having even when a deployment decides not to
    alter what it indexes.
    """
    found: dict[str, int] = {}
    for item in _selected(categories):
        count = len(item.pattern.findall(text))
        if count:
            found[item.name] = count
    return found


def redact(text: str, categories: frozenset[str] | None = None) -> tuple[str, int]:
    """Replace PII with placeholders. Returns ``(text, categories_hit)``."""
    result = text
    hits = 0
    for item in _selected(categories):
        replaced = item.pattern.sub(item.placeholder, result)
        if replaced != result:
            hits += 1
        result = replaced
    return result, hits
