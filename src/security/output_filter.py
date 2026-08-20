"""Output filter for redacting PII in LLM responses.

Scans generated answers for SSNs, credit card numbers and phone numbers
and replaces them with redacted placeholders.

Email addresses are matched by :mod:`src.security.pii` but deliberately
left alone here — enterprise answers are routinely "contact
facilities@company.com", and redacting the one actionable detail makes
the answer useless. They *are* redacted at ingest when
``INGEST_PII_MODE=redact``, because nobody reads a vector.

This is the last line, not the only one: without ingest-time screening
the vector store still holds the original text.
"""
from __future__ import annotations

import logging

from config import settings
from src.security.pii import ANSWER_CATEGORIES, redact

logger = logging.getLogger(__name__)


def filter_output(response: str) -> str:
    """Redact PII patterns from the LLM response.

    Only active when pii_detection_enabled=True.
    """
    if not settings.pii_detection_enabled:
        return response

    filtered, redacted_count = redact(response, ANSWER_CATEGORIES)

    if redacted_count > 0:
        logger.info("Output filter redacted %d PII pattern(s)", redacted_count)

    return filtered
