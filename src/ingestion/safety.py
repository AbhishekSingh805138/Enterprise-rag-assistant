"""Limits applied when parsing files a stranger uploaded.

PDF and DOCX parsers are a long-standing exploit surface, and the worker
runs them in-process on user-supplied bytes. Full isolation is a
deployment concern — read-only rootfs, dropped capabilities, seccomp, no
egress — and is configured in ``docker-compose.prod.yml``. What belongs
in the code is the part a container cannot express: bounds on what a
single document is allowed to consume, and what happens when it exceeds
them.

Two bounds, aimed at the two shapes of resource attack:

* **Expansion.** A 2 MB file that decompresses into gigabytes of text is
  the classic bomb. Upload size is already capped; extracted size was
  not, and it is the one that reaches memory and the embedding bill.
* **Time.** A malformed file that sends a parser into a pathological
  loop stalls the worker indefinitely, which stalls the queue behind it.

Both are treated as *permanent* failures. Retrying a decompression bomb
twice more just spends the budget on the same outcome, so these
dead-letter immediately and an operator sees them.

On the timeout, an honest limitation: Python cannot kill a thread, so a
parser stuck in a C extension keeps running in the background even after
this returns. What the deadline guarantees is that the *worker* stops
waiting and the queue keeps moving; the stuck thread dies with the
process. Hard preemption needs process isolation, which is what the
container hardening is for.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from config import settings

logger = logging.getLogger(__name__)


class DocumentTooLarge(RuntimeError):
    """Extracted text exceeded INGEST_MAX_EXTRACTED_CHARS."""


class ParseTimeout(RuntimeError):
    """Parsing exceeded INGEST_PARSE_TIMEOUT_S."""


def parse_with_timeout(fn, *args, **kwargs):
    """Run *fn*, raising :class:`ParseTimeout` past the deadline."""
    timeout = settings.ingest_parse_timeout_s
    if timeout <= 0:
        return fn(*args, **kwargs)

    # Deliberately not a `with` block: its __exit__ calls
    # shutdown(wait=True), which would block for as long as the stuck
    # parser runs — exactly the wait this deadline exists to avoid.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parse")
    try:
        future = pool.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeout as e:
            raise ParseTimeout(
                f"Parsing exceeded {timeout}s and was abandoned. The file is "
                f"malformed or deliberately pathological."
            ) from e
    finally:
        # Returns immediately either way; a stuck worker thread is a
        # daemon of this process and dies with it.
        pool.shutdown(wait=False, cancel_futures=True)
    return result


def enforce_extracted_size(documents) -> None:
    """Reject a document whose extracted text is implausibly large."""
    limit = settings.ingest_max_extracted_chars
    if limit <= 0:
        return
    total = sum(len(d.page_content or "") for d in documents)
    if total > limit:
        raise DocumentTooLarge(
            f"Extracted {total:,} characters, over the "
            f"INGEST_MAX_EXTRACTED_CHARS limit of {limit:,}. A file that "
            f"expands this much is a decompression bomb or a parser fault."
        )


def apply_pii_policy(documents, *, document_id: str = "") -> int:
    """Apply ``INGEST_PII_MODE`` to parsed documents, in place.

    Returns the number of PII categories seen. Three modes, because the
    right answer genuinely differs by deployment:

    ``off``     unchanged (the default — indexing behaviour is untouched)
    ``warn``    logs what was found and indexes it anyway, so a team can
                see that its vectors hold PII before deciding what to do
    ``redact``  strips it before embedding

    ``redact`` is not the default on purpose: it changes what is
    retrievable, and a corpus indexed before and after would be
    inconsistent. That is a decision for whoever owns the data, not one
    to make silently on their behalf.
    """
    from src.security.pii import redact, scan

    mode = settings.ingest_pii_mode.strip().lower()
    if mode == "off":
        return 0

    total: dict[str, int] = {}
    for doc in documents:
        text = doc.page_content or ""
        if mode == "redact":
            cleaned, _ = redact(text)
            found = scan(text)
            doc.page_content = cleaned
        else:
            found = scan(text)
        for name, count in found.items():
            total[name] = total.get(name, 0) + count

    if total:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(total.items()))
        if mode == "redact":
            logger.info("Redacted PII before embedding (%s) in %s", summary, document_id or "?")
        else:
            logger.warning(
                "PII detected in %s and indexed unredacted (%s). Vectors and "
                "their backups will contain it. Set INGEST_PII_MODE=redact to "
                "strip it before embedding.",
                document_id or "?", summary,
            )
    return len(total)
