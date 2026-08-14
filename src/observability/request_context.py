"""Ambient request/trace identifier shared by the API and the ingestion worker.

A single upload crosses three processes: the API that accepts it, the queue
that carries the event, and the worker that indexes it. Correlating those
log lines requires an id that rides along without being threaded through
every function signature — that is what the contextvar here is for.

The API sets it from the ``X-Request-ID`` header (generating one when the
caller does not supply it) and stamps it onto the published event; the
worker restores it from the event before processing. Deliberately free of
project imports so ``config.setup_logging`` can use it without a cycle.
"""
from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def new_request_id() -> str:
    """Generate a fresh request identifier."""
    return uuid.uuid4().hex[:16]


def get_request_id() -> str:
    """Return the current request id, or "" when running outside a request."""
    return _request_id.get()


def set_request_id(request_id: str | None):
    """Bind *request_id* to the current context. Returns a reset token.

    ``contextvars`` are copied into threads started by ``asyncio.to_thread``
    and ``contextvars.copy_context``, which is how the id survives the
    pipeline's worker threads.
    """
    return _request_id.set(request_id or new_request_id())


def reset_request_id(token) -> None:
    """Restore the previous request id bound before *token* was issued."""
    try:
        _request_id.reset(token)
    except (ValueError, LookupError):
        # Token from a different context (e.g. set in a thread, reset in the
        # parent). Nothing to restore — the child context is discarded anyway.
        pass
