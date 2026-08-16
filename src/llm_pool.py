"""Thread-safe LLM instance pool.

Caches chat model instances by (model, temperature) so the same
configuration is reused across nodes, avoiding redundant client creation
on every call.

Construction goes through :mod:`src.llm.providers`, so the vendor is a
configuration choice and a second vendor can stand in when the first is
unavailable. When no fallback is configured this returns the primary
model unchanged — byte-for-byte the previous behaviour.
"""
from __future__ import annotations

import logging
import threading

from langchain_core.runnables import Runnable

from config import settings
from src.llm.providers import (
    ChatModelWithFallbacks,
    ProviderNotAvailable,
    build_chat_model,
    fallback_spec,
)

logger = logging.getLogger(__name__)

_pool: dict[tuple[str, float], Runnable] = {}
_lock = threading.Lock()
# Logged once per process: a fallback that silently failed to build is
# worse than none, because the deployment believes it is covered.
_fallback_warned = False


def _build(model: str, temperature: float) -> Runnable:
    primary = build_chat_model(settings.llm_provider, model, temperature)

    spec = fallback_spec()
    if spec is None:
        return primary

    provider, fallback_model = spec
    try:
        secondary = build_chat_model(provider, fallback_model, temperature)
    except ProviderNotAvailable as e:
        global _fallback_warned
        if not _fallback_warned:
            _fallback_warned = True
            logger.error(
                "LLM_FALLBACK_PROVIDER=%s is configured but unusable (%s). "
                "Running WITHOUT failover — a primary outage will be a total outage.",
                provider,
                e,
            )
        return primary

    return ChatModelWithFallbacks(primary, [secondary])


def get_llm(temperature: float = 0, model: str | None = None) -> Runnable:
    """Return a cached chat model for the given params.

    Thread-safe: concurrent calls with the same key return the same object.
    """
    mdl = model or settings.llm_model
    key = (mdl, temperature)
    with _lock:
        if key not in _pool:
            _pool[key] = _build(mdl, temperature)
            logger.debug("LLM pool: created instance for %s", key)
        return _pool[key]


def reset_pool() -> None:
    """Clear the pool (for testing)."""
    global _fallback_warned
    with _lock:
        _pool.clear()
        _fallback_warned = False
