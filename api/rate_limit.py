"""Rate limiter construction and its storage backend.

The limiter used to be built with slowapi's default in-memory storage.
Counters then lived in one process, so N replicas allowed N x the
configured limit and every restart reset them to zero. That capped the
deployment at a single API process — which made a single crash a full
outage, so it was an availability problem before it was a scale one.

Pointing ``RATE_LIMIT_STORAGE_URI`` at Redis gives every replica one
shared counter. Two deliberate choices around that:

* **Redis must not become a new single point of failure.** With
  ``in_memory_fallback_enabled`` a Redis outage degrades to per-process
  limiting — imprecise, but the API keeps serving. Losing the limiter's
  accuracy is a far better failure than losing the API.
* **A misconfigured URI must be loud.** Silently falling back to memory
  would reproduce the exact bug this replaces, and it would look healthy.
  ``verify_storage()`` probes the backend at startup and logs an error.
"""
from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from config import settings

logger = logging.getLogger(__name__)

# Anything else is a single-process counter, whatever the scheme claims.
DISTRIBUTED_SCHEMES = ("redis://", "rediss://", "redis+sentinel://", "memcached://")


def storage_uri() -> str:
    """Configured storage URI, or slowapi's in-process default."""
    return settings.rate_limit_storage_uri.strip() or "memory://"


def is_distributed(uri: str | None = None) -> bool:
    """True when counters are shared across processes."""
    return (uri if uri is not None else storage_uri()).startswith(DISTRIBUTED_SCHEMES)


def rate_limit_key(request: Request) -> str:
    """Rate limit per API key when authenticated, else per client address.

    Keying purely on IP puts every user behind a corporate NAT or a
    single egress gateway into one bucket, where they exhaust each
    other's quota. The API key is the actual principal, so it is the
    correct unit — and ``api_key_id`` is already a truncated SHA-256, so
    no raw credential reaches Redis.

    Falls back to the client address when auth is disabled, which is the
    single-tenant development default and keeps behaviour unchanged there.
    """
    key_id = getattr(request.state, "api_key_id", "")
    if key_id:
        return f"key:{key_id}"
    return get_remote_address(request)


def build_limiter() -> Limiter:
    """Construct the application limiter from settings."""
    return Limiter(
        key_func=rate_limit_key,
        storage_uri=storage_uri(),
        # Degrade to per-process limiting if the shared backend is
        # unreachable, rather than failing every request.
        in_memory_fallback_enabled=True,
        # Tell clients their remaining quota and when it resets, so a
        # well-behaved caller can back off before it is rejected.
        headers_enabled=True,
    )


def verify_storage(limiter: Limiter) -> bool:
    """Probe the limiter's storage at startup. Returns True if usable.

    Called from the app lifespan. A wrong host or a Redis that is not up
    yet otherwise shows up only as quotas that mysteriously fail to apply
    across replicas, which is nearly impossible to notice in production.
    """
    uri = storage_uri()
    if not is_distributed(uri):
        if settings.auth_enabled:
            logger.warning(
                "Rate limiting uses in-process storage (%s). Counters are not "
                "shared, so N replicas permit N x the configured limit. Set "
                "RATE_LIMIT_STORAGE_URI=redis://<host>:6379/0 before scaling out.",
                uri,
            )
        return True

    try:
        healthy = bool(limiter.limiter.storage.check())
    except Exception as e:  # connection refused, auth failure, bad scheme
        logger.error(
            "Rate limit storage %s is unreachable (%s). Falling back to "
            "per-process counters — limits will NOT be shared across replicas.",
            _redact(uri),
            e,
        )
        return False

    if not healthy:
        logger.error(
            "Rate limit storage %s failed its health check. Falling back to "
            "per-process counters — limits will NOT be shared across replicas.",
            _redact(uri),
        )
        return False

    logger.info("Rate limiting backed by shared storage at %s", _redact(uri))
    return True


def _redact(uri: str) -> str:
    """Strip any password from a storage URI before it reaches the logs."""
    if "@" not in uri:
        return uri
    scheme, _, rest = uri.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"
