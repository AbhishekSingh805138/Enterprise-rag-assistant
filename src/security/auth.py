"""API authentication via static API keys or JWT.

Implements a FastAPI dependency that validates the Authorization header.
Supports two modes:
- Static API keys: comma-separated list in API_KEYS env var
- JWT: validates against a JWKS endpoint (future)

Gated behind AUTH_ENABLED feature flag (default: False).
"""
from __future__ import annotations

import hashlib
import logging
import secrets

from fastapi import HTTPException, Request

from config import settings
from src.security.access_control import KeyScopes, Scope, parse_api_keys

logger = logging.getLogger(__name__)


def api_key_identity(token: str) -> str:
    """Stable, non-reversible identifier for an API key (for session scoping/audit)."""
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def _get_key_scopes() -> KeyScopes:
    """Parse API_KEYS into keys and their per-key department scopes."""
    raw = settings.api_keys.strip()
    if not raw:
        return KeyScopes({})
    return parse_api_keys(raw)


def _get_valid_api_keys() -> frozenset[str]:
    """Parse comma-separated API keys from config."""
    return _get_key_scopes().keys


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency that validates the API key from the Authorization header.

    Usage:
        @app.post("/ask", dependencies=[Depends(verify_api_key)])

    When AUTH_ENABLED=false, this is a no-op.
    When AUTH_ENABLED=true, expects: Authorization: Bearer <api_key>
    """
    if not settings.auth_enabled:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Provide: Authorization: Bearer <api_key>",
        )

    # Extract token from "Bearer <token>"
    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization format. Use: Bearer <api_key>",
        )

    token = parts[1].strip()
    key_scopes = _get_key_scopes()
    valid_keys = key_scopes.keys

    if not valid_keys:
        logger.warning("AUTH_ENABLED=true but no API_KEYS configured — rejecting all requests")
        raise HTTPException(
            status_code=500,
            detail="Authentication is enabled but no API keys are configured.",
        )

    # Constant-time comparison against every key so response timing does not
    # reveal whether a prefix matched.
    matched = False
    matched_key = ""
    for key in valid_keys:
        if secrets.compare_digest(token, key):
            matched = True
            matched_key = key
    if not matched:
        logger.warning("Invalid API key attempt from %s", request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    # Expose a non-reversible key identity so endpoints can scope
    # per-user resources (e.g. conversation sessions) to the caller.
    request.state.api_key_id = api_key_identity(token)
    # Departments this key may retrieve from. None means unrestricted,
    # which is what an entry with no ``:scope`` suffix grants.
    request.state.departments = key_scopes.scope_for(matched_key)


def permitted_departments(request: Request) -> Scope:
    """Departments the caller may retrieve from; ``None`` is unrestricted.

    Auth being disabled is single-tenant development mode, where there is
    no principal to scope to — so it is unrestricted by construction.
    """
    if not settings.auth_enabled:
        return None
    return getattr(request.state, "departments", None)
