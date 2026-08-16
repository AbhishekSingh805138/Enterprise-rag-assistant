"""Per-key department scoping, enforced at retrieval.

The department filter arrives in the request body, and nothing tied it to
the caller. Any authenticated key could read ``legal`` or ``security``
documents simply by asking for them — and, worse, a request with *no*
filter searched every department at once, so confidential material could
surface in an answer without anyone asking for it. The ``access_level``
metadata the loader writes was never consulted.

This module makes the API key the authority on which departments a
caller may see:

  * ``API_KEYS=<key>:hr|general`` scopes a key to those departments.
  * ``API_KEYS=<key>`` (no suffix) stays unscoped — every department,
    which is the historical behaviour, so existing deployments are
    unchanged until they opt in.

Enforcement happens where retrieval is parameterised, not only at the
endpoint: a scoped caller always retrieves with an explicit department
filter, so "no filter" can no longer mean "the whole corpus".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The department taxonomy. ``general`` is the fallback the loader assigns
# to anything not filed under a department folder.
VALID_DEPARTMENTS = frozenset({
    "general", "hr", "legal", "engineering", "finance", "security", "operations",
})

# Departments whose documents the loader marks ``access_level=confidential``.
CONFIDENTIAL_DEPARTMENTS = frozenset({"legal", "security"})

WILDCARD = "*"

# ``None`` means unrestricted; a frozenset means exactly those departments.
Scope = frozenset[str] | None


class DepartmentForbidden(Exception):
    """Caller asked for a department outside their scope."""

    def __init__(self, requested: set[str], allowed: frozenset[str]):
        self.requested = requested
        self.allowed = allowed
        super().__init__(
            f"Not permitted for department(s): {', '.join(sorted(requested))}"
        )


@dataclass(frozen=True)
class KeyScopes:
    """Parsed ``API_KEYS``: the valid keys and each one's departments."""

    scopes: dict[str, Scope]

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self.scopes)

    def scope_for(self, key: str) -> Scope:
        return self.scopes.get(key)

    @property
    def any_scoped(self) -> bool:
        return any(v is not None for v in self.scopes.values())


def _parse_department_spec(spec: str) -> Scope | bool:
    """Interpret the part after ``:`` in an API_KEYS entry.

    Returns a frozenset of departments, ``None`` for the wildcard, or
    ``False`` when *spec* is not a department list at all — which is how
    a key that happens to contain a colon is told apart from a scope
    suffix. Every segment must be a known department for it to count.
    """
    spec = spec.strip()
    if not spec:
        return False
    if spec == WILDCARD:
        return None
    parts = [p.strip().lower() for p in spec.split("|")]
    if not all(parts):
        return False
    if not all(p in VALID_DEPARTMENTS for p in parts):
        return False
    return frozenset(parts)


def parse_api_keys(raw: str) -> KeyScopes:
    """Parse ``API_KEYS`` into keys and their department scopes.

    Accepted forms, comma separated::

        <key>                 unscoped — all departments (legacy behaviour)
        <key>:*               explicit wildcard, same as unscoped
        <key>:hr|general      restricted to those departments

    A key containing a colon is preserved as-is unless everything after
    the final colon is a valid department list, so an opaque token is
    never silently truncated into a shorter (and still accepted) key.
    """
    scopes: dict[str, Scope] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, sep, spec = entry.rpartition(":")
        if sep:
            parsed = _parse_department_spec(spec)
            if parsed is not False and key.strip():
                scopes[key.strip()] = parsed
                continue
        scopes[entry] = None
    return KeyScopes(scopes)


def normalise_requested(filter_dict: dict | None) -> set[str]:
    """Departments named by a caller-supplied filter, if any."""
    if not filter_dict:
        return set()
    value = filter_dict.get("department")
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, dict):
        # Chroma operator form, e.g. {"$in": ["hr", "legal"]}
        collected: set[str] = set()
        for op_value in value.values():
            if isinstance(op_value, str):
                collected.add(op_value.lower())
            elif isinstance(op_value, (list, tuple, set)):
                collected.update(str(v).lower() for v in op_value)
        return collected
    if isinstance(value, (list, tuple, set)):
        return {str(v).lower() for v in value}
    return set()


def department_filter(departments: frozenset[str]) -> dict:
    """Build the metadata filter clause for a set of departments."""
    if len(departments) == 1:
        return {"department": next(iter(departments))}
    return {"department": {"$in": sorted(departments)}}


def enforce_scope(filter_dict: dict | None, allowed: Scope) -> dict | None:
    """Constrain *filter_dict* to *allowed*, or raise DepartmentForbidden.

    * Unrestricted caller (``allowed is None``): the filter passes through
      untouched — identical to the previous behaviour.
    * Scoped caller with no department filter: one is *added*, so an
      unfiltered question can no longer sweep the whole corpus.
    * Scoped caller naming a permitted department: kept as asked, which
      preserves narrowing to a single department inside their scope.
    * Scoped caller naming anything else: rejected, rather than quietly
      returning empty results — a silent empty answer looks like "we have
      no policy on that", which is its own kind of wrong answer.
    """
    if allowed is None:
        return filter_dict

    requested = normalise_requested(filter_dict)
    forbidden = requested - allowed
    if forbidden:
        raise DepartmentForbidden(forbidden, allowed)

    constrained = dict(filter_dict or {})
    if not requested:
        constrained.update(department_filter(allowed))
    return constrained


def matches_filter(metadata: dict, filter_dict: dict | None) -> bool:
    """Whether *metadata* satisfies *filter_dict*, including ``$in``.

    In-process retrievers (BM25) filter by comparing metadata directly.
    Plain equality silently matches nothing once a filter uses ``$in``,
    which would make scoped callers lose sparse retrieval entirely while
    dense retrieval kept working — a subtle, per-user quality regression.
    """
    if not filter_dict:
        return True
    for field, expected in filter_dict.items():
        actual = metadata.get(field)
        if isinstance(expected, dict):
            for op, operand in expected.items():
                if op == "$in":
                    if actual not in operand:
                        return False
                elif op == "$nin":
                    if actual in operand:
                        return False
                elif op == "$eq":
                    if actual != operand:
                        return False
                elif op == "$ne":
                    if actual == operand:
                        return False
                else:  # unknown operator — refuse rather than over-return
                    logger.warning("Unsupported filter operator %r; excluding document", op)
                    return False
        elif actual != expected:
            return False
    return True
