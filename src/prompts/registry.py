"""A versioned, overridable registry for every prompt in the system.

Sixteen prompt constants sat in eight modules. Nothing recorded which
text produced which answer, so a drop in faithfulness could not be traced
to the prompt change that caused it, and changing any wording meant a
code deploy.

The registry is populated by ``register()`` calls made where each prompt
is *defined*, rather than by relocating the text into one file. That is
deliberate: prompts are the behaviour of an LLM system, and retyping
sixteen of them into a new module is an invitation to change one by a
character and never notice. Registering in place gives the same
capabilities — one inventory, versions, content hashes, overrides,
attribution in metrics — with a diff small enough to review.

What it buys:

* ``prompt_fingerprint()`` goes into every query metric, so an answer can
  be attributed to the exact prompt set that produced it.
* A content hash per prompt means editing the text without bumping the
  version is still visible; the version alone would be a promise, not a
  fact.
* ``PROMPT_OVERRIDE_DIR`` allows changing wording without a code change.
  Overrides must declare the same input variables as the built-in, which
  is checked at load — a prompt missing ``{context}`` would otherwise
  produce fluent, sourceless answers rather than an error.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from config import settings

logger = logging.getLogger(__name__)

Messages = list[tuple[str, str]]


@dataclass(frozen=True)
class PromptRecord:
    """One registered prompt and where its text came from."""

    name: str
    version: str
    messages: Messages
    description: str = ""
    source: str = "builtin"  # or "override:<path>"
    input_variables: tuple[str, ...] = field(default_factory=tuple)

    @property
    def content_hash(self) -> str:
        """Hash of the text, so an unversioned edit is still detectable."""
        payload = json.dumps(self.messages, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}+{self.content_hash}"


_registry: dict[str, PromptRecord] = {}
_overrides: dict[str, Messages] | None = None


def _input_variables(messages: Messages) -> tuple[str, ...]:
    """Template variables a message list requires."""
    return tuple(sorted(ChatPromptTemplate.from_messages(messages).input_variables))


def _override_dir() -> Path | None:
    raw = settings.prompt_override_dir.strip()
    return Path(raw) if raw else None


def load_overrides(force: bool = False) -> dict[str, Messages]:
    """Read prompt overrides from ``PROMPT_OVERRIDE_DIR``.

    Each file is ``<name>.json`` holding ``[["system", "..."], ["human",
    "..."]]``. Read once per process; ``force`` re-reads (used by tests
    and by a future reload hook).
    """
    global _overrides
    if _overrides is not None and not force:
        return _overrides

    _overrides = {}
    directory = _override_dir()
    if directory is None:
        return _overrides
    if not directory.is_dir():
        logger.warning(
            "PROMPT_OVERRIDE_DIR=%s is not a directory — no overrides loaded", directory
        )
        return _overrides

    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            messages = [(str(role), str(text)) for role, text in raw]
            if not messages:
                raise ValueError("override is empty")
        except Exception as e:
            logger.error("Ignoring invalid prompt override %s: %s", path.name, e)
            continue
        _overrides[path.stem] = messages
        logger.info("Loaded prompt override for %r from %s", path.stem, path.name)
    return _overrides


def register(
    name: str,
    version: str,
    messages: Messages,
    description: str = "",
) -> ChatPromptTemplate:
    """Record a prompt and return the template to use.

    Returns an override's template when one is configured and compatible,
    otherwise the built-in — so a bad override degrades to the shipped
    prompt rather than to a broken chain.
    """
    builtin_vars = _input_variables(messages)
    chosen, source = messages, "builtin"

    override = load_overrides().get(name)
    if override is not None:
        override_vars = _input_variables(override)
        if override_vars != builtin_vars:
            # A prompt that silently lost {context} would still produce
            # fluent text — just with nothing to ground it.
            logger.error(
                "Prompt override %r declares variables %s but %r requires %s; "
                "ignoring the override and using the built-in prompt.",
                name, list(override_vars), name, list(builtin_vars),
            )
        else:
            chosen, source = override, "override"

    record = PromptRecord(
        name=name,
        version=version,
        messages=[tuple(m) for m in chosen],  # type: ignore[misc]
        description=description,
        source=source,
        input_variables=builtin_vars,
    )
    if name in _registry and _registry[name].content_hash != record.content_hash:
        logger.debug("Prompt %r re-registered with different content", name)
    _registry[name] = record

    return ChatPromptTemplate.from_messages(chosen)


def get_prompt(name: str) -> PromptRecord | None:
    """The registered record for *name*, if its module has been imported."""
    return _registry.get(name)


def list_prompts() -> list[PromptRecord]:
    """Every registered prompt, by name. Only reflects imported modules."""
    return [_registry[k] for k in sorted(_registry)]


def prompt_fingerprint() -> str:
    """Short identifier for the whole active prompt set.

    Recorded on every query metric. Two answers with different
    fingerprints were produced by different prompts, which is what makes
    a quality regression attributable instead of merely observed.
    """
    if not _registry:
        return "none"
    payload = "|".join(r.label for r in list_prompts())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def reset_registry() -> None:
    """Clear registry and override cache (tests)."""
    global _overrides
    _registry.clear()
    _overrides = None
