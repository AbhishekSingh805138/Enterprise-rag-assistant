"""Data models for the knowledge graph."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str
    type: str  # e.g. "policy", "department", "person", "process"
    source_doc: str = ""


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    subject: str
    predicate: str  # e.g. "belongs_to", "requires", "applies_to"
    object: str
    source_doc: str = ""
    confidence: float = 1.0


@dataclass
class Triple:
    """A subject-predicate-object triple for graph construction."""
    subject: str
    predicate: str
    object: str
