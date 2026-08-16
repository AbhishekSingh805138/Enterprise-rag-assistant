"""Enterprise entity extraction using LLM structured output.

Extracts named entities relevant to enterprise knowledge bases:
policy names, departments, dates, people, document names, etc.
Falls back to regex-based extraction when LLM is unavailable.
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from config import settings
from src.prompts import register
from src.llm_pool import get_llm
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """A single extracted entity."""
    name: str = Field(description="The entity text as it appears in the query")
    type: str = Field(description="Entity type: department, policy, person, date, document, role, tool")


class EntityExtractionResult(BaseModel):
    """Structured output from the entity extractor."""
    entities: list[Entity] = Field(
        default_factory=list,
        description="List of extracted entities from the query",
    )


_EXTRACT_PROMPT_TEXT = (
    "You are an enterprise entity extractor. Extract named entities from the "
    "user's query that are relevant for searching a corporate knowledge base.\n\n"
    "Entity types:\n"
    "- **department**: Engineering, HR, Finance, Legal, Security, Sales, Marketing, etc.\n"
    "- **policy**: Named policies (PTO policy, NDA, security policy, etc.)\n"
    "- **person**: People mentioned by name or title\n"
    "- **date**: Dates, deadlines, or time references (Q3, 2024, last month, etc.)\n"
    "- **document**: Document names or references (handbook, contract, SLA, etc.)\n"
    "- **role**: Job titles or roles (engineer, manager, contractor, etc.)\n"
    "- **tool**: Tools or systems mentioned (Jira, Slack, GitHub, etc.)\n\n"
    "Only extract entities that are explicitly mentioned. Do not infer entities."
)

# --- Regex fallback patterns ---

_DEPARTMENT_PATTERN = re.compile(
    r"\b(engineering|hr|human resources|finance|legal|security|sales|marketing|"
    r"operations|product|design|support|it|compliance|procurement)\b",
    re.IGNORECASE,
)

_DOCUMENT_PATTERN = re.compile(
    r"\b(handbook|contract|agreement|policy|guideline|procedure|manual|"
    r"sla|nda|sow|rfc|spec|report)\b",
    re.IGNORECASE,
)

_ROLE_PATTERN = re.compile(
    r"\b(engineers?|managers?|directors?|vp|cto|ceo|cfo|analysts?|developers?|"
    r"contractors?|interns?|leads?|architects?|admins?|coordinators?)\b",
    re.IGNORECASE,
)

_DATE_PATTERN = re.compile(
    r"\b(Q[1-4]|20\d{2}|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|last (month|year|quarter)|"
    r"this (month|year|quarter)|next (month|year|quarter))\b",
    re.IGNORECASE,
)


def _regex_extract(query: str) -> list[Entity]:
    """Fast regex-based entity extraction fallback."""
    entities: list[Entity] = []
    seen: set[str] = set()

    for pattern, etype in [
        (_DEPARTMENT_PATTERN, "department"),
        (_DOCUMENT_PATTERN, "document"),
        (_ROLE_PATTERN, "role"),
        (_DATE_PATTERN, "date"),
    ]:
        for match in pattern.finditer(query):
            name = match.group(0)
            key = (name.lower(), etype)
            if key not in seen:
                seen.add(key)
                entities.append(Entity(name=name, type=etype))

    return entities


def extract_entities(query: str) -> list[Entity]:
    """Extract entities from a query using LLM with regex fallback."""
    if not query.strip():
        return []

    # Try LLM extraction first
    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = register(
            "entity_extract",
            "v1",
            [
            ("system", _EXTRACT_PROMPT_TEXT),
            ("human", "{query}"),
        ])
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        chain = prompt | get_llm().with_structured_output(EntityExtractionResult)
        result: EntityExtractionResult = cb.call(chain.invoke, {"query": query})
        logger.debug("LLM extracted %d entities from: %s", len(result.entities), query[:80])
        return result.entities

    except CircuitBreakerOpen as exc:
        logger.debug("LLM circuit open for entity extraction: %s", exc)
    except Exception:
        logger.debug("LLM entity extraction failed — using regex fallback", exc_info=True)

    # Regex fallback
    entities = _regex_extract(query)
    logger.debug("Regex extracted %d entities from: %s", len(entities), query[:80])
    return entities
