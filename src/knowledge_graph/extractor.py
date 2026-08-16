"""Entity-relationship extraction from document chunks using LLM structured output."""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from config import settings
from src.prompts import register
from src.knowledge_graph.models import Triple
from src.llm_pool import get_llm
from src.resilience.circuit_breaker import CircuitBreakerOpen, get_breaker

logger = logging.getLogger(__name__)


class ExtractionResult(BaseModel):
    """Structured output from the triple extractor."""
    triples: list[dict] = Field(
        default_factory=list,
        description=(
            "List of triples, each with 'subject', 'predicate', and 'object' keys. "
            "Example: {'subject': 'PTO policy', 'predicate': 'allows', 'object': '20 days per year'}"
        ),
    )


_EXTRACT_PROMPT = (
    "You are a knowledge graph extractor for an enterprise knowledge base. "
    "Extract entity-relationship triples from the given text chunk.\n\n"
    "Guidelines:\n"
    "- Extract key relationships between entities (policies, departments, people, processes)\n"
    "- Use clear predicate names: 'belongs_to', 'requires', 'applies_to', 'manages', "
    "'contains', 'defines', 'authorizes', 'restricts'\n"
    "- Each triple should be independently meaningful\n"
    "- Focus on factual relationships, not opinions\n"
    "- Extract 0-10 triples per chunk (0 if no clear relationships)\n\n"
    "Text chunk:\n{text}"
)


def extract_triples(text: str, source_doc: str = "") -> list[Triple]:
    """Extract entity-relationship triples from a text chunk.

    Args:
        text: The text chunk to extract from.
        source_doc: Source document identifier for provenance.

    Returns:
        List of Triple objects extracted from the text.
    """
    if not text.strip():
        return []

    if not settings.knowledge_graph_enabled:
        return []

    try:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = register(
            "kg_extract_entities",
            "v1",
            [
            ("system", _EXTRACT_PROMPT),
        ])
        cb = get_breaker(
            "llm",
            failure_threshold=settings.circuit_breaker_threshold,
            timeout=settings.circuit_breaker_timeout,
        )
        chain = prompt | get_llm().with_structured_output(ExtractionResult)
        result: ExtractionResult = cb.call(chain.invoke, {"text": text})

        triples = []
        for t in result.triples:
            if isinstance(t, dict) and all(k in t for k in ("subject", "predicate", "object")):
                triples.append(Triple(
                    subject=str(t["subject"]).strip(),
                    predicate=str(t["predicate"]).strip(),
                    object=str(t["object"]).strip(),
                ))

        logger.debug("Extracted %d triples from %s", len(triples), source_doc or "chunk")
        return triples

    except CircuitBreakerOpen as exc:
        logger.debug("LLM circuit open for KG extraction: %s", exc)
        return []
    except Exception:
        logger.debug("KG extraction failed", exc_info=True)
        return []
