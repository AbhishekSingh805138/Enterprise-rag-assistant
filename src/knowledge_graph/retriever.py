"""Knowledge graph retriever for graph-based document retrieval.

Extracts entities from the query, finds matching graph nodes,
traverses neighbors, and returns source documents as context.
"""
from __future__ import annotations

import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphRetriever(BaseRetriever):
    """Retriever that uses the knowledge graph to find relevant documents."""

    k: int = 4
    depth: int = 2

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve documents by finding entities in the knowledge graph."""
        from src.knowledge_graph.store import get_kg_store
        from src.retrieval.entity_extractor import _regex_extract

        store = get_kg_store()

        # Extract entities from query using regex (fast, no LLM call)
        entities = _regex_extract(query)
        entity_names = [e.name for e in entities]

        # Also try direct substring search for any multi-word matches
        words = query.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):
                phrase = " ".join(words[i:j])
                if len(phrase) > 3:  # skip very short phrases
                    matches = store.search_entities(phrase)
                    entity_names.extend(matches)

        # Deduplicate
        entity_names = list(dict.fromkeys(entity_names))

        if not entity_names:
            logger.debug("No entities found in query for KG retrieval")
            return []

        # Query neighbors for each entity
        all_relations: list[dict] = []
        seen_sources: set[str] = set()

        for entity in entity_names[:5]:  # limit entity lookups
            relations = store.query_neighbors(entity, depth=self.depth)
            for rel in relations:
                source = rel.get("source", "")
                if source and source not in seen_sources:
                    seen_sources.add(source)
                    all_relations.append(rel)

        if not all_relations:
            logger.debug("No graph relations found for entities: %s", entity_names[:5])
            return []

        # Convert relations to documents
        docs: list[Document] = []
        for rel in all_relations[:self.k]:
            content = f"{rel['subject']} {rel['predicate']} {rel['object']}"
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": rel.get("source", "knowledge_graph"),
                    "filename": rel.get("source", "knowledge_graph"),
                    "kg_subject": rel["subject"],
                    "kg_predicate": rel["predicate"],
                    "kg_object": rel["object"],
                },
            ))

        logger.info("KG retriever returned %d documents from %d relations",
                     len(docs), len(all_relations))
        return docs


def build_kg_retriever(k: int = 4, depth: int | None = None) -> KnowledgeGraphRetriever:
    """Build a knowledge graph retriever."""
    return KnowledgeGraphRetriever(
        k=k,
        depth=depth or settings.kg_max_depth,
    )
