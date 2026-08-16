"""Tests for Phase 3 retrieval strategies.

Covers:
  - Factory validation and routing
  - HybridRetriever RRF fusion logic
  - MultiQueryRetriever deduplication
  - RerankRetriever scoring and sorting
  - All strategies return list[Document]
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.factory import STRATEGIES, get_retriever

# === Factory tests ===

class TestFactory:
    def test_known_strategies(self):
        assert "dense" in STRATEGIES
        assert "hybrid" in STRATEGIES
        assert "multi_query" in STRATEGIES
        assert "rerank" in STRATEGIES

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            get_retriever(strategy="magic")

    @patch("src.retrieval.factory.get_retriever", wraps=get_retriever)
    def test_dense_delegates_to_chroma(self, mock_factory):
        """Dense strategy should call through to chroma_store.get_retriever."""
        with patch("src.vectorstore.chroma_store.get_retriever") as mock_dense:
            mock_dense.return_value = MagicMock()
            result = get_retriever(strategy="dense", k=5)
            mock_dense.assert_called_once_with(k=5, filter=None)


# === Hybrid retriever tests ===

class TestHybridRetriever:
    def _make_docs(self, texts: list[str]) -> list[Document]:
        return [
            Document(page_content=t, metadata={"filename": f"doc{i}.md"})
            for i, t in enumerate(texts)
        ]

    def test_rrf_fusion_ranks_correctly(self):
        """Doc appearing in both lists should rank higher than docs in only one."""
        from src.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(k=3)

        # Dense: A, B, C
        dense = self._make_docs(["Alpha document", "Beta document", "Charlie document"])
        # BM25: B, D, A (B appears first in both, A appears in both)
        bm25 = [
            (Document(page_content="Beta document", metadata={"filename": "doc1.md"}), 5.0),
            (Document(page_content="Delta document", metadata={"filename": "doc3.md"}), 3.0),
            (Document(page_content="Alpha document", metadata={"filename": "doc0.md"}), 1.0),
        ]

        fused = retriever._rrf_fuse(dense, bm25)
        texts = [d.page_content for d in fused]

        # Beta and Alpha appear in both lists → highest RRF scores
        assert texts[0] in ("Beta document", "Alpha document")
        assert len(fused) == 3

    def test_rrf_fusion_limits_to_k(self):
        from src.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(k=2)
        dense = self._make_docs(["A", "B", "C", "D"])
        bm25 = [(d, 1.0) for d in self._make_docs(["E", "F", "G"])]

        fused = retriever._rrf_fuse(dense, bm25)
        assert len(fused) == 2

    def test_rrf_fusion_handles_empty_bm25(self):
        from src.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(k=2)
        dense = self._make_docs(["A", "B"])

        fused = retriever._rrf_fuse(dense, [])
        assert len(fused) == 2

    def test_rrf_fusion_handles_empty_dense(self):
        from src.retrieval.hybrid import HybridRetriever

        retriever = HybridRetriever(k=2)
        bm25 = [(d, 1.0) for d in self._make_docs(["A", "B"])]

        fused = retriever._rrf_fuse([], bm25)
        assert len(fused) == 2

    def test_tokenize(self):
        from src.retrieval.hybrid import _tokenize

        tokens = _tokenize("Hello World FOO")
        assert tokens == ["hello", "world", "foo"]


# === Multi-query retriever tests ===

class TestMultiQueryRetriever:
    def test_deduplication(self):
        """Same document returned by multiple queries should appear only once."""
        from src.retrieval.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(k=4)

        shared_doc = Document(page_content="Shared document content", metadata={})
        unique_doc = Document(page_content="Unique to query 2", metadata={})

        with patch.object(retriever, "_get_dense_results") as mock_dense, \
             patch("src.retrieval.multi_query._generate_variants", return_value=["variant1"]):
            # Both the original query and variant return the shared doc
            mock_dense.side_effect = [
                [shared_doc],  # original query
                [shared_doc, unique_doc],  # variant
            ]
            results = retriever._get_relevant_documents("test query")

        texts = [d.page_content for d in results]
        assert texts.count("Shared document content") == 1
        assert "Unique to query 2" in texts

    def test_fallback_on_expansion_failure(self):
        """If LLM expansion fails, should still return results from original query."""
        from src.retrieval.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(k=4)
        doc = Document(page_content="Result from original", metadata={})

        with patch.object(retriever, "_get_dense_results", return_value=[doc]), \
             patch("src.retrieval.multi_query._generate_variants", side_effect=Exception("LLM down")):
            results = retriever._get_relevant_documents("test query")

        assert len(results) == 1
        assert results[0].page_content == "Result from original"

    def test_limits_to_k(self):
        from src.retrieval.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(k=2)
        docs = [
            Document(page_content=f"Doc {i}", metadata={})
            for i in range(10)
        ]

        with patch.object(retriever, "_get_dense_results", return_value=docs), \
             patch("src.retrieval.multi_query._generate_variants", return_value=[]):
            results = retriever._get_relevant_documents("test")

        assert len(results) == 2


# === Rerank retriever tests ===

class TestRerankRetriever:
    def test_rerank_sorts_by_score(self):
        """Documents should be returned in descending score order."""
        from src.retrieval.rerank import RelevanceScore, RerankRetriever

        retriever = RerankRetriever(k=2, fetch_k=4)

        docs = [
            Document(page_content="Low relevance", metadata={"filename": "a.md"}),
            Document(page_content="High relevance", metadata={"filename": "b.md"}),
            Document(page_content="Medium relevance", metadata={"filename": "c.md"}),
        ]

        scores = [
            RelevanceScore(score=2, reasoning="off-topic"),
            RelevanceScore(score=9, reasoning="directly answers"),
            RelevanceScore(score=6, reasoning="somewhat related"),
        ]

        with patch.object(retriever, "_get_candidates", return_value=docs):
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value.invoke.side_effect = scores

            with patch("src.retrieval.rerank.get_llm", return_value=mock_llm):
                results = retriever._get_relevant_documents("test query")

        assert len(results) == 2
        assert results[0].page_content == "High relevance"
        assert results[1].page_content == "Medium relevance"

    def test_rerank_handles_empty_candidates(self):
        from src.retrieval.rerank import RerankRetriever

        retriever = RerankRetriever(k=4, fetch_k=12)
        with patch.object(retriever, "_get_candidates", return_value=[]):
            results = retriever._get_relevant_documents("test")
        assert results == []

    def test_relevance_score_model(self):
        from src.retrieval.rerank import RelevanceScore

        score = RelevanceScore(score=8, reasoning="Directly relevant")
        assert score.score == 8
        assert score.reasoning == "Directly relevant"

    def test_build_rerank_retriever_default_k(self):
        from src.retrieval.rerank import build_rerank_retriever

        retriever = build_rerank_retriever()
        assert retriever.k == 4  # settings.top_k default
        assert retriever.fetch_k >= retriever.k * 3


# === Build function tests ===

class TestBuildFunctions:
    def test_build_hybrid_retriever(self):
        from src.retrieval.hybrid import build_hybrid_retriever

        r = build_hybrid_retriever(k=6)
        assert r.k == 6

    def test_build_multi_query_retriever(self):
        from src.retrieval.multi_query import build_multi_query_retriever

        r = build_multi_query_retriever(k=8)
        assert r.k == 8

    def test_build_rerank_retriever(self):
        from src.retrieval.rerank import build_rerank_retriever

        r = build_rerank_retriever(k=5)
        assert r.k == 5
        assert r.fetch_k == 15  # 5 * 3
