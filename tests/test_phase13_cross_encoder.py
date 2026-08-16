"""Phase 13: Cross-Encoder Reranker tests.

Tests for:
- CrossEncoderRetriever with mock model
- Factory registration of new strategies
- Composed retriever with cross-encoder option
- Config fields
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# CrossEncoderRetriever tests (with mocked model)
# ---------------------------------------------------------------------------


class TestCrossEncoderRetriever:
    """Test the cross-encoder reranking retriever."""

    @pytest.fixture(autouse=True)
    def reset_model(self):
        """Reset the cached cross-encoder model between tests."""
        import src.retrieval.cross_encoder_rerank as ce_mod
        ce_mod._cross_encoder = None
        yield
        ce_mod._cross_encoder = None

    def _make_docs(self, n=5) -> list[Document]:
        return [
            Document(page_content=f"doc-{i} content about topic", metadata={"filename": f"doc{i}.md"})
            for i in range(n)
        ]

    @patch("src.retrieval.cross_encoder_rerank._get_cross_encoder")
    @patch("src.vectorstore.chroma_store.get_retriever")
    def test_reranks_by_score(self, mock_dense, mock_ce):
        from src.retrieval.cross_encoder_rerank import CrossEncoderRetriever

        docs = self._make_docs(4)
        mock_dense.return_value.invoke.return_value = docs

        # Cross-encoder returns scores — doc-2 should rank highest
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.3, 0.9, 0.5]
        mock_ce.return_value = mock_model

        retriever = CrossEncoderRetriever(k=2, fetch_k=4)
        result = retriever.invoke("test query")

        assert len(result) == 2
        assert result[0].page_content == "doc-2 content about topic"
        assert result[1].page_content == "doc-3 content about topic"

    @patch("src.retrieval.cross_encoder_rerank._get_cross_encoder")
    @patch("src.vectorstore.chroma_store.get_retriever")
    def test_returns_empty_for_no_candidates(self, mock_dense, mock_ce):
        from src.retrieval.cross_encoder_rerank import CrossEncoderRetriever

        mock_dense.return_value.invoke.return_value = []
        retriever = CrossEncoderRetriever(k=4, fetch_k=10)
        result = retriever.invoke("test query")
        assert result == []

    @patch("src.retrieval.cross_encoder_rerank._get_cross_encoder")
    @patch("src.vectorstore.chroma_store.get_retriever")
    def test_batch_scoring(self, mock_dense, mock_ce):
        from src.retrieval.cross_encoder_rerank import CrossEncoderRetriever

        docs = self._make_docs(3)
        mock_dense.return_value.invoke.return_value = docs

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.8, 0.2]
        mock_ce.return_value = mock_model

        retriever = CrossEncoderRetriever(k=3, fetch_k=5)
        retriever.invoke("test")

        # Should have been called with 3 pairs
        call_args = mock_model.predict.call_args
        pairs = call_args[0][0]
        assert len(pairs) == 3
        assert all(p[0] == "test" for p in pairs)


# ---------------------------------------------------------------------------
# Factory registration tests
# ---------------------------------------------------------------------------


class TestFactoryRegistration:
    """Test that new strategies are registered correctly."""

    def test_cross_rerank_in_registry(self):
        from src.retrieval.factory import _REGISTRY, STRATEGIES
        assert "cross_rerank" in _REGISTRY
        assert "cross_rerank" in STRATEGIES

    def test_hybrid_cross_rerank_in_registry(self):
        from src.retrieval.factory import _REGISTRY, STRATEGIES
        assert "hybrid_cross_rerank" in _REGISTRY
        assert "hybrid_cross_rerank" in STRATEGIES

    def test_original_strategies_preserved(self):
        from src.retrieval.factory import STRATEGIES
        for s in ("dense", "hybrid", "multi_query", "rerank", "hybrid_rerank"):
            assert s in STRATEGIES

    def test_unknown_strategy_raises(self):
        from src.retrieval.factory import get_retriever
        with pytest.raises(ValueError, match="Unknown"):
            get_retriever("nonexistent_strategy")


# ---------------------------------------------------------------------------
# Composed retriever with cross-encoder
# ---------------------------------------------------------------------------


class TestComposedCrossEncoder:
    """Test the composed retriever with cross-encoder reranker type."""

    def test_composed_accepts_reranker_type(self):
        from src.retrieval.composed import ComposedRetriever
        r = ComposedRetriever(k=4, reranker_type="cross_encoder")
        assert r.reranker_type == "cross_encoder"

    def test_composed_default_is_llm(self):
        from src.retrieval.composed import ComposedRetriever
        r = ComposedRetriever(k=4)
        assert r.reranker_type == "llm"

    def test_build_composed_with_reranker_type(self):
        from src.retrieval.composed import build_composed_retriever
        r = build_composed_retriever(k=4, reranker_type="cross_encoder")
        assert r.reranker_type == "cross_encoder"


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


class TestCrossEncoderConfig:
    """Test cross-encoder config fields exist."""

    def test_config_fields_exist(self):
        from config import settings
        assert hasattr(settings, "cross_encoder_model")
        assert hasattr(settings, "cross_encoder_device")
        assert hasattr(settings, "cross_encoder_batch_size")

    def test_default_model(self):
        from config import settings
        assert "ms-marco" in settings.cross_encoder_model or "MiniLM" in settings.cross_encoder_model

    def test_default_device(self):
        from config import settings
        assert settings.cross_encoder_device == "cpu"

    def test_default_batch_size(self):
        from config import settings
        assert settings.cross_encoder_batch_size == 16


# ---------------------------------------------------------------------------
# Builder function
# ---------------------------------------------------------------------------


class TestBuildCrossEncoderRetriever:
    """Test the builder function."""

    def test_builds_with_defaults(self):
        from src.retrieval.cross_encoder_rerank import build_cross_encoder_retriever
        r = build_cross_encoder_retriever()
        assert r.k > 0
        assert r.fetch_k >= r.k

    def test_custom_k(self):
        from src.retrieval.cross_encoder_rerank import build_cross_encoder_retriever
        r = build_cross_encoder_retriever(k=8)
        assert r.k == 8

    def test_fetch_k_scales_with_k(self):
        from src.retrieval.cross_encoder_rerank import build_cross_encoder_retriever
        r = build_cross_encoder_retriever(k=3)
        assert r.fetch_k >= 9  # at least 3x

    def test_filter_passed_through(self):
        from src.retrieval.cross_encoder_rerank import build_cross_encoder_retriever
        r = build_cross_encoder_retriever(filter={"department": "hr"})
        assert r.filter == {"department": "hr"}
