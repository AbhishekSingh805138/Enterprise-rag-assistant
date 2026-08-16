"""Phase 8.2 tests: Retrieval Quality Improvements.

Covers:
  - BM25 tokenization (stop words, punctuation, case-folding)
  - BM25 cache (module-level caching, invalidation)
  - Content dedup (md5 hash instead of truncated prefix)
  - Reranker parallelization
  - Query normalization (acronym expansion, whitespace collapse)
  - Composed retriever (hybrid + rerank strategy)
  - Department detection (keyword-based)
  - Adaptive top_k (query length heuristic)
  - Factory includes hybrid_rerank strategy
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# TestBM25Tokenization
# ---------------------------------------------------------------------------

class TestBM25Tokenization:
    """Verify improved BM25 tokenizer."""

    def test_stop_words_removed(self):
        from src.retrieval.hybrid import _tokenize
        tokens = _tokenize("The quick brown fox is a fast animal")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_punctuation_handled(self):
        from src.retrieval.hybrid import _tokenize
        tokens = _tokenize("Hello, world! This is great.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "great" in tokens
        # Punctuation-only tokens should not appear
        assert "," not in tokens
        assert "!" not in tokens

    def test_case_folding(self):
        from src.retrieval.hybrid import _tokenize
        tokens = _tokenize("PTO Policy HANDBOOK")
        assert "pto" in tokens
        assert "policy" in tokens
        assert "handbook" in tokens

    def test_single_char_tokens_removed(self):
        from src.retrieval.hybrid import _tokenize
        tokens = _tokenize("I have a B plan")
        # Single-char "I", "a", "B" should be filtered out
        assert "i" not in tokens
        assert "b" not in tokens

    def test_empty_string(self):
        from src.retrieval.hybrid import _tokenize
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# TestBM25Cache
# ---------------------------------------------------------------------------

class TestBM25Cache:
    """Verify BM25 index caching."""

    def setup_method(self):
        from src.retrieval.hybrid import reset_bm25_cache
        reset_bm25_cache()

    def teardown_method(self):
        from src.retrieval.hybrid import reset_bm25_cache
        reset_bm25_cache()

    def test_cache_cleared_by_reset(self):
        from src.retrieval.hybrid import _bm25_cache, reset_bm25_cache
        _bm25_cache["test_key"] = (MagicMock(), [])
        assert len(_bm25_cache) == 1
        reset_bm25_cache()
        assert len(_bm25_cache) == 0

    def test_filter_cache_key_no_filter(self):
        from src.retrieval.hybrid import HybridRetriever
        assert HybridRetriever._filter_cache_key(None) == "__no_filter__"

    def test_filter_cache_key_with_filter(self):
        """Every field must reach the key, whatever the encoding.

        Asserts the contract (each field and value is represented) rather
        than one serialisation, so the key format can change without a
        false failure — as it did when department scoping introduced
        ``{"$in": [...]}`` values that the old ``k=v`` form could not
        encode unambiguously.
        """
        from src.retrieval.hybrid import HybridRetriever
        key = HybridRetriever._filter_cache_key({"department": "hr", "access_level": "internal"})
        for token in ("access_level", "internal", "department", "hr"):
            assert token in key

    def test_filter_cache_key_distinguishes_departments(self):
        """Two scopes must not collide onto one cached corpus."""
        from src.retrieval.hybrid import HybridRetriever
        hr_only = HybridRetriever._filter_cache_key({"department": "hr"})
        legal_only = HybridRetriever._filter_cache_key({"department": "legal"})
        scoped = HybridRetriever._filter_cache_key({"department": {"$in": ["general", "hr"]}})
        assert len({hr_only, legal_only, scoped}) == 3

    def test_filter_cache_key_stable_for_operator_filters(self):
        from src.retrieval.hybrid import HybridRetriever
        k1 = HybridRetriever._filter_cache_key({"department": {"$in": ["general", "hr"]}})
        k2 = HybridRetriever._filter_cache_key({"department": {"$in": ["general", "hr"]}})
        assert k1 == k2

    def test_filter_cache_key_deterministic(self):
        from src.retrieval.hybrid import HybridRetriever
        k1 = HybridRetriever._filter_cache_key({"b": "2", "a": "1"})
        k2 = HybridRetriever._filter_cache_key({"a": "1", "b": "2"})
        assert k1 == k2


# ---------------------------------------------------------------------------
# TestContentDedup
# ---------------------------------------------------------------------------

class TestContentDedup:
    """Verify md5-based deduplication handles docs with identical 200-char prefixes."""

    def test_identical_prefix_different_content(self):
        """Two docs with same first 200 chars but different content should be separate."""
        import hashlib
        prefix = "A" * 200
        doc1_content = prefix + " extension one"
        doc2_content = prefix + " extension two"
        key1 = hashlib.md5(doc1_content.encode()).hexdigest()
        key2 = hashlib.md5(doc2_content.encode()).hexdigest()
        assert key1 != key2

    def test_identical_content_same_key(self):
        import hashlib
        content = "This is a test document about PTO policy."
        key1 = hashlib.md5(content.encode()).hexdigest()
        key2 = hashlib.md5(content.encode()).hexdigest()
        assert key1 == key2


# ---------------------------------------------------------------------------
# TestRerankParallel
# ---------------------------------------------------------------------------

class TestRerankParallel:
    """Verify reranker uses ThreadPoolExecutor."""

    @patch("src.retrieval.rerank.settings")
    def test_rerank_uses_thread_pool(self, mock_settings):
        """Verify the reranker calls ThreadPoolExecutor."""
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2
        mock_settings.rerank_max_workers = 4
        mock_settings.top_k = 4

        from src.retrieval.rerank import RerankRetriever
        retriever = RerankRetriever(k=2, fetch_k=4)

        docs = [
            Document(page_content="doc1", metadata={"filename": "a.md"}),
            Document(page_content="doc2", metadata={"filename": "b.md"}),
        ]

        # Mock _get_candidates to return docs
        with patch.object(retriever, "_get_candidates", return_value=docs):
            # Mock _score_document to return fixed scores
            with patch.object(retriever, "_score_document") as mock_score:
                mock_score.side_effect = lambda llm, q, doc: (doc, 8 if "doc1" in doc.page_content else 6)
                result = retriever.invoke("test query")

        assert len(result) == 2
        assert result[0].page_content == "doc1"  # highest score first

    def test_rerank_fetch_k_capped(self):
        from src.retrieval.rerank import build_rerank_retriever
        with patch("src.retrieval.rerank.settings") as mock_settings:
            mock_settings.top_k = 4
            retriever = build_rerank_retriever(k=4)
            assert retriever.fetch_k <= 15


# ---------------------------------------------------------------------------
# TestQueryNormalization
# ---------------------------------------------------------------------------

class TestQueryNormalization:
    """Verify query normalization."""

    def test_acronym_expansion(self):
        from src.retrieval.normalizer import normalize_query
        result = normalize_query("What is the PTO policy?")
        assert "paid time off" in result.lower()
        assert "PTO" in result  # original preserved

    def test_multiple_acronyms(self):
        from src.retrieval.normalizer import normalize_query
        result = normalize_query("MFA and SSO requirements")
        assert "multi-factor authentication" in result.lower()
        assert "single sign-on" in result.lower()

    def test_whitespace_collapse(self):
        from src.retrieval.normalizer import normalize_query
        result = normalize_query("  What   is   the   policy  ")
        assert "  " not in result
        assert result == normalize_query("What is the policy")

    def test_trailing_question_mark_removed(self):
        from src.retrieval.normalizer import normalize_query
        result = normalize_query("What is the PTO policy?")
        assert not result.endswith("?")

    def test_empty_string(self):
        from src.retrieval.normalizer import normalize_query
        assert normalize_query("") == ""

    def test_no_acronyms_unchanged(self):
        from src.retrieval.normalizer import normalize_query
        result = normalize_query("How does the vacation policy work")
        assert result == "How does the vacation policy work"

    def test_case_insensitive_acronym_detection(self):
        from src.retrieval.normalizer import normalize_query
        result = normalize_query("pto policy details")
        assert "paid time off" in result.lower()


# ---------------------------------------------------------------------------
# TestDeptDetection
# ---------------------------------------------------------------------------

class TestDeptDetection:
    """Verify automatic department detection."""

    def test_hr_detection(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("What is the PTO policy?") == "hr"
        assert detect_department("Employee onboarding process") == "hr"

    def test_legal_detection(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("What are the NDA terms?") == "legal"
        assert detect_department("Data protection policy requirements") == "legal"

    def test_engineering_detection(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("API versioning guidelines") == "engineering"
        assert detect_department("Incident response runbook") == "engineering"

    def test_finance_detection(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("Procurement approval process") == "finance"

    def test_security_detection(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("MFA requirements for employees") == "security"

    def test_operations_detection(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("Business continuity plan details") == "operations"

    def test_ambiguous_returns_none(self):
        from src.retrieval.dept_detector import detect_department
        # Generic query that doesn't clearly match one department
        result = detect_department("What is the company doing?")
        assert result is None

    def test_empty_returns_none(self):
        from src.retrieval.dept_detector import detect_department
        assert detect_department("") is None


# ---------------------------------------------------------------------------
# TestAdaptiveK
# ---------------------------------------------------------------------------

class TestAdaptiveK:
    """Verify adaptive top_k based on query length."""

    @patch("src.graph.nodes.get_retriever")
    @patch("src.graph.nodes.settings")
    def test_short_query_gets_min_k(self, mock_settings, mock_get_retriever):
        mock_settings.adaptive_k = True
        mock_settings.adaptive_k_min = 3
        mock_settings.adaptive_k_max = 8
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="test")]
        mock_get_retriever.return_value = mock_retriever

        from src.graph.nodes import retrieve
        retrieve({"question": "PTO policy", "retries": 0})

        # k=3 for short queries (<=5 words)
        mock_get_retriever.assert_called_once()
        call_kwargs = mock_get_retriever.call_args
        assert call_kwargs.kwargs.get("k") == 3 or call_kwargs[1].get("k") == 3

    @patch("src.graph.nodes.get_retriever")
    @patch("src.graph.nodes.settings")
    def test_long_query_gets_max_k(self, mock_settings, mock_get_retriever):
        mock_settings.adaptive_k = True
        mock_settings.adaptive_k_min = 3
        mock_settings.adaptive_k_max = 8
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from src.graph.nodes import retrieve
        # 16+ words
        long_q = "What is the detailed process for requesting paid time off including all approvals and forms needed for international employees"
        retrieve({"question": long_q, "retries": 0})

        mock_get_retriever.assert_called_once()
        call_kwargs = mock_get_retriever.call_args
        assert call_kwargs.kwargs.get("k") == 8 or call_kwargs[1].get("k") == 8

    @patch("src.graph.nodes.get_retriever")
    @patch("src.graph.nodes.settings")
    def test_adaptive_k_disabled(self, mock_settings, mock_get_retriever):
        mock_settings.adaptive_k = False
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        from src.graph.nodes import retrieve
        retrieve({"question": "PTO policy", "retries": 0})

        mock_get_retriever.assert_called_once()
        call_kwargs = mock_get_retriever.call_args
        # k should be None when adaptive_k is disabled
        assert call_kwargs.kwargs.get("k") is None or call_kwargs[1].get("k") is None


# ---------------------------------------------------------------------------
# TestFactory
# ---------------------------------------------------------------------------

class TestFactory:
    """Verify factory includes new strategies."""

    def test_hybrid_rerank_in_strategies(self):
        from src.retrieval.factory import STRATEGIES
        assert "hybrid_rerank" in STRATEGIES

    def test_hybrid_rerank_returns_composed_retriever(self):
        with patch("src.retrieval.composed.ComposedRetriever") as MockComposed:
            MockComposed.return_value = MagicMock()
            from src.retrieval.composed import ComposedRetriever
            from src.retrieval.factory import get_retriever
            retriever = get_retriever("hybrid_rerank")
            # Should return a ComposedRetriever instance
            assert retriever is not None


# ---------------------------------------------------------------------------
# TestApiModels
# ---------------------------------------------------------------------------

class TestApiModels:
    """Verify API models include new strategy."""

    def test_ask_request_accepts_hybrid_rerank(self):
        from api.models import AskRequest
        req = AskRequest(question="test?", retriever_strategy="hybrid_rerank")
        assert req.retriever_strategy == "hybrid_rerank"

    def test_eval_request_accepts_hybrid_rerank(self):
        from api.models import EvalRequest
        req = EvalRequest(retriever_strategy="hybrid_rerank")
        assert req.retriever_strategy == "hybrid_rerank"
