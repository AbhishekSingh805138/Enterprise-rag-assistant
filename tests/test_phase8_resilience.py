"""Phase 8.1 tests: Resilience & Observability Foundation.

Covers:
  - LLM timeout / max_retries wiring
  - Grader error default (relevant=False)
  - Per-node tracing metric capture
  - Enhanced MetricsStore (IDK rate, grader rejection rate, latency percentiles, cost alerts)
  - IDK detection helper
  - QueryMetrics dataclass fields
  - API response node_latencies and is_idk
  - Config validation for new settings
"""
from __future__ import annotations

import sqlite3
import tempfile
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# TestConfigPhase8
# ---------------------------------------------------------------------------

class TestConfigPhase8:
    """Verify Phase 8 config additions and validation."""

    def test_llm_timeout_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.llm_timeout == 30

    def test_llm_max_retries_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.llm_max_retries == 2

    def test_cost_alert_threshold_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.cost_alert_threshold == 0.05

    def test_rerank_max_workers_default(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test")
        assert s.rerank_max_workers == 4

    def test_zero_llm_timeout_raises(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test", llm_timeout=0)
        with pytest.raises(ValueError, match="llm_timeout"):
            s.validate()

    def test_negative_llm_timeout_raises(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test", llm_timeout=-5)
        with pytest.raises(ValueError, match="llm_timeout"):
            s.validate()

    def test_negative_llm_max_retries_raises(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test", llm_max_retries=-1)
        with pytest.raises(ValueError, match="llm_max_retries"):
            s.validate()

    def test_zero_cost_alert_threshold_raises(self):
        from config import Settings
        s = Settings(openai_api_key="sk-test", cost_alert_threshold=0)
        with pytest.raises(ValueError, match="cost_alert_threshold"):
            s.validate()


# ---------------------------------------------------------------------------
# TestLLMTimeout
# ---------------------------------------------------------------------------

class TestLLMTimeout:
    """Verify _llm() passes timeout and max_retries from settings."""

    @patch("src.llm_pool.settings")
    def test_nodes_llm_passes_timeout(self, mock_settings):
        # _llm() delegates to the llm_pool, which reads src.llm_pool.settings
        # at instance-creation time and caches by (model, temperature) —
        # so patch the pool's settings and reset the pool around the test.
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_timeout = 45
        mock_settings.llm_max_retries = 3

        from src.llm_pool import reset_pool
        from src.graph.nodes import _llm

        reset_pool()
        try:
            llm = _llm()
            assert llm.request_timeout == 45
            assert llm.max_retries == 3
        finally:
            reset_pool()  # don't leak the mocked instance to other tests

    @patch("src.graph.nodes.settings")
    def test_nodes_llm_default_temperature(self, mock_settings):
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        from src.graph.nodes import _llm
        llm = _llm()
        assert llm.temperature == 0

    @patch("src.graph.nodes.settings")
    def test_nodes_llm_custom_temperature(self, mock_settings):
        mock_settings.llm_model = "gpt-4o-mini"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.llm_timeout = 30
        mock_settings.llm_max_retries = 2

        from src.graph.nodes import _llm
        llm = _llm(temperature=0.7)
        assert llm.temperature == 0.7


# ---------------------------------------------------------------------------
# TestGraderErrorDefault
# ---------------------------------------------------------------------------

class TestGraderErrorDefault:
    """Verify grader proceeds to generation (relevant=True) on transient errors.

    A grader exception is not evidence the documents are bad — defaulting
    to False would trigger the expensive rewrite→retry→web-search path on
    a transient failure. The critic still verifies claims downstream.
    """

    @patch("src.graph.nodes._llm")
    def test_grader_returns_true_on_exception(self, mock_llm):
        """When the grading chain raises, grade_documents should proceed to generate."""
        # Make with_structured_output return a chain whose invoke raises
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("LLM is down")
        mock_llm_instance = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured
        mock_llm.return_value = mock_llm_instance

        # Also need to patch the prompt pipe so (prompt | llm) yields our mock
        with patch("src.graph.nodes._grade_prompt") as mock_prompt:
            # prompt | llm.with_structured_output(...) needs to return something
            # whose invoke() raises. The pipe operator calls __or__ on the prompt.
            mock_chain = MagicMock()
            mock_chain.invoke.side_effect = RuntimeError("LLM is down")
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            from src.graph.nodes import grade_documents
            state = {
                "question": "test question",
                "documents": [Document(page_content="some content")],
            }
            result = grade_documents(state)
            assert result["relevant"] is True

    def test_grader_returns_false_on_empty_docs(self):
        """No documents -> relevant=False (no LLM call needed)."""
        from src.graph.nodes import grade_documents
        state = {"question": "test", "documents": []}
        result = grade_documents(state)
        assert result["relevant"] is False


# ---------------------------------------------------------------------------
# TestNodeMetrics
# ---------------------------------------------------------------------------

class TestNodeMetrics:
    """Verify the traced decorator records per-node metrics."""

    def setup_method(self):
        from src.graph.tracing import reset_node_metrics
        reset_node_metrics()

    def teardown_method(self):
        from src.graph.tracing import reset_node_metrics
        reset_node_metrics()

    def test_traced_records_timing(self):
        from src.graph.tracing import get_node_metrics, traced

        @traced
        def dummy_node(state: dict) -> dict:
            time.sleep(0.01)  # ~10ms
            return {"generation": "hello"}

        dummy_node({"question": "test"})
        metrics = get_node_metrics()
        assert "dummy_node" in metrics
        assert metrics["dummy_node"]["count"] == 1
        assert metrics["dummy_node"]["p50"] > 0

    def test_multiple_calls_accumulate(self):
        from src.graph.tracing import get_node_metrics, traced

        @traced
        def accumulator(state: dict) -> dict:
            return {"documents": []}

        for _ in range(5):
            accumulator({"question": "q"})

        metrics = get_node_metrics()
        assert metrics["accumulator"]["count"] == 5

    def test_get_last_run_latencies(self):
        from src.graph.tracing import get_last_run_latencies, traced

        @traced
        def latency_node(state: dict) -> dict:
            return {}

        latency_node({"question": "q"})
        latency_node({"question": "q"})

        lats = get_last_run_latencies()
        assert "latency_node" in lats
        assert isinstance(lats["latency_node"], float)

    def test_reset_clears_metrics(self):
        from src.graph.tracing import get_node_metrics, reset_node_metrics, traced

        @traced
        def resettable(state: dict) -> dict:
            return {}

        resettable({"question": "q"})
        assert "resettable" in get_node_metrics()

        reset_node_metrics()
        assert get_node_metrics() == {}

    def test_metrics_contain_expected_keys(self):
        from src.graph.tracing import get_node_metrics, traced

        @traced
        def keyed_node(state: dict) -> dict:
            return {"generation": "x"}

        keyed_node({"question": "q"})
        m = get_node_metrics()["keyed_node"]
        assert set(m.keys()) == {"count", "p50", "p95", "p99", "mean", "last"}


# ---------------------------------------------------------------------------
# TestIDKDetection
# ---------------------------------------------------------------------------

class TestIDKDetection:
    """Verify is_idk_response helper."""

    def test_detects_idk_phrases(self):
        from src.observability.cost_callback import is_idk_response

        assert is_idk_response("I don't have enough information to answer.")
        assert is_idk_response("I cannot answer this question based on the documents.")
        assert is_idk_response("There is no information available on this topic.")
        assert is_idk_response("Not enough information in the context.")
        assert is_idk_response("I am unable to answer based on the given context.")
        assert is_idk_response("No relevant information found in the documents.")

    def test_normal_answers_not_flagged(self):
        from src.observability.cost_callback import is_idk_response

        assert not is_idk_response("The company policy states that PTO is 20 days per year.")
        assert not is_idk_response("According to the handbook, remote work requires manager approval.")

    def test_case_insensitive(self):
        from src.observability.cost_callback import is_idk_response

        assert is_idk_response("I DON'T HAVE ENOUGH INFORMATION to answer.")


# ---------------------------------------------------------------------------
# TestQueryMetricsDataclass
# ---------------------------------------------------------------------------

class TestQueryMetricsDataclass:
    """Verify QueryMetrics has the new Phase 8 fields."""

    def test_has_is_idk_field(self):
        from src.observability.cost_callback import QueryMetrics
        m = QueryMetrics(
            thread_id="t1", question_preview="q", prompt_tokens=10,
            completion_tokens=5, total_tokens=15, estimated_cost_usd=0.001,
            latency_ms=100.0, retriever_strategy="dense", mode="naive",
        )
        assert m.is_idk is False  # default

    def test_has_grader_rejected_field(self):
        from src.observability.cost_callback import QueryMetrics
        m = QueryMetrics(
            thread_id="t1", question_preview="q", prompt_tokens=10,
            completion_tokens=5, total_tokens=15, estimated_cost_usd=0.001,
            latency_ms=100.0, retriever_strategy="dense", mode="graph",
            grader_rejected=1,
        )
        assert m.grader_rejected == 1

    def test_has_node_latencies_field(self):
        from src.observability.cost_callback import QueryMetrics
        lats = {"retrieve": 120.0, "generate": 300.0}
        m = QueryMetrics(
            thread_id="t1", question_preview="q", prompt_tokens=10,
            completion_tokens=5, total_tokens=15, estimated_cost_usd=0.001,
            latency_ms=100.0, retriever_strategy="dense", mode="graph",
            node_latencies=lats,
        )
        assert m.node_latencies == lats

    def test_is_dataclass(self):
        from src.observability.cost_callback import QueryMetrics
        field_names = {f.name for f in dataclass_fields(QueryMetrics)}
        assert "is_idk" in field_names
        assert "grader_rejected" in field_names
        assert "node_latencies" in field_names


# ---------------------------------------------------------------------------
# TestMetricsStoreEnhanced
# ---------------------------------------------------------------------------

class TestMetricsStoreEnhanced:
    """Verify enhanced MetricsStore methods from Phase 8."""

    @pytest.fixture()
    def store(self, tmp_path):
        from src.observability.metrics_store import MetricsStore
        db_path = str(tmp_path / "test_metrics.db")
        return MetricsStore(db_path)

    def _make_metrics(self, is_idk=False, grader_rejected=0, cost=0.001, latency=100.0):
        from src.observability.cost_callback import QueryMetrics
        return QueryMetrics(
            thread_id="test",
            question_preview="test question",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=cost,
            latency_ms=latency,
            retriever_strategy="dense",
            mode="naive",
            is_idk=is_idk,
            grader_rejected=grader_rejected,
        )

    def test_record_and_query_recent(self, store):
        store.record(self._make_metrics())
        recent = store.query_recent(5)
        assert len(recent) == 1
        assert recent[0]["cost_usd"] == 0.001

    def test_idk_rate_zero_when_no_idk(self, store):
        for _ in range(5):
            store.record(self._make_metrics(is_idk=False))
        assert store.idk_rate() == 0.0

    def test_idk_rate_calculated_correctly(self, store):
        store.record(self._make_metrics(is_idk=True))
        store.record(self._make_metrics(is_idk=False))
        store.record(self._make_metrics(is_idk=True))
        store.record(self._make_metrics(is_idk=False))
        rate = store.idk_rate()
        assert abs(rate - 0.5) < 0.01

    def test_idk_rate_with_limit(self, store):
        # First 3 non-idk, then 2 idk
        for _ in range(3):
            store.record(self._make_metrics(is_idk=False))
        for _ in range(2):
            store.record(self._make_metrics(is_idk=True))
        # Last 2 are both idk
        rate = store.idk_rate(n=2)
        assert abs(rate - 1.0) < 0.01

    def test_grader_rejection_rate(self, store):
        store.record(self._make_metrics(grader_rejected=1))
        store.record(self._make_metrics(grader_rejected=0))
        rate = store.grader_rejection_rate()
        assert abs(rate - 0.5) < 0.01

    def test_grader_rejection_rate_with_limit(self, store):
        store.record(self._make_metrics(grader_rejected=0))
        store.record(self._make_metrics(grader_rejected=1))
        rate = store.grader_rejection_rate(n=1)
        assert abs(rate - 1.0) < 0.01

    def test_latency_percentiles_empty(self, store):
        p = store.latency_percentiles()
        assert p == {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def test_latency_percentiles_basic(self, store):
        for lat in [100, 200, 300, 400, 500]:
            store.record(self._make_metrics(latency=lat))
        p = store.latency_percentiles()
        assert p["p50"] == 300.0

    def test_cost_alert_check_returns_expensive_queries(self, store):
        store.record(self._make_metrics(cost=0.001))
        store.record(self._make_metrics(cost=0.10))  # over default threshold
        alerts = store.cost_alert_check(threshold=0.05)
        assert len(alerts) == 1
        assert alerts[0]["cost_usd"] == 0.10

    def test_cost_alert_check_empty_when_under_threshold(self, store):
        store.record(self._make_metrics(cost=0.001))
        alerts = store.cost_alert_check(threshold=0.05)
        assert len(alerts) == 0

    def test_summary_includes_over_budget(self, store):
        store.record(self._make_metrics(cost=0.001))
        store.record(self._make_metrics(cost=0.03))  # over 0.02 budget
        s = store.summary()
        assert s["cnt"] == 2
        assert s["over_budget"] == 1

    def test_migration_idempotent(self, tmp_path):
        """Running migrations twice doesn't crash (idempotent ALTER TABLE)."""
        from src.observability.metrics_store import MetricsStore
        db_path = str(tmp_path / "test_migr.db")
        store1 = MetricsStore(db_path)
        store1.close()
        # Second init runs migrations again — should not raise
        store2 = MetricsStore(db_path)
        store2.close()


# ---------------------------------------------------------------------------
# TestAskResponseModel
# ---------------------------------------------------------------------------

class TestAskResponseModel:
    """Verify AskResponse has the new Phase 8 fields."""

    def test_node_latencies_optional(self):
        from api.models import AskResponse
        resp = AskResponse(
            answer="test", question="q", mode="naive",
            retriever_strategy="dense", cost_usd=0.001,
            latency_ms=100.0, tokens_used=50,
        )
        assert resp.node_latencies is None

    def test_node_latencies_populated(self):
        from api.models import AskResponse
        lats = {"retrieve": 100.0, "generate": 200.0}
        resp = AskResponse(
            answer="test", question="q", mode="graph",
            retriever_strategy="dense", cost_usd=0.001,
            latency_ms=300.0, tokens_used=50,
            node_latencies=lats,
        )
        assert resp.node_latencies == lats

    def test_is_idk_default_false(self):
        from api.models import AskResponse
        resp = AskResponse(
            answer="test", question="q", mode="naive",
            retriever_strategy="dense", cost_usd=0.001,
            latency_ms=100.0, tokens_used=50,
        )
        assert resp.is_idk is False

    def test_is_idk_set_true(self):
        from api.models import AskResponse
        resp = AskResponse(
            answer="idk", question="q", mode="naive",
            retriever_strategy="dense", cost_usd=0.001,
            latency_ms=100.0, tokens_used=50,
            is_idk=True,
        )
        assert resp.is_idk is True


# ---------------------------------------------------------------------------
# TestCostCallback
# ---------------------------------------------------------------------------

class TestCostCallback:
    """Verify CostCallbackHandler flush includes new fields."""

    def test_flush_returns_is_idk(self):
        from src.observability.cost_callback import CostCallbackHandler
        h = CostCallbackHandler()
        m = h.flush(
            thread_id="t", question="q", latency_ms=100,
            retriever_strategy="dense", mode="naive", is_idk=True,
        )
        assert m.is_idk is True

    def test_flush_returns_grader_rejected(self):
        from src.observability.cost_callback import CostCallbackHandler
        h = CostCallbackHandler()
        m = h.flush(
            thread_id="t", question="q", latency_ms=100,
            retriever_strategy="dense", mode="graph", grader_rejected=1,
        )
        assert m.grader_rejected == 1

    def test_flush_returns_node_latencies(self):
        from src.observability.cost_callback import CostCallbackHandler
        h = CostCallbackHandler()
        lats = {"retrieve": 50.0}
        m = h.flush(
            thread_id="t", question="q", latency_ms=100,
            retriever_strategy="dense", mode="graph", node_latencies=lats,
        )
        assert m.node_latencies == lats

    def test_flush_resets_counters(self):
        from src.observability.cost_callback import CostCallbackHandler
        h = CostCallbackHandler()
        h._prompt_tokens = 100
        h._completion_tokens = 50
        h._total_cost = 0.01
        m = h.flush(
            thread_id="t", question="q", latency_ms=100,
            retriever_strategy="dense", mode="naive",
        )
        assert m.total_tokens == 150
        # After flush, counters should be reset
        m2 = h.flush(
            thread_id="t", question="q", latency_ms=50,
            retriever_strategy="dense", mode="naive",
        )
        assert m2.total_tokens == 0
        assert m2.estimated_cost_usd == 0.0
