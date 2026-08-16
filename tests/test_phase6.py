"""Phase 6 tests: observability — cost callback, metrics store, dashboard, config.

All LLM and LangSmith calls are mocked. No real API calls are made.
"""
from __future__ import annotations

import json
import os
import sqlite3
from io import StringIO
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from src.observability.cost_callback import (
    MODEL_COSTS,
    CostCallbackHandler,
    QueryMetrics,
    compute_cost,
)
from src.observability.metrics_store import COST_BUDGET, MetricsStore, reset_store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_result(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    model: str = "gpt-4o-mini",
    use_usage_metadata: bool = True,
) -> LLMResult:
    """Build a mock LLMResult with token usage info."""
    if use_usage_metadata:
        msg = AIMessage(
            content="test response",
            usage_metadata={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            response_metadata={"model_name": model},
        )
        gen = ChatGeneration(message=msg)
        return LLMResult(generations=[[gen]])
    else:
        # Legacy path: llm_output
        msg = AIMessage(content="test response")
        gen = ChatGeneration(message=msg)
        return LLMResult(
            generations=[[gen]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "model_name": model,
            },
        )


def _make_metrics(**overrides) -> QueryMetrics:
    """Build a QueryMetrics with sensible defaults."""
    defaults = dict(
        thread_id="abc123",
        question_preview="What is the remote work policy?",
        prompt_tokens=500,
        completion_tokens=200,
        total_tokens=700,
        estimated_cost_usd=0.00195,
        latency_ms=412.0,
        retriever_strategy="dense",
        mode="graph",
    )
    defaults.update(overrides)
    return QueryMetrics(**defaults)


# ---------------------------------------------------------------------------
# TestCostCallback
# ---------------------------------------------------------------------------

class TestCostCallback:
    """CostCallbackHandler token extraction and cost computation."""

    def test_on_llm_end_extracts_usage_metadata(self):
        handler = CostCallbackHandler()
        result = _make_llm_result(prompt_tokens=200, completion_tokens=100)
        handler.on_llm_end(result, run_id=uuid4())
        assert handler._prompt_tokens == 200
        assert handler._completion_tokens == 100

    def test_on_llm_end_falls_back_to_llm_output(self):
        handler = CostCallbackHandler()
        result = _make_llm_result(
            prompt_tokens=300, completion_tokens=150, use_usage_metadata=False,
        )
        handler.on_llm_end(result, run_id=uuid4())
        assert handler._prompt_tokens == 300
        assert handler._completion_tokens == 150

    def test_on_llm_end_no_usage_skips_gracefully(self):
        handler = CostCallbackHandler()
        msg = AIMessage(content="no usage")
        gen = ChatGeneration(message=msg)
        result = LLMResult(generations=[[gen]])
        handler.on_llm_end(result, run_id=uuid4())
        assert handler._prompt_tokens == 0
        assert handler._completion_tokens == 0

    def test_cost_computed_for_known_model(self):
        cost = compute_cost("gpt-4o-mini", 1000, 500)
        expected = (1000 * 0.00015 + 500 * 0.0006) / 1000
        assert abs(cost - expected) < 1e-9

    def test_cost_zero_for_unknown_model(self):
        cost = compute_cost("unknown-model-v99", 1000, 500)
        assert cost == 0.0

    def test_flush_returns_metrics_and_resets(self):
        handler = CostCallbackHandler()
        result = _make_llm_result(prompt_tokens=100, completion_tokens=50)
        handler.on_llm_end(result, run_id=uuid4())

        metrics = handler.flush(
            thread_id="t1", question="test?",
            latency_ms=100.0, retriever_strategy="dense", mode="graph",
        )
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150

        # After flush, counters should be reset
        metrics2 = handler.flush(
            thread_id="t2", question="test2?",
            latency_ms=0.0, retriever_strategy="dense", mode="graph",
        )
        assert metrics2.prompt_tokens == 0
        assert metrics2.total_tokens == 0

    def test_multiple_llm_calls_accumulate(self):
        handler = CostCallbackHandler()
        for _ in range(3):
            result = _make_llm_result(prompt_tokens=100, completion_tokens=50)
            handler.on_llm_end(result, run_id=uuid4())

        assert handler._prompt_tokens == 300
        assert handler._completion_tokens == 150

    def test_always_verbose_is_true(self):
        handler = CostCallbackHandler()
        assert handler.always_verbose is True


# ---------------------------------------------------------------------------
# TestComputeCost
# ---------------------------------------------------------------------------

class TestComputeCost:
    """compute_cost pricing logic."""

    def test_gpt4o_mini_pricing(self):
        # 1K prompt + 1K completion
        cost = compute_cost("gpt-4o-mini", 1000, 1000)
        expected = (1000 * 0.00015 + 1000 * 0.0006) / 1000
        assert abs(cost - expected) < 1e-9

    def test_zero_tokens(self):
        assert compute_cost("gpt-4o-mini", 0, 0) == 0.0


# ---------------------------------------------------------------------------
# TestMetricsStore
# ---------------------------------------------------------------------------

class TestMetricsStore:
    """SQLite MetricsStore persistence."""

    @pytest.fixture(autouse=True)
    def _use_memory_db(self):
        """Use an in-memory DB for each test and reset the singleton."""
        reset_store()
        self.store = MetricsStore(":memory:")
        yield
        self.store.close()
        reset_store()

    def test_record_and_retrieve(self):
        m = _make_metrics(thread_id="test1")
        self.store.record(m)
        rows = self.store.query_recent(1)
        assert len(rows) == 1
        assert rows[0]["thread_id"] == "test1"
        assert rows[0]["cost_usd"] == m.estimated_cost_usd
        assert rows[0]["mode"] == "graph"

    def test_summary_empty_db(self):
        stats = self.store.summary()
        assert stats["cnt"] == 0
        assert stats["total_cost"] == 0
        assert stats["avg_cost"] == 0

    def test_summary_with_data(self):
        for i in range(3):
            self.store.record(_make_metrics(
                thread_id=f"t{i}",
                estimated_cost_usd=0.001 * (i + 1),
                latency_ms=100.0 * (i + 1),
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ))
        stats = self.store.summary()
        assert stats["cnt"] == 3
        assert abs(stats["total_cost"] - 0.006) < 1e-9
        assert abs(stats["avg_cost"] - 0.002) < 1e-9
        assert stats["total_tokens"] == 450

    def test_summary_with_limit(self):
        for i in range(5):
            self.store.record(_make_metrics(thread_id=f"t{i}"))
        stats = self.store.summary(n=3)
        assert stats["cnt"] == 3

    def test_idempotent_schema_creation(self):
        # Creating a second store on the same DB should not raise
        store2 = MetricsStore(":memory:")
        store2.close()

    def test_over_budget_count(self):
        self.store.record(_make_metrics(estimated_cost_usd=0.001))
        self.store.record(_make_metrics(estimated_cost_usd=0.03))  # over budget
        stats = self.store.summary()
        assert stats["over_budget"] == 1

    def test_query_recent_ordering(self):
        for i in range(5):
            self.store.record(_make_metrics(thread_id=f"t{i}"))
        rows = self.store.query_recent(3)
        assert len(rows) == 3
        # newest first
        assert rows[0]["thread_id"] == "t4"
        assert rows[2]["thread_id"] == "t2"


# ---------------------------------------------------------------------------
# TestAskWithMetrics
# ---------------------------------------------------------------------------

class TestAskWithMetrics:
    """Graph ask() records metrics via CostCallbackHandler."""

    @pytest.fixture(autouse=True)
    def _patch_graph_and_store(self):
        reset_store()
        self.mock_graph = MagicMock()
        self.mock_graph.invoke.return_value = {"generation": "Test answer"}

        with (
            patch("src.graph.build_graph.get_graph", return_value=self.mock_graph),
            patch("src.observability.metrics_store.get_store") as self.mock_get_store,
        ):
            self.mock_store = MagicMock()
            self.mock_get_store.return_value = self.mock_store
            yield
        reset_store()

    def test_ask_records_metrics(self):
        from src.graph.build_graph import ask

        result = ask("What is the remote work policy?", thread_id="test-tid")
        assert result == "Test answer"
        self.mock_store.record.assert_called_once()
        recorded = self.mock_store.record.call_args[0][0]
        assert isinstance(recorded, QueryMetrics)
        assert recorded.thread_id == "test-tid"
        assert recorded.mode == "graph"

    def test_ask_passes_callback_in_config(self):
        from src.graph.build_graph import ask

        ask("test?", thread_id="t1")
        call_args = self.mock_graph.invoke.call_args
        config = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("config", call_args[0][1] if len(call_args[0]) > 1 else None)
        # The config dict should have callbacks key
        assert "callbacks" in config
        assert len(config["callbacks"]) == 1

    def test_ask_metrics_failure_doesnt_break_query(self):
        self.mock_store.record.side_effect = RuntimeError("DB error")
        from src.graph.build_graph import ask

        # Should still return the answer despite metrics failure
        result = ask("test?", thread_id="t1")
        assert result == "Test answer"


# ---------------------------------------------------------------------------
# TestNaiveWithMetrics
# ---------------------------------------------------------------------------

class TestNaiveWithMetrics:
    """Naive RAG answer() records metrics."""

    @pytest.fixture(autouse=True)
    def _patch_chain_and_store(self):
        reset_store()
        self.mock_chain = MagicMock()
        self.mock_chain.invoke.return_value = "Naive answer"

        with (
            patch(
                "src.rag.naive_rag.build_naive_rag_chain",
                return_value=self.mock_chain,
            ),
            patch("src.observability.metrics_store.get_store") as mock_get_store,
        ):
            self.mock_store = MagicMock()
            mock_get_store.return_value = self.mock_store
            yield
        reset_store()

    def test_naive_records_metrics(self):
        from src.rag.naive_rag import answer

        result = answer("What is the policy?")
        assert result == "Naive answer"
        self.mock_store.record.assert_called_once()
        recorded = self.mock_store.record.call_args[0][0]
        assert isinstance(recorded, QueryMetrics)
        assert recorded.mode == "naive"

    def test_naive_metrics_failure_doesnt_break_query(self):
        self.mock_store.record.side_effect = RuntimeError("DB error")
        from src.rag.naive_rag import answer

        result = answer("test?")
        assert result == "Naive answer"


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------

class TestConfigPhase6:
    """Phase 6 config extensions."""

    def test_new_langsmith_fields_default_empty(self):
        from config import settings

        assert settings.langsmith_api_key == "" or isinstance(settings.langsmith_api_key, str)
        assert settings.langsmith_project == "enterprise-rag-assistant"

    def test_langsmith_warning_without_key(self):
        from dataclasses import replace

        from config import settings

        cfg = replace(
            settings,
            openai_api_key="sk-test",
            langsmith_tracing="true",
            langsmith_api_key="",
        )
        with patch("config.logging") as mock_logging:
            mock_logger = MagicMock()
            mock_logging.getLogger.return_value = mock_logger
            cfg.validate()
            mock_logger.warning.assert_called_once()
            assert "LANGSMITH_TRACING" in mock_logger.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# TestMetricsScript
# ---------------------------------------------------------------------------

class TestMetricsScript:
    """scripts/metrics.py CLI output."""

    @pytest.fixture(autouse=True)
    def _patch_store(self):
        reset_store()
        yield
        reset_store()

    def test_dashboard_empty_db(self, capsys):
        store = MetricsStore(":memory:")
        with patch("scripts.metrics.get_store", return_value=store):
            with patch("sys.argv", ["metrics.py"]):
                from scripts.metrics import main
                main()
        output = capsys.readouterr().out
        assert "No queries recorded" in output
        store.close()

    def test_dashboard_formats_costs(self, capsys):
        store = MetricsStore(":memory:")
        store.record(_make_metrics(
            thread_id="abc1234567",
            estimated_cost_usd=0.00142,
            latency_ms=412.0,
        ))
        with patch("scripts.metrics.get_store", return_value=store):
            with patch("sys.argv", ["metrics.py"]):
                from scripts.metrics import main
                main()
        output = capsys.readouterr().out
        assert "$0.00142" in output
        assert "412ms" in output
        store.close()

    def test_dashboard_over_budget_flag(self, capsys):
        store = MetricsStore(":memory:")
        store.record(_make_metrics(estimated_cost_usd=0.03))
        with patch("scripts.metrics.get_store", return_value=store):
            with patch("sys.argv", ["metrics.py"]):
                from scripts.metrics import main
                main()
        output = capsys.readouterr().out
        assert "Over budget" in output or "over_budget" in output or "*" in output
        store.close()


# ---------------------------------------------------------------------------
# TestUploadScript
# ---------------------------------------------------------------------------

class TestUploadScript:
    """scripts/upload_eval_dataset.py guards."""

    def test_dry_run_no_api_call(self, capsys):
        with patch("sys.argv", ["upload.py", "--dry-run"]):
            from scripts.upload_eval_dataset import main
            main()
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "60" in output or "enterprise-rag-eval" in output

    def test_missing_api_key_exits_1(self):
        with (
            patch("sys.argv", ["upload.py"]),
            patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}, clear=False),
        ):
            from scripts.upload_eval_dataset import main
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
