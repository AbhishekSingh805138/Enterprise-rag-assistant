"""Phase 27 — P1-4: the cost budget is enforced, not merely displayed.

``COST_BUDGET_PER_QUERY`` was read into a module constant and used to put
an asterisk beside expensive rows in a CLI report. Nothing could reject or
truncate anything, so a multi-part question could decompose into
sub-questions — each retrieving, grading, generating and criticising —
with no ceiling, and no daily cap behind it.

The two ceilings behave differently on purpose, and that difference is
what these tests mostly pin down: per query it degrades, per day it
denies.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.observability.cost_guard import (
    QueryBudget,
    allow,
    clear_query_budget,
    daily_cap_exceeded,
    daily_spend,
    get_query_budget,
    record_spend,
    remaining_sub_questions,
    reset_daily_cache,
    start_query_budget,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_query_budget()
    reset_daily_cache()
    yield
    clear_query_budget()
    reset_daily_cache()


# ---------------------------------------------------------------------------
# The budget itself
# ---------------------------------------------------------------------------

class TestQueryBudget:
    def test_spending_accumulates(self):
        budget = QueryBudget(limit=0.02)
        budget.add(0.005)
        budget.add(0.004)
        assert budget.spent == pytest.approx(0.009)

    def test_not_exceeded_below_the_limit(self):
        assert QueryBudget(limit=0.02, spent=0.019).exceeded is False

    def test_exceeded_at_the_limit(self):
        assert QueryBudget(limit=0.02, spent=0.02).exceeded is True

    def test_a_zero_limit_disables_the_ceiling(self):
        """`spent >= limit` would block every query at limit=0."""
        assert QueryBudget(limit=0.0, spent=1.0).exceeded is False
        assert QueryBudget(limit=0.0).remaining == float("inf")

    def test_a_negative_limit_also_disables_it(self):
        assert QueryBudget(limit=-1, spent=5).exceeded is False

    def test_remaining_never_goes_negative(self):
        assert QueryBudget(limit=0.02, spent=0.05).remaining == 0.0

    def test_allow_permits_while_in_budget(self):
        assert QueryBudget(limit=0.02, spent=0.001).allow("critic") is True

    def test_allow_refuses_once_spent(self):
        assert QueryBudget(limit=0.02, spent=0.02).allow("critic") is False

    def test_each_skipped_stage_is_recorded_once(self, caplog):
        """Recorded for metrics; logged once so it is visible, not spam."""
        budget = QueryBudget(limit=0.01, spent=0.02)
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                budget.allow("critic")
            budget.allow("expansion")
        assert budget.skipped == ["critic", "expansion"]
        assert caplog.text.count("skipping critic") == 1


class TestContextIntegration:
    def test_no_budget_in_scope_allows_everything(self):
        """Scripts, ingestion and tests must be unaffected."""
        assert get_query_budget() is None
        assert allow("anything") is True

    def test_spend_is_recorded_against_the_active_budget(self):
        budget = start_query_budget(limit=0.02)
        record_spend(0.003)
        assert budget.spent == pytest.approx(0.003)

    def test_recording_outside_a_query_is_a_no_op(self):
        record_spend(0.5)  # must not raise

    def test_the_limit_defaults_to_the_configured_budget(self):
        with patch("src.observability.cost_guard.settings") as s:
            s.cost_budget_per_query = 0.05
            assert start_query_budget().limit == 0.05

    def test_a_new_query_starts_from_zero(self):
        start_query_budget(limit=0.02)
        record_spend(0.02)
        assert allow("critic") is False
        start_query_budget(limit=0.02)
        assert allow("critic") is True

    def test_the_budget_survives_a_worker_thread(self):
        """The pipeline runs in a thread; contextvars must carry across."""
        import asyncio

        async def run():
            start_query_budget(limit=0.02)
            record_spend(0.02)
            return await asyncio.to_thread(allow, "critic")

        assert asyncio.run(run()) is False


# ---------------------------------------------------------------------------
# Where cost actually multiplies
# ---------------------------------------------------------------------------

class TestSubQuestionCeiling:
    def test_no_budget_leaves_the_plan_alone(self):
        assert remaining_sub_questions(5) == 5

    def test_a_single_sub_question_is_never_truncated(self):
        """Truncating to zero would answer nothing at all."""
        start_query_budget(limit=0.001)
        record_spend(0.10)
        assert remaining_sub_questions(1) == 1

    def test_a_plan_within_budget_is_untouched(self):
        start_query_budget(limit=0.10)
        record_spend(0.01)  # 9 more affordable
        assert remaining_sub_questions(5) == 5

    def test_an_expensive_plan_is_truncated(self, caplog):
        start_query_budget(limit=0.02)
        record_spend(0.008)  # 0.012 left, ~1 more at this rate
        with caplog.at_level(logging.WARNING):
            allowed = remaining_sub_questions(5)
        assert 1 <= allowed < 5
        assert "sub-questions" in caplog.text

    def test_at_least_one_sub_question_always_survives(self):
        start_query_budget(limit=0.02)
        record_spend(0.02)
        assert remaining_sub_questions(5) == 1

    def test_a_disabled_budget_never_truncates(self):
        start_query_budget(limit=0)
        record_spend(100.0)
        assert remaining_sub_questions(5) == 5


class TestPipelineDegradation:
    """Exceeding the budget must reduce elaboration, never deny service."""

    def test_query_expansion_is_skipped_when_over(self):
        from src.retrieval.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(k=2, per_query_k=2)
        start_query_budget(limit=0.01)
        record_spend(0.02)

        with (
            patch.object(retriever, "_get_dense_results", return_value=[]) as dense,
            patch("src.retrieval.multi_query._generate_variants") as expand,
        ):
            retriever._get_relevant_documents("what is the policy?")

        expand.assert_not_called()
        # The original query still runs — the caller still gets an answer.
        assert dense.call_count == 1

    def test_query_expansion_runs_when_in_budget(self):
        from src.retrieval.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever(k=2, per_query_k=2)
        start_query_budget(limit=0.10)

        with (
            patch.object(retriever, "_get_dense_results", return_value=[]),
            patch("src.retrieval.multi_query._generate_variants", return_value=["v1"]) as expand,
        ):
            retriever._get_relevant_documents("what is the policy?")

        expand.assert_called_once()

    def test_the_critic_is_skipped_when_over(self):
        from langchain_core.documents import Document

        from src.graph.nodes import critic

        start_query_budget(limit=0.01)
        record_spend(0.02)

        with patch("src.graph.nodes._llm") as llm, patch("src.graph.nodes.settings") as s:
            s.critic_mode = "always"
            result = critic({
                "question": "q",
                "generation": "The policy allows 3 days. (handbook.md)",
                "documents": [Document(page_content="3 days", metadata={})],
            })

        llm.assert_not_called()
        # Passed through rather than failed: an unverified answer beats none.
        assert result["critic_passed"] is True
        assert result["claims_removed"] == 0

    def test_sub_query_processing_stops_and_synthesises_what_it_has(self):
        from src.graph.planner import process_sub_query

        start_query_budget(limit=0.01)
        record_spend(0.02)

        with patch("src.graph.planner._process_single_sub_query") as run:
            result = process_sub_query({
                "sub_questions": ["a", "b", "c"],
                "current_sub_idx": 1,
                "sub_answers": ["answer to a"],
                "retriever_strategy": "dense",
            })

        run.assert_not_called()
        assert result["current_sub_idx"] == 3  # jump to the end
        assert result["sub_answers"] == ["answer to a"]
        assert result["budget_truncated"] is True

    def test_the_first_sub_question_always_runs(self):
        """Otherwise an over-budget query would answer nothing at all."""
        from src.graph.planner import process_sub_query

        start_query_budget(limit=0.01)
        record_spend(0.02)

        with patch(
            "src.graph.planner._process_single_sub_query", return_value=("ans", [])
        ) as run:
            process_sub_query({
                "sub_questions": ["a", "b"],
                "current_sub_idx": 0,
                "sub_answers": [],
                "retriever_strategy": "dense",
            })
        run.assert_called_once()

    def test_the_planner_truncates_its_own_plan(self):
        from src.graph.planner import PlanResult, planner

        start_query_budget(limit=0.02)
        record_spend(0.015)

        chain = MagicMock()
        chain.invoke.return_value = PlanResult(
            is_multi_part=True,
            sub_questions=["q1", "q2", "q3", "q4", "q5"],
            reasoning="multi",
        )
        with patch("src.graph.planner._llm") as llm:
            llm.return_value.with_structured_output.return_value = MagicMock()
            with patch("src.graph.planner._planner_prompt") as prompt:
                prompt.__or__ = MagicMock(return_value=chain)
                result = planner({"question": "a and b and c and d and e"})

        assert len(result["sub_questions"]) < 5


# ---------------------------------------------------------------------------
# Daily cap
# ---------------------------------------------------------------------------

class TestDailyCap:
    def test_disabled_by_default(self):
        with patch("src.observability.cost_guard.settings") as s:
            s.cost_daily_cap_usd = 0
            assert daily_cap_exceeded() is False

    def test_under_the_cap_is_allowed(self):
        with (
            patch("src.observability.cost_guard.settings") as s,
            patch("src.observability.cost_guard.daily_spend", return_value=1.0),
        ):
            s.cost_daily_cap_usd = 5.0
            assert daily_cap_exceeded() is False

    def test_at_the_cap_is_rejected(self, caplog):
        with (
            patch("src.observability.cost_guard.settings") as s,
            patch("src.observability.cost_guard.daily_spend", return_value=5.0),
            caplog.at_level(logging.ERROR),
        ):
            s.cost_daily_cap_usd = 5.0
            assert daily_cap_exceeded() is True
        assert "DAILY COST CAP REACHED" in caplog.text

    def test_a_store_failure_does_not_block_queries(self):
        """The cost check must never be the reason a query fails."""
        with patch(
            "src.observability.metrics_store.get_store", side_effect=RuntimeError("db gone")
        ):
            assert daily_spend(force=True) == 0.0

    def test_the_spend_is_cached_between_calls(self):
        """A SQL aggregate per request would put the check on the hot path."""
        store = MagicMock()
        store.spend_today.return_value = 2.5
        with patch("src.observability.metrics_store.get_store", return_value=store):
            assert daily_spend(force=True) == 2.5
            assert daily_spend() == 2.5
        assert store.spend_today.call_count == 1

    def test_force_bypasses_the_cache(self):
        store = MagicMock()
        store.spend_today.side_effect = [1.0, 3.0]
        with patch("src.observability.metrics_store.get_store", return_value=store):
            assert daily_spend(force=True) == 1.0
            assert daily_spend(force=True) == 3.0


class TestSpendToday:
    def _store(self, tmp_path):
        from src.observability.metrics_store import MetricsStore

        return MetricsStore(str(tmp_path / "m.db"))

    def _metrics(self, cost):
        from dataclasses import dataclass

        @dataclass
        class M:
            thread_id: str = "t"
            question_preview: str = "q?"
            mode: str = "graph"
            retriever_strategy: str = "dense"
            prompt_tokens: int = 1
            completion_tokens: int = 1
            total_tokens: int = 2
            estimated_cost_usd: float = cost
            latency_ms: float = 1.0
            is_idk: bool = False
            grader_rejected: int = 0

        return M()

    def test_an_empty_store_reports_zero(self, tmp_path):
        """COALESCE matters: SUM over no rows is NULL, not 0."""
        store = self._store(tmp_path)
        assert store.spend_today() == 0.0
        store.close()

    def test_it_sums_today(self, tmp_path):
        store = self._store(tmp_path)
        for cost in (0.01, 0.02, 0.03):
            store.record(self._metrics(cost))
        assert store.spend_today() == pytest.approx(0.06)
        store.close()

    def test_yesterday_does_not_count(self, tmp_path):
        store = self._store(tmp_path)
        store.record(self._metrics(0.05))
        with store._lock:
            store._conn.execute("UPDATE query_metrics SET ts = '2020-01-01T00:00:00+00:00'")
            store._conn.commit()
        assert store.spend_today() == 0.0
        store.close()


# ---------------------------------------------------------------------------
# The API boundary
# ---------------------------------------------------------------------------

class TestApiEnforcement:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        s = MagicMock()
        s.validate = MagicMock()
        s.auth_enabled = False
        s.chroma_collection = "c"
        s.log_level = "WARNING"
        s.debug_mode = False
        s.cors_origins = "http://localhost:8501"
        s.cors_allow_methods = "GET,POST,OPTIONS"
        s.cors_allow_headers = "Content-Type"
        s.async_ingestion = False
        s.guardrails_enabled = False
        s.rate_limit_storage_uri = ""
        with patch("api.app.settings", s), patch("src.security.auth.settings", s):
            from api.app import app

            with TestClient(app) as tc:
                yield tc

    def test_over_the_daily_cap_the_api_rejects(self, client):
        with patch("api.app.daily_cap_exceeded", return_value=True):
            resp = client.post("/ask", json={"question": "hello?"})
        assert resp.status_code == 503

    def test_the_rejection_says_when_to_come_back(self, client):
        with patch("api.app.daily_cap_exceeded", return_value=True):
            resp = client.post("/ask", json={"question": "hello?"})
        assert int(resp.headers["Retry-After"]) > 0

    def test_under_the_cap_the_query_runs(self, client):
        from api.models import AskResponse

        answer = AskResponse(
            answer="ok", question="hello?", mode="naive", retriever_strategy="dense",
            cost_usd=0.0, latency_ms=1.0, tokens_used=0,
        )
        with (
            patch("api.app.daily_cap_exceeded", return_value=False),
            patch("api.app._ask_sync", return_value=answer),
        ):
            resp = client.post("/ask", json={"question": "hello?"})
        assert resp.status_code == 200

    def test_a_budget_is_opened_for_every_query(self):
        """Without this, nothing downstream can see the spend."""
        import inspect

        import api.app

        source = inspect.getsource(api.app)
        assert source.count("start_query_budget()") == 3, (
            "each pipeline entry point (sync, streaming graph, streaming naive) "
            "must open a budget, or that path runs uncapped"
        )


class TestRetryAfterHelper:
    def test_it_is_within_a_day(self):
        from api.app import _seconds_until_utc_midnight

        assert 0 < _seconds_until_utc_midnight() <= 86400


class TestConfigDefaults:
    def test_the_daily_cap_ships_disabled(self):
        """Turning it on is a deployment decision, not a default."""
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(r'cost_daily_cap_usd:\s*float\s*=\s*float\(os\.getenv\(\s*"COST_DAILY_CAP_USD",\s*"0"\s*\)\)', source)

    def test_the_per_query_budget_keeps_its_shipped_value(self):
        import inspect
        import re

        import config

        source = inspect.getsource(config.Settings)
        assert re.search(r'cost_budget_per_query:.*"COST_BUDGET_PER_QUERY",\s*"0\.02"', source)
