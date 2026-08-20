"""Phase 27 — the daily cost cap covers the RAGAS endpoint too.

Two halves of one gap, found reviewing the Phase 27 changeset:

1. ``daily_cap_exceeded()`` was consulted only by ``/ask``. The RAGAS
   endpoint runs the whole suite — every item generates an answer, then
   four LLM-judged metrics score it — so the single most expensive
   operation in the system ignored the ceiling entirely.

2. The judge calls were never costed. Answer generation inside a run
   routes through ``graph_ask``/``naive_answer``, which record metrics;
   ragas calls its own LLM directly, so the larger half of a run's spend
   never reached the store and therefore never counted toward the cap
   that exists to stop exactly this.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.observability.cost_guard import clear_query_budget, reset_daily_cache


@pytest.fixture(autouse=True)
def _clean():
    clear_query_budget()
    reset_daily_cache()
    yield
    clear_query_budget()
    reset_daily_cache()


def _fake_ragas(cost_calls: int):
    """Stand-in for the ragas entry point that exercises passed callbacks."""
    from uuid import uuid4

    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    def _run(dataset, metrics=None, callbacks=None, **kwargs):
        for _ in range(cost_calls):
            msg = AIMessage(
                content="judged",
                usage_metadata={
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                },
                response_metadata={"model_name": "gpt-4o-mini"},
            )
            result = LLMResult(generations=[[ChatGeneration(message=msg)]])
            for handler in callbacks or []:
                handler.on_llm_end(result, run_id=uuid4())
        out = MagicMock()
        out._repr_dict = {"faithfulness": 0.9}
        return out

    return _run


# ---------------------------------------------------------------------------
# The endpoint honours the cap
# ---------------------------------------------------------------------------

class TestTheEndpointRespectsTheDailyCap:
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

    def test_over_the_cap_the_suite_is_refused(self, client):
        with patch("api.app.daily_cap_exceeded", return_value=True):
            resp = client.post("/eval", json={"limit": 1})
        assert resp.status_code == 503

    def test_the_refusal_says_when_to_come_back(self, client):
        with patch("api.app.daily_cap_exceeded", return_value=True):
            resp = client.post("/eval", json={"limit": 1})
        assert int(resp.headers["Retry-After"]) > 0

    def test_the_check_happens_before_any_work_starts(self, client):
        """Refusing after loading and scoring the set would save nothing."""
        with (
            patch("api.app.daily_cap_exceeded", return_value=True),
            patch("src.eval.ragas_eval.load_eval_set") as load,
        ):
            client.post("/eval", json={"limit": 1})
        load.assert_not_called()

    def test_under_the_cap_the_suite_still_runs(self, client):
        with (
            patch("api.app.daily_cap_exceeded", return_value=False),
            patch(
                "src.eval.ragas_eval.load_eval_set",
                return_value=[{"question": "q"}],
            ),
            patch(
                "src.eval.ragas_eval.evaluate",
                return_value={"faithfulness": 0.9},
            ),
            patch("src.retrieval.get_retriever", return_value=MagicMock()),
        ):
            resp = client.post("/eval", json={"limit": 1})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Judge spend is observed and recorded
# ---------------------------------------------------------------------------

class TestJudgeSpendIsRecorded:
    def test_a_cost_handler_reaches_the_judge(self):
        """Without this the judge's LLM calls are simply never observed."""
        from src.observability.cost_callback import CostCallbackHandler

        seen = {}

        def _capture(dataset, metrics=None, callbacks=None, **kwargs):
            seen["callbacks"] = callbacks
            out = MagicMock()
            out._repr_dict = {"faithfulness": 0.9}
            return out

        with (
            patch("ragas.evaluate", _capture),
            patch("datasets.Dataset.from_dict", return_value=MagicMock()),
            patch("src.eval.ragas_eval.record_judge_spend"),
        ):
            from src.eval.ragas_eval import run_ragas

            run_ragas({"question": ["q"]})

        assert any(
            isinstance(h, CostCallbackHandler)
            for h in (seen.get("callbacks") or [])
        ), "the judge must receive a CostCallbackHandler or its spend is invisible"

    def test_the_spend_reaches_the_metrics_store(self, tmp_path):
        """That row is what `spend_today()` — and so the cap — reads."""
        from src.observability.metrics_store import MetricsStore

        store = MetricsStore(str(tmp_path / "m.db"))
        assert store.spend_today() == 0.0

        with (
            patch("ragas.evaluate", _fake_ragas(cost_calls=4)),
            patch("datasets.Dataset.from_dict", return_value=MagicMock()),
            patch(
                "src.observability.metrics_store.get_store",
                return_value=store,
            ),
        ):
            from src.eval.ragas_eval import run_ragas

            run_ragas({"question": ["q"]})

        # 4 judge calls x (1000 in + 500 out) on gpt-4o-mini
        expected = 4 * (1000 * 0.00015 + 500 * 0.0006) / 1000
        assert store.spend_today() == pytest.approx(expected)

    def test_the_recorded_row_is_identifiable(self, tmp_path):
        """Suite spend must not read as ordinary user query traffic."""
        from src.observability.metrics_store import MetricsStore

        store = MetricsStore(str(tmp_path / "m.db"))
        with (
            patch("ragas.evaluate", _fake_ragas(cost_calls=1)),
            patch("datasets.Dataset.from_dict", return_value=MagicMock()),
            patch(
                "src.observability.metrics_store.get_store",
                return_value=store,
            ),
        ):
            from src.eval.ragas_eval import run_ragas

            run_ragas({"question": ["q"]})

        assert store.query_recent(1)[0]["mode"] == "eval"

    def test_a_recording_failure_does_not_lose_the_scores(self):
        """Cost bookkeeping must never be why a finished run is thrown away."""
        with (
            patch("ragas.evaluate", _fake_ragas(cost_calls=1)),
            patch("datasets.Dataset.from_dict", return_value=MagicMock()),
            patch(
                "src.eval.ragas_eval.record_judge_spend",
                side_effect=RuntimeError("store down"),
            ),
        ):
            from src.eval.ragas_eval import run_ragas

            assert run_ragas({"question": ["q"]}) == {"faithfulness": 0.9}
