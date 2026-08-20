"""Phase 27 — P1-3: RAG quality is measured continuously and can fail a build.

RAGAS existed and worked, and never ran in CI. Retrieval and prompt
changes shipped with no measurement of whether faithfulness or context
precision moved — the single most valuable missing test for a RAG system,
because correctness here is statistical and unit tests cannot see it.

The gate has to survive two opposite failure modes. Too strict and it
fires on the LLM judge's own run-to-run noise, teaching everyone to
re-run until green; too loose and a real regression clears it. The
tolerance-against-baseline design is what balances those, and most of
what follows pins that behaviour down.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.ragas_gate import (
    DEFAULT_TOLERANCE,
    FLOORS,
    compare,
    load_baseline,
    save_baseline,
)

HEALTHY = {
    "faithfulness": 0.88,
    "answer_relevancy": 0.91,
    "context_precision": 0.80,
    "context_recall": 0.85,
}

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ragas-nightly.yml"


def baseline_of(scores: dict) -> dict:
    return {"recorded_at": "2026-08-01T00:00:00+00:00", "scores": dict(scores)}


# ---------------------------------------------------------------------------
# Absolute floors
# ---------------------------------------------------------------------------

class TestFloors:
    def test_healthy_scores_pass(self):
        failures, _ = compare(HEALTHY, None, DEFAULT_TOLERANCE)
        assert failures == []

    @pytest.mark.parametrize("metric", sorted(FLOORS))
    def test_each_metric_has_an_enforced_floor(self, metric):
        scores = dict(HEALTHY, **{metric: FLOORS[metric] - 0.1})
        failures, _ = compare(scores, None, DEFAULT_TOLERANCE)
        assert any(metric in f and "floor" in f for f in failures)

    def test_a_score_exactly_at_the_floor_passes(self):
        scores = dict(HEALTHY, faithfulness=FLOORS["faithfulness"])
        failures, _ = compare(scores, None, DEFAULT_TOLERANCE)
        assert failures == []

    def test_a_missing_metric_fails_rather_than_passing(self):
        """A broken harness reporting nothing must not read as green."""
        failures, _ = compare({}, None, DEFAULT_TOLERANCE)
        assert len(failures) == len(FLOORS)
        assert all("not reported" in f for f in failures)

    def test_a_non_numeric_score_is_treated_as_missing(self):
        failures, _ = compare(dict(HEALTHY, faithfulness="n/a"), None, DEFAULT_TOLERANCE)
        assert any("faithfulness" in f and "not reported" in f for f in failures)


# ---------------------------------------------------------------------------
# Regression against baseline
# ---------------------------------------------------------------------------

class TestRegressionDetection:
    def test_identical_scores_pass(self):
        failures, _ = compare(HEALTHY, baseline_of(HEALTHY), DEFAULT_TOLERANCE)
        assert failures == []

    def test_noise_within_tolerance_does_not_fail(self):
        """A gate that fires on judge noise gets re-run until green."""
        noisy = {k: v - (DEFAULT_TOLERANCE - 0.01) for k, v in HEALTHY.items()}
        failures, _ = compare(noisy, baseline_of(HEALTHY), DEFAULT_TOLERANCE)
        assert failures == []

    def test_a_real_regression_fails_even_above_the_floor(self):
        """0.88 -> 0.71 clears every floor and is exactly what to catch."""
        regressed = dict(HEALTHY, faithfulness=0.71)
        failures, _ = compare(regressed, baseline_of(HEALTHY), DEFAULT_TOLERANCE)
        assert len(failures) == 1
        assert "regressed 0.1700" in failures[0]
        assert FLOORS["faithfulness"] < 0.71  # the floor would have missed it

    def test_the_failure_names_both_numbers_and_the_tolerance(self):
        failures, _ = compare(dict(HEALTHY, faithfulness=0.70), baseline_of(HEALTHY), 0.05)
        message = failures[0]
        assert "0.7000" in message and "0.8800" in message and "0.05" in message

    def test_an_improvement_is_noted_not_failed(self):
        improved = dict(HEALTHY, faithfulness=0.96)
        failures, notes = compare(improved, baseline_of(HEALTHY), DEFAULT_TOLERANCE)
        assert failures == []
        assert any("improved" in n for n in notes)

    def test_a_metric_absent_from_baseline_is_skipped(self):
        """A newly added metric must not fail its first run."""
        partial = baseline_of({"faithfulness": 0.88})
        failures, _ = compare(HEALTHY, partial, DEFAULT_TOLERANCE)
        assert failures == []

    def test_a_tighter_tolerance_catches_a_smaller_drop(self):
        slipped = dict(HEALTHY, faithfulness=0.85)
        assert compare(slipped, baseline_of(HEALTHY), 0.05)[0] == []
        assert compare(slipped, baseline_of(HEALTHY), 0.01)[0] != []

    def test_the_default_tolerance_exceeds_typical_judge_noise(self):
        """Tighter than the noise floor is how a gate gets disabled."""
        assert DEFAULT_TOLERANCE >= 0.02


class TestNoBaseline:
    def test_floors_still_apply(self):
        failures, _ = compare(dict(HEALTHY, faithfulness=0.1), None, DEFAULT_TOLERANCE)
        assert failures

    def test_it_says_regression_detection_is_off(self):
        """Silently checking less than you think is the trap here."""
        _, notes = compare(HEALTHY, None, DEFAULT_TOLERANCE)
        assert any("No baseline" in n and "--update-baseline" in n for n in notes)


# ---------------------------------------------------------------------------
# Baseline persistence
# ---------------------------------------------------------------------------

class TestBaselineFile:
    def test_a_missing_file_reads_as_no_baseline(self, tmp_path):
        assert load_baseline(tmp_path / "absent.json") is None

    def test_a_corrupt_file_does_not_crash_the_gate(self, tmp_path, capsys):
        path = tmp_path / "baseline.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_baseline(path) is None
        assert "could not read baseline" in capsys.readouterr().err

    def test_a_round_trip_preserves_the_scores(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline(HEALTHY, {"mode": "graph"}, path)
        assert load_baseline(path)["scores"] == HEALTHY

    def test_the_baseline_records_what_produced_it(self, tmp_path):
        """A score without its prompt set cannot be compared to anything."""
        path = tmp_path / "baseline.json"
        save_baseline(
            HEALTHY,
            {"mode": "graph", "retriever": "hybrid", "items": 12, "prompt_set": "abc123"},
            path,
        )
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert recorded["prompt_set"] == "abc123"
        assert recorded["mode"] == "graph"
        assert recorded["retriever"] == "hybrid"
        assert recorded["recorded_at"]

    def test_non_numeric_scores_are_not_persisted(self, tmp_path):
        path = tmp_path / "baseline.json"
        save_baseline(dict(HEALTHY, note="ran twice"), {}, path)
        assert "note" not in json.loads(path.read_text(encoding="utf-8"))["scores"]


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------

class TestNightlyWorkflow:
    @pytest.fixture(scope="class")
    def workflow(self):
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def _triggers(self, workflow):
        # PyYAML parses a bare `on:` key as the boolean True.
        return workflow.get("on") or workflow.get(True)

    def test_it_runs_on_a_schedule(self, workflow):
        assert "schedule" in self._triggers(workflow)

    def test_it_does_not_run_on_pull_requests(self, workflow):
        """Non-deterministic and token-costly: PR CI would go flaky."""
        triggers = self._triggers(workflow)
        assert "pull_request" not in triggers
        assert "push" not in triggers

    def test_it_can_be_triggered_by_hand(self, workflow):
        assert "workflow_dispatch" in self._triggers(workflow)

    def test_the_gate_step_can_fail_the_build(self, workflow):
        """The whole point — the harness alone always exits 0."""
        steps = workflow["jobs"]["quality-gate"]["steps"]
        assert any("scripts.ragas_gate" in str(s.get("run", "")) for s in steps)

    def test_the_cache_is_disabled_for_the_run(self, workflow):
        """A cache hit would score a stored answer, not the pipeline."""
        env = str(workflow["jobs"]["quality-gate"]["steps"])
        assert "SEMANTIC_CACHE_ENABLED" in env

    def test_the_cost_ceiling_is_lifted_for_the_run(self, workflow):
        """Otherwise a budget change would read as a quality regression."""
        env = str(workflow["jobs"]["quality-gate"]["steps"])
        assert "COST_BUDGET_PER_QUERY" in env

    def test_it_has_a_timeout(self, workflow):
        assert workflow["jobs"]["quality-gate"]["timeout-minutes"] > 0


class TestGateAndHarnessAgree:
    def test_the_floors_match_the_harness_targets(self):
        """Two sets of thresholds would eventually disagree."""
        import inspect
        import re

        import src.eval.ragas_eval as harness

        source = inspect.getsource(harness.main)
        for metric, floor in FLOORS.items():
            assert re.search(rf'"{metric}":\s*{floor}', source), (
                f"{metric} floor {floor} does not match the harness's printed target"
            )
