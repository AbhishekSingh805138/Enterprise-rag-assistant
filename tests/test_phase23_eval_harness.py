"""Phase 23 — RAGAS evaluation harness (src/eval/ragas_eval.py).

The harness is how retrieval changes get defended with numbers, so the
behaviour that matters is that it never silently drops an eval item: a
question that errors must still produce a row, or the reported scores are
computed over a different set than the one you think you evaluated.

`ragas` and `datasets` are stubbed via sys.modules — the real packages are
heavy to import and their scoring is not what is under test here.
"""
from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.eval.ragas_eval import (
    collect_predictions,
    evaluate,
    load_eval_set,
    run_ragas,
    save_results,
)

EVAL_ITEMS = [
    {"question": "What is the PTO policy?", "ground_truth": "20 days per year."},
    {"question": "What are payment terms?", "ground_truth": "Net 30."},
]


class StubRetriever:
    def __init__(self, docs=None, error=None):
        self.docs = docs if docs is not None else [Document(page_content="ctx")]
        self.error = error
        self.calls = []

    def invoke(self, query):
        self.calls.append(query)
        if self.error:
            raise self.error
        return self.docs


@pytest.fixture
def fake_ragas(monkeypatch):
    """Stub `ragas` + `datasets` with recording fakes."""
    captured = {}

    datasets_mod = types.ModuleType("datasets")

    class Dataset:
        @staticmethod
        def from_dict(rows):
            captured["rows"] = rows
            return {"__dataset__": rows}

    datasets_mod.Dataset = Dataset

    ragas_mod = types.ModuleType("ragas")
    metrics_mod = types.ModuleType("ragas.metrics")

    class Result:
        def __init__(self, scores):
            self._repr_dict = scores

    def _evaluate(dataset, metrics=None):
        captured["dataset"] = dataset
        captured["metrics"] = metrics
        return Result({
            "faithfulness": 0.812345,
            "answer_relevancy": 0.7654,
            "context_precision": 0.61,
            "context_recall": 0.55,
        })

    ragas_mod.evaluate = _evaluate
    ragas_mod.metrics = metrics_mod
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        setattr(metrics_mod, name, f"<metric:{name}>")

    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "ragas", ragas_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_mod)
    return captured


# ---------------------------------------------------------------------------
# load_eval_set
# ---------------------------------------------------------------------------

class TestLoadEvalSet:
    def test_loads_the_bundled_eval_set(self):
        items = load_eval_set()
        assert items
        assert {"question", "ground_truth"} <= set(items[0])

    def test_limit_truncates(self):
        assert len(load_eval_set(limit=2)) == 2

    def test_limit_larger_than_the_set_returns_everything(self):
        full = load_eval_set()
        assert len(load_eval_set(limit=len(full) + 500)) == len(full)

    def test_custom_path_is_honoured(self, tmp_path):
        p = tmp_path / "custom.json"
        p.write_text(json.dumps(EVAL_ITEMS), encoding="utf-8")
        assert len(load_eval_set(path=p)) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eval_set(path=tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# collect_predictions
# ---------------------------------------------------------------------------

class TestCollectPredictions:
    def test_builds_the_four_ragas_columns(self):
        rows = collect_predictions(EVAL_ITEMS, lambda q: "answer", StubRetriever())
        assert set(rows) == {"question", "answer", "contexts", "ground_truth"}

    def test_every_column_has_one_entry_per_item(self):
        rows = collect_predictions(EVAL_ITEMS, lambda q: "a", StubRetriever())
        assert all(len(v) == len(EVAL_ITEMS) for v in rows.values())

    def test_contexts_are_page_contents_not_documents(self):
        retriever = StubRetriever([Document(page_content="chunk-A"), Document(page_content="chunk-B")])
        rows = collect_predictions(EVAL_ITEMS[:1], lambda q: "a", retriever)
        assert rows["contexts"][0] == ["chunk-A", "chunk-B"]

    def test_ground_truth_is_carried_through(self):
        rows = collect_predictions(EVAL_ITEMS, lambda q: "a", StubRetriever())
        assert rows["ground_truth"] == ["20 days per year.", "Net 30."]

    def test_answer_fn_receives_each_question(self):
        seen = []
        collect_predictions(EVAL_ITEMS, lambda q: seen.append(q) or "a", StubRetriever())
        assert seen == [i["question"] for i in EVAL_ITEMS]

    def test_retriever_failure_still_produces_a_row(self):
        """Dropping the row would silently change the denominator of the scores."""
        rows = collect_predictions(
            EVAL_ITEMS, lambda q: "a", StubRetriever(error=RuntimeError("chroma down"))
        )
        assert len(rows["question"]) == 2
        assert rows["contexts"] == [[], []]
        assert rows["answer"] == ["(error)", "(error)"]

    def test_answer_failure_still_produces_a_row(self):
        def _boom(q):
            raise RuntimeError("rate limited")

        rows = collect_predictions(EVAL_ITEMS[:1], _boom, StubRetriever())
        assert rows["answer"] == ["(error)"]
        assert rows["question"] == [EVAL_ITEMS[0]["question"]]

    def test_one_failure_does_not_poison_the_others(self):
        calls = {"n": 0}

        def flaky(q):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            return "good answer"

        rows = collect_predictions(EVAL_ITEMS, flaky, StubRetriever())
        assert rows["answer"] == ["(error)", "good answer"]

    def test_empty_eval_set_yields_empty_columns(self):
        rows = collect_predictions([], lambda q: "a", StubRetriever())
        assert rows == {"question": [], "answer": [], "contexts": [], "ground_truth": []}


# ---------------------------------------------------------------------------
# run_ragas
# ---------------------------------------------------------------------------

class TestRunRagas:
    def test_returns_the_metric_scores(self, fake_ragas):
        scores = run_ragas({"question": [], "answer": [], "contexts": [], "ground_truth": []})
        assert scores["faithfulness"] == pytest.approx(0.812345)

    def test_all_four_metrics_are_requested(self, fake_ragas):
        run_ragas({"question": []})
        assert len(fake_ragas["metrics"]) == 4

    def test_rows_are_passed_through_to_the_dataset(self, fake_ragas):
        rows = {"question": ["q"], "answer": ["a"], "contexts": [["c"]], "ground_truth": ["g"]}
        run_ragas(rows)
        assert fake_ragas["rows"] == rows

    def test_result_is_a_plain_dict(self, fake_ragas):
        assert isinstance(run_ragas({"question": []}), dict)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_end_to_end_with_a_supplied_eval_set(self, fake_ragas):
        scores = evaluate(lambda q: "a", StubRetriever(), eval_set=EVAL_ITEMS)
        assert "faithfulness" in scores
        assert len(fake_ragas["rows"]["question"]) == 2

    def test_falls_back_to_the_bundled_set_with_a_limit(self, fake_ragas):
        evaluate(lambda q: "a", StubRetriever(), limit=3)
        assert len(fake_ragas["rows"]["question"]) == 3


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_writes_the_payload(self, tmp_path):
        out = tmp_path / "res.json"
        save_results({"faithfulness": 0.9}, "graph", str(out), num_items=5, retriever="hybrid")
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["mode"] == "graph"
        assert payload["retriever"] == "hybrid"
        assert payload["num_items"] == 5

    def test_returns_the_path_written(self, tmp_path):
        out = tmp_path / "res.json"
        assert save_results({"f": 1.0}, "naive", str(out)) == str(out)

    def test_scores_are_rounded_for_readability(self, tmp_path):
        out = tmp_path / "res.json"
        save_results({"faithfulness": 0.123456789}, "naive", str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["scores"]["faithfulness"] == 0.1235

    def test_non_float_scores_survive_untouched(self, tmp_path):
        out = tmp_path / "res.json"
        save_results({"note": "partial run"}, "naive", str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["scores"]["note"] == "partial run"

    def test_payload_carries_a_timestamp(self, tmp_path):
        out = tmp_path / "res.json"
        save_results({"f": 1.0}, "naive", str(out))
        assert json.loads(out.read_text(encoding="utf-8"))["timestamp"]

    def test_default_filename_encodes_mode_retriever_and_timestamp(self, tmp_path, monkeypatch):
        """With no --output, results land in eval_results/ under a sortable name."""
        import src.eval.ragas_eval as module

        class FakePath:
            """Stands in for Path(__file__) so the write lands in tmp_path."""

            def __init__(self, *_):
                pass

            @property
            def parent(self):
                return self

            def __truediv__(self, other):
                return tmp_path / other

        monkeypatch.setattr(module, "Path", FakePath)
        written = save_results({"f": 1.0}, "graph", None, retriever="hybrid")

        name = written.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        assert name.startswith("graph_hybrid_")
        assert name.endswith(".json")
        assert (tmp_path / "eval_results").is_dir()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def _run(self, argv, tmp_path, mode_patch):
        from src.eval import ragas_eval

        out = tmp_path / "out.json"
        full_argv = ["ragas_eval", "--output", str(out), *argv]
        with (
            patch.object(sys, "argv", full_argv),
            patch("config.settings", MagicMock()),
            patch("config.setup_logging", MagicMock()),
            patch("src.retrieval.get_retriever", return_value=StubRetriever()) as get_ret,
            patch.object(ragas_eval, "collect_predictions", return_value={"question": ["q"]}),
            patch.object(ragas_eval, "run_ragas", return_value={
                "faithfulness": 0.80, "answer_relevancy": 0.60,
                "context_precision": 0.75, "unknown_metric": 0.5, "note": "text",
            }),
            mode_patch,
        ):
            ragas_eval.main()
        return out, get_ret

    def test_naive_mode_runs_and_saves(self, tmp_path):
        out, _ = self._run([], tmp_path, patch("src.rag.naive_rag.answer", return_value="a"))
        assert out.exists()
        assert json.loads(out.read_text(encoding="utf-8"))["mode"] == "naive"

    def test_graph_mode_selects_the_graph_pipeline(self, tmp_path):
        out, _ = self._run(
            ["--mode", "graph"], tmp_path,
            patch("src.graph.build_graph.ask", return_value="a"),
        )
        assert json.loads(out.read_text(encoding="utf-8"))["mode"] == "graph"

    def test_retriever_flag_is_passed_to_the_factory(self, tmp_path):
        out, get_ret = self._run(
            ["--retriever", "hybrid"], tmp_path,
            patch("src.rag.naive_rag.answer", return_value="a"),
        )
        get_ret.assert_called_once_with(strategy="hybrid")

    def test_limit_flag_truncates_the_eval_set(self, tmp_path):
        from src.eval import ragas_eval

        out = tmp_path / "out.json"
        with (
            patch.object(sys, "argv", ["e", "--limit", "2", "--output", str(out)]),
            patch("config.settings", MagicMock()),
            patch("config.setup_logging", MagicMock()),
            patch("src.retrieval.get_retriever", return_value=StubRetriever()),
            patch("src.rag.naive_rag.answer", return_value="a"),
            patch.object(ragas_eval, "load_eval_set", return_value=EVAL_ITEMS) as load,
            patch.object(ragas_eval, "collect_predictions", return_value={"question": ["q"]}),
            patch.object(ragas_eval, "run_ragas", return_value={"faithfulness": 0.9}),
        ):
            ragas_eval.main()
        load.assert_called_once_with(limit=2)

    def test_scores_are_printed_with_pass_fail_against_prd_targets(self, tmp_path, capsys):
        self._run([], tmp_path, patch("src.rag.naive_rag.answer", return_value="a"))
        printed = capsys.readouterr().out
        assert "PASS" in printed          # faithfulness 0.80 >= 0.65
        assert "FAIL" in printed          # answer_relevancy 0.60 < 0.70
        assert "unknown_metric" in printed  # no target -> printed without status
