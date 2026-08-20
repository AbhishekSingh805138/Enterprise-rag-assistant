"""RAGAS regression gate — the CI job that can actually fail.

``src.eval.ragas_eval`` computes the scores and prints PASS/FAIL beside
them, then exits 0 regardless. So retrieval and prompt changes shipped
with no measurement of whether faithfulness or context precision moved.
For a RAG system that is the most valuable missing test: correctness here
is statistical, and no unit test can see it.

Two thresholds, because they catch different things:

* **Floors** are absolute. Below these the system is not fit to serve,
  whatever it scored last week.
* **Regression** is relative to a recorded baseline. A drop from 0.86 to
  0.71 clears every floor and is still the thing you want to know about
  before it reaches users.

Deliberately kept out of PR CI. It costs real tokens per run and the
scores are non-deterministic, so gating pull requests on it would produce
flaky failures that teams learn to re-run until green — worse than no
gate. It belongs on a schedule, against a known corpus.

Usage:
    python -m scripts.ragas_gate --mode graph --retriever hybrid
    python -m scripts.ragas_gate --update-baseline    # after a real improvement
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parent.parent / "src" / "eval" / "baseline.json"

# Below these the system is not fit to serve, regardless of history.
# Same values the harness already prints against, so the gate and the
# report cannot disagree.
FLOORS = {
    "faithfulness": 0.65,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.70,
}

# How far a metric may fall below baseline before it is a regression.
# Not zero: these scores are produced by an LLM judge over a small set,
# so run-to-run noise of a few points is expected. A tolerance tighter
# than the noise floor produces failures nobody can act on, which is how
# a gate gets disabled.
DEFAULT_TOLERANCE = 0.05


def load_baseline(path: Path = BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: could not read baseline {path}: {e}", file=sys.stderr)
        return None


def save_baseline(scores: dict, meta: dict, path: Path = BASELINE_PATH) -> None:
    payload = {
        "recorded_at": datetime.now(UTC).isoformat(),
        **meta,
        "scores": {
            k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def compare(scores: dict, baseline: dict | None, tolerance: float) -> tuple[list[str], list[str]]:
    """Compare *scores* against floors and baseline. Returns (failures, notes)."""
    failures: list[str] = []
    notes: list[str] = []

    for metric, floor in sorted(FLOORS.items()):
        value = scores.get(metric)
        if not isinstance(value, (int, float)):
            # A missing metric is a failure, not a pass. Treating it as
            # absent-therefore-fine is how a broken harness reports green.
            failures.append(f"{metric}: not reported by the run")
            continue
        if value < floor:
            failures.append(f"{metric}: {value:.4f} is below the floor of {floor:.2f}")

    if baseline is None:
        notes.append(
            "No baseline recorded — only absolute floors were checked. "
            "Run with --update-baseline on a known-good commit to enable "
            "regression detection."
        )
        return failures, notes

    previous = baseline.get("scores", {})
    for metric, before in sorted(previous.items()):
        now = scores.get(metric)
        if not isinstance(now, (int, float)) or not isinstance(before, (int, float)):
            continue
        delta = now - before
        if delta < -tolerance:
            failures.append(
                f"{metric}: {now:.4f} regressed {abs(delta):.4f} from baseline "
                f"{before:.4f} (tolerance {tolerance:.2f})"
            )
        elif delta > tolerance:
            notes.append(f"{metric} improved by {delta:.4f} — consider --update-baseline")
    return failures, notes


def _measure(args) -> tuple[dict, int]:
    """Run the existing harness and return (scores, item_count)."""
    from src.eval.ragas_eval import collect_predictions, load_eval_set, run_ragas
    from src.retrieval import get_retriever

    dataset = load_eval_set(limit=args.limit)
    retriever = get_retriever(strategy=args.retriever)

    if args.mode == "graph":
        from src.graph.build_graph import ask

        answer_fn = ask
    else:
        from src.rag.naive_rag import answer_question

        def answer_fn(q: str) -> str:
            return answer_question(q, retriever_strategy=args.retriever)

    rows = collect_predictions(dataset, answer_fn, retriever)
    return run_ragas(rows), len(dataset)


def main() -> int:
    from config import settings, setup_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["naive", "graph"], default="graph")
    parser.add_argument(
        "--retriever",
        choices=["dense", "hybrid", "multi_query", "rerank", "hybrid_rerank"],
        default="hybrid",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"Allowed drop from baseline before failing (default {DEFAULT_TOLERANCE})",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Record this run as the new baseline instead of gating on it.",
    )
    parser.add_argument(
        "--scores-file", type=str, default=None,
        help="Read scores from a previous run instead of measuring again.",
    )
    args = parser.parse_args()

    setup_logging()
    settings.validate()

    if args.scores_file:
        payload = json.loads(Path(args.scores_file).read_text(encoding="utf-8"))
        scores, count = payload.get("scores", payload), payload.get("items", 0)
    else:
        scores, count = _measure(args)

    meta = {
        "mode": args.mode,
        "retriever": args.retriever,
        "items": count,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
    }
    try:
        from src.prompts import prompt_fingerprint

        meta["prompt_set"] = prompt_fingerprint()
    except Exception:
        pass

    if args.update_baseline:
        save_baseline(scores, meta)
        print(f"Baseline updated: {BASELINE_PATH}")
        for metric, value in sorted(scores.items()):
            if isinstance(value, (int, float)):
                print(f"  {metric:>20}: {value:.4f}")
        return 0

    baseline = load_baseline()
    failures, notes = compare(scores, baseline, args.tolerance)

    print(f"\nRAGAS gate — {args.mode}/{args.retriever}, {count} items")
    if baseline:
        print(
            f"Baseline recorded {baseline.get('recorded_at', '?')} "
            f"(prompt_set={baseline.get('prompt_set', '?')})"
        )
    print("-" * 66)
    previous = (baseline or {}).get("scores", {})
    for metric in sorted(set(scores) | set(FLOORS)):
        value = scores.get(metric)
        if not isinstance(value, (int, float)):
            continue
        before = previous.get(metric)
        delta = f"{value - before:+.4f}" if isinstance(before, (int, float)) else "    n/a"
        floor = FLOORS.get(metric)
        floor_text = f"floor {floor:.2f}" if floor else "no floor"
        print(f"  {metric:>20}: {value:.4f}  ({delta} vs baseline, {floor_text})")

    for note in notes:
        print(f"\nnote: {note}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nIf this change is a deliberate trade-off, re-record the "
            "baseline with --update-baseline and say why in the commit."
        )
        return 1

    print("\nPASSED — no metric regressed beyond tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
