"""Phase 2: evaluation harness with RAGAS.

This is the single most important file in the project. It turns "the answers
feel better" into numbers you can defend. Run it against the naive baseline
first, record the scores, then re-run after every retrieval change.

Metrics:
  - faithfulness      : is the answer grounded in the retrieved context?
  - answer_relevancy  : does the answer address the question?
  - context_precision : are the retrieved chunks actually relevant?
  - context_recall    : did we retrieve everything needed? (needs ground truth)

Usage:
    python -m src.eval.ragas_eval                       # evaluate naive pipeline
    python -m src.eval.ragas_eval --mode graph           # evaluate CRAG pipeline
    python -m src.eval.ragas_eval --limit 10             # quick test with 10 items
    python -m src.eval.ragas_eval --output results.json  # save detailed results
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set(path: Path | None = None, limit: int | None = None) -> list[dict]:
    """Load the Q/A eval set from JSON."""
    p = path or EVAL_SET_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    logger.info("Loaded %d eval items from %s", len(data), p)
    return data


def collect_predictions(
    eval_set: list[dict],
    answer_fn: Callable[[str], str],
    retriever,
) -> dict:
    """Run the answer_fn and retriever over each eval item.

    Returns a dict with keys: question, answer, contexts, ground_truth
    suitable for RAGAS Dataset.from_dict().
    """
    rows: dict[str, list] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    total = len(eval_set)
    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        gt = item["ground_truth"]
        logger.info("[%d/%d] Evaluating: %s", i, total, q[:80])

        try:
            docs = retriever.invoke(q)
            ans = answer_fn(q)
        except Exception:
            logger.exception("Failed on question: %s", q[:80])
            docs = []
            ans = "(error)"

        rows["question"].append(q)
        rows["answer"].append(ans)
        rows["contexts"].append([d.page_content for d in docs])
        rows["ground_truth"].append(gt)

    return rows


def run_ragas(rows: dict) -> dict:
    """Run RAGAS evaluation over collected predictions.

    Returns a dict of metric_name -> score.
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(rows)
    result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    # ragas 0.2.x EvaluationResult stores mean scores in _repr_dict
    return dict(result._repr_dict)


def evaluate(
    answer_fn: Callable[[str], str],
    retriever,
    eval_set: list[dict] | None = None,
    limit: int | None = None,
) -> dict:
    """End-to-end evaluation: load data, collect predictions, run RAGAS.

    Returns the RAGAS scores dict.
    """
    items = eval_set or load_eval_set(limit=limit)
    rows = collect_predictions(items, answer_fn, retriever)
    return run_ragas(rows)


def save_results(
    scores: dict,
    mode: str,
    output_path: str | None = None,
    num_items: int = 0,
    retriever: str = "dense",
) -> str:
    """Save evaluation results to a JSON file."""
    results_dir = Path(__file__).parent.parent.parent / "eval_results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = output_path or str(results_dir / f"{mode}_{retriever}_{timestamp}.json")

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode,
        "retriever": retriever,
        "num_items": num_items,
        "scores": {k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()},
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Results saved to %s", filename)
    return filename


def main() -> None:
    from config import settings, setup_logging

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation suite.")
    parser.add_argument(
        "--mode", choices=["naive", "graph"], default="naive",
        help="Pipeline to evaluate (default: naive)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N items (for quick testing)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "hybrid", "multi_query", "rerank"],
        default="dense",
        help="Retrieval strategy (default: dense)",
    )
    args = parser.parse_args()

    setup_logging()
    settings.validate()

    # Load eval set
    eval_set = load_eval_set(limit=args.limit)
    print(f"\n{'='*60}")
    print(f"RAGAS Evaluation — mode: {args.mode}, retriever: {args.retriever}, items: {len(eval_set)}")
    print(f"{'='*60}\n")

    # Set up answer function and retriever using the factory
    from src.retrieval import get_retriever
    retriever = get_retriever(strategy=args.retriever)

    if args.mode == "graph":
        from src.graph.build_graph import ask as graph_ask
        answer_fn = lambda q: graph_ask(q, retriever_strategy=args.retriever)
    else:
        from src.rag.naive_rag import answer as naive_answer
        answer_fn = lambda q: naive_answer(q, retriever_strategy=args.retriever)

    # Collect predictions
    print("Collecting predictions...")
    start = time.time()
    rows = collect_predictions(eval_set, answer_fn, retriever)
    predict_time = time.time() - start
    print(f"  Predictions collected in {predict_time:.1f}s")

    # Run RAGAS
    print("\nRunning RAGAS metrics (this calls the LLM for each metric)...")
    start = time.time()
    scores = run_ragas(rows)
    eval_time = time.time() - start
    print(f"  RAGAS evaluation completed in {eval_time:.1f}s")

    # Display results
    print(f"\n{'='*60}")
    print(f"  RAGAS Scores — {args.mode} pipeline, {args.retriever} retriever")
    print(f"{'='*60}")

    # PRD baseline targets for reference
    prd_targets = {
        "faithfulness": 0.65,
        "answer_relevancy": 0.70,
        "context_precision": 0.60,
        "context_recall": 0.70,
    }

    for metric, value in sorted(scores.items()):
        if isinstance(value, float):
            target = prd_targets.get(metric)
            status = ""
            if target is not None:
                status = " PASS" if value >= target else " FAIL"
            print(f"  {metric:>20}: {value:.4f}  (target >= {target:.2f}){status}" if target else f"  {metric:>20}: {value:.4f}")

    print(f"\n  Total time: {predict_time + eval_time:.1f}s")
    print(f"  Items evaluated: {len(eval_set)}")
    print(f"{'='*60}\n")

    # Save results
    filepath = save_results(scores, args.mode, args.output, len(eval_set), args.retriever)
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
