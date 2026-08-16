"""CLI dashboard for query cost and latency metrics.

Usage:
    python -m scripts.metrics              # last 20 queries
    python -m scripts.metrics --last 50    # last 50 queries
    python -m scripts.metrics --all        # all recorded queries
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime

from config import setup_logging
from src.observability.metrics_store import COST_BUDGET, get_store


def _format_row(i: int, row: dict) -> str:
    """Format a single metrics row for display."""
    thread = row["thread_id"][:10]
    mode = row["mode"]
    retriever = row["retriever"]
    latency = f"{row['latency_ms']:.0f}ms"
    tokens = f"{row['total_tok']:,}"
    cost = f"${row['cost_usd']:.5f}"
    flag = " *" if row["cost_usd"] > COST_BUDGET else ""
    return f"  {i:<3} {thread:<12} {mode:<7} {retriever:<13} {latency:>8} {tokens:>8} {cost:>10}{flag}"


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Query cost & latency dashboard",
    )
    parser.add_argument(
        "--last", type=int, default=20,
        help="Number of recent queries to show (default: 20)",
    )
    parser.add_argument(
        "--all", action="store_true", default=False,
        help="Show all recorded queries",
    )
    args = parser.parse_args()

    store = get_store()
    n = None if args.all else args.last
    rows = store.query_recent(n) if n else store.query_recent(999_999)
    stats = store.summary(n)

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    scope = "all queries" if args.all else f"last {args.last} queries"

    print()
    print("=" * 78)
    print("  Query Cost & Latency Dashboard")
    print(f"  {scope}  (as of {now})")
    print("=" * 78)

    if not rows:
        print("\n  No queries recorded yet.\n")
        print("=" * 78)
        return

    # Header
    print(f"  {'#':<3} {'THREAD':<12} {'MODE':<7} {'RETRIEVER':<13} {'LATENCY':>8} {'TOKENS':>8} {'COST':>10}")
    print("-" * 78)

    # Rows (reversed so newest is at bottom, matching natural reading order)
    for i, row in enumerate(reversed(rows), 1):
        print(_format_row(i, row))

    # Summary
    print("-" * 78)
    print(f"  SUMMARY ({scope})")
    print(f"  Total cost:        ${stats['total_cost']:.5f}")
    print(f"  Avg cost/query:    ${stats['avg_cost']:.5f}  [target: < ${COST_BUDGET}]")
    print(f"  Avg latency:       {stats['avg_latency']:.0f}ms")
    print(f"  Total tokens:      {stats['total_tokens']:,}")
    print(f"  Total queries:     {stats['cnt']}")
    print(f"  Over budget:       {stats['over_budget']}")
    if stats["over_budget"] > 0:
        print(f"  * = cost exceeds ${COST_BUDGET} budget")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
