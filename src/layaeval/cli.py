"""Command line interface for LayaEval."""

from __future__ import annotations

import argparse
import json
import sys

from ._version import __version__
from .dataset import EvalDataset
from .evaluator import Evaluator
from .metrics import ALL_METRICS, get_metric


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="layaeval",
        description="LayaEval: Agent & RAG Evaluation powered by the Laya MCP Server",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-metrics
    subparsers.add_parser("list-metrics", help="List all available RAG and Agent metrics")

    # run
    run_p = subparsers.add_parser("run", help="Run evaluation on a JSONL dataset")
    run_p.add_argument("dataset", help="Path to evaluation JSONL file")
    run_p.add_argument("--metrics", "-m", nargs="+", help="Metric names or IDs to evaluate")
    run_p.add_argument("--output", "-o", help="Path to write report JSON")
    run_p.add_argument("--model", default="auto", help="Laya model checkpoint ('auto', 'english')")
    run_p.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Display hardware info and real-time progress bar (default)",
    )
    run_p.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable progress bar and diagnostic banner",
    )

    # compare
    cmp_p = subparsers.add_parser("compare", help="Compare candidate report against baseline")
    cmp_p.add_argument("baseline", help="Baseline report JSON file")
    cmp_p.add_argument("candidate", help="Candidate report JSON file")

    args = parser.parse_args(argv)

    if args.command == "list-metrics":
        print("Available LayaEval Metrics:\n")
        print(f"{'ID':<10} {'Name':<32} {'Type':<8}")
        print("-" * 52)
        for m_cls in ALL_METRICS:
            print(f"{m_cls.metric_id:<10} {m_cls.metric_name:<32} {m_cls.question_type:<8}")
        return 0

    if args.command == "run":
        dataset = EvalDataset.from_jsonl(args.dataset)
        selected_metrics = []
        if args.metrics:
            for name in args.metrics:
                try:
                    selected_metrics.append(get_metric(name))
                except ValueError:
                    print(f"Error: Unknown metric '{name}'", file=sys.stderr)
                    return 1
        else:
            selected_metrics = [m_cls() for m_cls in ALL_METRICS]

        print(
            f"Loaded {len(dataset)} examples. Starting evaluation with {len(selected_metrics)} metrics..."
        )
        with Evaluator(model=args.model, progress=args.progress) as ev:
            report = ev.evaluate_dataset(dataset, selected_metrics, progress=args.progress)

        print("\n" + report.summary())
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report.to_json())
            print(f"\nSaved report to {args.output}")
        return 0

    if args.command == "compare":
        with open(args.baseline, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        with open(args.candidate, "r", encoding="utf-8") as f:
            cand_data = json.load(f)

        [r for r in base_data.get("results", [])]
        [r for r in cand_data.get("results", [])]

        # Simple delta output
        delta_rate = cand_data.get("pass_rate", 0) - base_data.get("pass_rate", 0)
        print("=" * 60)
        print(f"Baseline Pass Rate:  {base_data.get('pass_rate', 0):.2%}")
        print(f"Candidate Pass Rate: {cand_data.get('pass_rate', 0):.2%}")
        print(f"Delta:               {delta_rate:+.2%}")
        print("=" * 60)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
