"""Evaluation report and comparison utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .metrics._base import EvalResult


@dataclass
class EvalReport:
    """Consolidated evaluation report over one or more metric results."""

    results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def total_evaluations(self) -> int:
        """Alias for total evaluation results count."""
        return self.total

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_count / len(self.results)

    def per_metric_summary(self) -> dict[str, dict[str, Any]]:
        """Compute aggregated metrics (pass rate, mean score/noul, count)."""
        metrics: dict[str, list[EvalResult]] = {}
        for r in self.results:
            metrics.setdefault(r.metric_name, []).append(r)

        out: dict[str, dict[str, Any]] = {}
        for name, res_list in metrics.items():
            scores = [r.score for r in res_list if r.score is not None]
            nouls = [r.noul for r in res_list if r.noul is not None]
            passes = sum(1 for r in res_list if r.passed)
            count = len(res_list)

            entry: dict[str, Any] = {
                "count": count,
                "passed": passes,
                "pass_rate": passes / count if count else 0.0,
            }
            if scores:
                entry["mean_score"] = float(np.mean(scores))
            if nouls:
                entry["mean_noul"] = float(np.mean(nouls))

            out[name] = entry
        return out

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total": self.total,
            "passed": self.passed_count,
            "pass_rate": round(self.pass_rate, 4),
            "metadata": self.metadata,
            "per_metric": self.per_metric_summary(),
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Render a readable summary table of the evaluation results."""
        lines = []
        lines.append("=" * 70)
        lines.append(
            f"LayaEval Evaluation Report  (Total: {self.total}, Pass Rate: {self.pass_rate:.1%})"
        )
        lines.append("=" * 70)
        lines.append(
            f"{'Metric':<28} {'Type':<8} {'Count':<7} {'Pass Rate':<12} {'Mean Value':<10}"
        )
        lines.append("-" * 70)

        summary_data = self.per_metric_summary()
        for name, stats in summary_data.items():
            val_str = "-"
            if "mean_score" in stats:
                val_str = f"{stats['mean_score']:.2f}/4.0"
                m_type = "score"
            elif "mean_noul" in stats:
                val_str = f"{stats['mean_noul']:.2%}"
                m_type = "noul"
            else:
                m_type = "choice"

            lines.append(
                f"{name:<28} {m_type:<8} {stats['count']:<7} {stats['pass_rate']:<12.1%} {val_str:<10}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def compare(self, baseline: EvalReport) -> dict[str, Any]:
        """Compare this report (candidate) against a baseline report.

        Returns delta per metric and overall pass_rate difference.
        """
        curr = self.per_metric_summary()
        base = baseline.per_metric_summary()

        diff: dict[str, Any] = {
            "candidate_pass_rate": self.pass_rate,
            "baseline_pass_rate": baseline.pass_rate,
            "delta_pass_rate": self.pass_rate - baseline.pass_rate,
            "metrics": {},
        }

        all_keys = sorted(set(curr.keys()) | set(base.keys()))
        for k in all_keys:
            c_info = curr.get(k, {})
            b_info = base.get(k, {})

            m_diff: dict[str, Any] = {
                "candidate_passed": c_info.get("passed", 0),
                "baseline_passed": b_info.get("passed", 0),
                "candidate_pass_rate": c_info.get("pass_rate", 0.0),
                "baseline_pass_rate": b_info.get("pass_rate", 0.0),
                "delta_pass_rate": c_info.get("pass_rate", 0.0) - b_info.get("pass_rate", 0.0),
            }
            if "mean_score" in c_info and "mean_score" in b_info:
                m_diff["delta_mean_score"] = c_info["mean_score"] - b_info["mean_score"]
            if "mean_noul" in c_info and "mean_noul" in b_info:
                m_diff["delta_mean_noul"] = c_info["mean_noul"] - b_info["mean_noul"]

            diff["metrics"][k] = m_diff

        return diff
