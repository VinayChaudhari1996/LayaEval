"""Unit tests for EvalReport and comparisons."""

from layaeval.metrics._base import EvalResult
from layaeval.report import EvalReport


def test_eval_report_summary_and_compare():
    r1 = EvalResult(
        metric_id="RAG-006",
        metric_name="groundedness",
        score=3.5,
        passed=True,
        threshold=3.0,
    )
    r2 = EvalResult(
        metric_id="RAG-006",
        metric_name="groundedness",
        score=2.0,
        passed=False,
        threshold=3.0,
    )
    r3 = EvalResult(
        metric_id="RAG-008",
        metric_name="hallucination",
        noul=0.1,
        passed=True,
        threshold=0.3,
    )

    report = EvalReport(results=[r1, r2, r3])
    assert report.total == 3
    assert report.passed_count == 2
    assert abs(report.pass_rate - (2 / 3)) < 1e-4

    summary = report.summary()
    assert "groundedness" in summary
    assert "hallucination" in summary

    d = report.to_dict()
    assert d["total"] == 3
    assert "groundedness" in d["per_metric"]

    # Baseline comparison
    baseline = EvalReport(results=[r1, r1, r3])  # all 3 passed
    comparison = report.compare(baseline)
    assert comparison["candidate_pass_rate"] < comparison["baseline_pass_rate"]
    assert "groundedness" in comparison["metrics"]
