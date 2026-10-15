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


def test_eval_report_save_and_load(tmp_path):
    r1 = EvalResult(
        metric_id="RAG-006",
        metric_name="groundedness",
        score=3.8,
        confidence=0.94,
        passed=True,
        threshold=3.0,
    )
    r2 = EvalResult(
        metric_id="RAG-008",
        metric_name="hallucination",
        noul=0.06,
        confidence=0.95,
        passed=True,
        threshold=0.2,
    )
    report = EvalReport(results=[r1, r2], metadata={"model": "test-v1"})

    filepath = tmp_path / "subdir" / "eval_report.json"
    report.save(filepath)

    assert filepath.exists()

    loaded = EvalReport.load(filepath)
    assert loaded.total == 2
    assert loaded.passed_count == 2
    assert loaded.metadata == {"model": "test-v1"}
    assert loaded.results[0].metric_name == "groundedness"
    assert loaded.results[0].score == 3.8
    assert loaded.results[0].confidence == 0.94
    assert loaded.results[1].metric_name == "hallucination"
    assert loaded.results[1].noul == 0.06
    assert loaded.results[1].confidence == 0.95
