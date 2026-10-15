"""Unit tests for LayaEval CLI commands."""

from layaeval.cli import main
from layaeval.metrics._base import EvalResult
from layaeval.report import EvalReport


def test_cli_list_metrics(capsys):
    exit_code = main(["list-metrics"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Available LayaEval Metrics" in captured.out
    assert "groundedness" in captured.out
    assert "task_completed" in captured.out


def test_cli_compare(tmp_path, capsys):
    r1 = EvalResult(metric_id="RAG-006", metric_name="groundedness", score=3.5, passed=True)
    r2 = EvalResult(metric_id="RAG-006", metric_name="groundedness", score=2.0, passed=False)

    base = EvalReport(results=[r1, r1])  # 100% pass rate
    cand = EvalReport(results=[r1, r2])  # 50% pass rate

    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"
    base.save(base_path)
    cand.save(cand_path)

    exit_code = main(["compare", str(base_path), str(cand_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Baseline Pass Rate:  100.00%" in captured.out
    assert "Candidate Pass Rate: 50.00%" in captured.out
    assert "Delta:               -50.00%" in captured.out
