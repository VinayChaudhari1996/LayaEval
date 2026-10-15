"""Unit tests for Evaluator orchestrator and dataset evaluation."""

from unittest.mock import MagicMock

from layaeval.dataset import EvalDataset, EvalExample
from layaeval.evaluator import Evaluator
from layaeval.mcp.client import LayaMCPClient
from layaeval.metrics.agent import TaskCompletion
from layaeval.metrics.rag import Groundedness, HallucinationRisk


def test_evaluator_single_metric():
    mock_client = MagicMock(spec=LayaMCPClient)
    mock_client.predict.return_value = {
        "groundedness": {
            "score": 3.8,
            "confidence": 0.94,
            "answer_confidence": 0.91,
        }
    }

    ev = Evaluator(client=mock_client)
    res = ev.evaluate(
        Groundedness(),
        retrieved_context="Python was created by Guido van Rossum.",
        response="Guido created Python.",
        threshold=3.0,
    )

    assert res.passed is True
    assert res.score == 3.8
    assert res.confidence == 0.94
    mock_client.predict.assert_called_once()


def test_evaluator_multi_metric_single_pass():
    mock_client = MagicMock(spec=LayaMCPClient)
    mock_client.predict.return_value = {
        "groundedness": {"score": 4.0, "confidence": 0.9},
        "hallucination": {"noul": 0.05, "confidence": 0.95},
    }

    ev = Evaluator(client=mock_client)
    results = ev.evaluate_metrics(
        [Groundedness(), HallucinationRisk(threshold=0.2)],
        retrieved_context="Context info",
        response="Response info",
    )

    assert len(results) == 2
    assert results[0].metric_name == "groundedness"
    assert results[0].passed is True
    assert results[1].metric_name == "hallucination"
    assert results[1].passed is True

    # Verify predict was called with merged questions in one single call
    assert mock_client.predict.call_count == 1
    call_args = mock_client.predict.call_args[1]
    assert "groundedness" in call_args["questions"]
    assert "hallucination" in call_args["questions"]


def test_evaluator_dataset():
    mock_client = MagicMock(spec=LayaMCPClient)
    mock_client.predict.return_value = {"task_completed": {"noul": 0.95, "confidence": 0.92}}

    examples = [
        EvalExample(
            state={"task": "do x", "agent_trace": "did x", "final_result": "x done"},
            metrics=["task_completed"],
        ),
        EvalExample(
            state={"task": "do y", "agent_trace": "did y", "final_result": "y done"},
            metrics=["task_completed"],
        ),
    ]
    dataset = EvalDataset(examples)

    ev = Evaluator(client=mock_client)
    report = ev.evaluate_dataset(dataset, [TaskCompletion()])

    assert report.total == 2
    assert report.pass_rate == 1.0
    summary_str = report.summary()
    assert "task_completed" in summary_str


def test_top_level_evaluate_string_and_class():
    mock_client = MagicMock(spec=LayaMCPClient)
    mock_client.predict.return_value = {"groundedness": {"score": 3.5, "confidence": 0.9}}
    ev = Evaluator(client=mock_client)

    # 1. String identifier
    from layaeval import evaluate

    res1 = evaluate("groundedness", evaluator=ev, retrieved_context="ctx", response="resp")
    assert res1.passed is True
    assert res1.score == 3.5

    # 2. Metric class (uninstantiated)
    res2 = evaluate(Groundedness, evaluator=ev, retrieved_context="ctx", response="resp")
    assert res2.passed is True
    assert res2.score == 3.5

    # 3. List of strings in unified evaluate()
    mock_client.predict.return_value = {
        "groundedness": {"score": 3.5},
        "hallucination": {"noul": 0.1},
    }
    results = evaluate(
        ["groundedness", "hallucination"],
        evaluator=ev,
        retrieved_context="ctx",
        response="resp",
    )
    assert len(results) == 2
    assert results[0].metric_name == "groundedness"
    assert results[1].metric_name == "hallucination"


def test_metric_callable_and_evaluate_method():
    mock_client = MagicMock(spec=LayaMCPClient)
    mock_client.predict.return_value = {"groundedness": {"score": 3.8, "confidence": 0.95}}
    ev = Evaluator(client=mock_client)

    g = Groundedness(threshold=3.0)

    # Calling metric as a function
    res = g(evaluator=ev, retrieved_context="ctx", response="resp")
    assert res.score == 3.8
    assert res.passed is True

    # Calling metric.evaluate()
    res2 = g.evaluate(evaluator=ev, retrieved_context="ctx", response="resp")
    assert res2.score == 3.8
    assert res2.passed is True


def test_get_metric_and_list_metrics():
    import pytest

    from layaeval import get_metric, list_metrics

    # By ID
    m1 = get_metric("RAG-006")
    assert m1.metric_name == "groundedness"

    # By Name
    m2 = get_metric("groundedness")
    assert m2.metric_id == "RAG-006"

    # By Class Name
    m3 = get_metric("Groundedness", threshold=2.5)
    assert m3.default_threshold == 2.5

    # List metrics
    all_catalog = list_metrics()
    assert len(all_catalog) == 42

    rag_catalog = list_metrics(category="rag")
    assert len(rag_catalog) == 20

    agent_catalog = list_metrics(category="agent")
    assert len(agent_catalog) == 22

    # Error with suggestion
    with pytest.raises(ValueError, match="Did you mean: groundedness"):
        get_metric("groundednes")


def test_hardware_detection():
    from layaeval.hardware import detect_hardware

    hw = detect_hardware()
    assert hw.device_type in ("cuda", "mps", "cpu")
    assert hw.cores >= 1
    assert len(hw.description) > 0
    assert str(hw) == hw.description


def test_progress_bar_and_evaluator_progress():
    from layaeval.progress import ProgressBar

    items = [1, 2, 3]
    collected = []
    pbar = ProgressBar(items, desc="Testing", enabled=True)
    for x in pbar:
        collected.append(x)
    assert collected == items

    # Manual update
    manual_pbar = ProgressBar(total=5, desc="Manual", enabled=True)
    manual_pbar.update(2)
    manual_pbar.set_postfix_str("2 done")
    manual_pbar.update(3)
    manual_pbar.close()


def test_evaluate_dataset_with_progress():
    mock_client = MagicMock(spec=LayaMCPClient)
    mock_client.predict.return_value = {"groundedness": {"score": 3.9, "confidence": 0.95}}

    dataset = EvalDataset(
        [
            EvalExample(state={"retrieved_context": "c1", "response": "r1"}),
            EvalExample(state={"retrieved_context": "c2", "response": "r2"}),
        ]
    )

    ev = Evaluator(client=mock_client, progress=True)
    report = ev.evaluate_dataset(dataset, metrics=["groundedness"], progress=True)
    assert report.total == 2
    assert report.pass_rate == 1.0


def test_mock_mcp_client_and_demo_mode():
    import warnings

    from layaeval.mcp.client import MockMCPClient

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = MockMCPClient()
        assert len(w) == 1
        assert "MockMCPClient" in str(w[-1].message)

    res = client.predict(
        state={"retrieved_context": "Paris is capital", "response": "Paris is capital"},
        questions={"groundedness": {"type": "score"}},
    )
    assert res["groundedness"]["score"] == 3.0
    assert res["groundedness"]["confidence"] == 0.50

