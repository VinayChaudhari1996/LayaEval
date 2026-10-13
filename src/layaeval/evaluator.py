"""Evaluator orchestrator running metrics through the Laya MCP client."""

from __future__ import annotations

import atexit
import threading
from collections.abc import Sequence
from typing import Any

from .dataset import EvalDataset
from .mcp.client import LayaMCPClient
from .metrics._base import EvalResult, Metric
from .metrics.registry import get_metric, resolve_metrics
from .report import EvalReport

_DEFAULT_EVALUATOR: Evaluator | None = None
_DEFAULT_LOCK = threading.Lock()


def default_evaluator() -> Evaluator:
    """Get or create the process-wide default Evaluator.

    Automatically cleans up its MCP server subprocess on exit.
    """
    global _DEFAULT_EVALUATOR
    with _DEFAULT_LOCK:
        if _DEFAULT_EVALUATOR is None:
            _DEFAULT_EVALUATOR = Evaluator()
            atexit.register(close_default_evaluator)
        return _DEFAULT_EVALUATOR


def close_default_evaluator() -> None:
    """Stop and cleanup the default Evaluator process."""
    global _DEFAULT_EVALUATOR
    with _DEFAULT_LOCK:
        if _DEFAULT_EVALUATOR is not None:
            try:
                _DEFAULT_EVALUATOR.stop()
            except Exception:
                pass
            _DEFAULT_EVALUATOR = None


class Evaluator:
    """Evaluates agent and RAG states using Laya typed questions over MCP."""

    def __init__(
        self,
        client: LayaMCPClient | None = None,
        laya_server_cmd: Sequence[str] | None = None,
        model: str = "auto",
        timeout: float = 60.0,
        demo: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            client: Optional pre-configured LayaMCPClient instance.
            laya_server_cmd: Command to start laya-mcp-server if no client passed.
            model: Model identifier to route to ('auto', 'english', etc.).
            timeout: Subprocess communication timeout in seconds.
            demo: If True, uses DemoMCPClient for simulated offline execution.
        """
        if demo and client is None:
            from .mcp.client import DemoMCPClient
            self.client = DemoMCPClient()
            self._external_client = True
        else:
            self._external_client = client is not None
            self.client = client or LayaMCPClient(command=laya_server_cmd, timeout=timeout)
        self.model = model
        self.timeout = timeout
        self.demo = demo

    def __enter__(self) -> Evaluator:
        if not self._external_client:
            self.client.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._external_client:
            self.client.stop()

    def start(self) -> None:
        """Start underlying MCP client if not running."""
        self.client.start()

    def stop(self) -> None:
        """Stop underlying MCP client."""
        if not self._external_client:
            self.client.stop()

    def evaluate(
        self,
        target: Metric | type[Metric] | str | Sequence[Metric | type[Metric] | str] | EvalDataset | None = None,
        *,
        metrics: Sequence[Metric | type[Metric] | str] | None = None,
        dataset: EvalDataset | None = None,
        threshold: float | None = None,
        thresholds: dict[str, float] | None = None,
        name: str | None = None,
        **state_kwargs: Any,
    ) -> EvalResult | list[EvalResult] | EvalReport:
        """Universal evaluation method.

        Accepts:
            - A single metric (instance, class, or string) -> returns `EvalResult`
            - A sequence of metrics -> returns `list[EvalResult]` in one forward pass
            - An EvalDataset (or `dataset=...`) -> returns `EvalReport`

        Examples:
            # Single metric
            ev.evaluate(Groundedness(), retrieved_context=ctx, response=resp)
            ev.evaluate("groundedness", retrieved_context=ctx, response=resp)

            # Multiple metrics in one forward pass
            ev.evaluate(["groundedness", "hallucination"], retrieved_context=ctx, response=resp)

            # Dataset
            ev.evaluate(dataset, metrics=["groundedness", "hallucination"])
        """
        # Case 1: Dataset evaluation
        if isinstance(target, EvalDataset) or dataset is not None:
            ds = target if isinstance(target, EvalDataset) else dataset
            assert ds is not None
            eval_metrics = metrics or []
            return self.evaluate_dataset(ds, metrics=eval_metrics, name=name)

        # Case 2: Multi-metric evaluation (sequence passed as target or metrics keyword)
        if isinstance(target, (list, tuple, set)):
            return self.evaluate_metrics(
                list(target),
                thresholds=thresholds,
                **state_kwargs,
            )

        if metrics is not None:
            return self.evaluate_metrics(
                metrics,
                thresholds=thresholds,
                **state_kwargs,
            )

        # Case 3: Single metric evaluation
        if target is None:
            raise ValueError(
                "evaluate() requires a target metric, list of metrics, or EvalDataset."
            )

        metric_obj = get_metric(target, threshold=threshold)
        state = metric_obj.build_state(**state_kwargs)
        questions = metric_obj.build_questions()

        answers = self.client.predict(state=state, questions=questions, model=self.model)
        return metric_obj.parse_result(answers, threshold=threshold, state=state)

    def evaluate_metrics(
        self,
        metrics: Sequence[Metric | type[Metric] | str],
        thresholds: dict[str, float] | None = None,
        **state_kwargs: Any,
    ) -> list[EvalResult]:
        """Evaluate multiple metrics against a shared state in one forward pass.

        Combines all questions into a single request to leverage LAYA's
        multi-question batch efficiency.
        """
        if not metrics:
            return []

        resolved_metrics = resolve_metrics(metrics, thresholds=thresholds)
        thresholds = thresholds or {}
        combined_questions: dict[str, Any] = {}
        validated_state: dict[str, Any] = {}

        for m in resolved_metrics:
            st = m.build_state(**state_kwargs)
            validated_state.update(st)
            combined_questions.update(m.build_questions())

        answers = self.client.predict(
            state=validated_state,
            questions=combined_questions,
            model=self.model,
        )

        results: list[EvalResult] = []
        for m in resolved_metrics:
            t = thresholds.get(m.metric_name, thresholds.get(m.metric_id, m.default_threshold))
            res = m.parse_result(answers, threshold=t, state=validated_state)
            results.append(res)
        return results

    def evaluate_dataset(
        self,
        dataset: EvalDataset,
        metrics: Sequence[Metric | type[Metric] | str],
        name: str | None = None,
    ) -> EvalReport:
        """Evaluate an entire dataset across a suite of metrics.

        Args:
            dataset: EvalDataset instance containing examples.
            metrics: List of metrics to evaluate on each example.
            name: Optional name for report metadata.

        Returns:
            An EvalReport with aggregate pass rates and comparison helpers.
        """
        resolved_metrics = resolve_metrics(metrics)
        all_results: list[EvalResult] = []
        metric_lookup = {m.metric_name.lower(): m for m in resolved_metrics}
        metric_id_lookup = {m.metric_id.lower(): m for m in resolved_metrics}

        for example in dataset:
            # Determine which metrics apply for this example
            if example.metrics:
                active_metrics: list[Metric] = []
                for name_or_id in example.metrics:
                    key = name_or_id.lower()
                    m = metric_lookup.get(key, metric_id_lookup.get(key))
                    if m is None:
                        # Fallback to resolving directly
                        try:
                            m = get_metric(name_or_id)
                        except ValueError:
                            m = None
                    if m is not None:
                        active_metrics.append(m)
            else:
                active_metrics = list(resolved_metrics)

            if not active_metrics:
                continue

            results = self.evaluate_metrics(
                active_metrics,
                thresholds=example.thresholds,
                **example.state,
            )
            all_results.extend(results)

        report = EvalReport(
            results=all_results,
            metadata={"dataset_size": len(dataset), "name": name or "eval_run"},
        )
        return report


def evaluate(
    target: Metric | type[Metric] | str | Sequence[Metric | type[Metric] | str] | EvalDataset | None = None,
    *,
    metrics: Sequence[Metric | type[Metric] | str] | None = None,
    dataset: EvalDataset | None = None,
    threshold: float | None = None,
    thresholds: dict[str, float] | None = None,
    evaluator: Evaluator | None = None,
    model: str | None = None,
    name: str | None = None,
    demo: bool = False,
    **state_kwargs: Any,
) -> EvalResult | list[EvalResult] | EvalReport:
    """High-level, ultra-simple evaluation entry point.

    Args:
        target: A single metric (instance, class, or string), a sequence of metrics,
            or an EvalDataset.
        metrics: Optional sequence of metrics if `dataset` is provided or if target is None.
        dataset: Optional EvalDataset instance.
        threshold: Optional score threshold override for a single metric.
        thresholds: Optional mapping of metric name/id to custom threshold.
        evaluator: Optional custom Evaluator instance. If None, uses default_evaluator().
        model: Optional model identifier override (e.g. 'auto', 'english').
        name: Optional run name for dataset reports.
        demo: Optional flag to use simulated client for offline testing / demonstrations.
        **state_kwargs: Evaluation input state (query, retrieved_context, response, etc.).

    Returns:
        An EvalResult (single metric), a list of EvalResults (multiple metrics),
        or an EvalReport (dataset).

    Examples:
        import layaeval

        # 1. Evaluate single metric by name or instance:
        res = layaeval.evaluate("groundedness", retrieved_context=ctx, response=resp)

        # 2. Evaluate multiple metrics in one neural forward pass:
        results = layaeval.evaluate(
            ["groundedness", "hallucination"],
            retrieved_context=ctx,
            response=resp,
        )

        # 3. Evaluate a dataset:
        report = layaeval.evaluate(dataset, metrics=["groundedness", "hallucination"])
    """
    if evaluator is None and demo:
        ev = Evaluator(demo=True)
    else:
        ev = evaluator or default_evaluator()
    if model is not None:
        prev_model = ev.model
        ev.model = model
        try:
            return ev.evaluate(
                target,
                metrics=metrics,
                dataset=dataset,
                threshold=threshold,
                thresholds=thresholds,
                name=name,
                **state_kwargs,
            )
        finally:
            ev.model = prev_model

    return ev.evaluate(
        target,
        metrics=metrics,
        dataset=dataset,
        threshold=threshold,
        thresholds=thresholds,
        name=name,
        **state_kwargs,
    )
