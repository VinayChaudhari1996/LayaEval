"""Registry and discovery utilities for LayaEval metrics."""

from __future__ import annotations

import difflib
from typing import Any

from ._base import Metric
from .agent import (
    AgentAbstention,
    AgentAnswerCompleteness,
    AgentCorrectness,
    AgentGroundedness,
    AgentHallucinationRisk,
    AgentResponseRelevance,
    GoalAchievement,
    IndirectPromptInjectionDetection,
    IntentResolution,
    NavigationEfficiency,
    PolicyCompliance,
    PromptInjectionDetection,
    TaskAdherence,
    TaskCompletion,
    ToolCallAccuracy,
    ToolInputAccuracy,
    ToolNecessity,
    ToolOutputUtilization,
    ToolRedundancy,
    ToolSelection,
    ToolSelectionCorrectness,
    ToolSequenceCorrectness,
)
from .rag import (
    AbstentionCorrectness,
    Answerability,
    CitationCompleteness,
    CitationCorrectness,
    CitationRelevance,
    ClaimSupport,
    ContextConflictDetection,
    ContextCoverage,
    ContextSufficiency,
    ContextUtilization,
    ContradictionDetection,
    DocumentRelevance,
    FactualCorrectness,
    Groundedness,
    HallucinationRisk,
    ResponseCompleteness,
    ResponseRelevance,
    RetrievalCompleteness,
    RetrievalRelevance,
    StrictGroundedness,
)

RAG_METRICS: tuple[type[Metric], ...] = (
    RetrievalRelevance,
    DocumentRelevance,
    ContextSufficiency,
    ContextCoverage,
    RetrievalCompleteness,
    Groundedness,
    StrictGroundedness,
    HallucinationRisk,
    ResponseRelevance,
    ResponseCompleteness,
    FactualCorrectness,
    Answerability,
    AbstentionCorrectness,
    ContextUtilization,
    CitationCorrectness,
    CitationCompleteness,
    CitationRelevance,
    ClaimSupport,
    ContradictionDetection,
    ContextConflictDetection,
)

AGENT_METRICS: tuple[type[Metric], ...] = (
    IntentResolution,
    TaskCompletion,
    TaskAdherence,
    GoalAchievement,
    ToolSelection,
    ToolSelectionCorrectness,
    ToolCallAccuracy,
    ToolInputAccuracy,
    ToolOutputUtilization,
    ToolNecessity,
    ToolRedundancy,
    ToolSequenceCorrectness,
    NavigationEfficiency,
    AgentGroundedness,
    AgentHallucinationRisk,
    AgentResponseRelevance,
    AgentAnswerCompleteness,
    AgentCorrectness,
    AgentAbstention,
    PolicyCompliance,
    PromptInjectionDetection,
    IndirectPromptInjectionDetection,
)

ALL_METRICS: tuple[type[Metric], ...] = RAG_METRICS + AGENT_METRICS

# Build fast lookup indexes
_BY_NAME: dict[str, type[Metric]] = {}
_BY_ID: dict[str, type[Metric]] = {}
_BY_CLASS_NAME: dict[str, type[Metric]] = {}

for _cls in ALL_METRICS:
    _BY_NAME[_cls.metric_name.lower()] = _cls
    _BY_ID[_cls.metric_id.lower()] = _cls
    _BY_CLASS_NAME[_cls.__name__.lower()] = _cls


def get_metric(
    identifier: Metric | type[Metric] | str,
    threshold: float | None = None,
    **kwargs: Any,
) -> Metric:
    """Resolve a metric from an instance, a class, or a string identifier.

    Args:
        identifier: A Metric instance, Metric subclass, or string
            (e.g. 'groundedness', 'RAG-006', 'Groundedness').
        threshold: Optional default threshold overriding the metric's default.
        **kwargs: Additional keyword arguments passed to the Metric constructor.

    Returns:
        An instantiated Metric object.

    Raises:
        ValueError: If the identifier cannot be resolved to a known Metric.
    """
    if isinstance(identifier, Metric):
        if threshold is not None:
            identifier.default_threshold = float(threshold)
        return identifier

    if isinstance(identifier, type) and issubclass(identifier, Metric):
        init_kwargs = dict(kwargs)
        if threshold is not None:
            init_kwargs["threshold"] = threshold
        return identifier(**init_kwargs)

    if isinstance(identifier, str):
        key = identifier.strip().lower()

        cls = _BY_NAME.get(key) or _BY_ID.get(key) or _BY_CLASS_NAME.get(key)
        if cls is not None:
            init_kwargs = dict(kwargs)
            if threshold is not None:
                init_kwargs["threshold"] = threshold
            return cls(**init_kwargs)

        all_keys = sorted(
            set(list(_BY_NAME.keys()) + list(_BY_ID.keys()) + list(_BY_CLASS_NAME.keys()))
        )
        matches = difflib.get_close_matches(key, all_keys, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
        raise ValueError(f"Unknown metric '{identifier}'.{hint}")

    raise TypeError(
        f"Expected Metric instance, Metric class, or string identifier, got {type(identifier).__name__}"
    )


def resolve_metrics(
    metrics: Metric | type[Metric] | str | list[Any] | tuple[Any, ...],
    thresholds: dict[str, float] | None = None,
) -> list[Metric]:
    """Resolve a single metric or list of metrics into instantiated Metric objects."""
    thresholds = thresholds or {}
    items = [metrics] if not isinstance(metrics, (list, tuple)) else list(metrics)
    resolved: list[Metric] = []
    for item in items:
        m = get_metric(item)
        t = thresholds.get(m.metric_name, thresholds.get(m.metric_id))
        if t is not None:
            m.default_threshold = float(t)
        resolved.append(m)
    return resolved


def list_metrics(category: str | None = None) -> list[dict[str, Any]]:
    """Return catalog information for all supported metrics.

    Args:
        category: Optional filter: 'rag' or 'agent'.

    Returns:
        List of dictionaries with metric metadata.
    """
    category_lower = category.lower() if category else None
    catalog: list[dict[str, Any]] = []

    for cls in ALL_METRICS:
        is_rag = cls in RAG_METRICS
        cat = "rag" if is_rag else "agent"
        if category_lower and cat != category_lower:
            continue

        catalog.append(
            {
                "id": cls.metric_id,
                "name": cls.metric_name,
                "class_name": cls.__name__,
                "category": cat,
                "question_type": cls.question_type,
                "state_fields": list(cls.state_fields),
                "optional_fields": list(cls.optional_fields),
                "doc": (cls.__doc__ or "").strip().split("\n")[0],
            }
        )

    return catalog
