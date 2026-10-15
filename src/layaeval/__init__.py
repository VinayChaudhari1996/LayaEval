"""LayaEval: Agent & RAG Evaluation Metrics powered by the Laya MCP Server."""

from __future__ import annotations

from ._version import __version__
from .dataset import EvalDataset, EvalExample
from .evaluator import Evaluator, close_default_evaluator, default_evaluator, evaluate
from .hardware import HardwareInfo, detect_hardware
from .mcp.client import DemoMCPClient, LayaMCPClient, MCPClientError, MockMCPClient
from .metrics import agent, rag
from .metrics._base import EvalResult, Metric
from .metrics.agent import (
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
from .metrics.rag import (
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
from .metrics.registry import (
    AGENT_METRICS,
    ALL_METRICS,
    RAG_METRICS,
    get_metric,
    list_metrics,
    resolve_metrics,
)
from .progress import ProgressBar
from .report import EvalReport

__all__ = [
    "__version__",
    "evaluate",
    "Evaluator",
    "default_evaluator",
    "close_default_evaluator",
    "EvalReport",
    "EvalDataset",
    "EvalExample",
    "EvalResult",
    "Metric",
    "get_metric",
    "resolve_metrics",
    "list_metrics",
    "ALL_METRICS",
    "RAG_METRICS",
    "AGENT_METRICS",
    "rag",
    "agent",
    "LayaMCPClient",
    "MockMCPClient",
    "DemoMCPClient",
    "MCPClientError",
    "detect_hardware",
    "HardwareInfo",
    "ProgressBar",
    # 20 RAG metrics
    "RetrievalRelevance",
    "DocumentRelevance",
    "ContextSufficiency",
    "ContextCoverage",
    "RetrievalCompleteness",
    "Groundedness",
    "StrictGroundedness",
    "HallucinationRisk",
    "ResponseRelevance",
    "ResponseCompleteness",
    "FactualCorrectness",
    "Answerability",
    "AbstentionCorrectness",
    "ContextUtilization",
    "CitationCorrectness",
    "CitationCompleteness",
    "CitationRelevance",
    "ClaimSupport",
    "ContradictionDetection",
    "ContextConflictDetection",
    # 22 Agent metrics
    "IntentResolution",
    "TaskCompletion",
    "TaskAdherence",
    "GoalAchievement",
    "ToolSelection",
    "ToolSelectionCorrectness",
    "ToolCallAccuracy",
    "ToolInputAccuracy",
    "ToolOutputUtilization",
    "ToolNecessity",
    "ToolRedundancy",
    "ToolSequenceCorrectness",
    "NavigationEfficiency",
    "AgentGroundedness",
    "AgentHallucinationRisk",
    "AgentResponseRelevance",
    "AgentAnswerCompleteness",
    "AgentCorrectness",
    "AgentAbstention",
    "PolicyCompliance",
    "PromptInjectionDetection",
    "IndirectPromptInjectionDetection",
    "test",
]


def test(args: list[str] | None = None) -> int:
    """Run LayaEval's built-in unit test suite."""
    try:
        import pytest
    except ImportError:
        raise ImportError(
            "pytest is required to run the test suite. Install with `pip install pytest`."
        ) from None
    from pathlib import Path

    test_dir = Path(__file__).parent / "tests"
    cmd = [str(test_dir)]
    if args:
        cmd.extend(args)
    return pytest.main(cmd)
