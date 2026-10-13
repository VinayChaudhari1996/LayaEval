"""Unit tests for all 42 RAG and Agent evaluation metrics."""

import pytest

from layaeval.metrics.agent import (
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
from layaeval.metrics.rag import (
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


def test_groundedness_metric():
    m = Groundedness(threshold=3.0)
    assert m.metric_id == "RAG-006"
    assert m.question_type == "score"

    # State validation
    with pytest.raises(ValueError, match="requires state fields"):
        m.build_state(retrieved_context="Some context")

    st = m.build_state(retrieved_context="Sky is blue.", response="The sky is blue.")
    assert st["retrieved_context"] == "Sky is blue."
    assert st["response"] == "The sky is blue."

    # Question template
    q = m.build_questions()
    assert "groundedness" in q
    assert q["groundedness"]["type"] == "score"

    # Parsing passing result
    ans = {
        "groundedness": {
            "score": 4.0,
            "confidence": 0.95,
            "answer_confidence": 0.92,
        }
    }
    res = m.parse_result(ans)
    assert res.passed is True
    assert res.score == 4.0
    assert res.confidence == 0.95

    # Parsing failing result
    ans_fail = {"groundedness": {"score": 2.0}}
    res_fail = m.parse_result(ans_fail)
    assert res_fail.passed is False
    assert res_fail.score == 2.0


def test_hallucination_risk_metric():
    m = HallucinationRisk(threshold=0.3)
    assert m.metric_id == "RAG-008"
    assert m.question_type == "noul"

    # Low risk passes
    ans_pass = {"hallucination": {"noul": 0.1, "confidence": 0.9}}
    res_pass = m.parse_result(ans_pass)
    assert res_pass.passed is True
    assert res_pass.noul == 0.1

    # High risk fails
    ans_fail = {"hallucination": {"noul": 0.8, "confidence": 0.85}}
    res_fail = m.parse_result(ans_fail)
    assert res_fail.passed is False


def test_all_rag_metrics_instantiation():
    rag_classes = [
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
    ]
    assert len(rag_classes) == 20
    for cls in rag_classes:
        inst = cls()
        assert inst.metric_id.startswith("RAG-")
        q = inst.build_questions()
        assert len(q) == 1
        assert len(inst.state_fields) > 0


def test_all_agent_metrics_instantiation():
    agent_classes = [
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
    ]
    assert len(agent_classes) == 22
    for cls in agent_classes:
        inst = cls()
        assert inst.metric_id.startswith("AG-")
        q = inst.build_questions()
        assert len(q) == 1
        assert len(inst.state_fields) > 0


def test_intent_resolution_choice_parsing():
    m = IntentResolution()
    ans = {"intent": {"choice": "troubleshooting", "confidence": 0.98}}
    res = m.parse_result(ans)
    assert res.passed is True
    assert res.choice == "troubleshooting"
    assert res.confidence == 0.98
