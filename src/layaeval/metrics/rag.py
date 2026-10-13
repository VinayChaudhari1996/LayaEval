"""20 LAYA-supported RAG Evaluation Metrics (RAG-001 -> RAG-020)."""

from __future__ import annotations

from ._base import Metric


class RetrievalRelevance(Metric):
    """RAG-001: Retrieval Relevance.
    Judges how relevant the retrieved context is to the user's query as a whole.
    """

    metric_id = "RAG-001"
    metric_name = "retrieval_relevance"
    question_type = "score"
    state_fields = ("query", "retrieved_context")
    _question_key = "retrieval_relevance"
    _question_template = {
        "retrieval_relevance": {
            "type": "score",
            "instructions": "How relevant is the retrieved context to the user's query?",
            "criteria": [
                "not relevant",
                "slightly relevant",
                "moderately relevant",
                "highly relevant",
                "directly relevant",
            ],
        }
    }


class DocumentRelevance(Metric):
    """RAG-002: Document Relevance.
    Judges how relevant an individual document/chunk is to the user's query.
    """

    metric_id = "RAG-002"
    metric_name = "document_relevance"
    question_type = "score"
    state_fields = ("query", "document")
    _question_key = "document_relevance"
    _question_template = {
        "document_relevance": {
            "type": "score",
            "instructions": "How relevant is this document to the user's query?",
            "criteria": [
                "irrelevant",
                "weakly relevant",
                "partially relevant",
                "strongly relevant",
                "highly relevant",
            ],
        }
    }


class ContextSufficiency(Metric):
    """RAG-003: Context Sufficiency.
    Binary typed decision: Is the supplied context sufficient to answer the query?
    """

    metric_id = "RAG-003"
    metric_name = "context_sufficient"
    question_type = "noul"
    state_fields = ("query", "retrieved_context")
    _question_key = "context_sufficient"
    _question_template = {
        "context_sufficient": {
            "type": "noul",
            "instructions": "Is the supplied context sufficient to answer the user's query without needing important missing information?",
        }
    }


class ContextCoverage(Metric):
    """RAG-004: Context Coverage.
    How completely does the retrieved context cover the information needed?
    """

    metric_id = "RAG-004"
    metric_name = "context_coverage"
    question_type = "score"
    state_fields = ("query", "retrieved_context")
    _question_key = "context_coverage"
    _question_template = {
        "context_coverage": {
            "type": "score",
            "instructions": "How completely does the retrieved context cover the information needed to answer the user's query?",
            "criteria": [
                "none",
                "limited",
                "partial",
                "mostly complete",
                "complete",
            ],
        }
    }


class RetrievalCompleteness(Metric):
    """RAG-005: Retrieval Completeness.
    How completely does the retrieved evidence contain important information needed?
    """

    metric_id = "RAG-005"
    metric_name = "retrieval_completeness"
    question_type = "score"
    state_fields = ("query", "retrieved_context")
    optional_fields = ("optional_reference",)
    _question_key = "retrieval_completeness"
    _question_template = {
        "retrieval_completeness": {
            "type": "score",
            "instructions": "How completely does the retrieved evidence contain the important information needed to answer the query?",
            "criteria": [
                "missing most information",
                "missing important information",
                "partially complete",
                "mostly complete",
                "complete",
            ],
        }
    }


class Groundedness(Metric):
    """RAG-006: Groundedness.
    How well is the response supported by the supplied context?
    """

    metric_id = "RAG-006"
    metric_name = "groundedness"
    question_type = "score"
    state_fields = ("retrieved_context", "response")
    _question_key = "groundedness"
    _question_template = {
        "groundedness": {
            "type": "score",
            "instructions": "How well is the response supported by the supplied context?",
            "criteria": [
                "unsupported",
                "mostly unsupported",
                "partially supported",
                "mostly supported",
                "fully supported",
            ],
        }
    }


class StrictGroundedness(Metric):
    """RAG-007: Strict Groundedness.
    Binary decision: Is every material claim in the response supported by context?
    """

    metric_id = "RAG-007"
    metric_name = "strictly_grounded"
    question_type = "noul"
    state_fields = ("retrieved_context", "response")
    _question_key = "strictly_grounded"
    _question_template = {
        "strictly_grounded": {
            "type": "noul",
            "instructions": "Is every material claim in the response supported by the supplied context?",
        }
    }


class HallucinationRisk(Metric):
    """RAG-008: Hallucination Risk.
    Binary decision: Does response contain material info unsupported by context?
    Pass condition: P(true) <= threshold.
    """

    metric_id = "RAG-008"
    metric_name = "hallucination"
    question_type = "noul"
    state_fields = ("retrieved_context", "response")
    _question_key = "hallucination"
    _question_template = {
        "hallucination": {
            "type": "noul",
            "instructions": "Does the response contain any material information that is unsupported by the supplied context?",
        }
    }

    def __init__(self, threshold: float = 0.3):
        super().__init__(threshold=threshold)


class ResponseRelevance(Metric):
    """RAG-009: Response Relevance.
    How directly and appropriately does the response answer the user's query?
    """

    metric_id = "RAG-009"
    metric_name = "response_relevance"
    question_type = "score"
    state_fields = ("query", "response")
    _question_key = "response_relevance"
    _question_template = {
        "response_relevance": {
            "type": "score",
            "instructions": "How directly and appropriately does the response answer the user's query?",
            "criteria": [
                "does not answer",
                "weakly answers",
                "partially answers",
                "mostly answers",
                "directly answers",
            ],
        }
    }


class ResponseCompleteness(Metric):
    """RAG-010: Response Completeness.
    How completely does the response cover the important parts of the query?
    """

    metric_id = "RAG-010"
    metric_name = "response_completeness"
    question_type = "score"
    state_fields = ("query", "response")
    optional_fields = ("optional_context_or_reference",)
    _question_key = "response_completeness"
    _question_template = {
        "response_completeness": {
            "type": "score",
            "instructions": "How completely does the response cover the important parts of the user's query?",
            "criteria": [
                "misses nearly all",
                "misses major parts",
                "partially complete",
                "mostly complete",
                "fully complete",
            ],
        }
    }


class FactualCorrectness(Metric):
    """RAG-011: Factual Correctness.
    How correct is the response compared with the provided ground-truth answer?
    """

    metric_id = "RAG-011"
    metric_name = "factual_correctness"
    question_type = "score"
    state_fields = ("query", "response", "ground_truth")
    _question_key = "factual_correctness"
    _question_template = {
        "factual_correctness": {
            "type": "score",
            "instructions": "How correct is the response compared with the provided ground-truth answer?",
            "criteria": [
                "incorrect",
                "mostly incorrect",
                "partially correct",
                "mostly correct",
                "fully correct",
            ],
        }
    }


class Answerability(Metric):
    """RAG-012: Answerability.
    Can the user's query be answered from the supplied context?
    """

    metric_id = "RAG-012"
    metric_name = "answerable"
    question_type = "noul"
    state_fields = ("query", "retrieved_context")
    _question_key = "answerable"
    _question_template = {
        "answerable": {
            "type": "noul",
            "instructions": "Can the user's query be answered from the supplied context?",
        }
    }


class AbstentionCorrectness(Metric):
    """RAG-013: Abstention Correctness.
    Is the response's decision to answer or abstain appropriate?
    """

    metric_id = "RAG-013"
    metric_name = "abstention_correct"
    question_type = "noul"
    state_fields = ("query", "retrieved_context", "response")
    _question_key = "abstention_correct"
    _question_template = {
        "abstention_correct": {
            "type": "noul",
            "instructions": "Given the query and context, is the response's decision to answer or abstain appropriate?",
        }
    }


class ContextUtilization(Metric):
    """RAG-014: Context Utilization.
    How appropriately does the response use useful info present in context?
    """

    metric_id = "RAG-014"
    metric_name = "context_utilization"
    question_type = "score"
    state_fields = ("retrieved_context", "response")
    _question_key = "context_utilization"
    _question_template = {
        "context_utilization": {
            "type": "score",
            "instructions": "How appropriately does the response use the useful information present in the supplied context?",
            "criteria": [
                "does not use useful context",
                "uses little useful context",
                "uses some useful context",
                "uses most useful context",
                "uses the relevant context appropriately",
            ],
        }
    }


class CitationCorrectness(Metric):
    """RAG-015: Citation Correctness.
    How correctly do the supplied citations support claims they are attached to?
    """

    metric_id = "RAG-015"
    metric_name = "citation_correctness"
    question_type = "score"
    state_fields = ("query", "response", "citations", "cited_source_text")
    _question_key = "citation_correctness"
    _question_template = {
        "citation_correctness": {
            "type": "score",
            "instructions": "How correctly do the supplied citations support the claims they are attached to?",
            "criteria": [
                "do not support",
                "weakly support",
                "partially support",
                "mostly support",
                "fully support",
            ],
        }
    }


class CitationCompleteness(Metric):
    """RAG-016: Citation Completeness.
    How completely are the important claims in the response supported by citations?
    """

    metric_id = "RAG-016"
    metric_name = "citation_completeness"
    question_type = "score"
    state_fields = ("response", "citations")
    _question_key = "citation_completeness"
    _question_template = {
        "citation_completeness": {
            "type": "score",
            "instructions": "How completely are the important claims in the response supported by citations?",
            "criteria": [
                "no important claims cited",
                "few important claims cited",
                "some important claims cited",
                "most important claims cited",
                "all important claims appropriately cited",
            ],
        }
    }


class CitationRelevance(Metric):
    """RAG-017: Citation Relevance.
    How relevant is the cited source to the claim it is intended to support?
    """

    metric_id = "RAG-017"
    metric_name = "citation_relevance"
    question_type = "score"
    state_fields = ("claim", "citation", "source_text")
    _question_key = "citation_relevance"
    _question_template = {
        "citation_relevance": {
            "type": "score",
            "instructions": "How relevant is the cited source to the claim it is intended to support?",
            "criteria": [
                "irrelevant",
                "weakly relevant",
                "partially relevant",
                "highly relevant",
                "directly relevant",
            ],
        }
    }


class ClaimSupport(Metric):
    """RAG-018: Claim Support.
    Does the provided evidence directly support this claim?
    """

    metric_id = "RAG-018"
    metric_name = "claim_supported"
    question_type = "noul"
    state_fields = ("claim", "evidence")
    _question_key = "claim_supported"
    _question_template = {
        "claim_supported": {
            "type": "noul",
            "instructions": "Does the provided evidence directly support this claim?",
        }
    }


class ContradictionDetection(Metric):
    """RAG-019: Contradiction Detection.
    Does the response materially contradict the supplied context?
    Pass condition: P(true) <= threshold.
    """

    metric_id = "RAG-019"
    metric_name = "contradiction"
    question_type = "noul"
    state_fields = ("context", "response")
    _question_key = "contradiction"
    _question_template = {
        "contradiction": {
            "type": "noul",
            "instructions": "Does the response materially contradict the supplied context?",
        }
    }

    def __init__(self, threshold: float = 0.2):
        super().__init__(threshold=threshold)


class ContextConflictDetection(Metric):
    """RAG-020: Context Conflict Detection.
    Do multiple retrieved documents contain material conflict relevant to query?
    Pass condition: P(true) <= threshold.
    """

    metric_id = "RAG-020"
    metric_name = "context_conflict"
    question_type = "noul"
    state_fields = ("query", "retrieved_context")
    _question_key = "context_conflict"
    _question_template = {
        "context_conflict": {
            "type": "noul",
            "instructions": "Do the supplied retrieved documents contain a material conflict relevant to the user's query?",
        }
    }

    def __init__(self, threshold: float = 0.3):
        super().__init__(threshold=threshold)
