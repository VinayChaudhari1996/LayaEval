"""Base class and data containers for LayaEval metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """The result of evaluating a single metric on an input state."""

    metric_id: str
    metric_name: str
    score: float | None = None
    noul: float | None = None
    choice: str | None = None
    confidence: float | None = None
    answer_confidence: float | None = None
    passed: bool = False
    threshold: float = 0.0
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "score": self.score,
            "noul": self.noul,
            "choice": self.choice,
            "confidence": self.confidence,
            "answer_confidence": self.answer_confidence,
            "passed": self.passed,
            "threshold": self.threshold,
            "reason": self.reason,
            "raw": self.raw,
        }


class Metric:
    """Abstract base class for all LayaEval evaluation metrics."""

    metric_id: str = "GEN-000"
    metric_name: str = "generic_metric"
    question_type: str = "score"  # "score", "noul", "choice"
    state_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    _question_key: str = ""
    _question_template: dict[str, Any] = {}

    def __init__(self, threshold: float | None = None):
        """Initialize metric with optional default threshold."""
        if threshold is not None:
            self.default_threshold = float(threshold)
        elif self.question_type == "score":
            self.default_threshold = 3.0  # default 3 on 0..4 rubric
        elif self.question_type == "noul":
            self.default_threshold = 0.5  # default P(true) >= 0.5
        else:
            self.default_threshold = 0.0

    def __call__(
        self,
        evaluator: Any | None = None,
        threshold: float | None = None,
        **state_kwargs: Any,
    ) -> EvalResult:
        """Evaluate this metric directly as a callable.

        Example:
            groundedness = Groundedness(threshold=3.0)
            res = groundedness(retrieved_context="...", response="...")
        """
        from ..evaluator import default_evaluator

        ev = evaluator or default_evaluator()
        return ev.evaluate(self, threshold=threshold, **state_kwargs)

    def evaluate(
        self,
        evaluator: Any | None = None,
        threshold: float | None = None,
        **state_kwargs: Any,
    ) -> EvalResult:
        """Evaluate this metric against input state."""
        return self(evaluator=evaluator, threshold=threshold, **state_kwargs)

    @property
    def question_key(self) -> str:
        """The key used in the questions dict."""
        return self._question_key or self.metric_name

    def build_state(self, **kwargs) -> dict[str, Any]:
        """Validate required state fields and return the state dict for LAYA."""
        missing = [f for f in self.state_fields if f not in kwargs or kwargs[f] is None]
        if missing:
            raise ValueError(
                f"Metric {self.metric_name} ({self.metric_id}) requires state fields: "
                f"{', '.join(missing)}"
            )
        state: dict[str, Any] = {}
        for key in self.state_fields:
            state[key] = kwargs[key]
        for key in self.optional_fields:
            if key in kwargs and kwargs[key] is not None:
                state[key] = kwargs[key]
        return state

    def build_questions(self) -> dict[str, Any]:
        """Generate the typed question dictionary to send to LAYA."""
        return dict(self._question_template)

    def parse_result(
        self,
        answer: dict[str, Any],
        threshold: float | None = None,
        state: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Parse raw LAYA answer output into a structured EvalResult."""
        t = self.default_threshold if threshold is None else float(threshold)
        raw_ans = answer.get(self.question_key, answer)

        score_val: float | None = None
        noul_val: float | None = None
        choice_val: str | None = None
        passed = False
        reason = None

        if self.question_type == "score":
            if "score" in raw_ans and raw_ans["score"] is not None:
                score_val = float(raw_ans["score"])
                passed = score_val >= t
                reason = f"score {score_val} {'≥' if passed else '<'} threshold {t}"
        elif self.question_type == "noul":
            if "noul" in raw_ans and raw_ans["noul"] is not None:
                noul_val = float(raw_ans["noul"])
                # For risk/hallucination/contradiction metrics, pass condition may be <= max
                if (
                    "risk" in self.metric_name
                    or "contradiction" in self.metric_name
                    or "conflict" in self.metric_name
                    or "injection" in self.metric_name
                    or "hallucination" in self.metric_name
                    or "redundant" in self.metric_name
                ):
                    passed = noul_val <= t
                    reason = f"risk {noul_val} {'≤' if passed else '>'} threshold {t}"
                else:
                    passed = noul_val >= t
                    reason = f"noul P(true) {noul_val} {'≥' if passed else '<'} threshold {t}"
        elif self.question_type == "choice":
            choice_val = raw_ans.get("choice")
            passed = bool(choice_val is not None)
            reason = f"selected choice: {choice_val}"

        conf = raw_ans.get("confidence")
        ans_conf = raw_ans.get("answer_confidence")

        return EvalResult(
            metric_id=self.metric_id,
            metric_name=self.metric_name,
            score=score_val,
            noul=noul_val,
            choice=choice_val,
            confidence=float(conf) if conf is not None else None,
            answer_confidence=float(ans_conf) if ans_conf is not None else None,
            passed=passed,
            threshold=t,
            reason=reason,
            raw=raw_ans,
            state=state or {},
        )
