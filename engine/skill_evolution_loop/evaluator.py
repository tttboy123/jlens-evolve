"""Three-layer evaluation gates for structural, native, and holdout validity."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import ContractError
from .student_adapter import StudentAttempt


@dataclass(frozen=True)
class NativeOutcome:
    resolved: bool
    safe: bool
    detail: str = ""
    infrastructure_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "safe": self.safe,
            "detail": self.detail,
            "infrastructure_error": self.infrastructure_error,
        }


@dataclass(frozen=True)
class EvaluationBaseline:
    feedback_native_rate: float
    holdout_native_rate: float

    def __post_init__(self) -> None:
        for value in (self.feedback_native_rate, self.holdout_native_rate):
            if not 0.0 <= value <= 1.0:
                raise ContractError("baseline rates must be between zero and one")

    def to_dict(self) -> dict[str, float]:
        return {
            "feedback_native_rate": self.feedback_native_rate,
            "holdout_native_rate": self.holdout_native_rate,
        }


@dataclass(frozen=True)
class EvaluationPolicy:
    minimum_structural_rate: float
    minimum_feedback_native_rate: float
    minimum_feedback_native_gain: float
    minimum_holdout_native_rate: float
    holdout_regression_tolerance: float

    def __post_init__(self) -> None:
        for value in (
            self.minimum_structural_rate,
            self.minimum_feedback_native_rate,
            self.minimum_holdout_native_rate,
            self.holdout_regression_tolerance,
        ):
            if not 0.0 <= value <= 1.0:
                raise ContractError(
                    "evaluation policy rates must be between zero and one"
                )
        if not -1.0 <= self.minimum_feedback_native_gain <= 1.0:
            raise ContractError(
                "feedback native gain must be between minus one and one"
            )

    @classmethod
    def strict(cls) -> EvaluationPolicy:
        return cls(
            minimum_structural_rate=1.0,
            minimum_feedback_native_rate=1.0,
            minimum_feedback_native_gain=0.01,
            minimum_holdout_native_rate=1.0,
            holdout_regression_tolerance=0.0,
        )


@dataclass(frozen=True)
class RoundEvaluation:
    structural_rate: float
    feedback_native_rate: float
    holdout_native_rate: float
    feedback_native_gain: float
    holdout_native_gain: float
    safety_regression: bool
    infrastructure_errors: tuple[str, ...]
    native_outcomes: dict[str, NativeOutcome]
    converged: bool

    @property
    def score(self) -> tuple[float, float, float]:
        return (
            self.holdout_native_rate,
            self.feedback_native_rate,
            self.structural_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "structural_rate": self.structural_rate,
            "feedback_native_rate": self.feedback_native_rate,
            "holdout_native_rate": self.holdout_native_rate,
            "feedback_native_gain": self.feedback_native_gain,
            "holdout_native_gain": self.holdout_native_gain,
            "safety_regression": self.safety_regression,
            "infrastructure_errors": list(self.infrastructure_errors),
            "native_outcomes": {
                key: value.to_dict() for key, value in self.native_outcomes.items()
            },
            "converged": self.converged,
        }


NativeEvaluator = Callable[[StudentAttempt], NativeOutcome]


class LoopEvaluator:
    """Require format validity, native gain, safety, and holdout non-regression."""

    def __init__(self, policy: EvaluationPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        attempts: Sequence[StudentAttempt],
        *,
        native_evaluator: NativeEvaluator,
        baseline: EvaluationBaseline,
    ) -> RoundEvaluation:
        if not attempts:
            raise ContractError("round evaluation requires attempts")
        task_ids = [row.task.task_id for row in attempts]
        if len(set(task_ids)) != len(task_ids):
            raise ContractError("round evaluation task ids must be unique")
        feedback = [row for row in attempts if row.task.cohort == "feedback"]
        holdout = [row for row in attempts if row.task.cohort == "holdout"]
        if not feedback or not holdout:
            raise ContractError("round evaluation requires feedback and holdout tasks")
        structural_rate = sum(row.structural_valid for row in attempts) / len(attempts)
        outcomes: dict[str, NativeOutcome] = {}
        errors: list[str] = []
        for attempt in attempts:
            if not attempt.structural_valid:
                outcomes[attempt.task.task_id] = NativeOutcome(
                    resolved=False,
                    safe=True,
                    detail=f"structural rejection: {attempt.failure_reason}",
                )
                continue
            try:
                outcome = native_evaluator(attempt)
                if not isinstance(outcome, NativeOutcome):
                    raise TypeError("native evaluator returned an invalid outcome")
            except Exception as exc:
                outcome = NativeOutcome(
                    resolved=False,
                    safe=False,
                    detail=str(exc),
                    infrastructure_error=True,
                )
            outcomes[attempt.task.task_id] = outcome
            if outcome.infrastructure_error:
                errors.append(attempt.task.task_id)

        feedback_rate = self._resolved_rate(feedback, outcomes)
        holdout_rate = self._resolved_rate(holdout, outcomes)
        feedback_gain = feedback_rate - baseline.feedback_native_rate
        holdout_gain = holdout_rate - baseline.holdout_native_rate
        safety_regression = any(not row.safe for row in outcomes.values())
        converged = all(
            (
                structural_rate >= self.policy.minimum_structural_rate,
                feedback_rate >= self.policy.minimum_feedback_native_rate,
                feedback_gain >= self.policy.minimum_feedback_native_gain,
                holdout_rate >= self.policy.minimum_holdout_native_rate,
                holdout_gain >= -self.policy.holdout_regression_tolerance,
                not safety_regression,
                not errors,
            )
        )
        return RoundEvaluation(
            structural_rate=structural_rate,
            feedback_native_rate=feedback_rate,
            holdout_native_rate=holdout_rate,
            feedback_native_gain=feedback_gain,
            holdout_native_gain=holdout_gain,
            safety_regression=safety_regression,
            infrastructure_errors=tuple(errors),
            native_outcomes=outcomes,
            converged=converged,
        )

    @staticmethod
    def _resolved_rate(
        attempts: Sequence[StudentAttempt], outcomes: dict[str, NativeOutcome]
    ) -> float:
        return sum(
            outcomes[row.task.task_id].resolved
            and outcomes[row.task.task_id].safe
            and not outcomes[row.task.task_id].infrastructure_error
            for row in attempts
        ) / len(attempts)
