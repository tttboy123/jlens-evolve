"""Bounded, auditable driver for the project-local Skill feedback loop."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ContractError,
    FailureEvidence,
    FeedbackPackage,
    LoopAuthorization,
    LoopRevision,
    ParentModelRequest,
    ParentModelResponse,
    canonical_json,
)
from .evaluator import (
    EvaluationBaseline,
    LoopEvaluator,
    NativeEvaluator,
    RoundEvaluation,
)
from .parent_model import ParentModelAdapter
from .registry import LoopRevisionRegistry
from .student_adapter import StudentAdapter, StudentAttempt, StudentTask


@dataclass(frozen=True)
class LoopConfig:
    """Hard stopping limits for one loop run."""

    max_rounds: int
    no_progress_patience: int

    def __post_init__(self) -> None:
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise ContractError("max_rounds must be a positive integer")
        if type(self.no_progress_patience) is not int or self.no_progress_patience < 1:
            raise ContractError("no_progress_patience must be a positive integer")


@dataclass(frozen=True)
class LoopResult:
    """Terminal loop state, including the rollback-safe best revision."""

    status: str
    rounds_completed: int
    final_revision: LoopRevision
    best_revision: LoopRevision
    evaluations: tuple[RoundEvaluation, ...]

    def __post_init__(self) -> None:
        if self.status not in {"converged", "no-progress", "exhausted"}:
            raise ContractError("invalid loop result status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rounds_completed": self.rounds_completed,
            "final_revision": self.final_revision.to_dict(),
            "best_revision": self.best_revision.to_dict(),
            "evaluations": [row.to_dict() for row in self.evaluations],
        }


class LoopDriver:
    """Evaluate, feed back failures, regenerate, and stop fail-closed."""

    def __init__(
        self,
        *,
        student: StudentAdapter,
        evaluator: LoopEvaluator,
        parent: ParentModelAdapter,
        registry: LoopRevisionRegistry,
        evidence_root: Path,
        authorization: LoopAuthorization,
        config: LoopConfig,
    ) -> None:
        self.student = student
        self.evaluator = evaluator
        self.parent = parent
        self.registry = registry
        self.evidence_root = evidence_root.resolve()
        self.authorization = authorization
        self.config = config

    def run(
        self,
        *,
        initial_revision: LoopRevision,
        tasks: Sequence[StudentTask],
        native_evaluator: NativeEvaluator,
        baseline: EvaluationBaseline,
    ) -> LoopResult:
        initial_revision.validate()
        self.authorization.validate()
        self.authorization.assert_active()
        if not tasks:
            raise ContractError("loop requires student tasks")
        task_ids = [task.task_id for task in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ContractError("loop task ids must be unique")
        for task in tasks:
            task.validate()

        latest = self.registry.latest(initial_revision.skill_id)
        if latest is None:
            self.registry.append(initial_revision)
        elif latest.fingerprint != initial_revision.fingerprint:
            raise ContractError("initial revision is not the registry head")

        current = initial_revision
        best = initial_revision
        best_score: tuple[float, float, float] | None = None
        no_progress_count = 0
        rejected_fingerprints: list[str] = []
        evaluations: list[RoundEvaluation] = []

        for round_index in range(self.config.max_rounds):
            attempts = [self.student.run(task, current) for task in tasks]
            evaluation = self.evaluator.evaluate(
                attempts,
                native_evaluator=native_evaluator,
                baseline=baseline,
            )
            evaluations.append(evaluation)
            score_improved = best_score is None or evaluation.score > best_score
            if score_improved:
                best, best_score = current, evaluation.score

            if evaluation.converged:
                self._freeze_round(
                    round_index=round_index,
                    revision=current,
                    attempts=attempts,
                    evaluation=evaluation,
                    baseline=baseline,
                    feedback=None,
                    parent_request=None,
                    parent_response=None,
                    next_revision=None,
                    terminal_status="converged",
                )
                return LoopResult(
                    status="converged",
                    rounds_completed=round_index + 1,
                    final_revision=current,
                    best_revision=current,
                    evaluations=tuple(evaluations),
                )

            rejected_fingerprints.append(current.fingerprint)
            if round_index + 1 >= self.config.max_rounds:
                self._freeze_round(
                    round_index=round_index,
                    revision=current,
                    attempts=attempts,
                    evaluation=evaluation,
                    baseline=baseline,
                    feedback=None,
                    parent_request=None,
                    parent_response=None,
                    next_revision=None,
                    terminal_status="exhausted",
                )
                return LoopResult(
                    status="exhausted",
                    rounds_completed=round_index + 1,
                    final_revision=best,
                    best_revision=best,
                    evaluations=tuple(evaluations),
                )

            feedback = self._feedback(
                round_index=round_index,
                revision=current,
                attempts=attempts,
                evaluation=evaluation,
                no_progress=no_progress_count > 0,
                rejected_fingerprints=rejected_fingerprints,
            )
            request = ParentModelRequest.create(
                feedback=feedback,
                current_revision=current,
            )
            response = self.parent.generate(
                call_id=f"{current.skill_id}-round-{round_index:03d}",
                request=request,
                authorization=self.authorization,
            )
            next_revision = self._next_revision(current, response, round_index + 1)
            equivalent = self._same_mechanism(current, next_revision)
            if (round_index > 0 and not score_improved) or equivalent:
                no_progress_count += 1
            else:
                no_progress_count = 0
            self.registry.append(next_revision)

            terminal_status = (
                "no-progress"
                if no_progress_count >= self.config.no_progress_patience
                else None
            )
            self._freeze_round(
                round_index=round_index,
                revision=current,
                attempts=attempts,
                evaluation=evaluation,
                baseline=baseline,
                feedback=feedback,
                parent_request=request,
                parent_response=response,
                next_revision=next_revision,
                terminal_status=terminal_status,
            )
            if terminal_status is not None:
                return LoopResult(
                    status=terminal_status,
                    rounds_completed=round_index + 1,
                    final_revision=best,
                    best_revision=best,
                    evaluations=tuple(evaluations),
                )
            current = next_revision

        raise AssertionError("bounded loop did not terminate")

    @staticmethod
    def _feedback(
        *,
        round_index: int,
        revision: LoopRevision,
        attempts: Sequence[StudentAttempt],
        evaluation: RoundEvaluation,
        no_progress: bool,
        rejected_fingerprints: list[str],
    ) -> FeedbackPackage:
        evidence: list[FailureEvidence] = []
        for attempt in attempts:
            if attempt.task.cohort != "feedback":
                continue
            outcome = evaluation.native_outcomes[attempt.task.task_id]
            reason = attempt.failure_reason
            if reason is None:
                if outcome.infrastructure_error:
                    reason = "eval-infra"
                elif not outcome.safe:
                    reason = "regression"
                else:
                    reason = "native-unresolved"
            detail = outcome.detail or attempt.detail or reason
            apply_error = attempt.detail if reason == "apply-fail" else None
            evidence.append(
                FailureEvidence.create(
                    task_id=attempt.task.task_id,
                    reason_code=reason,
                    diagnostic_summary=detail,
                    raw_output_sha256=attempt.raw_output_sha256,
                    extracted_edit_sha256=attempt.patch_sha256,
                    apply_error=apply_error,
                )
            )
        if not evidence:
            raise ContractError("failed round produced no feedback-arm evidence")
        return FeedbackPackage.create(
            current_round=round_index,
            arm_evidence=evidence,
            previous_eval_note=revision.eval_note,
            no_progress=no_progress,
            rejected_fingerprints=list(dict.fromkeys(rejected_fingerprints)),
        )

    @staticmethod
    def _next_revision(
        current: LoopRevision,
        response: ParentModelResponse,
        source_round: int,
    ) -> LoopRevision:
        revision_id = f"{current.skill_id}-r{source_round:03d}-{response.sha256[:8]}"
        return LoopRevision.create(
            skill_id=current.skill_id,
            revision_id=revision_id,
            parent_revision_id=current.revision_id,
            source_round=source_round,
            protocol=response.protocol,
            skill_text=response.skill_text,
            prompt_template=response.prompt_template,
            eval_note=response.eval_note,
        )

    @staticmethod
    def _same_mechanism(left: LoopRevision, right: LoopRevision) -> bool:
        return (
            left.protocol,
            left.skill_text,
            left.prompt_template,
        ) == (
            right.protocol,
            right.skill_text,
            right.prompt_template,
        )

    def _freeze_round(
        self,
        *,
        round_index: int,
        revision: LoopRevision,
        attempts: Sequence[StudentAttempt],
        evaluation: RoundEvaluation,
        baseline: EvaluationBaseline,
        feedback: FeedbackPackage | None,
        parent_request: ParentModelRequest | None,
        parent_response: ParentModelResponse | None,
        next_revision: LoopRevision | None,
        terminal_status: str | None,
    ) -> None:
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        target = self.evidence_root / f"round-{round_index:03d}"
        if target.exists():
            raise ContractError(f"round evidence already exists: {target.name}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}-", dir=self.evidence_root)
        )
        try:
            artifact_hashes: dict[str, str] = {}
            attempt_rows: list[dict[str, Any]] = []
            for attempt in attempts:
                raw_name = f"{attempt.task.task_id}.raw.txt"
                patch_name = f"{attempt.task.task_id}.patch"
                raw_path = temporary / raw_name
                patch_path = temporary / patch_name
                raw_path.write_text(attempt.raw_output, encoding="utf-8")
                patch_path.write_text(attempt.patch, encoding="utf-8")
                artifact_hashes[raw_name] = self._file_sha256(raw_path)
                artifact_hashes[patch_name] = self._file_sha256(patch_path)
                attempt_rows.append(attempt.to_dict())
            content = {
                "schema_version": 1,
                "round_index": round_index,
                "terminal_status": terminal_status,
                "revision": revision.to_dict(),
                "baseline": baseline.to_dict(),
                "evaluation": evaluation.to_dict(),
                "attempts": attempt_rows,
                "feedback": feedback.to_dict() if feedback else None,
                "parent_request": parent_request.to_dict() if parent_request else None,
                "parent_response": (
                    parent_response.to_dict() if parent_response else None
                ),
                "next_revision": next_revision.to_dict() if next_revision else None,
                "artifact_sha256": artifact_hashes,
            }
            report = {
                **content,
                "evidence_sha256": hashlib.sha256(
                    canonical_json(content).encode("utf-8")
                ).hexdigest(),
            }
            (temporary / "ROUND.json").write_text(
                canonical_json(report) + "\n", encoding="utf-8"
            )
            temporary.replace(target)
        except Exception:
            for path in temporary.iterdir():
                path.unlink(missing_ok=True)
            temporary.rmdir()
            raise

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
