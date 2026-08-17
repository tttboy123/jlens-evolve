"""Generic diagnosis-frozen, bounded multi-realization Student adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import ContractError, LoopRevision, canonical_json, sha256_json
from .realization_candidates import (
    FrozenDiagnosis,
    RealizationCandidate,
    select_realization_candidate,
)
from .student_adapter import StudentAdapter, StudentAttempt, StudentTask, _sha256_text

DiagnosisProvider = Callable[[StudentTask, LoopRevision], FrozenDiagnosis]
SeedCandidateProvider = Callable[
    [StudentTask, LoopRevision], tuple[FrozenDiagnosis, StudentAttempt]
]
CandidateRunner = Callable[
    [StudentTask, LoopRevision, FrozenDiagnosis, int], StudentAttempt
]


class DiagnosisFrozenRealizationAdapter:
    """Generate bounded implementations under one diagnosis and select structurally.

    The selection policy never reads tests or native labels. It is intended as a
    mechanism adapter; official native evaluation remains the sole capability gate.
    """

    def __init__(
        self,
        *,
        diagnosis_provider: DiagnosisProvider | None = None,
        seed_candidate_provider: SeedCandidateProvider | None = None,
        candidate_runner: CandidateRunner,
        maximum_candidates: int,
        candidate_generation_config: dict[str, Any] | None = None,
    ) -> None:
        if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 8:
            raise ContractError("realization candidate budget must be between 1 and 8")
        if (diagnosis_provider is None) == (seed_candidate_provider is None):
            raise ContractError(
                "exactly one diagnosis or seed candidate provider is required"
            )
        self.diagnosis_provider = diagnosis_provider
        self.seed_candidate_provider = seed_candidate_provider
        self.candidate_runner = candidate_runner
        self.maximum_candidates = maximum_candidates
        self.candidate_generation_config = dict(candidate_generation_config or {})
        # PairedExperimentRunner reads optional generation traces from
        # ``adapter.generator``. This adapter owns the aggregate candidate trace.
        self.generator = self
        self._last_evidence: dict[str, Any] | None = None
        self._last_trace: tuple[str, ...] = ()
        self._last_trace_results: tuple[dict[str, Any], ...] = ()

    def experiment_config(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "adapter_contract": "diagnosis-frozen-multi-realization-v1",
            "maximum_candidates": self.maximum_candidates,
            "selection_policy": "minimal-changed-lines-then-id-v1",
            "native_labels_visible_to_selection": False,
            "candidate_generation": self.candidate_generation_config,
        }

    def run(self, task: StudentTask, revision: LoopRevision) -> StudentAttempt:
        task.validate()
        revision.validate()
        self._last_evidence = None
        self._last_trace = ()
        self._last_trace_results = ()
        attempts: list[StudentAttempt] = []
        if self.seed_candidate_provider is not None:
            diagnosis, seed_attempt = self.seed_candidate_provider(task, revision)
            self._validate_attempt(seed_attempt, task, revision)
            attempts.append(seed_attempt)
        else:
            assert self.diagnosis_provider is not None
            diagnosis = self.diagnosis_provider(task, revision)
        if not isinstance(diagnosis, FrozenDiagnosis):
            raise ContractError("diagnosis provider returned an invalid contract")
        diagnosis.validate()

        candidates: list[RealizationCandidate] = []
        trace_results: list[dict[str, Any]] = []
        for index in range(len(attempts), self.maximum_candidates):
            try:
                attempt = self.candidate_runner(task, revision, diagnosis, index)
            except Exception as exc:
                attempt = StudentAdapter._failure(
                    task,
                    revision,
                    "",
                    "eval-infra",
                    f"candidate runner failed: {exc}",
                )
            self._validate_attempt(attempt, task, revision)
            attempts.append(attempt)

        for index, attempt in enumerate(attempts):
            candidate_id = f"candidate-{index + 1:03d}"
            candidate = RealizationCandidate.create(
                candidate_id=candidate_id,
                diagnosis_sha256=diagnosis.fingerprint,
                raw_output_sha256=attempt.raw_output_sha256,
                patch=attempt.patch,
                structural_valid=attempt.structural_valid,
                failure_reason=attempt.failure_reason,
            )
            candidates.append(candidate)
            trace_results.append(
                {
                    "status": (
                        "structural-valid"
                        if attempt.structural_valid
                        else "structural-rejected"
                    ),
                    "candidate_id": candidate_id,
                    **(
                        {}
                        if attempt.structural_valid
                        else {
                            "failure_reason": attempt.failure_reason,
                            "detail": attempt.detail,
                        }
                    ),
                }
            )

        selection = select_realization_candidate(
            diagnosis=diagnosis,
            candidates=candidates,
            maximum_candidates=self.maximum_candidates,
        )
        candidate_rows = [
            {
                "candidate_id": candidate.candidate_id,
                "diagnosis_sha256": candidate.diagnosis_sha256,
                "raw_output_sha256": candidate.raw_output_sha256,
                "patch_sha256": candidate.patch_sha256,
                "changed_lines": candidate.changed_lines,
                "structural_valid": candidate.structural_valid,
                "failure_reason": candidate.failure_reason,
                "detail": attempts[index].detail,
            }
            for candidate in candidates
        ]
        evidence_content = {
            "schema_version": 1,
            "contract": "diagnosis-frozen-multi-realization-v1",
            "diagnosis": diagnosis.to_dict(),
            "maximum_candidates": self.maximum_candidates,
            "candidates": candidate_rows,
            "selection": selection.to_dict(),
            "native_labels_visible_to_selection": False,
            "network_calls_performed": False,
        }
        self._last_evidence = {
            **evidence_content,
            "evidence_sha256": sha256_json(evidence_content),
        }
        self._last_trace = tuple(attempt.raw_output for attempt in attempts)
        self._last_trace_results = tuple(trace_results)

        selected_id = selection.selected_candidate_id
        if selected_id is not None:
            selected_index = int(selected_id.rsplit("-", 1)[1]) - 1
            return attempts[selected_index]
        raw = canonical_json(
            {
                "schema_version": 1,
                "status": "unresolved",
                "diagnosis_sha256": diagnosis.fingerprint,
                "candidate_decisions": list(selection.candidate_decisions),
            }
        )
        return StudentAdapter._failure(
            task,
            revision,
            raw,
            "unresolved",
            "no structurally eligible realization candidate",
        )

    @staticmethod
    def _validate_attempt(
        attempt: StudentAttempt, task: StudentTask, revision: LoopRevision
    ) -> None:
        if not isinstance(attempt, StudentAttempt):
            raise ContractError("candidate runner returned an invalid attempt")
        if (
            attempt.task != task
            or attempt.revision_id != revision.revision_id
            or attempt.raw_output_sha256 != _sha256_text(attempt.raw_output)
        ):
            raise ContractError("candidate attempt boundary is invalid")

    def realization_evidence(self) -> dict[str, Any] | None:
        return None if self._last_evidence is None else dict(self._last_evidence)

    def generation_trace(self) -> tuple[str, ...]:
        return self._last_trace

    def generation_trace_kinds(self) -> tuple[str, ...]:
        return tuple(
            f"realization-candidate-{index + 1:03d}"
            for index in range(len(self._last_trace))
        )

    def generation_prompt_trace(self) -> tuple[None, ...]:
        return tuple(None for _output in self._last_trace)

    def generation_trace_results(self) -> tuple[dict[str, Any], ...]:
        return self._last_trace_results
