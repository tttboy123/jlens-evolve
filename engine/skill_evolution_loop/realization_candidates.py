"""Diagnosis-frozen, bounded selection for patch realization candidates."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .contracts import FAILURE_REASON_CODES, ContractError, sha256_json

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SELECTION_POLICY = "minimal-changed-lines-then-id-v1"


def _require_nonempty(label: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be non-empty")


@dataclass(frozen=True)
class FrozenDiagnosis:
    """One immutable diagnosis shared by all realization candidates in an arm."""

    schema_version: int
    defect: str
    trigger: str
    desired_boundary: str

    @classmethod
    def create(
        cls, *, defect: str, trigger: str, desired_boundary: str
    ) -> FrozenDiagnosis:
        diagnosis = cls(
            schema_version=1,
            defect=defect,
            trigger=trigger,
            desired_boundary=desired_boundary,
        )
        diagnosis.validate()
        return diagnosis

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported frozen diagnosis schema")
        for label, value in (
            ("diagnosis defect", self.defect),
            ("diagnosis trigger", self.trigger),
            ("diagnosis desired_boundary", self.desired_boundary),
        ):
            _require_nonempty(label, value)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "defect": self.defect,
            "trigger": self.trigger,
            "desired_boundary": self.desired_boundary,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class RealizationCandidate:
    """One materialized candidate bound to a frozen diagnosis."""

    schema_version: int
    candidate_id: str
    diagnosis_sha256: str
    raw_output_sha256: str
    patch: str
    structural_valid: bool
    failure_reason: str | None

    @classmethod
    def create(cls, **fields: Any) -> RealizationCandidate:
        candidate = cls(schema_version=1, **fields)
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported realization candidate schema")
        if _IDENTIFIER.fullmatch(self.candidate_id) is None:
            raise ContractError("invalid realization candidate_id")
        for label, value in (
            ("diagnosis_sha256", self.diagnosis_sha256),
            ("raw_output_sha256", self.raw_output_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ContractError(f"invalid realization {label}")
        if type(self.structural_valid) is not bool:
            raise ContractError("realization structural_valid must be boolean")
        if self.structural_valid:
            if self.failure_reason is not None:
                raise ContractError("valid realization cannot have failure_reason")
            _require_nonempty("valid realization patch", self.patch)
        elif self.failure_reason not in FAILURE_REASON_CODES:
            raise ContractError("invalid realization failure_reason")

    @property
    def patch_sha256(self) -> str:
        return hashlib.sha256(self.patch.encode("utf-8")).hexdigest()

    @property
    def changed_lines(self) -> int:
        return sum(
            1
            for line in self.patch.splitlines()
            if (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )


@dataclass(frozen=True)
class RealizationSelection:
    """Auditable deterministic decision over one bounded candidate set."""

    schema_version: int
    diagnosis_sha256: str
    maximum_candidates: int
    selection_policy: str
    selected_candidate_id: str | None
    candidate_decisions: tuple[dict[str, Any], ...]

    _FIELDS = frozenset(
        {
            "schema_version",
            "diagnosis_sha256",
            "maximum_candidates",
            "selection_policy",
            "selected_candidate_id",
            "candidate_decisions",
            "evidence_sha256",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RealizationSelection:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid realization selection fields")
        decisions = data["candidate_decisions"]
        if not isinstance(decisions, list) or any(
            not isinstance(row, dict) for row in decisions
        ):
            raise ContractError("invalid realization candidate decisions")
        selection = cls(
            schema_version=data["schema_version"],
            diagnosis_sha256=str(data["diagnosis_sha256"]),
            maximum_candidates=data["maximum_candidates"],
            selection_policy=str(data["selection_policy"]),
            selected_candidate_id=(
                None
                if data["selected_candidate_id"] is None
                else str(data["selected_candidate_id"])
            ),
            candidate_decisions=tuple(dict(row) for row in decisions),
        )
        selection.validate()
        if data["evidence_sha256"] != selection.evidence_sha256:
            raise ContractError("realization selection evidence sha256 mismatch")
        return selection

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported realization selection schema")
        if _SHA256.fullmatch(self.diagnosis_sha256) is None:
            raise ContractError("invalid selection diagnosis_sha256")
        if type(self.maximum_candidates) is not int or not (
            1 <= self.maximum_candidates <= 8
        ):
            raise ContractError("realization candidate budget must be between 1 and 8")
        if self.selection_policy != _SELECTION_POLICY:
            raise ContractError("unsupported realization selection policy")
        if len(self.candidate_decisions) > self.maximum_candidates:
            raise ContractError("realization candidate budget exceeded")

        selected_ids: list[str] = []
        candidate_ids: list[str] = []
        for decision in self.candidate_decisions:
            candidate_id = decision.get("candidate_id")
            status = decision.get("status")
            if (
                not isinstance(candidate_id, str)
                or _IDENTIFIER.fullmatch(candidate_id) is None
                or not isinstance(status, str)
            ):
                raise ContractError("invalid realization candidate decision")
            candidate_ids.append(candidate_id)
            expected = {"candidate_id", "status"}
            if status == "structural-invalid":
                expected.add("failure_reason")
                if decision.get("failure_reason") not in FAILURE_REASON_CODES:
                    raise ContractError("invalid candidate decision failure_reason")
            elif status in {"eligible", "selected"}:
                expected.update({"changed_lines", "patch_sha256"})
                if (
                    type(decision.get("changed_lines")) is not int
                    or decision["changed_lines"] < 1
                    or not isinstance(decision.get("patch_sha256"), str)
                    or _SHA256.fullmatch(decision["patch_sha256"]) is None
                ):
                    raise ContractError("invalid eligible candidate decision")
                if status == "selected":
                    selected_ids.append(candidate_id)
            elif status not in {"duplicate-patch", "diagnosis-mismatch"}:
                raise ContractError("invalid realization candidate decision status")
            if set(decision) != expected:
                raise ContractError("invalid realization candidate decision fields")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ContractError("realization candidate ids must be unique")
        expected_selected = selected_ids[0] if len(selected_ids) == 1 else None
        if len(selected_ids) > 1 or self.selected_candidate_id != expected_selected:
            raise ContractError("realization selected candidate is inconsistent")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "diagnosis_sha256": self.diagnosis_sha256,
            "maximum_candidates": self.maximum_candidates,
            "selection_policy": self.selection_policy,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_decisions": list(self.candidate_decisions),
        }

    @property
    def evidence_sha256(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._content_dict(), "evidence_sha256": self.evidence_sha256}


def select_realization_candidate(
    *,
    diagnosis: FrozenDiagnosis,
    candidates: list[RealizationCandidate],
    maximum_candidates: int,
) -> RealizationSelection:
    """Select a minimal eligible patch without opening tests or native labels."""

    diagnosis.validate()
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 8:
        raise ContractError("realization candidate budget must be between 1 and 8")
    if len(candidates) > maximum_candidates:
        raise ContractError("realization candidate budget exceeded")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ContractError("realization candidate ids must be unique")

    decisions: list[dict[str, Any]] = []
    eligible: list[RealizationCandidate] = []
    patch_ids: set[str] = set()
    for candidate in candidates:
        candidate.validate()
        if not candidate.structural_valid:
            decisions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": "structural-invalid",
                    "failure_reason": candidate.failure_reason,
                }
            )
            continue
        if candidate.diagnosis_sha256 != diagnosis.fingerprint:
            decisions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": "diagnosis-mismatch",
                }
            )
            continue
        if candidate.patch_sha256 in patch_ids:
            decisions.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": "duplicate-patch",
                }
            )
            continue
        patch_ids.add(candidate.patch_sha256)
        eligible.append(candidate)
        decisions.append(
            {
                "candidate_id": candidate.candidate_id,
                "status": "eligible",
                "changed_lines": candidate.changed_lines,
                "patch_sha256": candidate.patch_sha256,
            }
        )

    selected = min(
        eligible,
        key=lambda candidate: (candidate.changed_lines, candidate.candidate_id),
        default=None,
    )
    if selected is not None:
        for decision in decisions:
            if decision["candidate_id"] == selected.candidate_id:
                decision["status"] = "selected"
                break
    selection = RealizationSelection(
        schema_version=1,
        diagnosis_sha256=diagnosis.fingerprint,
        maximum_candidates=maximum_candidates,
        selection_policy=_SELECTION_POLICY,
        selected_candidate_id=(None if selected is None else selected.candidate_id),
        candidate_decisions=tuple(decisions),
    )
    selection.validate()
    return selection
