"""Single promotion authority for v3 product assets."""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable

from evolve.contracts import Claim, ClaimClassification, ClaimGrade, ContractViolation


class GateDecision(StrEnum):
    APPROVED = "approved"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class GovernanceService:
    """Evaluate evidence without generating or mutating candidates.

    Approval is intentionally a decision value.  Applying it to a registry is a
    separate, explicitly authorized product operation and is not exposed by the
    registries' ordinary candidate-writing interface.
    """

    def evaluate(
        self,
        *,
        candidate_id: str,
        claims: Iterable[Claim],
        candidate_active: bool,
        human_approval: bool,
    ) -> GateDecision:
        if not candidate_id.strip():
            raise ContractViolation("candidate_id must be non-empty")
        if candidate_active:
            raise ContractViolation("candidates must enter governance inactive")
        rows = tuple(claims)
        if not rows or any(row.candidate_id != candidate_id for row in rows):
            raise ContractViolation("claims must match the candidate")
        if any(row.classification is ClaimClassification.INFRA_FAILURE for row in rows):
            return GateDecision.BLOCKED
        if any(
            row.classification
            in {ClaimClassification.REGRESSION, ClaimClassification.NEUTRAL}
            for row in rows
        ):
            return GateDecision.REJECTED
        if not all(row.classification is ClaimClassification.GAIN for row in rows):
            return GateDecision.REJECTED
        highest_grade = max(row.grade for row in rows)
        if not human_approval or highest_grade < ClaimGrade.E3:
            return GateDecision.HUMAN_APPROVAL_REQUIRED
        return GateDecision.APPROVED


__all__ = ["GateDecision", "GovernanceService"]
