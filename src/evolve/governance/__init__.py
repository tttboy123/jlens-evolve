"""Single promotion authority for v3 product assets."""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable

from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    content_sha256,
)
from evolve.evidence import CandidateEvidenceState
from evolve.registry import CandidateRecord, CapabilityRecord, RejectedRecord

from .decisions import (
    DecisionLogBusy,
    DecisionLogError,
    GovernanceDecisionAuthority,
    PromotionDecision,
    PromotionDecisionLog,
    decision_identity,
)


class GateDecision(StrEnum):
    APPROVED = "approved"
    HUMAN_APPROVAL_REQUIRED = "human_approval_required"
    NO_CHANGE = "no_change"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class GovernanceService:
    """Evaluate evidence without generating or mutating candidates.

    Approval is intentionally a decision value.  Applying it to a registry is a
    separate, explicitly authorized product operation and is not exposed by the
    registries' ordinary candidate-writing interface.
    """

    def __init__(
        self, *, authority: GovernanceDecisionAuthority | None = None
    ) -> None:
        self._authority = authority

    def evaluate(
        self,
        *,
        candidate_id: str,
        claims: Iterable[Claim],
        candidate_active: bool,
        human_approval: bool,
        evidence_grade: ClaimGrade | None = None,
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
        if any(row.classification is ClaimClassification.REGRESSION for row in rows):
            return GateDecision.REJECTED
        if any(row.classification is ClaimClassification.NEUTRAL for row in rows):
            return GateDecision.NO_CHANGE
        if not all(row.classification is ClaimClassification.GAIN for row in rows):
            return GateDecision.REJECTED
        # When an aggregate projection is supplied it is authoritative.  A
        # self-reported E3 Claim may not override an E2 aggregate that failed
        # ReceiptStore/trusted-observer replay.
        highest_grade = (
            evidence_grade
            if evidence_grade is not None
            else max(row.grade for row in rows)
        )
        if not human_approval or highest_grade < ClaimGrade.E3:
            return GateDecision.HUMAN_APPROVAL_REQUIRED
        return GateDecision.APPROVED

    def decide(
        self,
        *,
        candidate: CandidateRecord,
        evidence: CandidateEvidenceState,
        claims: Iterable[Claim],
        human_approval: bool,
        decided_at: str,
        log: PromotionDecisionLog,
    ) -> PromotionDecision:
        rows = tuple(claims)
        claim_ids = tuple(sorted(row.claim_id for row in rows))
        if evidence.candidate_id != candidate.candidate_id:
            raise ContractViolation("aggregate evidence candidate identity mismatch")
        if claim_ids != tuple(sorted(evidence.claim_ids)):
            raise ContractViolation("aggregate evidence claim identity mismatch")
        if set(candidate.source_claim_ids) != set(claim_ids):
            raise ContractViolation("candidate source claims do not match governance")
        _validate_evidence_state(evidence, rows)
        gate = self.evaluate(
            candidate_id=candidate.candidate_id,
            claims=rows,
            candidate_active=candidate.active,
            human_approval=human_approval,
            evidence_grade=evidence.grade,
        )
        rationale = f"{gate}: {evidence.rationale}"
        identity_payload = {
            "candidate_id": candidate.candidate_id,
            "candidate_revision_id": candidate.revision_id,
            "gate_decision": gate,
            "evidence_grade": evidence.grade,
            "claim_ids": claim_ids,
            "prediction_evidence_ids": evidence.prediction_evidence_ids,
            "human_approval": human_approval,
            "decided_at": decided_at,
            "rationale": rationale,
            "evidence_state_sha256": content_sha256(evidence),
            "authority_key_id": (
                self._authority.key_id if self._authority is not None else None
            ),
        }
        decision = PromotionDecision(
            decision_id=decision_identity(identity_payload),
            candidate_id=candidate.candidate_id,
            candidate_revision_id=candidate.revision_id,
            gate_decision=gate,
            evidence_grade=evidence.grade,
            claim_ids=claim_ids,
            prediction_evidence_ids=evidence.prediction_evidence_ids,
            human_approval=human_approval,
            decided_at=decided_at,
            rationale=rationale,
            evidence_state_sha256=identity_payload["evidence_state_sha256"],
            authority_key_id=identity_payload["authority_key_id"],
        )
        if gate is GateDecision.APPROVED:
            if self._authority is None:
                raise ContractViolation(
                    "approved decisions require configured governance authority"
                )
            decision = self._authority.sign(decision)
        log.append(decision)
        return decision

    @staticmethod
    def to_capability(
        *,
        candidate: CandidateRecord,
        decision: PromotionDecision,
        capability_id: str,
    ) -> CapabilityRecord:
        _validate_decision_candidate(candidate, decision)
        if decision.gate_decision is not GateDecision.APPROVED:
            raise ContractViolation("only approved decisions create capabilities")
        return CapabilityRecord(
            capability_id=capability_id,
            revision_id=candidate.revision_id,
            capability_kind=candidate.candidate_kind,
            evidence_claim_ids=decision.claim_ids,
            artifact_sha256=candidate.artifact_sha256,
            promotion_decision_id=decision.decision_id,
            source_candidate_id=candidate.candidate_id,
        )

    @staticmethod
    def to_rejected(
        *, candidate: CandidateRecord, decision: PromotionDecision
    ) -> RejectedRecord:
        _validate_decision_candidate(candidate, decision)
        if decision.gate_decision is not GateDecision.REJECTED:
            raise ContractViolation("only rejected decisions create rejected records")
        return RejectedRecord(
            candidate_id=candidate.candidate_id,
            revision_id=candidate.revision_id,
            candidate_kind=candidate.candidate_kind,
            evidence_claim_ids=decision.claim_ids,
            promotion_decision_id=decision.decision_id,
            reason=decision.rationale,
            artifact_sha256=candidate.artifact_sha256,
        )


def _validate_decision_candidate(
    candidate: CandidateRecord, decision: PromotionDecision
) -> None:
    if (
        decision.candidate_id != candidate.candidate_id
        or decision.candidate_revision_id != candidate.revision_id
        or set(decision.claim_ids) != set(candidate.source_claim_ids)
    ):
        raise ContractViolation("promotion decision candidate identity mismatch")


def _validate_evidence_state(
    evidence: CandidateEvidenceState, claims: tuple[Claim, ...]
) -> None:
    counts = {
        classification: sum(claim.classification is classification for claim in claims)
        for classification in ClaimClassification
    }
    if (
        evidence.task_count != len(claims)
        or evidence.gain_count != counts[ClaimClassification.GAIN]
        or evidence.neutral_count != counts[ClaimClassification.NEUTRAL]
        or evidence.regression_count != counts[ClaimClassification.REGRESSION]
        or evidence.infra_failure_count != counts[ClaimClassification.INFRA_FAILURE]
    ):
        raise ContractViolation("aggregate evidence classification counts mismatch")
    if evidence.infra_failure_count and evidence.grade is not ClaimGrade.E1:
        raise ContractViolation("infrastructure evidence cannot exceed E1")
    if evidence.grade is ClaimGrade.E3 and (
        not evidence.e3_eligible
        or evidence.task_count < 3
        or evidence.project_count < 2
        or evidence.gain_count < 2
        or evidence.regression_count
        or evidence.infra_failure_count
        or evidence.prediction_consistent_task_count != evidence.task_count
        or len(evidence.prediction_evidence_ids) < evidence.task_count * 2
        or not evidence.mechanism_id
    ):
        raise ContractViolation("E3 evidence does not satisfy mechanism replay gates")


__all__ = [
    "DecisionLogBusy",
    "DecisionLogError",
    "GateDecision",
    "GovernanceDecisionAuthority",
    "GovernanceService",
    "PromotionDecision",
    "PromotionDecisionLog",
]
