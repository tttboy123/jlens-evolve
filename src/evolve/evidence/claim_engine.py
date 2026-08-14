"""Derive auditable outcome claims from strictly aligned native evidence."""

from __future__ import annotations

import hashlib

from evolve.alignment import MatchedNativePair
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    canonical_json,
)

from .evidence_graph import EvidenceGraph


class ClaimEngine:
    """The sole classification boundary; it never promotes a candidate."""

    def __init__(self, graph: EvidenceGraph) -> None:
        self._graph = graph

    def classify_pair(
        self,
        candidate_id: str,
        pair: MatchedNativePair,
        *,
        supersedes: Claim | None = None,
    ) -> Claim:
        baseline = pair.baseline.payload
        taught = pair.taught.payload
        evaluator_errors = tuple(
            value
            for value in (baseline["evaluator_error"], taught["evaluator_error"])
            if value not in (None, "")
        )
        if evaluator_errors:
            classification = ClaimClassification.INFRA_FAILURE
            rationale = "native evaluator infrastructure error: " + "; ".join(
                str(value) for value in evaluator_errors
            )
        else:
            baseline_resolved = baseline["resolved"]
            taught_resolved = taught["resolved"]
            if not isinstance(baseline_resolved, bool) or not isinstance(
                taught_resolved, bool
            ):
                raise ContractViolation(
                    "native resolved fields must be booleans when "
                    "evaluator_error is empty"
                )
            classification, rationale = _classify_outcomes(
                baseline_resolved, taught_resolved
            )

        evidence_ids = (pair.baseline.evidence_id, pair.taught.evidence_id)
        supersedes_id = supersedes.claim_id if supersedes is not None else None
        identity = hashlib.sha256(
            canonical_json(
                {
                    "candidate_id": candidate_id,
                    "classification": classification,
                    "evidence_ids": evidence_ids,
                    "supersedes_claim_id": supersedes_id,
                }
            ).encode("utf-8")
        ).hexdigest()
        claim = Claim(
            claim_id=f"claim-{identity}",
            candidate_id=candidate_id,
            grade=(
                ClaimGrade.E1
                if classification is ClaimClassification.INFRA_FAILURE
                else ClaimGrade.E2
            ),
            classification=classification,
            evidence_ids=evidence_ids,
            rationale=rationale,
            supersedes_claim_id=supersedes_id,
        )
        return self._graph.append_claim(claim)


def _classify_outcomes(
    baseline_resolved: bool,
    taught_resolved: bool,
) -> tuple[ClaimClassification, str]:
    if not baseline_resolved and taught_resolved:
        return ClaimClassification.GAIN, "baseline failed and taught resolved"
    if baseline_resolved and not taught_resolved:
        return ClaimClassification.REGRESSION, "baseline resolved and taught failed"
    if baseline_resolved:
        return ClaimClassification.NEUTRAL, "baseline and taught both resolved"
    return ClaimClassification.NEUTRAL, "baseline and taught both failed"
