"""Derive auditable outcome claims from strictly aligned native evidence."""

from __future__ import annotations

import hashlib

from evolve.alignment import MatchedNativePair
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    MatchedCounterfactualPair,
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
        counterfactual_pair: MatchedCounterfactualPair | None = None,
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

        if classification is not ClaimClassification.INFRA_FAILURE:
            if counterfactual_pair is not None:
                _validate_counterfactual_binding(
                    candidate_id, pair, counterfactual_pair
                )
                _validate_graph_binding(self._graph, counterfactual_pair)

        evidence_ids = (
            counterfactual_pair.evidence_ids
            if counterfactual_pair is not None
            else (pair.baseline.evidence_id, pair.taught.evidence_id)
        )
        supersedes_id = supersedes.claim_id if supersedes is not None else None
        identity = hashlib.sha256(
            canonical_json(
                {
                    "candidate_id": candidate_id,
                    "classification": classification,
                    "evidence_ids": evidence_ids,
                    "counterfactual_pair_sha256": (
                        counterfactual_pair.content_sha256
                        if counterfactual_pair is not None
                        else None
                    ),
                    "supersedes_claim_id": supersedes_id,
                }
            ).encode("utf-8")
        ).hexdigest()
        claim = Claim(
            claim_id=f"claim-{identity}",
            candidate_id=candidate_id,
            grade=(
                ClaimGrade.E2
                if counterfactual_pair is not None
                and classification is not ClaimClassification.INFRA_FAILURE
                else ClaimGrade.E1
            ),
            classification=classification,
            evidence_ids=evidence_ids,
            rationale=rationale,
            supersedes_claim_id=supersedes_id,
            counterfactual_pair_sha256=(
                counterfactual_pair.content_sha256
                if counterfactual_pair is not None
                else None
            ),
            counterfactual_receipt_ids=(
                counterfactual_pair.receipt_ids
                if counterfactual_pair is not None
                else ()
            ),
        )
        return self._graph.append_claim(
            claim,
            counterfactual_pair=counterfactual_pair,
        )


def _validate_counterfactual_binding(
    candidate_id: str,
    native_pair: MatchedNativePair,
    counterfactual_pair: MatchedCounterfactualPair,
) -> None:
    if counterfactual_pair.candidate_id != candidate_id:
        raise ContractViolation("counterfactual candidate does not match claim")
    if (
        counterfactual_pair.baseline.native_outcome_evidence_id
        != native_pair.baseline.evidence_id
        or counterfactual_pair.taught.native_outcome_evidence_id
        != native_pair.taught.evidence_id
    ):
        raise ContractViolation("counterfactual native evidence does not match pair")
    for field, expected in native_pair.matched_identity.items():
        if getattr(counterfactual_pair, field) != expected:
            raise ContractViolation(
                f"counterfactual {field} does not match native alignment"
            )


def _validate_graph_binding(
    graph: EvidenceGraph, pair: MatchedCounterfactualPair
) -> None:
    frozen = {row.evidence_id: row for row in graph.list_evidence()}
    expected = (
        (
            pair.baseline.external_trace_evidence_id,
            pair.baseline.external_trace_evidence_sha256,
            pair.baseline.external_trace_receipt_id,
            pair.baseline.external_trace_artifact_sha256,
        ),
        (
            pair.baseline.native_outcome_evidence_id,
            pair.baseline.native_outcome_evidence_sha256,
            pair.baseline.native_outcome_receipt_id,
            pair.baseline.native_outcome_artifact_sha256,
        ),
        (
            pair.taught.external_trace_evidence_id,
            pair.taught.external_trace_evidence_sha256,
            pair.taught.external_trace_receipt_id,
            pair.taught.external_trace_artifact_sha256,
        ),
        (
            pair.taught.native_outcome_evidence_id,
            pair.taught.native_outcome_evidence_sha256,
            pair.taught.native_outcome_receipt_id,
            pair.taught.native_outcome_artifact_sha256,
        ),
    )
    for evidence_id, evidence_sha256, receipt_id, artifact_sha256 in expected:
        envelope = frozen.get(evidence_id)
        if (
            envelope is None
            or envelope.content_sha256 != evidence_sha256
            or envelope.receipt_ids != (receipt_id,)
            or envelope.artifact_sha256 != artifact_sha256
        ):
            raise ContractViolation(
                f"counterfactual evidence {evidence_id} is not frozen in EvidenceGraph"
            )


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
