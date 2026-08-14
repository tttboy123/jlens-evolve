"""Independent verification of a campaign before the outer loop consumes it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolve.contracts import ClaimClassification, ClaimGrade
from evolve.evidence import EvidenceGraph, ReceiptStore

from .config import AutonomousEvolutionError


@dataclass(frozen=True, slots=True)
class VerifiedCampaignClaim:
    task_id: str
    claim_id: str
    classification: str
    grade: str
    counterfactual_pair_sha256: str | None
    counterfactual_receipt_ids: tuple[str, ...]


class CampaignOutcomeVerifier:
    """Rebuild receipts/evidence and match the public result to those facts."""

    def verify(
        self,
        *,
        round_root: str | Path,
        result: Mapping[str, Any],
        selected_task_ids: Sequence[str],
        candidate_id: str,
        candidate_revision_id: str,
        candidate_bundle_sha256: str,
    ) -> tuple[VerifiedCampaignClaim, ...]:
        root = Path(round_root).resolve()
        if any(
            result.get(name) is not False
            for name in (
                "holdout_opened",
                "burned_holdout_opened",
                "capability_active",
            )
        ):
            raise AutonomousEvolutionError(
                "campaign safety invariant was not independently preserved"
            )
        if result.get("campaign_status") != "completed":
            raise AutonomousEvolutionError("campaign did not complete")
        if result.get("execution_statuses") != ["completed"] * 6:
            raise AutonomousEvolutionError("campaign did not complete six executions")
        receipt_store = ReceiptStore(root / "receipt-store")
        graph = EvidenceGraph(root / "evidence-graph")
        try:
            EvidenceGraph.rebuild(graph.root, receipt_store)
            receipts = receipt_store.list_receipts()
            claims = graph.latest_claims()
        except Exception as error:
            raise AutonomousEvolutionError(
                "campaign authoritative receipt/evidence replay failed"
            ) from error
        if not receipts or not claims:
            raise AutonomousEvolutionError(
                "campaign has no authoritative receipts or claims"
            )
        by_id = {claim.claim_id: claim for claim in claims}
        evidence_by_id = {
            envelope.evidence_id: envelope for envelope in graph.list_evidence()
        }
        raw_claims = result.get("claims")
        if not isinstance(raw_claims, list) or len(raw_claims) != len(
            selected_task_ids
        ):
            raise AutonomousEvolutionError("campaign claim projection is incomplete")
        projected: list[VerifiedCampaignClaim] = []
        seen_tasks: set[str] = set()
        for row in raw_claims:
            if not isinstance(row, Mapping):
                raise AutonomousEvolutionError("campaign claim projection is invalid")
            task_id = row.get("task_id")
            task_revision_id = row.get("task_revision_id")
            claim_id = row.get("claim_id")
            classification = row.get("classification")
            if (
                not isinstance(task_id, str)
                or not isinstance(task_revision_id, str)
                or not isinstance(claim_id, str)
                or not isinstance(classification, str)
            ):
                raise AutonomousEvolutionError("campaign claim identity is invalid")
            authoritative = by_id.get(claim_id)
            authoritative_task_revision_ids = {
                evidence_by_id[evidence_id].payload.get("task_revision_id")
                for evidence_id in authoritative.evidence_ids
            } if authoritative is not None else set()
            if (
                authoritative is None
                or authoritative.candidate_id != candidate_id
                or str(authoritative.classification) != classification
                or authoritative_task_revision_ids != {task_revision_id}
                or row.get("grade") != str(authoritative.grade)
                or row.get("counterfactual_pair_sha256")
                != authoritative.counterfactual_pair_sha256
                or tuple(row.get("counterfactual_receipt_ids", ()))
                != authoritative.counterfactual_receipt_ids
            ):
                raise AutonomousEvolutionError(
                    "campaign claim projection disagrees with authoritative graph"
                )
            if authoritative.classification is not ClaimClassification.INFRA_FAILURE:
                if (
                    authoritative.grade not in {ClaimGrade.E2, ClaimGrade.E3}
                    or authoritative.counterfactual_pair_sha256 is None
                    or len(authoritative.evidence_ids) != 4
                    or len(authoritative.counterfactual_receipt_ids) != 6
                ):
                    raise AutonomousEvolutionError(
                        "native campaign claim lacks complete E2 counterfactual lineage"
                    )
            seen_tasks.add(task_id)
            projected.append(
                VerifiedCampaignClaim(
                    task_id=task_id,
                    claim_id=claim_id,
                    classification=classification,
                    grade=str(authoritative.grade),
                    counterfactual_pair_sha256=(
                        authoritative.counterfactual_pair_sha256
                    ),
                    counterfactual_receipt_ids=(
                        authoritative.counterfactual_receipt_ids
                    ),
                )
            )
        if seen_tasks != set(selected_task_ids):
            raise AutonomousEvolutionError(
                "campaign claims do not match selected feedback tasks"
            )
        self._verify_model_receipts(
            receipts=receipts,
            candidate_revision_id=candidate_revision_id,
            candidate_bundle_sha256=candidate_bundle_sha256,
            task_count=len(selected_task_ids),
        )
        return tuple(projected)

    @staticmethod
    def _verify_model_receipts(
        *,
        receipts: Sequence[Any],
        candidate_revision_id: str,
        candidate_bundle_sha256: str,
        task_count: int,
    ) -> None:
        model = [receipt for receipt in receipts if receipt.kind == "model"]
        baseline = [receipt for receipt in model if receipt.payload.get("arm") == "baseline"]
        taught = [receipt for receipt in model if receipt.payload.get("arm") == "taught"]
        if len(baseline) != task_count or len(taught) != task_count:
            raise AutonomousEvolutionError(
                "campaign model receipt pairing is incomplete"
            )
        for receipt in baseline:
            payload = receipt.payload
            if (
                payload.get("candidate_consumed") is not False
                or payload.get("candidate_revision_id") is not None
                or payload.get("candidate_bundle_sha256") is not None
                or payload.get("candidate_prompt") is not None
            ):
                raise AutonomousEvolutionError(
                    "baseline model receipt consumed proposed candidate content"
                )
        for receipt in taught:
            payload = receipt.payload
            if (
                payload.get("candidate_consumed") is not True
                or payload.get("candidate_revision_id") != candidate_revision_id
                or payload.get("candidate_bundle_sha256")
                != candidate_bundle_sha256
            ):
                raise AutonomousEvolutionError(
                    "taught model receipt candidate lineage mismatch"
                )
