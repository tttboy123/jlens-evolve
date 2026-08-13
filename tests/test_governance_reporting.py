from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    Receipt,
)
from evolve.governance import GateDecision, GovernanceService
from evolve.reporting import AuditVerifier, CampaignReportProjector

SHA = "a" * 64


def _claim(classification: ClaimClassification, grade: ClaimGrade) -> Claim:
    return Claim(
        claim_id=f"claim-{classification}-{grade}",
        candidate_id="candidate-1",
        grade=grade,
        classification=classification,
        evidence_ids=("evidence-1",),
        rationale="native matched evidence",
        supersedes_claim_id=None,
    )


def test_only_governance_can_approve_inactive_candidate_and_requires_human() -> None:
    service = GovernanceService()

    pending = service.evaluate(
        candidate_id="candidate-1",
        claims=[_claim(ClaimClassification.GAIN, ClaimGrade.E2)],
        candidate_active=False,
        human_approval=False,
    )
    assert pending is GateDecision.HUMAN_APPROVAL_REQUIRED

    approved = service.evaluate(
        candidate_id="candidate-1",
        claims=[_claim(ClaimClassification.GAIN, ClaimGrade.E3)],
        candidate_active=False,
        human_approval=True,
    )
    assert approved is GateDecision.APPROVED


@pytest.mark.parametrize(
    "classification,expected",
    [
        (ClaimClassification.NEUTRAL, GateDecision.REJECTED),
        (ClaimClassification.REGRESSION, GateDecision.REJECTED),
        (ClaimClassification.INFRA_FAILURE, GateDecision.BLOCKED),
    ],
)
def test_governance_fails_closed_on_non_gain_claims(
    classification: ClaimClassification, expected: GateDecision
) -> None:
    assert (
        GovernanceService().evaluate(
            candidate_id="candidate-1",
            claims=[_claim(classification, ClaimGrade.E3)],
            candidate_active=False,
            human_approval=True,
        )
        is expected
    )


def test_report_is_rebuilt_from_receipts_and_claims_without_hand_counts(
    tmp_path: Path,
) -> None:
    receipts = [
        Receipt(
            receipt_id="r1",
            campaign_id="campaign-1",
            plan_id="p1",
            sequence=1,
            kind="native_evaluation",
            created_at="2026-08-14T00:00:00Z",
            payload={"task_id": "sphinx-7757", "arm": "baseline", "resolved": False},
            artifact_sha256=SHA,
        ),
        Receipt(
            receipt_id="r2",
            campaign_id="campaign-1",
            plan_id="p2",
            sequence=2,
            kind="cost",
            created_at="2026-08-14T00:00:01Z",
            payload={"cost_cny": 0.25},
            artifact_sha256=SHA,
        ),
    ]
    report = CampaignReportProjector().project(
        campaign_id="campaign-1",
        receipts=receipts,
        claims=[_claim(ClaimClassification.NEUTRAL, ClaimGrade.E1)],
        final_commit_sha="f" * 40,
    )

    assert report["counts"] == {"neutral": 1}
    assert report["actual_api_spend_cny"] == 0.25
    assert report["receipt_count"] == 2

    paths = CampaignReportProjector().write(report, tmp_path)
    verified = AuditVerifier().verify_manifest(paths.manifest_path, root=tmp_path)
    assert verified == len(json.loads(paths.manifest_path.read_text())["entries"])
    assert paths.json_path.is_file() and paths.markdown_path.is_file()


def test_manifest_rejects_missing_or_modified_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("original", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "path": "artifact.txt",
                        "sha256": hashlib.sha256(b"original").hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact.write_text("modified", encoding="utf-8")

    with pytest.raises(ContractViolation, match="hash mismatch"):
        AuditVerifier().verify_manifest(manifest, root=tmp_path)
