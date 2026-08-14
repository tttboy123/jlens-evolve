from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolve.contracts import Claim, ClaimClassification, ClaimGrade, canonical_json
from evolve.governance import (
    DecisionLogError,
    GateDecision,
    GovernanceDecisionAuthority,
    GovernanceService,
    PromotionDecision,
    PromotionDecisionLog,
)
from evolve.kernel import DurableCostLedger, LedgerIntegrityError
from evolve.registry import CapabilityRecord, CapabilityRegistry, RegistryViolation


def _claim(classification: ClaimClassification, *, grade: ClaimGrade = ClaimGrade.E1) -> Claim:
    return Claim(
        claim_id=f"claim-{classification}",
        candidate_id="candidate-1",
        grade=grade,
        classification=classification,
        evidence_ids=("evidence-1",),
        rationale="matched native evidence",
        supersedes_claim_id=None,
    )


@pytest.mark.parametrize(
    ("classification", "expected"),
    (
        (ClaimClassification.NEUTRAL, GateDecision.NO_CHANGE),
        (ClaimClassification.REGRESSION, GateDecision.REJECTED),
        (ClaimClassification.INFRA_FAILURE, GateDecision.BLOCKED),
    ),
)
def test_governance_projects_accurate_non_gain_states(
    classification: ClaimClassification, expected: GateDecision
) -> None:
    assert GovernanceService().evaluate(
        candidate_id="candidate-1",
        claims=(_claim(classification),),
        candidate_active=False,
        human_approval=True,
    ) is expected


def test_self_reported_e3_claim_cannot_override_e2_aggregate() -> None:
    claim = Claim(
        claim_id="claim-self-reported-e3",
        candidate_id="candidate-1",
        grade=ClaimGrade.E3,
        classification=ClaimClassification.GAIN,
        evidence_ids=tuple(f"evidence-{index}" for index in range(4)),
        rationale="shape-valid but not receipt-replayed",
        supersedes_claim_id=None,
        counterfactual_pair_sha256="a" * 64,
        counterfactual_receipt_ids=tuple(
            f"receipt-{index}" for index in range(6)
        ),
    )

    assert GovernanceService().evaluate(
        candidate_id="candidate-1",
        claims=(claim,),
        candidate_active=False,
        human_approval=True,
        evidence_grade=ClaimGrade.E2,
    ) is GateDecision.HUMAN_APPROVAL_REQUIRED


def test_capability_registry_requires_e3_human_approval_decision(tmp_path: Path) -> None:
    class Decisions:
        def all(self) -> tuple[PromotionDecision, ...]:
            return (
                PromotionDecision(
                    decision_id="decision-forged",
                    candidate_id="candidate-1",
                    candidate_revision_id="revision-1",
                    gate_decision=GateDecision.APPROVED,
                    evidence_grade=ClaimGrade.E3,
                    claim_ids=("claim-1",),
                    prediction_evidence_ids=("trusted-evidence-1",),
                    human_approval=False,
                    decided_at="2026-08-14T00:00:00Z",
                    rationale="approval bit is missing",
                ),
            )

    record = CapabilityRecord(
        capability_id="capability-1",
        revision_id="revision-1",
        capability_kind="operator",
        evidence_claim_ids=("claim-1",),
        artifact_sha256="a" * 64,
        promotion_decision_id="decision-forged",
        source_candidate_id="candidate-1",
    )
    with pytest.raises(RegistryViolation, match="authoritative promotion decision log"):
        CapabilityRegistry(tmp_path / "capabilities.jsonl", decision_log=Decisions()).append(record)


def test_forged_approved_decision_cannot_enter_authoritative_log_or_capability(
    tmp_path: Path,
) -> None:
    authority = GovernanceDecisionAuthority(
        key_id="governance-production-key",
        secret_key=b"p" * 32,
    )
    log = PromotionDecisionLog(
        tmp_path / "promotion-decisions.jsonl", authority=authority
    )
    forged = PromotionDecision(
        decision_id="decision-forged",
        candidate_id="candidate-1",
        candidate_revision_id="revision-1",
        gate_decision=GateDecision.APPROVED,
        evidence_grade=ClaimGrade.E3,
        claim_ids=("claim-1",),
        prediction_evidence_ids=("trusted-evidence-1",),
        human_approval=True,
        decided_at="2026-08-14T00:00:00Z",
        rationale="hand-built approval bypass",
        evidence_state_sha256="a" * 64,
        authority_key_id=authority.key_id,
        authority_signature_hmac_sha256="b" * 64,
    )

    with pytest.raises(DecisionLogError, match="governance authority"):
        log.append(forged)

    assert log.all() == ()


def _ledger(path: Path) -> DurableCostLedger:
    return DurableCostLedger(
        path,
        campaign_id="campaign-1",
        max_cost_cny=10,
        max_model_calls=3,
    )


def _pop_line(path: Path, index: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.pop(index)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cost_ledger_events_form_a_restart_verifiable_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = _ledger(path)
    ledger.reserve("reservation-1", cost_cny=2, model_calls=1)
    ledger.record(
        "reservation-1",
        result_id="result-1",
        actual_cost_cny=1.25,
        actual_model_calls=1,
    )

    events = ledger.events()
    assert [event["sequence"] for event in events] == [0, 1, 2]
    assert events[0]["previous_event_sha256"] == "0" * 64
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]
    assert events[2]["previous_event_sha256"] == events[1]["event_sha256"]
    assert all(
        event["event_sha256"]
        == hashlib.sha256(
            canonical_json({k: v for k, v in event.items() if k != "event_sha256"}).encode()
        ).hexdigest()
        for event in events
    )
    assert _ledger(path).snapshot().spent_cost_cny == 1.25


@pytest.mark.parametrize("attack", ("delete-middle", "delete-tail", "reorder", "duplicate"))
def test_cost_ledger_detects_line_structure_attacks(tmp_path: Path, attack: str) -> None:
    path = tmp_path / f"{attack}.jsonl"
    ledger = _ledger(path)
    ledger.reserve("reservation-1", cost_cny=2, model_calls=1)
    ledger.record(
        "reservation-1",
        result_id="result-1",
        actual_cost_cny=1,
        actual_model_calls=1,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    if attack == "delete-middle":
        lines.pop(1)
    elif attack == "delete-tail":
        lines.pop()
    elif attack == "reorder":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines.insert(2, lines[1])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerIntegrityError):
        _ledger(path)


def test_cost_ledger_detects_partial_tail_and_tampering(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = _ledger(path)
    ledger.reserve("reservation-1", cost_cny=2, model_calls=1)

    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(LedgerIntegrityError, match="partial final event"):
        _ledger(path)

    clean = tmp_path / "tampered.jsonl"
    ledger = _ledger(clean)
    ledger.reserve("reservation-1", cost_cny=2, model_calls=1)
    lines = clean.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])
    payload["cost_cny"] = 9
    lines[-1] = json.dumps(payload)
    clean.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="hash mismatch"):
        _ledger(clean)


def test_cost_ledger_recovers_one_fsynced_event_after_head_update_crash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    ledger = _ledger(path)
    ledger.reserve("reservation-1", cost_cny=2, model_calls=1)
    prior_head = ledger.head_path.read_bytes()
    ledger.record(
        "reservation-1",
        result_id="result-1",
        actual_cost_cny=1,
        actual_model_calls=1,
    )
    # Fault injection: the event reached the append-only log, then the process
    # died before its atomic head replacement became durable.
    ledger.head_path.write_bytes(prior_head)

    recovered = _ledger(path)
    assert recovered.snapshot().spent_cost_cny == 1
    assert json.loads(recovered.head_path.read_text())["event_count"] == 3


def test_cost_ledger_refuses_ambiguous_multi_event_head_recovery(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = _ledger(path)
    genesis_head = ledger.head_path.read_bytes()
    ledger.reserve("reservation-1", cost_cny=2, model_calls=1)
    ledger.record(
        "reservation-1",
        result_id="result-1",
        actual_cost_cny=1,
        actual_model_calls=1,
    )
    ledger.head_path.write_bytes(genesis_head)

    with pytest.raises(LedgerIntegrityError, match="head mismatch"):
        _ledger(path)
