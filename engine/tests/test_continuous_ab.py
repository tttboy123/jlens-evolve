from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from continuous_ab import (
    ABContractError,
    ArmResult,
    BaselineContract,
    ChangeSetCadence,
    ChangeSetRegistry,
    ContinuousSchedule,
    FinalSealedAuditor,
    MatchedRoundLedger,
    PermanentBaselineAuthority,
    PromotionGate,
)


def _baseline() -> BaselineContract:
    return BaselineContract(
        experiment_id="continuous-ab-v1",
        agent_program_sha256="a" * 64,
        model="codex",
        reasoning="low",
        token_budget=4096,
        timeout_seconds=1800,
        tools=("shell", "apply_patch"),
        retries=0,
        evaluator_epoch="benchmark-native-v1",
    )


def _result(arm: str, *, resolved: bool, family: str = "swe") -> ArmResult:
    return ArmResult(
        arm=arm,
        resolved=resolved,
        regression_failures=0,
        safe=True,
        input_tokens=1000,
        output_tokens=500,
        elapsed_seconds=60.0,
        benchmark_family=family,
        evaluator_epoch="benchmark-native-v1",
    )


def test_round_requires_both_frozen_matched_predictions_before_evaluation(
    tmp_path: Path,
):
    ledger = MatchedRoundLedger.create(
        tmp_path / "round.json",
        round_id="round-001",
        task_uid="task-001",
        baseline=_baseline(),
        evolved_agent_sha256="b" * 64,
    )
    baseline_patch = tmp_path / "baseline.patch"
    evolved_patch = tmp_path / "evolved.patch"
    baseline_patch.write_text("baseline\n", encoding="utf-8")
    evolved_patch.write_text("evolved\n", encoding="utf-8")

    with pytest.raises(ABContractError, match="both arms"):
        ledger.freeze_predictions({"baseline": baseline_patch})

    ledger.freeze_predictions({"baseline": baseline_patch, "evolved": evolved_patch})
    with pytest.raises(ABContractError, match="both arm results"):
        ledger.record_results({"baseline": _result("baseline", resolved=True)})

    ledger.record_results(
        {
            "baseline": _result("baseline", resolved=False),
            "evolved": _result("evolved", resolved=True),
        }
    )
    ledger.retire()

    assert MatchedRoundLedger.load(tmp_path / "round.json").phase == "retired"


def test_round_rejects_prediction_changed_after_freeze(tmp_path: Path):
    ledger = MatchedRoundLedger.create(
        tmp_path / "round.json",
        round_id="round-001",
        task_uid="task-001",
        baseline=_baseline(),
        evolved_agent_sha256="b" * 64,
    )
    baseline_patch = tmp_path / "baseline.patch"
    evolved_patch = tmp_path / "evolved.patch"
    baseline_patch.write_text("baseline\n", encoding="utf-8")
    evolved_patch.write_text("evolved\n", encoding="utf-8")
    ledger.freeze_predictions({"baseline": baseline_patch, "evolved": evolved_patch})

    baseline_patch.write_text("tampered after freeze\n", encoding="utf-8")

    with pytest.raises(ABContractError, match="frozen prediction.*tampered"):
        ledger.record_results(
            {
                "baseline": _result("baseline", resolved=False),
                "evolved": _result("evolved", resolved=True),
            }
        )


def test_native_evidence_pair_is_frozen_before_result_admission(tmp_path: Path):
    ledger = MatchedRoundLedger.create(
        tmp_path / "round.json",
        round_id="round-001",
        task_uid="task-001",
        baseline=_baseline(),
        evolved_agent_sha256="b" * 64,
    )
    predictions = {}
    for arm in ("baseline", "evolved"):
        prediction = tmp_path / f"{arm}.patch"
        prediction.write_text(f"{arm}\n", encoding="utf-8")
        predictions[arm] = prediction
    ledger.freeze_predictions(predictions)
    results = {
        "baseline": _result("baseline", resolved=False),
        "evolved": _result("evolved", resolved=True),
    }
    evidence_paths = {}
    for arm in ("baseline", "evolved"):
        payload = {
            "schema_version": "1.0",
            "identity": {
                "round_id": "round-001",
                "arm": arm,
                "agent_program_sha256": "a" * 64 if arm == "baseline" else "b" * 64,
                "baseline_contract_sha256": _baseline().contract_sha256,
                "evaluator_epoch": "benchmark-native-v1",
            },
            "result": asdict(results[arm]),
        }
        payload["integrity_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path = tmp_path / f"{arm}-native.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        evidence_paths[arm] = path

    ledger.record_native_evidence(evidence_paths, results)
    evidence_paths["evolved"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ABContractError, match="native evidence.*tampered"):
        ledger.record_results(results)


def test_round_ledger_detects_json_tampering_and_freezes_matched_arm_contracts(
    tmp_path: Path,
):
    path = tmp_path / "round.json"
    ledger = MatchedRoundLedger.create(
        path,
        round_id="round-001",
        task_uid="task-001",
        baseline=_baseline(),
        evolved_agent_sha256="b" * 64,
    )
    baseline_contract = ledger.payload["arm_contracts"]["baseline"]
    evolved_contract = ledger.payload["arm_contracts"]["evolved"]
    assert {
        key: value
        for key, value in baseline_contract.items()
        if key != "agent_program_sha256"
    } == {
        key: value
        for key, value in evolved_contract.items()
        if key != "agent_program_sha256"
    }

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["evolved_agent_sha256"] = "c" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ABContractError, match="ledger.*tampered"):
        MatchedRoundLedger.load(path)


def test_permanent_baseline_hash_cannot_be_replaced(tmp_path: Path):
    path = tmp_path / "round.json"
    MatchedRoundLedger.create(
        path,
        round_id="round-001",
        task_uid="task-001",
        baseline=_baseline(),
        evolved_agent_sha256="b" * 64,
    )
    changed = _baseline().replace(agent_program_sha256="c" * 64)

    with pytest.raises(ABContractError, match="permanent shadow baseline"):
        MatchedRoundLedger.resume(path, baseline=changed)


def test_experiment_baseline_authority_is_idempotent_but_immutable(tmp_path: Path):
    authority = PermanentBaselineAuthority(tmp_path / "baseline.json")

    first = authority.freeze(_baseline())
    second = authority.freeze(_baseline())

    assert first["baseline_contract_sha256"] == second["baseline_contract_sha256"]
    assert second["freeze_count"] == 1
    with pytest.raises(ABContractError, match="permanent shadow baseline"):
        authority.freeze(_baseline().replace(agent_program_sha256="c" * 64))


def test_changeset_cadence_allows_at_most_one_proposal_per_ten_rounds():
    cadence = ChangeSetCadence(interval=10)

    assert cadence.can_propose(completed_rounds=9, last_proposal_round=0) is False
    assert cadence.can_propose(completed_rounds=10, last_proposal_round=0) is True
    assert cadence.can_propose(completed_rounds=19, last_proposal_round=10) is False
    assert cadence.can_propose(completed_rounds=20, last_proposal_round=10) is True


def test_300_round_schedule_has_eight_fixed_promotion_blocks_then_sealed():
    schedule = ContinuousSchedule().build()

    assert len(schedule["rounds"]) == 300
    assert schedule["partition_counts"] == {
        "search": 160,
        "promotion": 80,
        "final_sealed": 60,
    }
    assert schedule["proposal_after_rounds"] == [20, 50, 80, 110, 140, 170, 200, 230]
    assert schedule["rounds"][-1]["partition"] == "final_sealed"


def test_promotion_gate_uses_paired_wins_regressions_safety_and_cost():
    pairs = []
    for index in range(8):
        pairs.append(
            (
                _result("baseline", resolved=False, family="swe"),
                _result("evolved", resolved=True, family="swe"),
            )
        )
    for index in range(4):
        pairs.append(
            (
                _result("baseline", resolved=True, family="terminal"),
                _result("evolved", resolved=True, family="terminal"),
            )
        )

    decision = PromotionGate().evaluate(pairs)

    assert decision.approved is True
    assert decision.wins == 8
    assert decision.losses == 0
    assert decision.one_sided_sign_p <= 0.05

    unsafe = list(pairs)
    unsafe[0] = (
        unsafe[0][0],
        unsafe[0][1].replace(safe=False),
    )
    assert PromotionGate().evaluate(unsafe).approved is False


def test_changeset_registry_enforces_cadence_promotion_and_rollback(tmp_path: Path):
    registry = ChangeSetRegistry.create(
        tmp_path / "changesets.json",
        initial_agent_sha256="a" * 64,
        cadence=ChangeSetCadence(interval=10),
    )
    forward = tmp_path / "forward.patch"
    rollback = tmp_path / "rollback.patch"
    forward.write_text("forward\n", encoding="utf-8")
    rollback.write_text("rollback\n", encoding="utf-8")

    with pytest.raises(ABContractError, match="cadence"):
        registry.propose(
            completed_rounds=9,
            candidate_agent_sha256="b" * 64,
            forward_patch=forward,
            rollback_patch=rollback,
        )
    proposal = registry.propose(
        completed_rounds=10,
        candidate_agent_sha256="b" * 64,
        forward_patch=forward,
        rollback_patch=rollback,
    )
    with pytest.raises(ABContractError, match="one ChangeSet"):
        registry.propose(
            completed_rounds=10,
            candidate_agent_sha256="c" * 64,
            forward_patch=forward,
            rollback_patch=rollback,
        )

    rejected = PromotionGate().evaluate(
        [(_result("baseline", resolved=True), _result("evolved", resolved=False))]
    )
    with pytest.raises(ABContractError, match="failed promotion"):
        registry.promote(proposal["proposal_id"], rejected)

    approved_pairs = [
        (_result("baseline", resolved=False), _result("evolved", resolved=True))
        for _ in range(10)
    ]
    approved = PromotionGate().evaluate(approved_pairs)
    promoted = registry.promote(proposal["proposal_id"], approved)

    assert promoted["active_agent_sha256"] == "b" * 64
    rolled_back = registry.rollback(proposal["proposal_id"], reason="operator-test")
    assert rolled_back["active_agent_sha256"] == "a" * 64
    assert (
        ChangeSetRegistry.load(tmp_path / "changesets.json").active_agent_sha256
        == "a" * 64
    )


def test_changeset_registry_detects_active_ref_tampering(tmp_path: Path):
    path = tmp_path / "changesets.json"
    ChangeSetRegistry.create(
        path,
        initial_agent_sha256="a" * 64,
        cadence=ChangeSetCadence(interval=10),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["active_agent_sha256"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ABContractError, match="registry.*tampered"):
        ChangeSetRegistry.load(path)


def test_promotion_requires_all_ten_predeclared_pairs():
    incomplete = [
        (_result("baseline", resolved=False), _result("evolved", resolved=True))
        for _ in range(9)
    ]

    decision = PromotionGate().evaluate(incomplete)

    assert decision.approved is False
    assert decision.checks["complete_promotion_block"] is False


def test_final_sealed_auditor_opens_once_and_requires_60_unused_tasks(
    tmp_path: Path,
):
    auditor = FinalSealedAuditor(tmp_path / "sealed.json")
    tasks = [f"sealed-{index:03d}" for index in range(60)]

    audit = auditor.open(candidate_sha256="d" * 64, task_uids=tasks)

    assert audit["opened_once"] is True
    assert audit["task_count"] == 60
    with pytest.raises(ABContractError, match="already opened"):
        auditor.open(candidate_sha256="d" * 64, task_uids=tasks)
    with pytest.raises(ABContractError, match="at least 60"):
        FinalSealedAuditor(tmp_path / "too-small.json").open(
            candidate_sha256="d" * 64,
            task_uids=tasks[:59],
        )
