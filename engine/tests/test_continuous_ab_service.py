from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from benchmark_adapters import (
    BenchmarkRegistry,
    BenchmarkTask,
    StaticBenchmarkAdapter,
    TaskPool,
)
from continuous_ab import (
    ABContractError,
    ArmResult,
    BaselineContract,
    MatchedRoundLedger,
)
from continuous_ab_service import ContinuousABService
from native_result_adapter import NormalizedAdmission


def _task(index: int) -> BenchmarkTask:
    return BenchmarkTask(
        benchmark_id="fixture",
        benchmark_revision="fixture-frozen-v1",
        instance_id=f"task-{index}",
        task_family="fixture",
        language="python",
        repo="example/repo",
        base_commit=f"commit-{index}",
        environment_ref=f"fixture://environment/{index}",
        grader_ref=f"fixture://grader/{index}",
        instruction_ref=f"fixture://instruction/{index}",
        source_url="https://example.test/fixture",
        license_id="test-only",
        overlap_keys=(f"fixture://task/{index}",),
        content_sha256=f"{index:064x}"[-64:],
    )


def _pool(path: Path) -> None:
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id="fixture",
            revision="fixture-frozen-v1",
            executable=True,
            tasks=tuple(_task(index) for index in range(70)),
        )
    )
    TaskPool.build(
        registry=registry,
        seed_material="service-test",
        target_count=70,
        promotion_count=5,
        final_sealed_count=60,
    ).save(path)


def _scheduled_pool(path: Path) -> None:
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id="fixture",
            revision="fixture-frozen-v1",
            executable=True,
            tasks=tuple(_task(index) for index in range(90)),
        )
    )
    TaskPool.build(
        registry=registry,
        seed_material="scheduled-service-test",
        target_count=90,
        promotion_count=10,
        final_sealed_count=60,
    ).save(path)


def _baseline() -> BaselineContract:
    return BaselineContract(
        experiment_id="service-test",
        agent_program_sha256="a" * 64,
        model="codex",
        reasoning="low",
        token_budget=4096,
        timeout_seconds=1800,
        tools=("shell", "apply_patch"),
        retries=0,
        evaluator_epoch="fixture-v1",
    )


def _result(arm: str, resolved: bool) -> ArmResult:
    return ArmResult(
        arm=arm,
        resolved=resolved,
        regression_failures=0,
        safe=True,
        input_tokens=100,
        output_tokens=50,
        elapsed_seconds=1,
        benchmark_family="fixture",
        evaluator_epoch="fixture-v1",
    )


def _finish_round(
    service: ContinuousABService,
    planned: dict,
    root: Path,
    *,
    baseline_resolved: bool,
    evolved_resolved: bool,
) -> None:
    prediction_dir = root / planned["round_id"]
    prediction_dir.mkdir(parents=True)
    baseline_patch = prediction_dir / "baseline.patch"
    evolved_patch = prediction_dir / "evolved.patch"
    baseline_patch.write_text("baseline\n", encoding="utf-8")
    evolved_patch.write_text("evolved\n", encoding="utf-8")
    service.freeze_predictions(
        planned["round_id"],
        {"baseline": baseline_patch, "evolved": evolved_patch},
    )
    service.record_results(
        planned["round_id"],
        {
            "baseline": _result("baseline", baseline_resolved),
            "evolved": _result("evolved", evolved_resolved),
        },
    )


def test_service_runs_recoverable_matched_round_and_retires_task(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )

    planned = service.plan_round(
        partition="search",
        evolved_agent_sha256="b" * 64,
    )
    baseline_patch = tmp_path / "baseline.patch"
    evolved_patch = tmp_path / "evolved.patch"
    baseline_patch.write_text("baseline\n", encoding="utf-8")
    evolved_patch.write_text("evolved\n", encoding="utf-8")
    service.freeze_predictions(
        planned["round_id"],
        {"baseline": baseline_patch, "evolved": evolved_patch},
    )
    service.record_results(
        planned["round_id"],
        {
            "baseline": _result("baseline", False),
            "evolved": _result("evolved", True),
        },
    )

    restored = ContinuousABService.load(tmp_path / "runtime")
    round_state = restored.round_state(planned["round_id"])
    pool = TaskPool.load(tmp_path / "runtime" / "TASK_POOL.json")
    task = next(row for row in pool.records if row.task_uid == planned["task_uid"])

    assert round_state["phase"] == "retired"
    assert task.state == "retired"
    assert restored.completed_round_count == 1
    assert (
        restored.plan_round(partition="search", evolved_agent_sha256="b" * 64)[
            "task_uid"
        ]
        != planned["task_uid"]
    )


def test_service_admits_paired_native_evidence_and_retires_task(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )
    planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
    predictions = {}
    for arm in ("baseline", "evolved"):
        prediction = tmp_path / f"{arm}.patch"
        prediction.write_text(f"{arm}\n", encoding="utf-8")
        predictions[arm] = prediction
    service.freeze_predictions(planned["round_id"], predictions)
    admissions = {}
    for arm, resolved in (("baseline", False), ("evolved", True)):
        result = _result(arm, resolved)
        payload = {
            "schema_version": "1.0",
            "identity": {
                "round_id": planned["round_id"],
                "arm": arm,
                "agent_program_sha256": "a" * 64 if arm == "baseline" else "b" * 64,
                "baseline_contract_sha256": _baseline().contract_sha256,
                "evaluator_epoch": "fixture-v1",
            },
            "result": asdict(result),
        }
        payload["integrity_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        evidence_path = tmp_path / f"{arm}-native.json"
        evidence_path.write_text(json.dumps(payload), encoding="utf-8")
        admissions[arm] = NormalizedAdmission(result, evidence_path, payload)

    service.record_native_results(planned["round_id"], admissions)

    state = service.round_state(planned["round_id"])
    assert state["phase"] == "retired"
    assert len(state["native_evidence"]) == 2


def test_reconcile_finishes_crash_after_evaluation_before_retirement(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )
    planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
    baseline_patch = tmp_path / "baseline.patch"
    evolved_patch = tmp_path / "evolved.patch"
    baseline_patch.write_text("baseline\n", encoding="utf-8")
    evolved_patch.write_text("evolved\n", encoding="utf-8")
    service.freeze_predictions(
        planned["round_id"],
        {"baseline": baseline_patch, "evolved": evolved_patch},
    )
    ledger = MatchedRoundLedger.load(
        tmp_path / "runtime" / "rounds" / f"{planned['round_id']}.json"
    )
    ledger.record_results(
        {
            "baseline": _result("baseline", False),
            "evolved": _result("evolved", True),
        }
    )

    reconciled = ContinuousABService.load(tmp_path / "runtime").reconcile_round(
        planned["round_id"]
    )

    assert reconciled["phase"] == "retired"
    assert reconciled["reconciled"] is True


def test_planning_reconciles_crash_window_before_selecting_next_task(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )
    first = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
    runtime_pool_path = tmp_path / "runtime/TASK_POOL.json"
    runtime_pool = TaskPool.load(runtime_pool_path)
    opened = next(
        row for row in runtime_pool.records if row.task_uid == first["task_uid"]
    )
    opened.state = "unopened"
    runtime_pool.save(runtime_pool_path)

    second = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)

    assert second["task_uid"] != first["task_uid"]
    reconciled_pool = TaskPool.load(runtime_pool_path)
    assert (
        next(
            row for row in reconciled_pool.records if row.task_uid == first["task_uid"]
        ).state
        == "search"
    )


def test_service_enforces_predeclared_partition_order(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )

    with pytest.raises(ABContractError, match="expected partition is search"):
        service.plan_round(partition="promotion", evolved_agent_sha256="b" * 64)


def test_search_cycle_freezes_one_evolved_candidate(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )
    service.plan_round(partition="search", evolved_agent_sha256="b" * 64)

    with pytest.raises(ABContractError, match="search cycle candidate"):
        service.plan_round(partition="search", evolved_agent_sha256="c" * 64)


def test_service_manifest_detects_tampering(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _pool(pool_path)
    runtime = tmp_path / "runtime"
    ContinuousABService.initialize(
        runtime,
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )
    service_path = runtime / "SERVICE.json"
    payload = json.loads(service_path.read_text(encoding="utf-8"))
    payload["final_sealed_opened"] = True
    service_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ABContractError, match="service manifest.*tampered"):
        ContinuousABService.load(runtime)


def test_promotion_requires_registered_candidate_and_ten_new_pairs(tmp_path: Path):
    pool_path = tmp_path / "TASK_POOL.json"
    _scheduled_pool(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )
    for _ in range(20):
        planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
        _finish_round(
            service,
            planned,
            tmp_path / "predictions",
            baseline_resolved=False,
            evolved_resolved=False,
        )

    with pytest.raises(ABContractError, match="pending ChangeSet"):
        service.plan_round(partition="promotion", evolved_agent_sha256="c" * 64)

    forward = tmp_path / "forward.patch"
    rollback = tmp_path / "rollback.patch"
    forward.write_text("forward\n", encoding="utf-8")
    rollback.write_text("rollback\n", encoding="utf-8")
    proposal = service.propose_changeset(
        candidate_agent_sha256="c" * 64,
        forward_patch=forward,
        rollback_patch=rollback,
    )
    with pytest.raises(ABContractError, match="pending ChangeSet candidate"):
        service.plan_round(partition="promotion", evolved_agent_sha256="b" * 64)

    for _ in range(10):
        planned = service.plan_round(
            partition="promotion", evolved_agent_sha256="c" * 64
        )
        _finish_round(
            service,
            planned,
            tmp_path / "predictions",
            baseline_resolved=False,
            evolved_resolved=True,
        )
    decision = service.decide_changeset(proposal["proposal_id"])

    assert decision["status"] == "promoted"
    assert decision["active_agent_sha256"] == "c" * 64


def test_final_sealed_opens_once_after_search_and_freezes_candidate(tmp_path: Path):
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id="fixture",
            revision="fixture-frozen-v1",
            executable=True,
            tasks=tuple(_task(index) for index in range(60)),
        )
    )
    pool_path = tmp_path / "TASK_POOL.json"
    TaskPool.build(
        registry=registry,
        seed_material="sealed-test",
        target_count=60,
        promotion_count=0,
        final_sealed_count=60,
    ).save(pool_path)
    service = ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )

    opened = service.open_final_sealed(candidate_sha256="b" * 64)

    assert opened["task_count"] == 60
    with pytest.raises(ABContractError, match="already opened"):
        service.open_final_sealed(candidate_sha256="b" * 64)
    with pytest.raises(ABContractError, match="frozen final candidate"):
        service.plan_round(partition="final_sealed", evolved_agent_sha256="c" * 64)
    planned = service.plan_round(
        partition="final_sealed", evolved_agent_sha256="b" * 64
    )
    assert planned["partition"] == "final_sealed"
