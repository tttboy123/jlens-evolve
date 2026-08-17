"""No-model pass^3 smoke for the v2.1 continuous matched A/B protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from benchmark_adapters import TaskPool
from continuous_ab import (
    ABContractError,
    ArmResult,
    BaselineContract,
    ChangeSetCadence,
    ChangeSetRegistry,
    MatchedRoundLedger,
    PermanentBaselineAuthority,
    PromotionGate,
)
from continuous_ab_service import ContinuousABService


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_smoke(stage_dir: Path, output_dir: Path) -> dict[str, Any]:
    stage_dir = stage_dir.resolve()
    output_dir = output_dir.resolve()
    frozen_pool_path = stage_dir / "configs/benchmark-pool/TASK_POOL.json"
    source_pool_sha256_before = _sha256_file(frozen_pool_path)
    ready_runtime = stage_dir / "runs/ready/runtime"
    baseline_payload = PermanentBaselineAuthority(
        ready_runtime / "permanent-baseline.json"
    ).load()
    baseline = BaselineContract.from_dict(baseline_payload["baseline"])
    seed_candidate = json.loads(
        (stage_dir / "configs/SEED_CANDIDATE.json").read_text(encoding="utf-8")
    )
    service = ContinuousABService.initialize(
        output_dir / "runtime",
        frozen_pool_path=frozen_pool_path,
        baseline=baseline,
        initial_active_agent_sha256=baseline.agent_program_sha256,
    )
    planned = service.plan_round(
        partition="search",
        evolved_agent_sha256=seed_candidate["candidate_agent_sha256"],
    )
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    baseline_prediction = prediction_dir / "baseline.patch"
    evolved_prediction = prediction_dir / "evolved.patch"
    baseline_prediction.write_text(
        "# protocol smoke placeholder; not a benchmark answer\n",
        encoding="utf-8",
    )
    evolved_prediction.write_text(
        "# protocol smoke placeholder; not a benchmark answer\n",
        encoding="utf-8",
    )
    service.freeze_predictions(
        planned["round_id"],
        {
            "baseline": baseline_prediction,
            "evolved": evolved_prediction,
        },
    )
    task_family = planned["task_contract"]["task_family"]
    neutral_pair = {
        arm: ArmResult(
            arm=arm,
            resolved=False,
            regression_failures=0,
            safe=True,
            input_tokens=100,
            output_tokens=50,
            elapsed_seconds=1.0,
            benchmark_family=task_family,
            evaluator_epoch=baseline.evaluator_epoch,
        )
        for arm in ("baseline", "evolved")
    }
    service.record_results(planned["round_id"], neutral_pair)
    restored = ContinuousABService.load(output_dir / "runtime")
    round_state = restored.round_state(planned["round_id"])
    runtime_pool = TaskPool.load(output_dir / "runtime/TASK_POOL.json")
    retired = next(
        record
        for record in runtime_pool.records
        if record.task_uid == planned["task_uid"]
    )
    baseline_replacement_rejected = False
    try:
        PermanentBaselineAuthority(
            output_dir / "runtime/permanent-baseline.json"
        ).freeze(baseline.replace(agent_program_sha256="f" * 64))
    except ABContractError:
        baseline_replacement_rejected = True
    promotion = PromotionGate().evaluate(
        [(neutral_pair["baseline"], neutral_pair["evolved"])]
    )
    early_promotion_rejected = False
    try:
        service.plan_round(
            partition="promotion",
            evolved_agent_sha256=seed_candidate["candidate_agent_sha256"],
        )
    except ABContractError:
        early_promotion_rejected = True

    crash_round = service.plan_round(
        partition="search",
        evolved_agent_sha256=seed_candidate["candidate_agent_sha256"],
    )
    crash_pool = TaskPool.load(output_dir / "runtime/TASK_POOL.json")
    crash_record = next(
        record
        for record in crash_pool.records
        if record.task_uid == crash_round["task_uid"]
    )
    crash_record.state = "unopened"
    crash_pool.save(output_dir / "runtime/TASK_POOL.json")
    service.reconcile_all_rounds()
    reconciled_pool = TaskPool.load(output_dir / "runtime/TASK_POOL.json")
    planned_round_crash_reconciled = (
        next(
            record
            for record in reconciled_pool.records
            if record.task_uid == crash_round["task_uid"]
        ).state
        == "search"
    )

    probe_dir = output_dir / "tamper-probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_baseline = probe_dir / "baseline.patch"
    probe_evolved = probe_dir / "evolved.patch"
    probe_baseline.write_text("baseline\n", encoding="utf-8")
    probe_evolved.write_text("evolved\n", encoding="utf-8")
    probe_ledger = MatchedRoundLedger.create(
        probe_dir / "round.json",
        round_id="tamper-probe",
        task_uid="tamper-probe-task",
        baseline=baseline,
        evolved_agent_sha256=seed_candidate["candidate_agent_sha256"],
    )
    probe_ledger.freeze_predictions(
        {"baseline": probe_baseline, "evolved": probe_evolved}
    )
    probe_baseline.write_text("tampered\n", encoding="utf-8")
    prediction_tamper_rejected = False
    try:
        probe_ledger.record_results(neutral_pair)
    except ABContractError:
        prediction_tamper_rejected = True

    tampered_registry = probe_dir / "changesets.json"
    shutil.copyfile(output_dir / "runtime/changesets.json", tampered_registry)
    registry_payload = json.loads(tampered_registry.read_text(encoding="utf-8"))
    registry_payload["active_agent_sha256"] = "f" * 64
    _atomic_json(tampered_registry, registry_payload)
    registry_tamper_rejected = False
    try:
        ChangeSetRegistry.load(tampered_registry)
    except ABContractError:
        registry_tamper_rejected = True
    checks = {
        "frozen_pool_unchanged": _sha256_file(frozen_pool_path)
        == source_pool_sha256_before,
        "matched_pair_completed": round_state["phase"] == "retired"
        and round_state["events"][-2]["matched"] is True,
        "prediction_pair_frozen": len(round_state["predictions"]) == 2
        and all(item["frozen"] for item in round_state["predictions"]),
        "runtime_task_retired": retired.state == "retired",
        "retired_task_cannot_reenter": not any(
            record.task_uid == planned["task_uid"] and record.state == "unopened"
            for record in runtime_pool.records
        ),
        "permanent_baseline_replacement_rejected": baseline_replacement_rejected,
        "changeset_not_allowed_before_round_10": not ChangeSetCadence(
            interval=10
        ).can_propose(completed_rounds=9, last_proposal_round=0),
        "neutral_candidate_not_promoted": promotion.approved is False,
        "prediction_tamper_rejected": prediction_tamper_rejected,
        "planned_round_crash_reconciled": planned_round_crash_reconciled,
        "early_promotion_rejected": early_promotion_rejected,
        "registry_tamper_rejected": registry_tamper_rejected,
        "final_sealed_remains_unopened": sum(
            record.assigned_partition == "final_sealed" and record.state == "unopened"
            for record in runtime_pool.records
        )
        == 60,
    }
    facts = {
        "completed_rounds": restored.completed_round_count,
        "task_uid": planned["task_uid"],
        "benchmark_id": planned["benchmark_id"],
        "baseline_agent_sha256": baseline.agent_program_sha256,
        "evolved_agent_sha256": seed_candidate["candidate_agent_sha256"],
        "wins": promotion.wins,
        "losses": promotion.losses,
        "ties": promotion.ties,
    }
    fingerprint_payload = {"checks": checks, "facts": facts}
    result = {
        "schema_version": "1.0",
        "status": "protocol_verified" if all(checks.values()) else "rejected",
        "quality_claim": "none",
        "explanation": (
            "This smoke uses deterministic placeholder predictions and neutral results. "
            "It proves lifecycle and recovery wiring only; it does not evaluate an Agent."
        ),
        "checks": checks,
        "facts": facts,
        "outcome_fingerprint": _sha256_json(fingerprint_payload),
    }
    _atomic_json(output_dir / "SMOKE_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    if args.passes < 1:
        raise ABContractError("passes must be positive")
    results = [
        run_smoke(args.stage_dir, args.output_root / f"pass-{index}")
        for index in range(1, args.passes + 1)
    ]
    fingerprints = [result["outcome_fingerprint"] for result in results]
    aggregate = {
        "schema_version": "1.0",
        "status": (
            "pass3_verified"
            if len(results) == 3
            and len(set(fingerprints)) == 1
            and all(result["status"] == "protocol_verified" for result in results)
            else "rejected"
        ),
        "pass_count": len(results),
        "stable_outcome": len(set(fingerprints)) == 1,
        "quality_claim": "none",
        "outcome_fingerprints": fingerprints,
    }
    _atomic_json(args.output_root / "PASS3_AGGREGATE.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0 if aggregate["status"] == "pass3_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
