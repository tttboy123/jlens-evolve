"""Build a zero-side-effect pilot plan and enforce explicit paid-run caps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark_adapters import TaskPool
from continuous_ab import BaselineContract, PermanentBaselineAuthority


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_pilot_plan(stage_dir: Path) -> dict[str, Any]:
    """Select identities only; task state and task content remain unopened."""

    stage_dir = stage_dir.resolve()
    pool = TaskPool.load(stage_dir / "configs/benchmark-pool/TASK_POOL.json")
    baseline_payload = PermanentBaselineAuthority(
        stage_dir / "runs/ready/runtime/permanent-baseline.json"
    ).load()
    baseline = BaselineContract.from_dict(baseline_payload["baseline"])
    candidate = json.loads(
        (stage_dir / "configs/SEED_CANDIDATE.json").read_text(encoding="utf-8")
    )
    search = [
        record
        for record in pool.records
        if record.assigned_partition == "search" and record.state == "unopened"
    ][:10]
    if len(search) != 10 or len({record.benchmark_id for record in search}) != 4:
        raise ValueError("pilot requires ten unopened tasks spanning four adapters")

    baseline_arm = baseline.to_dict()
    evolved_arm = baseline.replace(
        agent_program_sha256=candidate["candidate_agent_sha256"]
    ).to_dict()
    matched_keys = set(baseline_arm) - {"agent_program_sha256"}
    matched = all(baseline_arm[key] == evolved_arm[key] for key in matched_keys)
    rounds = []
    forbidden = {"patch", "fix_patch", "test_patch", "problem_statement"}
    for index, record in enumerate(search, start=1):
        if forbidden & set(record.task_contract):
            raise ValueError("pilot scheduler contract exposes task content or gold")
        contract = {
            "round_id": f"round-{index:06d}",
            "task_uid": record.task_uid,
            "benchmark_id": record.benchmark_id,
            "instance_id": record.instance_id,
            "partition": "search",
            "baseline_contract_sha256": baseline.contract_sha256,
            "baseline_agent_sha256": baseline.agent_program_sha256,
            "evolved_agent_sha256": candidate["candidate_agent_sha256"],
            "matched_contract": matched,
        }
        contract["round_contract_sha256"] = _sha256_json(contract)
        rounds.append(contract)
    plan = {
        "schema_version": "1.0",
        "status": "predeclared_no_tasks_opened",
        "partition": "search",
        "matched_rounds": 10,
        "maximum_agent_calls": 20,
        "changeset_calls": 0,
        "promotion_tasks_opened": 0,
        "final_sealed_tasks_opened": 0,
        "baseline_contract_sha256": baseline.contract_sha256,
        "rounds": rounds,
    }
    plan["plan_sha256"] = _sha256_json(plan)
    return plan


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def evaluate_authorization(
    authorization: dict[str, Any], budget_gate: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed unless the user supplied all three explicit external caps."""

    recommended = budget_gate["recommended_first_authorization"]
    failures = []
    if (
        not isinstance(authorization.get("authorization_id"), str)
        or not authorization["authorization_id"].strip()
    ):
        failures.append("authorization_id_missing")
    if authorization.get("permission_to_consume_real_codex_calls") is not True:
        failures.append("codex_call_permission_missing")
    if (
        authorization.get("permission_to_create_and_release_ephemeral_cloud_instance")
        is not True
    ):
        failures.append("ephemeral_cloud_permission_missing")

    calls = _positive_number(authorization.get("maximum_agent_calls"))
    cloud = _positive_number(authorization.get("cloud_budget_cap_cny"))
    hours = _positive_number(authorization.get("cloud_runtime_cap_hours"))
    instances = _positive_number(authorization.get("instance_count_cap"))
    if calls is None or calls < recommended["maximum_agent_calls"]:
        failures.append("agent_call_cap_below_20_or_missing")
    if cloud is None:
        failures.append("cloud_budget_cap_missing")
    if hours is None:
        failures.append("cloud_runtime_cap_missing")
    if instances is None:
        failures.append("instance_count_cap_missing")

    effective = None
    if not failures:
        effective = {
            "maximum_agent_calls": int(recommended["maximum_agent_calls"]),
            "cloud_budget_cap_cny": min(
                float(recommended["cloud_budget_cap_cny"]), cloud
            ),
            "cloud_runtime_cap_hours": min(
                float(recommended["cloud_runtime_cap_hours"]), hours
            ),
            "instance_count_cap": min(
                int(recommended["instance_count_cap"]), int(instances)
            ),
        }
    return {
        "schema_version": "1.0",
        "status": "HUMAN_REQUIRED" if failures else "authorized",
        "authorization_id": authorization.get("authorization_id"),
        "failures": failures,
        "effective_caps": effective,
        "paid_actions_dispatched": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    plan = build_pilot_plan(args.stage_dir)
    budget = json.loads((args.stage_dir / "BUDGET_GATE.json").read_text())
    authorization = (
        json.loads(args.authorization.read_text()) if args.authorization else {}
    )
    preflight = evaluate_authorization(authorization, budget)
    if args.plan_output:
        _write_json(args.plan_output, plan)
    if args.preflight_output:
        _write_json(args.preflight_output, preflight)
    print(json.dumps({"plan": plan, "preflight": preflight}, indent=2))
    return 0 if preflight["status"] == "authorized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
