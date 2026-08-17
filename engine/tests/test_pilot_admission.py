from __future__ import annotations

import hashlib
from pathlib import Path

from pilot_admission import build_pilot_plan, evaluate_authorization


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pilot_plan_predeclares_ten_matched_search_tasks_without_opening_pool():
    root = Path(__file__).resolve().parents[1]
    stage = root / "artifacts/v2.1.0/v2.1.0-continuous-ab"
    pool_path = stage / "configs/benchmark-pool/TASK_POOL.json"
    before = _sha256(pool_path)

    plan = build_pilot_plan(stage)

    assert plan["status"] == "predeclared_no_tasks_opened"
    assert plan["matched_rounds"] == 10
    assert plan["maximum_agent_calls"] == 20
    assert len(plan["rounds"]) == 10
    assert {row["partition"] for row in plan["rounds"]} == {"search"}
    assert {row["benchmark_id"] for row in plan["rounds"]} == {
        "swe-bench-verified",
        "swe-bench-multilingual",
        "multi-swe-bench-flash",
        "terminal-bench-2",
    }
    assert all(row["baseline_contract_sha256"] for row in plan["rounds"])
    assert all(row["matched_contract"] is True for row in plan["rounds"])
    assert _sha256(pool_path) == before


def test_authorization_remains_human_required_until_all_hard_caps_are_explicit():
    budget_gate = {
        "recommended_first_authorization": {
            "maximum_agent_calls": 20,
            "cloud_budget_cap_cny": 200,
            "cloud_runtime_cap_hours": 24,
            "instance_count_cap": 1,
        }
    }
    missing = evaluate_authorization({}, budget_gate)
    approved = evaluate_authorization(
        {
            "authorization_id": "user-pilot-001",
            "permission_to_consume_real_codex_calls": True,
            "permission_to_create_and_release_ephemeral_cloud_instance": True,
            "maximum_agent_calls": 20,
            "cloud_budget_cap_cny": 200,
            "cloud_runtime_cap_hours": 24,
            "instance_count_cap": 1,
        },
        budget_gate,
    )

    assert missing["status"] == "HUMAN_REQUIRED"
    assert approved["status"] == "authorized"
    assert approved["effective_caps"] == {
        "maximum_agent_calls": 20,
        "cloud_budget_cap_cny": 200.0,
        "cloud_runtime_cap_hours": 24.0,
        "instance_count_cap": 1,
    }
