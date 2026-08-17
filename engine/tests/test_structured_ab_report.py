from __future__ import annotations

from structured_ab_report import evaluate_structured_ab


def _report(mode: str, public: float, holdout: float, unique_behaviors: int) -> dict:
    return {
        "run": {
            "run_id": f"{mode}-seed",
            "task_id": "task",
            "config_hash": "config",
            "evaluator_hash": "evaluator",
            "initial_hash": "initial",
            "model_id": "model",
            "iterations_requested_total": 10,
            "experience_mode": "off",
            "experiment_seed": 20260802,
            "initial_holdout_score": 0.0,
            "final_holdout_score": holdout,
            "best_public_score": public,
            "proposal_controller_id": f"{mode}-v4",
            "proposal_controller_mode": mode,
            "proposal_controller_sha256": f"{mode}-config",
            "proposal_controller_implementation_sha256": "same-implementation",
            "proposal_controller_calls_per_request": 2,
            "proposal_controller_endpoint_verified": True,
        },
        "admission": {
            "candidate_attempts": 10,
            "unique_behavior_signatures": unique_behaviors,
            "accepted_parent_regressions": 0,
        },
    }


def _stats(mode: str, calls: int = 20) -> dict:
    return {
        "mode": mode,
        "protocol_version": "structured-mutation-v4",
        "requests": 10,
        "upstream_calls": calls,
        "structured_plans": 7 if mode == "structured-mutation" else 0,
        "deterministic_transforms": 7 if mode == "structured-mutation" else 0,
        "model_repairs_selected": 5 if mode == "structured-mutation" else 0,
        "deterministic_fallbacks": 2 if mode == "structured-mutation" else 0,
    }


def _audits(hashes: list[str]) -> list[dict]:
    return [{"selected_ast_sha256": value} for value in hashes]


def test_single_seed_capability_can_pass_but_cannot_formally_promote():
    result = evaluate_structured_ab(
        _report("planner-control", 0.23, 0.0, 1),
        _report("structured-mutation", 0.46, 0.0, 4),
        _stats("planner-control"),
        _stats("structured-mutation"),
        _audits(["same"] * 10),
        _audits(["a", "a", "b", "c", "c", "d", "e", "e", "f", "f"]),
        completed_seed_count=1,
    )

    assert result["fixed_call_budget"] is True
    assert result["structural_novelty_improved"] is True
    assert result["capability_trial_pass"] is True
    assert result["formal_seed_requirement_met"] is False
    assert result["promotion_decision"] == "capability_pass_formal_pending"


def test_structured_ab_rejects_unequal_model_call_budget():
    result = evaluate_structured_ab(
        _report("planner-control", 0.23, 0.0, 1),
        _report("structured-mutation", 0.46, 0.0, 4),
        _stats("planner-control"),
        _stats("structured-mutation", calls=22),
        _audits(["same"] * 10),
        _audits(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        completed_seed_count=1,
    )

    assert result["fixed_call_budget"] is False
    assert result["capability_trial_pass"] is False
    assert result["promotion_decision"] == "rejected"


def test_structured_ab_rejects_mismatched_experiment_seed():
    control = _report("planner-control", 0.23, 0.0, 1)
    treatment = _report("structured-mutation", 0.46, 0.0, 4)
    treatment["run"]["experiment_seed"] = 20260803

    result = evaluate_structured_ab(
        control,
        treatment,
        _stats("planner-control"),
        _stats("structured-mutation"),
        _audits(["same"] * 10),
        _audits(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]),
        completed_seed_count=1,
    )

    assert "experiment_seed" in result["contract_mismatches"]
    assert result["capability_trial_pass"] is False
