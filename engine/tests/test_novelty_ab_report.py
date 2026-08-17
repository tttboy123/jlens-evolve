from __future__ import annotations

from novelty_ab_report import evaluate_novelty_ab


def _report(
    *,
    mode: str,
    public: float,
    holdout: float,
    exact: int,
    ast_duplicates: int,
    unique_asts: int,
    unique_behaviors: int,
) -> dict:
    return {
        "run": {
            "run_id": mode,
            "task_id": "task",
            "config_hash": "config",
            "evaluator_hash": "evaluator",
            "initial_hash": "initial",
            "model_id": "model",
            "iterations_requested_total": 10,
            "experience_mode": "off",
            "initial_holdout_score": 0.0,
            "final_holdout_score": holdout,
            "best_public_score": public,
            "proposal_controller_id": f"{mode}-v2",
            "proposal_controller_mode": mode,
            "proposal_controller_sha256": f"{mode}-hash",
            "proposal_controller_calls_per_request": 2,
            "proposal_controller_endpoint_verified": True,
            "proposal_controller_implementation_sha256": f"{mode}-implementation",
        },
        "admission": {
            "candidate_attempts": 10,
            "unique_source_hashes": unique_asts,
            "unique_ast_hashes": unique_asts,
            "unique_behavior_signatures": unique_behaviors,
            "accepted_parent_regressions": 0,
            "rejection_reasons": {
                "exact_duplicate": exact,
                "ast_duplicate": ast_duplicates,
            },
        },
    }


def _stats(mode: str, calls: int = 20) -> dict:
    return {
        "mode": mode,
        "requests": 10,
        "upstream_calls": calls,
        "first_duplicates": 5,
        "retry_feedback_requests": 5 if mode == "duplicate-aware" else 0,
        "selected_second": 5 if mode == "duplicate-aware" else 0,
        "selected_novel": 8,
        "stagnation_triggers": 2 if mode == "duplicate-aware" else 0,
        "stagnation_detector_version": "global-best-v2",
    }


def test_novelty_ab_promotes_fixed_budget_noninferior_duplicate_reduction():
    control = _report(
        mode="shadow-control",
        public=0.8,
        holdout=0.5,
        exact=2,
        ast_duplicates=4,
        unique_asts=4,
        unique_behaviors=3,
    )
    treatment = _report(
        mode="duplicate-aware",
        public=0.8,
        holdout=2 / 3,
        exact=0,
        ast_duplicates=1,
        unique_asts=7,
        unique_behaviors=4,
    )

    result = evaluate_novelty_ab(
        control, treatment, _stats("shadow-control"), _stats("duplicate-aware")
    )

    assert result["contract_matched"] is True
    assert result["fixed_call_budget"] is True
    assert result["performance_noninferior"] is True
    assert result["structural_novelty_improved"] is True
    assert result["agent_optimization_pass"] is True


def test_novelty_ab_rejects_unequal_upstream_call_budget():
    control = _report(
        mode="shadow-control",
        public=0.8,
        holdout=0.5,
        exact=2,
        ast_duplicates=4,
        unique_asts=4,
        unique_behaviors=3,
    )
    treatment = _report(
        mode="duplicate-aware",
        public=0.9,
        holdout=2 / 3,
        exact=0,
        ast_duplicates=1,
        unique_asts=7,
        unique_behaviors=4,
    )

    result = evaluate_novelty_ab(
        control,
        treatment,
        _stats("shadow-control"),
        _stats("duplicate-aware", calls=22),
    )

    assert result["fixed_call_budget"] is False
    assert result["agent_optimization_pass"] is False


def test_novelty_ab_rejects_unbound_treatment_implementation():
    control = _report(
        mode="shadow-control",
        public=0.8,
        holdout=0.5,
        exact=2,
        ast_duplicates=4,
        unique_asts=4,
        unique_behaviors=3,
    )
    treatment = _report(
        mode="duplicate-aware",
        public=0.8,
        holdout=2 / 3,
        exact=0,
        ast_duplicates=1,
        unique_asts=7,
        unique_behaviors=4,
    )
    treatment["run"]["proposal_controller_implementation_sha256"] = None

    result = evaluate_novelty_ab(
        control, treatment, _stats("shadow-control"), _stats("duplicate-aware")
    )

    assert result["controller_binding_valid"] is False
    assert result["agent_optimization_pass"] is False
