from __future__ import annotations

from agent_ab_report import evaluate_agent_ab


def _report(
    *,
    policy: str,
    strategy_id: str | None,
    public: float,
    holdout: float,
    unique_sources: int,
    exact_duplicates: int,
    policy_schedule: list[str] | None = None,
):
    return {
        "run": {
            "run_id": policy,
            "task_id": "task",
            "config_hash": "config",
            "evaluator_hash": "evaluator",
            "initial_hash": "initial",
            "model_id": "model",
            "iterations_requested": 10,
            "iterations_requested_total": 10,
            "experience_mode": "off",
            "initial_holdout_score": 0.0,
            "final_holdout_score": holdout,
            "best_public_score": public,
            "operator_policy_id": policy,
            "operator_policy_schedule": policy_schedule or [policy],
            "agent_strategy_id": strategy_id,
            "agent_strategy_sha256": "strategy-hash" if strategy_id else None,
        },
        "admission": {
            "candidate_attempts": 10,
            "unique_source_hashes": unique_sources,
            "unique_ast_hashes": unique_sources,
            "unique_behavior_signatures": 3,
            "accepted_parent_regressions": 0,
            "rejection_reasons": {"exact_duplicate": exact_duplicates},
        },
    }


def test_agent_ab_promotes_only_noninferior_more_diverse_treatment():
    control = _report(
        policy="focused-v1",
        strategy_id=None,
        public=0.7,
        holdout=0.5,
        unique_sources=3,
        exact_duplicates=5,
    )
    treatment = _report(
        policy="jlens-guided-v1",
        strategy_id="jlens-agent",
        public=0.8,
        holdout=2 / 3,
        unique_sources=5,
        exact_duplicates=1,
    )

    result = evaluate_agent_ab(control, treatment)

    assert result["contract_matched"] is True
    assert result["strategy_binding_valid"] is True
    assert result["performance_noninferior"] is True
    assert result["diversity_improved"] is True
    assert result["agent_optimization_pass"] is True


def test_agent_ab_rejects_diversity_gain_that_hurts_holdout():
    control = _report(
        policy="focused-v1",
        strategy_id=None,
        public=0.7,
        holdout=0.5,
        unique_sources=3,
        exact_duplicates=5,
    )
    treatment = _report(
        policy="jlens-guided-v1",
        strategy_id="jlens-agent",
        public=0.8,
        holdout=0.0,
        unique_sources=5,
        exact_duplicates=1,
    )

    result = evaluate_agent_ab(control, treatment)

    assert result["diversity_improved"] is True
    assert result["performance_noninferior"] is False
    assert result["agent_optimization_pass"] is False


def test_agent_ab_accepts_audited_bootstrap_then_jlens_schedule():
    control = _report(
        policy="focused-v1",
        strategy_id=None,
        public=0.7,
        holdout=0.5,
        unique_sources=3,
        exact_duplicates=5,
    )
    treatment = _report(
        policy="jlens-guided-v1",
        policy_schedule=["focused-v1", "jlens-guided-v1"],
        strategy_id="jlens-agent",
        public=0.8,
        holdout=2 / 3,
        unique_sources=5,
        exact_duplicates=1,
    )

    result = evaluate_agent_ab(control, treatment)

    assert result["contract_matched"] is True
    assert result["strategy_binding_valid"] is True
    assert result["treatment_policy_schedule"] == [
        "focused-v1",
        "jlens-guided-v1",
    ]
