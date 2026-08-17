from __future__ import annotations

from structured_multiseed_report import evaluate_multiseed


def _seed_result(seed: int, *, treatment_duplicates: int = 4) -> dict:
    return {
        "schema_version": 1,
        "structured_ab": {
            "experiment_seed": seed,
            "contract_matched": True,
            "controller_binding_valid": True,
            "fixed_call_budget": True,
            "operator_execution_valid": True,
            "performance_noninferior": True,
            "no_accepted_regression": True,
            "capability_trial_pass": True,
            "control_candidate_attempts": 10,
            "treatment_candidate_attempts": 10,
            "control_structural_duplicates": 9,
            "treatment_structural_duplicates": treatment_duplicates,
            "control_structural_duplicate_rate": 0.9,
            "treatment_structural_duplicate_rate": treatment_duplicates / 10,
            "control_unique_asts": 1,
            "treatment_unique_asts": 6,
            "control_unique_behaviors": 1,
            "treatment_unique_behaviors": 4,
            "control_public_score": 0.23,
            "treatment_public_score": 0.46,
            "control_holdout_score": 0.0,
            "treatment_holdout_score": 0.0,
        },
    }


def test_three_distinct_passing_seeds_approve_formal_promotion():
    result = evaluate_multiseed(
        [_seed_result(20260802), _seed_result(20260803), _seed_result(20260804)]
    )

    assert result["seed_count"] == 3
    assert result["distinct_seed_count"] == 3
    assert result["pooled_duplicate_rate_improved"] is True
    assert result["nonhigher_duplicate_seed_count"] == 3
    assert result["formal_promotion_pass"] is True
    assert result["promotion_decision"] == "approved"


def test_duplicate_seed_or_failed_performance_rejects_formal_promotion():
    duplicate_seed = _seed_result(20260802)
    bad = _seed_result(20260803)
    bad["structured_ab"]["performance_noninferior"] = False
    bad["structured_ab"]["capability_trial_pass"] = False

    result = evaluate_multiseed([_seed_result(20260802), duplicate_seed, bad])

    assert result["distinct_seed_count"] == 2
    assert result["all_seed_capability_pass"] is False
    assert result["formal_promotion_pass"] is False
    assert result["promotion_decision"] == "rejected"


def test_formal_gate_allows_one_nonworse_seed_without_strict_capability_gain():
    neutral = _seed_result(20260804, treatment_duplicates=9)
    neutral["structured_ab"]["capability_trial_pass"] = False

    result = evaluate_multiseed(
        [_seed_result(20260802), _seed_result(20260803), neutral]
    )

    assert result["all_seed_capability_pass"] is False
    assert result["nonhigher_duplicate_seed_count"] == 3
    assert result["pooled_duplicate_rate_improved"] is True
    assert result["formal_promotion_pass"] is True
