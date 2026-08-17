from __future__ import annotations

import pytest

from self_improvement_eval import evaluate_psi, evaluate_psi_ab, evaluate_rsi


def test_rsi_requires_recursive_gain_and_operator_improvement():
    events = [
        {"iteration": 1, "accepted": True, "parent_score": 0.2, "child_score": 0.3},
        {"iteration": 2, "accepted": True, "parent_score": 0.3, "child_score": 0.5},
        {
            "iteration": 3,
            "accepted": True,
            "parent_score": 0.5,
            "child_score": 0.7,
            "operator_revision": "policy-v2",
            "pre_revision_yield": 0.2,
            "post_revision_yield": 0.5,
        },
    ]

    result = evaluate_rsi(events)

    assert result["strict_improvement_depth"] == 3
    assert result["operator_improved"] is True
    assert result["rsi_pass"] is True


def test_rsi_does_not_mislabel_candidate_search_as_recursive():
    result = evaluate_rsi(
        [{"iteration": 1, "accepted": True, "parent_score": 0.2, "child_score": 0.5}]
    )

    assert result["candidate_improved"] is True
    assert result["operator_improved"] is False
    assert result["rsi_pass"] is False


def test_psi_reports_resume_and_cross_task_transfer_separately():
    manifests = [
        {
            "run_id": "r1",
            "task_id": "a",
            "resumed_from": None,
            "initial_holdout_score": 0.2,
            "final_holdout_score": 0.4,
            "retrieved_lesson_sources": [],
        },
        {
            "run_id": "r2",
            "task_id": "b",
            "resumed_from": "checkpoint_10",
            "resume_parent_run_id": "r1",
            "pre_resume_best_score": 0.4,
            "final_holdout_score": 0.5,
            "initial_holdout_score": 0.3,
            "retrieved_lesson_sources": [{"task_id": "a", "lesson_id": "l1"}],
        },
    ]

    result = evaluate_psi(manifests)

    assert result["same_search_resume_pass"] is True
    assert result["cross_task_transfer_observed"] is True
    assert result["cross_task_holdout_gain"] == 0.2
    assert result["psi_pass"] is True


def test_psi_ab_requires_matched_contract_and_cross_task_provenance():
    common = {
        "psi_experiment_id": "payout-ab-1",
        "task_id": "payout-record-cleaning-v1",
        "task_family": "record-cleaning",
        "config_hash": "same",
        "evaluator_hash": "same",
        "initial_hash": "same",
        "search_protocol_hash": "same",
        "model_id": "coder-7b",
        "iterations_requested": 20,
        "initial_holdout_score": 0.0,
    }
    rows = [
        {
            **common,
            "run_id": "control",
            "psi_arm": "control",
            "experience_mode": "off",
            "best_public_score": 0.7,
            "final_holdout_score": 0.5,
            "retrieved_lesson_sources": [],
        },
        {
            **common,
            "run_id": "transfer",
            "psi_arm": "transfer",
            "experience_mode": "cross-task",
            "best_public_score": 0.8,
            "final_holdout_score": 2 / 3,
            "retrieved_lesson_sources": [
                {"task_id": "transaction-record-cleaning-v1", "lesson_id": "l1"}
            ],
        },
    ]

    result = evaluate_psi_ab(rows, experiment_id="payout-ab-1")

    assert result["contract_matched"] is True
    assert result["cross_task_provenance"] is True
    assert result["public_score_delta_vs_control"] == pytest.approx(0.1)
    assert result["holdout_delta_vs_control"] == pytest.approx(1 / 6)
    assert result["strict_transfer_benefit"] is True
    assert result["psi_ab_pass"] is True


def test_psi_ab_rejects_mismatched_protocol_even_when_transfer_scores_higher():
    rows = [
        {
            "psi_experiment_id": "ab",
            "psi_arm": "control",
            "task_id": "task-b",
            "search_protocol_hash": "one",
            "initial_holdout_score": 0.0,
            "final_holdout_score": 0.1,
            "retrieved_lesson_sources": [],
        },
        {
            "psi_experiment_id": "ab",
            "psi_arm": "transfer",
            "task_id": "task-b",
            "search_protocol_hash": "two",
            "initial_holdout_score": 0.0,
            "final_holdout_score": 1.0,
            "retrieved_lesson_sources": [{"task_id": "task-a", "lesson_id": "lesson"}],
        },
    ]

    result = evaluate_psi_ab(rows, experiment_id="ab")

    assert result["contract_matched"] is False
    assert result["psi_ab_pass"] is False
