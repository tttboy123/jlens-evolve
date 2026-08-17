from __future__ import annotations

import pytest

from experiment_report import (
    compare_model_runs,
    summarize_psi,
    verify_experience_snapshot,
)


def report(model: str, public: float, holdout: float, duration: float):
    return {
        "run": {
            "task_id": "transaction-record-cleaning-v1",
            "model_id": model,
            "search_protocol_hash": "protocol",
            "evaluator_hash": "evaluator",
            "initial_hash": "initial",
            "iterations_requested": 20,
            "experience_mode": "off",
            "operator_policy_id": "focused-v1",
            "best_public_score": public,
            "final_holdout_score": holdout,
            "duration_seconds": duration,
        },
        "admission": {
            "candidate_attempts": 20,
            "accepted": 4,
            "rejected": 16,
            "accept_rate": 0.2,
            "accepted_parent_regressions": 0,
            "unique_behavior_signatures": 5,
            "best_passed_cases": 12,
        },
    }


def test_model_comparison_requires_model_only_matched_protocol():
    control = report("general-4b", 0.7, 0.5, 10)
    treatment = report("coder-7b", 0.9, 2 / 3, 20)

    result = compare_model_runs(control, treatment)

    assert result["contract_matched"] is True
    assert result["models_differ"] is True
    assert result["public_score_delta"] == pytest.approx(0.2)
    assert result["holdout_score_delta"] == pytest.approx(1 / 6)
    assert result["public_winner"] == "treatment"


def test_model_comparison_rejects_changed_evaluator():
    control = report("general-4b", 0.7, 0.5, 10)
    treatment = report("coder-7b", 0.9, 0.6, 20)
    treatment["run"]["evaluator_hash"] = "changed"

    result = compare_model_runs(control, treatment)

    assert result["contract_matched"] is False
    assert "evaluator_hash" in result["contract_mismatches"]


def test_psi_summary_requires_resume_and_matched_cross_task_ab():
    resume_report = {"psi": {"same_search_resume_pass": True, "resume_trials": 4}}
    psi_ab = {
        "psi_ab_pass": True,
        "strict_transfer_benefit": False,
    }

    result = summarize_psi(resume_report, psi_ab)

    assert result == {
        "definition": "persistent self-improvement",
        "same_search_resume_pass": True,
        "resume_trials": 4,
        "cross_task_ab_pass": True,
        "strict_transfer_benefit": False,
        "psi_pass": True,
    }


def test_psi_summary_fails_when_cross_task_ab_is_not_noninferior():
    resume_report = {"psi": {"same_search_resume_pass": True, "resume_trials": 4}}

    result = summarize_psi(
        resume_report,
        {"psi_ab_pass": False, "strict_transfer_benefit": False},
    )

    assert result["psi_pass"] is False


def test_experience_snapshot_accepts_append_only_control_and_unchanged_transfer(
    tmp_path,
):
    source = tmp_path / "source.jsonl"
    control = tmp_path / "control.jsonl"
    transfer = tmp_path / "transfer.jsonl"
    source.write_bytes(b'{"lesson":"source"}\n')
    control.write_bytes(source.read_bytes() + b'{"lesson":"control-result"}\n')
    transfer.write_bytes(source.read_bytes())

    result = verify_experience_snapshot(source, control, transfer)

    assert result["snapshot_matched"] is True
    assert result["control_prefix_matches"] is True
    assert result["transfer_prefix_matches"] is True
    assert len(result["source_sha256"]) == 64
