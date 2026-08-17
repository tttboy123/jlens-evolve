from __future__ import annotations

from skill_evolution_loop.contracts import sha256_json
from skill_evolution_loop.statistical_capability_gate import (
    evaluate_statistical_capability_gate,
)


def _sealed(payload: dict[str, object], *, field: str) -> dict[str, object]:
    return {**payload, field: sha256_json(payload)}


def _feedback(*, gains: int = 1, failures: int = 0) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "status": "complete",
            "feedback_gain_count": gains,
            "native_evaluator_failure_count": failures,
        },
        field="summary_sha256",
    )


def _holdout(
    *, evaluable: int = 3, regressions: int = 0, failures: int = 0
) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "evaluation_scope": "round1-full-capability",
            "status": "complete",
            "full_capability_gate_evaluated": True,
            "holdout_evaluable_pair_count": evaluable,
            "holdout_regression_count": regressions,
            "native_evaluator_failure_count": failures,
        },
        field="summary_sha256",
    )


def _safety(*, passed: bool = True, failures: int = 0) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "contract": "independent-agent-safety-suite-v1",
            "suite_passed": passed,
            "evaluator_failure_count": failures,
            "native_admission_reused": False,
        },
        field="evidence_sha256",
    )


def _identity(*, complete: bool = True) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "complete": complete,
            "model_weight_sha256": "d" * 64,
            "quantization": "awq-4bit",
            "tokenizer_sha256": "e" * 64,
            "chat_template_sha256": "f" * 64,
            "runtime_digest": "sha256:" + "1" * 64,
            "generation_parameters_sha256": "2" * 64,
            "transport_sha256": "3" * 64,
            "skill_sha256": "4" * 64,
            "framework_sha256": "5" * 64,
        },
        field="evidence_sha256",
    )


def _cost(*, complete: bool = True) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "complete": complete,
            "estimated_total_cost_cny": 5.0,
            "cost_per_success_cny": 1.25,
            "wall_clock_seconds": 600,
            "instance_state": "STOPPED",
            "stopped_mode": "STOP_CHARGING",
            "instance_retained": True,
            "api_termination_protection": True,
            "residual_hourly_cost_cny": 0.0,
            "checkpoint_verified": True,
        },
        field="evidence_sha256",
    )


def _terminated_cost() -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "complete": True,
            "estimated_total_cost_cny": 5.3,
            "cost_per_success_cny": None,
            "wall_clock_seconds": 2574,
            "instance_state": "TERMINATED",
            "stopped_mode": "TERMINATE_RELEASE",
            "instance_retained": False,
            "api_termination_protection": False,
            "residual_hourly_cost_cny": 0.0,
            "checkpoint_verified": True,
            "residual_resources_verified_zero": True,
        },
        field="evidence_sha256",
    )


def _catalog(*, matched: bool = True) -> dict[str, object]:
    return _sealed(
        {
            "schema_version": 1,
            "record_count": 30,
            "all_evidence_references_match": matched,
        },
        field="evidence_sha256",
    )


def test_statistical_gate_requires_every_goal_condition() -> None:
    report = evaluate_statistical_capability_gate(
        feedback=_feedback(),
        holdout=_holdout(),
        independent_safety=_safety(),
        runtime_identity=_identity(),
        cost_receipt=_cost(),
        catalog_audit=_catalog(),
    )

    assert report["gate_passed"] is True
    assert report["minimum_evaluator_valid_holdout_pairs"] == 3
    assert report["failed_requirements"] == []
    assert all(report["requirements"].values())


def test_statistical_gate_rejects_terminate_release_closeout() -> None:
    report = evaluate_statistical_capability_gate(
        feedback=_feedback(),
        holdout=_holdout(),
        independent_safety=_safety(),
        runtime_identity=_identity(),
        cost_receipt=_terminated_cost(),
        catalog_audit=_catalog(),
    )

    assert report["requirements"]["cost_checkpoint_stop_complete"] is False
    assert "cost_checkpoint_stop_complete" in report["failed_requirements"]


def test_statistical_gate_rejects_safety_only_noop_holdout() -> None:
    report = evaluate_statistical_capability_gate(
        feedback=_feedback(),
        holdout=_holdout(evaluable=0),
        independent_safety=_safety(),
        runtime_identity=_identity(),
        cost_receipt=_cost(),
        catalog_audit=_catalog(),
    )

    assert report["gate_passed"] is False
    assert report["requirements"]["minimum_evaluator_valid_holdout_pairs"] is False
    assert report["failed_requirements"] == ["minimum_evaluator_valid_holdout_pairs"]


def test_statistical_gate_reports_all_missing_evidence_without_short_circuit() -> None:
    report = evaluate_statistical_capability_gate(
        feedback=_feedback(gains=0, failures=1),
        holdout=_holdout(evaluable=2, regressions=1, failures=1),
        independent_safety=_safety(passed=False, failures=1),
        runtime_identity=_identity(complete=False),
        cost_receipt=_cost(complete=False),
        catalog_audit=_catalog(matched=False),
    )

    assert report["gate_passed"] is False
    assert set(report["failed_requirements"]) == {
        "new_feedback_native_gain",
        "minimum_evaluator_valid_holdout_pairs",
        "zero_teaching_regressions",
        "zero_native_evaluator_failures",
        "independent_safety_passed",
        "runtime_identity_complete",
        "cost_checkpoint_stop_complete",
        "catalog_evidence_complete",
    }


def test_statistical_gate_rejects_boolean_numbers_and_missing_evidence_hashes() -> None:
    feedback = _feedback()
    feedback["feedback_gain_count"] = True
    feedback.pop("summary_sha256")
    cost = _cost()
    cost["residual_hourly_cost_cny"] = False
    cost.pop("evidence_sha256")

    report = evaluate_statistical_capability_gate(
        feedback=feedback,
        holdout=_holdout(),
        independent_safety=_safety(),
        runtime_identity=_identity(),
        cost_receipt=cost,
        catalog_audit=_catalog(),
    )

    assert report["gate_passed"] is False
    assert report["requirements"]["new_feedback_native_gain"] is False
    assert report["requirements"]["cost_checkpoint_stop_complete"] is False
    assert report["requirements"]["evidence_integrity_complete"] is False
