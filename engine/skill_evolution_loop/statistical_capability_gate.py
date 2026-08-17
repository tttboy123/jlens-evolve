"""Strict completion gate for statistically meaningful Skill transfer."""

from __future__ import annotations

import re
from typing import Any

from .contracts import sha256_json

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _sealed_evidence_valid(value: dict[str, Any], *, field: str) -> bool:
    digest = value.get(field)
    if value.get("schema_version") != 1 or not isinstance(digest, str):
        return False
    if _SHA256.fullmatch(digest) is None:
        return False
    content = {key: item for key, item in value.items() if key != field}
    return digest == sha256_json(content)


def evaluate_statistical_capability_gate(
    *,
    feedback: dict[str, Any],
    holdout: dict[str, Any],
    independent_safety: dict[str, Any],
    runtime_identity: dict[str, Any],
    cost_receipt: dict[str, Any],
    catalog_audit: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every Goal requirement without safety-only substitutions."""

    failure_counts = (
        feedback.get("native_evaluator_failure_count"),
        holdout.get("native_evaluator_failure_count"),
        independent_safety.get("evaluator_failure_count"),
    )
    failure_counts_valid = all(
        type(value) is int and value >= 0 for value in failure_counts
    )
    evaluator_failures = sum(failure_counts) if failure_counts_valid else None
    evidence_integrity_complete = all(
        (
            _sealed_evidence_valid(feedback, field="summary_sha256"),
            _sealed_evidence_valid(holdout, field="summary_sha256"),
            _sealed_evidence_valid(independent_safety, field="evidence_sha256"),
            _sealed_evidence_valid(runtime_identity, field="evidence_sha256"),
            _sealed_evidence_valid(cost_receipt, field="evidence_sha256"),
            _sealed_evidence_valid(catalog_audit, field="evidence_sha256"),
        )
    )
    requirements = {
        "new_feedback_native_gain": (
            feedback.get("status") == "complete"
            and type(feedback.get("feedback_gain_count")) is int
            and feedback["feedback_gain_count"] >= 1
        ),
        "minimum_evaluator_valid_holdout_pairs": (
            holdout.get("evaluation_scope") == "round1-full-capability"
            and holdout.get("status") == "complete"
            and holdout.get("full_capability_gate_evaluated") is True
            and type(holdout.get("holdout_evaluable_pair_count")) is int
            and holdout["holdout_evaluable_pair_count"] >= 3
        ),
        "zero_teaching_regressions": (
            type(holdout.get("holdout_regression_count")) is int
            and holdout["holdout_regression_count"] == 0
        ),
        "zero_native_evaluator_failures": (
            failure_counts_valid and evaluator_failures == 0
        ),
        "independent_safety_passed": (
            independent_safety.get("contract") == "independent-agent-safety-suite-v1"
            and independent_safety.get("suite_passed") is True
            and independent_safety.get("native_admission_reused") is False
        ),
        "runtime_identity_complete": runtime_identity.get("complete") is True,
        "cost_checkpoint_stop_complete": (
            cost_receipt.get("complete") is True
            and cost_receipt.get("checkpoint_verified") is True
            and type(cost_receipt.get("residual_hourly_cost_cny")) in (int, float)
            and cost_receipt["residual_hourly_cost_cny"] == 0.0
            and cost_receipt.get("instance_state") == "STOPPED"
            and cost_receipt.get("stopped_mode") == "STOP_CHARGING"
            and cost_receipt.get("instance_retained") is True
            and cost_receipt.get("api_termination_protection") is True
        ),
        "catalog_evidence_complete": (
            catalog_audit.get("all_evidence_references_match") is True
        ),
        "evidence_integrity_complete": evidence_integrity_complete,
    }
    failed = [name for name, passed in requirements.items() if not passed]
    evidence_refs = {
        "feedback_summary_sha256": feedback.get("summary_sha256"),
        "holdout_summary_sha256": holdout.get("summary_sha256"),
        "independent_safety_evidence_sha256": independent_safety.get("evidence_sha256"),
        "runtime_identity_evidence_sha256": runtime_identity.get("evidence_sha256"),
        "cost_receipt_evidence_sha256": cost_receipt.get("evidence_sha256"),
        "catalog_audit_evidence_sha256": catalog_audit.get("evidence_sha256"),
    }
    content = {
        "schema_version": 1,
        "contract": "statistical-skill-transfer-capability-v2",
        "minimum_evaluator_valid_holdout_pairs": 3,
        "requirements": requirements,
        "failed_requirements": failed,
        "evaluator_failure_count": evaluator_failures,
        "gate_passed": not failed,
        "safety_only_pairs_count_as_evaluator_valid": False,
        "evidence_refs": evidence_refs,
    }
    return {**content, "evidence_sha256": sha256_json(content)}
