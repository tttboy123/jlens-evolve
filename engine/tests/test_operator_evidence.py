from __future__ import annotations

from operator_evidence import (
    build_operator_evidence,
    merge_operator_evidence,
    propose_operator_policy,
    render_operator_skill_candidate,
)


def test_merge_operator_evidence_sums_independent_windows():
    first = {
        "schema_version": 1,
        "run_id": "seed-a",
        "audit_rows": 2,
        "matched_public_candidates": 2,
        "operators": {
            "canonicalize_before_predicate": {
                "attempts": 2,
                "accepted": 1,
                "public_improvements": 1,
                "postcondition_valid": 2,
            }
        },
    }
    second = {
        "schema_version": 1,
        "run_id": "seed-b",
        "audit_rows": 3,
        "matched_public_candidates": 3,
        "operators": {
            "canonicalize_before_predicate": {
                "attempts": 1,
                "accepted": 0,
                "public_improvements": 0,
                "postcondition_valid": 1,
            },
            "free_form_rewrite": {"attempts": 2, "evaluator_valid": 0},
        },
    }

    merged = merge_operator_evidence([first, second])

    assert merged["run_ids"] == ["seed-a", "seed-b"]
    assert merged["audit_rows"] == 5
    assert merged["matched_public_candidates"] == 5
    assert merged["operators"]["canonicalize_before_predicate"]["attempts"] == 3
    assert merged["operators"]["canonicalize_before_predicate"]["accepted"] == 1
    assert merged["operators"]["free_form_rewrite"]["attempts"] == 2
    assert "holdout" not in str(merged).lower()


def test_operator_evidence_joins_audit_to_public_candidate_without_hidden_data():
    audits = [
        {
            "operator_id": "canonicalize_before_predicate",
            "selected_source_sha256": "source-a",
            "deterministic_transform_applied": True,
            "repair_postcondition_valid": True,
        },
        {
            "operator_id": "canonicalize_before_predicate",
            "selected_source_sha256": "source-b",
            "deterministic_transform_applied": True,
            "repair_postcondition_valid": False,
        },
        {
            "operator_id": "canonicalize_before_predicate",
            "selected_source_sha256": "proxy-formatted-source-c",
            "selected_ast_sha256": "ast-c",
            "deterministic_transform_applied": True,
            "repair_postcondition_valid": True,
        },
    ]
    events = [
        {
            "event_type": "candidate",
            "run_id": "other-run",
            "source_hash": "source-a",
            "accepted": False,
            "gained_cases": [],
            "admission_reasons": ["parent_regression"],
            "metrics": {"evaluator_valid": 1.0, "combined_score": 0.1},
        },
        {
            "event_type": "candidate",
            "run_id": "target-run",
            "source_hash": "source-a",
            "accepted": True,
            "gained_cases": ["case_filter_normalized_status"],
            "admission_reasons": [],
            "metrics": {"evaluator_valid": 1.0, "combined_score": 0.4},
        },
        {
            "event_type": "candidate",
            "run_id": "target-run",
            "source_hash": "source-b",
            "accepted": False,
            "gained_cases": [],
            "admission_reasons": ["ast_duplicate"],
            "metrics": {"evaluator_valid": 1.0, "combined_score": 0.4},
        },
        {
            "event_type": "candidate",
            "run_id": "target-run",
            "source_hash": "runtime-normalized-source-c",
            "ast_hash": "ast-c",
            "accepted": True,
            "gained_cases": ["case_drop_empty_user"],
            "admission_reasons": [],
            "metrics": {"evaluator_valid": 1.0, "combined_score": 0.5},
        },
        {
            "event_type": "holdout_verification",
            "source_hash": "source-a",
            "child_holdout_score": 1.0,
        },
    ]

    evidence = build_operator_evidence(audits, events, run_id="target-run")

    row = evidence["operators"]["canonicalize_before_predicate"]
    assert row["attempts"] == 3
    assert row["matched_public_candidates"] == 3
    assert row["accepted"] == 2
    assert row["public_improvements"] == 2
    assert row["structural_duplicates"] == 1
    assert "holdout" not in str(evidence).lower()


def test_policy_revision_is_candidate_and_normalizes_operator_weights():
    evidence = {
        "operators": {
            "canonicalize_before_predicate": {
                "attempts": 4,
                "accepted": 3,
                "public_improvements": 2,
                "postcondition_valid": 4,
            },
            "finite_numeric_guard": {
                "attempts": 2,
                "accepted": 1,
                "public_improvements": 0,
                "postcondition_valid": 2,
            },
        }
    }

    policy = propose_operator_policy(evidence, parent_policy_id="operator-policy-v1")

    assert policy["status"] == "candidate"
    assert policy["parent_policy_id"] == "operator-policy-v1"
    assert policy["rsi_pass"] is False
    assert abs(sum(policy["operator_weights"].values()) - 1.0) < 1e-12
    assert (
        policy["operator_weights"]["canonicalize_before_predicate"]
        > policy["operator_weights"]["finite_numeric_guard"]
    )


def test_operator_skill_candidate_contains_schema_and_evidence_not_task_code():
    evidence = {
        "operators": {
            "finite_numeric_guard": {
                "attempts": 3,
                "accepted": 2,
                "public_improvements": 1,
                "postcondition_valid": 3,
            }
        }
    }

    skill = render_operator_skill_candidate("finite_numeric_guard", evidence)

    assert "status: candidate" in skill
    assert "attempts: 3" in skill
    assert "def solve" not in skill
    assert "holdout" not in skill.lower()
