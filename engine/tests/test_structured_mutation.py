from __future__ import annotations

import json

from admission_policy import source_fingerprints
from structured_mutation import (
    MutationPlan,
    apply_mutation_plan,
    derive_fallback_plan,
    extract_current_program,
    extract_public_target_failure,
    parse_mutation_plan,
    postcondition_satisfied,
)


def _payload(source: str, target: str) -> dict:
    return {
        "model": "local-model",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Evaluator evidence:\n"
                    f"### target_failure\n```\n{{'id': '{target}'}}\n```\n\n"
                    f"Current program:\n```python\n{source}\n```\n"
                ),
            }
        ],
    }


def test_extracts_only_current_program_and_public_target():
    reference = "def solve(records):\n    return ['reference']"
    current = "def solve(records):\n    return []"
    payload = _payload(current, "filter_normalized_status")
    payload["messages"][0]["content"] = (
        f"Accepted reference:\n```python\n{reference}\n```\n\n"
        + payload["messages"][0]["content"]
    )

    assert extract_current_program(payload) == current
    assert extract_public_target_failure(payload) == "filter_normalized_status"


def test_parses_only_allowlisted_operator_for_matching_public_failure():
    content = json.dumps(
        {
            "schema_version": 1,
            "operator_id": "canonicalize_before_predicate",
            "target_symbol": "solve",
            "public_failure": "filter_normalized_status",
            "preserve": ["aggregation", "sorting"],
        }
    )

    plan = parse_mutation_plan(content, "filter_normalized_status")

    assert plan.operator_id == "canonicalize_before_predicate"
    assert plan.public_failure == "filter_normalized_status"


def test_invalid_model_plan_falls_back_to_public_failure_mapping():
    plan = derive_fallback_plan("reject_invalid_amounts")

    assert plan == MutationPlan(
        schema_version=1,
        operator_id="finite_numeric_guard",
        target_symbol="solve",
        public_failure="reject_invalid_amounts",
        preserve=(),
    )


def test_canonicalize_operator_changes_status_predicate_ast():
    source = """def solve(records):
    return [row for row in records if row.get("status") == "paid"]
"""
    plan = derive_fallback_plan("filter_normalized_status")

    result = apply_mutation_plan(source, plan)

    assert result.changed is True
    assert postcondition_satisfied(result.source, plan) is True
    assert ".strip().lower()" in result.source
    assert source_fingerprints(source)[1] != source_fingerprints(result.source)[1]


def test_numeric_guard_rejects_bool_non_numeric_non_finite_and_non_positive():
    source = """def solve(records):
    output = []
    for row in records:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).strip().lower() != "paid":
            continue
        amount = row.get("amount")
        output.append((str(row.get("user", "")).strip().lower(), amount))
    return output
"""
    plan = derive_fallback_plan("reject_invalid_amounts")

    result = apply_mutation_plan(source, plan)
    namespace: dict = {"__builtins__": __builtins__}
    exec(result.source, namespace, namespace)  # noqa: S102
    records = [
        {"user": "bool", "amount": True, "status": "paid"},
        {"user": "text", "amount": "9", "status": "paid"},
        {"user": "nan", "amount": float("nan"), "status": "paid"},
        {"user": "inf", "amount": float("inf"), "status": "paid"},
        {"user": "zero", "amount": 0, "status": "paid"},
        {"user": "ok", "amount": 2, "status": "paid"},
    ]

    assert result.changed is True
    assert postcondition_satisfied(result.source, plan) is True
    assert namespace["solve"](records) == [("ok", 2)]
