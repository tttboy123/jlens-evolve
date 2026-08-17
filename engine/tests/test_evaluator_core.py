from __future__ import annotations

import math

import pytest

from evaluator_core import (
    CASE_GROUPS,
    HOLDOUT_CASES,
    score_callable,
    score_holdout_callable,
    validate_candidate_source,
)


def reference_solution(records):
    totals: dict[str, float] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        user = row.get("user")
        status = row.get("status")
        amount = row.get("amount")
        if not isinstance(user, str) or not user.strip():
            continue
        if not isinstance(status, str) or status.strip().lower() != "paid":
            continue
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            continue
        if not math.isfinite(float(amount)) or amount <= 0:
            continue
        normalized_user = user.strip().lower()
        totals[normalized_user] = totals.get(normalized_user, 0.0) + float(amount)
    return sorted(
        ((user, round(total, 2)) for user, total in totals.items()),
        key=lambda item: (-item[1], item[0]),
    )


def test_reference_solution_passes_every_deterministic_case_group():
    result = score_callable(reference_solution)

    assert result["combined_score"] == pytest.approx(1.0)
    assert set(CASE_GROUPS) <= result.keys()
    assert all(result[group] == pytest.approx(1.0) for group in CASE_GROUPS)
    assert result["passed_cases"] == result["total_cases"]


def test_reference_solution_passes_hidden_holdout_cases():
    result = score_holdout_callable(reference_solution)

    assert result["passed_cases"] == len(HOLDOUT_CASES)
    assert result["holdout_pass_rate"] == pytest.approx(1.0)
    assert {row["id"] for row in result["case_results"]}.isdisjoint(
        {"basic_paid_rows", "filter_non_paid"}
    )


def test_partial_solution_produces_component_level_failures():
    def partial(records):
        return [
            (row["user"].strip().lower(), round(row["amount"], 2))
            for row in records
            if isinstance(row, dict) and row.get("status") == "paid"
        ]

    result = score_callable(partial)

    assert 0.0 < result["combined_score"] < 1.0
    assert result["basic"] == pytest.approx(1.0)
    assert result["aggregation"] < 1.0
    assert result["validation"] < 1.0


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef solve(records): return []",
        "def solve(records):\n    return open('/tmp/x').read()",
        "def solve(records):\n    return eval('[]')",
    ],
)
def test_static_validator_rejects_unsafe_candidate_code(source):
    valid, reasons = validate_candidate_source(source)

    assert not valid
    assert reasons


def test_static_validator_accepts_pure_python_candidate():
    valid, reasons = validate_candidate_source(
        "def solve(records):\n"
        "    totals = {}\n"
        "    for row in records:\n"
        "        totals[row['user']] = totals.get(row['user'], 0) + row['amount']\n"
        "    return sorted(totals.items())\n"
    )

    assert valid
    assert reasons == []
