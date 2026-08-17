"""Deterministic public and sealed partitions for refund record cleaning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SHARED_CORE_PATH = Path(__file__).resolve().parents[2] / "evaluator_core.py"
_SHARED_SPEC = importlib.util.spec_from_file_location(
    "_evolve_shared_refund_evaluator_core", _SHARED_CORE_PATH
)
if _SHARED_SPEC is None or _SHARED_SPEC.loader is None:
    raise ImportError(f"cannot load shared evaluator core: {_SHARED_CORE_PATH}")
_SHARED_CORE = importlib.util.module_from_spec(_SHARED_SPEC)
_SHARED_SPEC.loader.exec_module(_SHARED_CORE)

CASE_GROUPS = _SHARED_CORE.CASE_GROUPS
GROUP_WEIGHTS = _SHARED_CORE.GROUP_WEIGHTS
score_cases = _SHARED_CORE.score_cases

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "basic_approved_refunds",
        "group": "basic",
        "records": [
            {"customer": "Bob", "refund_amount": 3, "decision": "approved"},
            {"customer": "Alice", "refund_amount": 2, "decision": "approved"},
        ],
        "expected": [("bob", 3.0), ("alice", 2.0)],
    },
    {
        "id": "filter_non_approved",
        "group": "filtering",
        "records": [
            {"customer": "Alice", "refund_amount": 5, "decision": "approved"},
            {"customer": "Mallory", "refund_amount": 50, "decision": "denied"},
            {"customer": "Eve", "refund_amount": 40, "decision": "review"},
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "normalize_decision",
        "group": "filtering",
        "records": [
            {"customer": "Alice", "refund_amount": 2, "decision": " APPROVED "},
            {"customer": "Bob", "refund_amount": 1, "decision": "Approved"},
        ],
        "expected": [("alice", 2.0), ("bob", 1.0)],
    },
    {
        "id": "normalize_and_aggregate_customer",
        "group": "normalization",
        "records": [
            {"customer": " Alice ", "refund_amount": 2, "decision": "approved"},
            {"customer": "ALICE", "refund_amount": 3, "decision": "approved"},
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "drop_empty_customer",
        "group": "normalization",
        "records": [
            {"customer": " ", "refund_amount": 30, "decision": "approved"},
            {"customer": "Bob", "refund_amount": 4, "decision": "approved"},
        ],
        "expected": [("bob", 4.0)],
    },
    {
        "id": "aggregate_after_filtering",
        "group": "aggregation",
        "records": [
            {"customer": "a", "refund_amount": 1.25, "decision": "approved"},
            {"customer": "a", "refund_amount": 90, "decision": "denied"},
            {"customer": "a", "refund_amount": 2.75, "decision": "approved"},
        ],
        "expected": [("a", 4.0)],
    },
    {
        "id": "reject_invalid_values",
        "group": "validation",
        "records": [
            {"customer": "a", "refund_amount": True, "decision": "approved"},
            {"customer": "b", "refund_amount": "9", "decision": "approved"},
            {"customer": "c", "refund_amount": float("nan"), "decision": "approved"},
            {"customer": "ok", "refund_amount": 3, "decision": "approved"},
        ],
        "expected": [("ok", 3.0)],
    },
    {
        "id": "reject_nonpositive_and_nonfinite",
        "group": "validation",
        "records": [
            {"customer": "a", "refund_amount": 0, "decision": "approved"},
            {"customer": "b", "refund_amount": -1, "decision": "approved"},
            {"customer": "c", "refund_amount": float("inf"), "decision": "approved"},
            {"customer": "ok", "refund_amount": 1, "decision": "approved"},
        ],
        "expected": [("ok", 1.0)],
    },
    {
        "id": "round_after_aggregation",
        "group": "rounding",
        "records": [
            {"customer": "a", "refund_amount": 1.234, "decision": "approved"},
            {"customer": "a", "refund_amount": 2.345, "decision": "approved"},
        ],
        "expected": [("a", 3.58)],
    },
    {
        "id": "sort_total_then_customer",
        "group": "sorting",
        "records": [
            {"customer": "z", "refund_amount": 5, "decision": "approved"},
            {"customer": "a", "refund_amount": 9, "decision": "approved"},
            {"customer": "b", "refund_amount": 5, "decision": "approved"},
        ],
        "expected": [("a", 9.0), ("b", 5.0), ("z", 5.0)],
    },
)

HOLDOUT_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "holdout_permutation_invariance",
        "group": "robustness",
        "records": [
            {"customer": "b", "refund_amount": 2, "decision": "approved"},
            {"customer": "a", "refund_amount": 5, "decision": "approved"},
            {"customer": "b", "refund_amount": 4, "decision": "approved"},
        ],
        "expected": [("b", 6.0), ("a", 5.0)],
    },
    {
        "id": "holdout_malformed_insertion_invariance",
        "group": "robustness",
        "records": [
            None,
            {"customer": "ok", "refund_amount": 4, "decision": "approved"},
            {"customer": [], "refund_amount": 99, "decision": "approved"},
            {"customer": "x", "refund_amount": 10, "decision": "denied"},
        ],
        "expected": [("ok", 4.0)],
    },
    {
        "id": "holdout_decision_identity_normalization",
        "group": "normalization",
        "records": [
            {"customer": " ACME ", "refund_amount": 2, "decision": " Approved "},
            {"customer": "acme", "refund_amount": 3, "decision": "APPROVED"},
        ],
        "expected": [("acme", 5.0)],
    },
    {
        "id": "holdout_numeric_equivalence_and_bool_rejection",
        "group": "validation",
        "records": [
            {"customer": "a", "refund_amount": 2, "decision": "approved"},
            {"customer": "a", "refund_amount": 2.0, "decision": "approved"},
            {"customer": "a", "refund_amount": False, "decision": "approved"},
            {"customer": "a", "refund_amount": float("-inf"), "decision": "approved"},
        ],
        "expected": [("a", 4.0)],
    },
    {
        "id": "holdout_split_merge_aggregation",
        "group": "aggregation",
        "records": [
            {"customer": "a", "refund_amount": 1.11, "decision": "approved"},
            {"customer": "a", "refund_amount": 2.22, "decision": "approved"},
            {"customer": "a", "refund_amount": 3.33, "decision": "approved"},
        ],
        "expected": [("a", 6.66)],
    },
    {
        "id": "holdout_unique_finite_sorted_output",
        "group": "sorting",
        "records": [
            {"customer": "z", "refund_amount": 3, "decision": "approved"},
            {"customer": "a", "refund_amount": 3, "decision": "approved"},
            {"customer": "m", "refund_amount": 8, "decision": "approved"},
            {"customer": "m", "refund_amount": 2, "decision": "approved"},
        ],
        "expected": [("m", 10.0), ("a", 3.0), ("z", 3.0)],
    },
)


def score_callable(solve):
    return score_cases(
        solve, CASES, case_groups=CASE_GROUPS, group_weights=GROUP_WEIGHTS
    )


def score_holdout_callable(solve):
    result = score_cases(
        solve, HOLDOUT_CASES, case_groups=CASE_GROUPS, group_weights=GROUP_WEIGHTS
    )
    result["holdout_pass_rate"] = result["passed_cases"] / result["total_cases"]
    return result
