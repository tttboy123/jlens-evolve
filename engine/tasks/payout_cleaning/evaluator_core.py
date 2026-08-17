"""Deterministic public and hidden partitions for payout record cleaning."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_SHARED_CORE_PATH = Path(__file__).resolve().parents[2] / "evaluator_core.py"
_SHARED_SPEC = importlib.util.spec_from_file_location(
    "_evolve_shared_evaluator_core", _SHARED_CORE_PATH
)
if _SHARED_SPEC is None or _SHARED_SPEC.loader is None:
    raise ImportError(f"cannot load shared evaluator core: {_SHARED_CORE_PATH}")
_SHARED_CORE = importlib.util.module_from_spec(_SHARED_SPEC)
_SHARED_SPEC.loader.exec_module(_SHARED_CORE)

CASE_GROUPS = _SHARED_CORE.CASE_GROUPS
GROUP_WEIGHTS = _SHARED_CORE.GROUP_WEIGHTS
ast_complexity = _SHARED_CORE.ast_complexity
load_candidate = _SHARED_CORE.load_candidate
score_cases = _SHARED_CORE.score_cases

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "basic_settled_usd_rows",
        "group": "basic",
        "records": [
            {"account": "Bob", "value": 3, "state": "settled", "currency": "USD"},
            {
                "account": "Alice",
                "value": 2,
                "state": "settled",
                "currency": "USD",
            },
        ],
        "expected": [("bob", 3.0), ("alice", 2.0)],
    },
    {
        "id": "filter_non_settled",
        "group": "filtering",
        "records": [
            {
                "account": "Alice",
                "value": 5,
                "state": "settled",
                "currency": "USD",
            },
            {
                "account": "Mallory",
                "value": 100,
                "state": "reversed",
                "currency": "USD",
            },
            {
                "account": "Eve",
                "value": 50,
                "state": "pending",
                "currency": "USD",
            },
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "filter_normalized_state",
        "group": "filtering",
        "records": [
            {
                "account": "Alice",
                "value": 2,
                "state": " SETTLED ",
                "currency": "USD",
            },
            {
                "account": "Bob",
                "value": 1,
                "state": "Settled",
                "currency": "USD",
            },
        ],
        "expected": [("alice", 2.0), ("bob", 1.0)],
    },
    {
        "id": "filter_and_normalize_currency",
        "group": "filtering",
        "records": [
            {
                "account": "Alice",
                "value": 4,
                "state": "settled",
                "currency": " usd ",
            },
            {
                "account": "Mallory",
                "value": 90,
                "state": "settled",
                "currency": "EUR",
            },
        ],
        "expected": [("alice", 4.0)],
    },
    {
        "id": "normalize_account_identity",
        "group": "normalization",
        "records": [
            {
                "account": " Alice ",
                "value": 2,
                "state": "settled",
                "currency": "USD",
            },
            {
                "account": "ALICE",
                "value": 3,
                "state": "settled",
                "currency": "USD",
            },
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "drop_empty_account",
        "group": "normalization",
        "records": [
            {"account": " ", "value": 30, "state": "settled", "currency": "USD"},
            {"account": "Bob", "value": 4, "state": "settled", "currency": "USD"},
        ],
        "expected": [("bob", 4.0)],
    },
    {
        "id": "aggregate_repeated_rows",
        "group": "aggregation",
        "records": [
            {
                "account": "alice",
                "value": 1.25,
                "state": "settled",
                "currency": "USD",
            },
            {
                "account": "alice",
                "value": 2.75,
                "state": "settled",
                "currency": "USD",
            },
            {"account": "bob", "value": 3, "state": "settled", "currency": "USD"},
        ],
        "expected": [("alice", 4.0), ("bob", 3.0)],
    },
    {
        "id": "aggregate_after_filtering",
        "group": "aggregation",
        "records": [
            {"account": "alice", "value": 5, "state": "settled", "currency": "USD"},
            {"account": "alice", "value": 50, "state": "pending", "currency": "USD"},
            {"account": "alice", "value": 1, "state": "settled", "currency": "USD"},
        ],
        "expected": [("alice", 6.0)],
    },
    {
        "id": "reject_invalid_values",
        "group": "validation",
        "records": [
            {"account": "a", "value": True, "state": "settled", "currency": "USD"},
            {"account": "b", "value": "9", "state": "settled", "currency": "USD"},
            {"account": "c", "value": None, "state": "settled", "currency": "USD"},
            {"account": "ok", "value": 3, "state": "settled", "currency": "USD"},
        ],
        "expected": [("ok", 3.0)],
    },
    {
        "id": "reject_nonpositive_and_nonfinite",
        "group": "validation",
        "records": [
            {"account": "a", "value": 0, "state": "settled", "currency": "USD"},
            {"account": "b", "value": -2, "state": "settled", "currency": "USD"},
            {
                "account": "c",
                "value": float("nan"),
                "state": "settled",
                "currency": "USD",
            },
            {
                "account": "d",
                "value": float("inf"),
                "state": "settled",
                "currency": "USD",
            },
            {"account": "ok", "value": 1, "state": "settled", "currency": "USD"},
        ],
        "expected": [("ok", 1.0)],
    },
    {
        "id": "round_after_aggregation",
        "group": "rounding",
        "records": [
            {
                "account": "alice",
                "value": 1.234,
                "state": "settled",
                "currency": "USD",
            },
            {
                "account": "alice",
                "value": 2.345,
                "state": "settled",
                "currency": "USD",
            },
        ],
        "expected": [("alice", 3.58)],
    },
    {
        "id": "sort_total_descending",
        "group": "sorting",
        "records": [
            {"account": "a", "value": 1, "state": "settled", "currency": "USD"},
            {"account": "b", "value": 9, "state": "settled", "currency": "USD"},
            {"account": "c", "value": 4, "state": "settled", "currency": "USD"},
        ],
        "expected": [("b", 9.0), ("c", 4.0), ("a", 1.0)],
    },
    {
        "id": "sort_ties_by_account",
        "group": "sorting",
        "records": [
            {"account": "zoe", "value": 5, "state": "settled", "currency": "USD"},
            {"account": "amy", "value": 5, "state": "settled", "currency": "USD"},
        ],
        "expected": [("amy", 5.0), ("zoe", 5.0)],
    },
    {
        "id": "ignore_malformed_rows",
        "group": "robustness",
        "records": [
            None,
            "bad",
            {},
            {"account": "a", "state": "settled", "currency": "USD"},
            {"account": "ok", "value": 2, "state": "settled", "currency": "USD"},
        ],
        "expected": [("ok", 2.0)],
    },
)

HOLDOUT_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "holdout_permutation_invariance",
        "group": "robustness",
        "records": [
            {"account": "b", "value": 2, "state": "settled", "currency": "USD"},
            {"account": "a", "value": 5, "state": "settled", "currency": "USD"},
            {"account": "b", "value": 4, "state": "settled", "currency": "USD"},
        ],
        "expected": [("b", 6.0), ("a", 5.0)],
    },
    {
        "id": "holdout_malformed_insertion_invariance",
        "group": "robustness",
        "records": [
            None,
            {"account": "ok", "value": 4, "state": "settled", "currency": "USD"},
            {"account": [], "value": 99, "state": "settled", "currency": "USD"},
            {"account": "x", "value": 10, "state": "pending", "currency": "USD"},
        ],
        "expected": [("ok", 4.0)],
    },
    {
        "id": "holdout_state_currency_identity_normalization",
        "group": "normalization",
        "records": [
            {
                "account": " ACME ",
                "value": 2,
                "state": " Settled ",
                "currency": " usd ",
            },
            {"account": "acme", "value": 3, "state": "SETTLED", "currency": "USD"},
        ],
        "expected": [("acme", 5.0)],
    },
    {
        "id": "holdout_numeric_equivalence_and_bool_rejection",
        "group": "validation",
        "records": [
            {"account": "a", "value": 2, "state": "settled", "currency": "USD"},
            {"account": "a", "value": 2.0, "state": "settled", "currency": "USD"},
            {"account": "a", "value": False, "state": "settled", "currency": "USD"},
            {
                "account": "a",
                "value": float("-inf"),
                "state": "settled",
                "currency": "USD",
            },
        ],
        "expected": [("a", 4.0)],
    },
    {
        "id": "holdout_split_merge_aggregation",
        "group": "aggregation",
        "records": [
            {"account": "a", "value": 1.11, "state": "settled", "currency": "USD"},
            {"account": "a", "value": 2.22, "state": "settled", "currency": "USD"},
            {"account": "a", "value": 3.33, "state": "settled", "currency": "USD"},
        ],
        "expected": [("a", 6.66)],
    },
    {
        "id": "holdout_unique_finite_sorted_output",
        "group": "sorting",
        "records": [
            {"account": "z", "value": 3, "state": "settled", "currency": "USD"},
            {"account": "a", "value": 3, "state": "settled", "currency": "USD"},
            {"account": "m", "value": 8, "state": "settled", "currency": "USD"},
            {"account": "m", "value": 2, "state": "settled", "currency": "USD"},
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


def score_program_path(path: str | Path) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    solve, reasons = load_candidate(path)
    if solve is None:
        return {
            **{group: 0.0 for group in CASE_GROUPS},
            "combined_score": 0.0,
            "weighted_score": 0.0,
            "passed_cases": 0,
            "total_cases": len(CASES),
            "case_results": [],
            "rejection_reasons": reasons,
            "ast_complexity": ast_complexity(source),
            "evaluator_valid": 0.0,
        }
    result = score_callable(solve)
    result["ast_complexity"] = ast_complexity(source)
    result["evaluator_valid"] = 1.0
    return result
