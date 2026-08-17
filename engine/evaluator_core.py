"""Deterministic, side-effect-bounded evaluator for the evolution task."""

from __future__ import annotations

import ast
import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

CASE_GROUPS = (
    "basic",
    "filtering",
    "normalization",
    "aggregation",
    "validation",
    "rounding",
    "sorting",
    "robustness",
)

GROUP_WEIGHTS = {
    "basic": 0.10,
    "filtering": 0.15,
    "normalization": 0.15,
    "aggregation": 0.15,
    "validation": 0.15,
    "rounding": 0.10,
    "sorting": 0.10,
    "robustness": 0.10,
}

CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "basic_paid_rows",
        "group": "basic",
        "records": [
            {"user": "Bob", "amount": 3, "status": "paid"},
            {"user": "Alice", "amount": 2, "status": "paid"},
        ],
        "expected": [("bob", 3.0), ("alice", 2.0)],
    },
    {
        "id": "filter_non_paid",
        "group": "filtering",
        "records": [
            {"user": "Alice", "amount": 5, "status": "paid"},
            {"user": "Mallory", "amount": 100, "status": "refunded"},
            {"user": "Eve", "amount": 50, "status": "pending"},
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "filter_normalized_status",
        "group": "filtering",
        "records": [
            {"user": "Alice", "amount": 2, "status": " PAID "},
            {"user": "Bob", "amount": 1, "status": "Paid"},
        ],
        "expected": [("alice", 2.0), ("bob", 1.0)],
    },
    {
        "id": "normalize_user_identity",
        "group": "normalization",
        "records": [
            {"user": " Alice ", "amount": 2, "status": "paid"},
            {"user": "ALICE", "amount": 3, "status": "paid"},
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "drop_empty_user",
        "group": "normalization",
        "records": [
            {"user": " ", "amount": 30, "status": "paid"},
            {"user": "Bob", "amount": 4, "status": "paid"},
        ],
        "expected": [("bob", 4.0)],
    },
    {
        "id": "aggregate_repeated_rows",
        "group": "aggregation",
        "records": [
            {"user": "alice", "amount": 1.25, "status": "paid"},
            {"user": "alice", "amount": 2.75, "status": "paid"},
            {"user": "bob", "amount": 3, "status": "paid"},
        ],
        "expected": [("alice", 4.0), ("bob", 3.0)],
    },
    {
        "id": "aggregate_after_filtering",
        "group": "aggregation",
        "records": [
            {"user": "alice", "amount": 5, "status": "paid"},
            {"user": "alice", "amount": 50, "status": "pending"},
            {"user": "alice", "amount": 1, "status": "paid"},
        ],
        "expected": [("alice", 6.0)],
    },
    {
        "id": "reject_invalid_amounts",
        "group": "validation",
        "records": [
            {"user": "a", "amount": True, "status": "paid"},
            {"user": "b", "amount": "9", "status": "paid"},
            {"user": "c", "amount": None, "status": "paid"},
            {"user": "d", "amount": float("nan"), "status": "paid"},
            {"user": "e", "amount": float("inf"), "status": "paid"},
            {"user": "ok", "amount": 2, "status": "paid"},
        ],
        "expected": [("ok", 2.0)],
    },
    {
        "id": "reject_non_positive_amounts",
        "group": "validation",
        "records": [
            {"user": "a", "amount": 0, "status": "paid"},
            {"user": "b", "amount": -1, "status": "paid"},
            {"user": "c", "amount": 0.01, "status": "paid"},
        ],
        "expected": [("c", 0.01)],
    },
    {
        "id": "round_after_aggregation",
        "group": "rounding",
        "records": [
            {"user": "alice", "amount": 1.234, "status": "paid"},
            {"user": "alice", "amount": 0.002, "status": "paid"},
            {"user": "bob", "amount": 1.236, "status": "paid"},
        ],
        "expected": [("alice", 1.24), ("bob", 1.24)],
    },
    {
        "id": "sort_total_then_user",
        "group": "sorting",
        "records": [
            {"user": "zoe", "amount": 5, "status": "paid"},
            {"user": "alice", "amount": 10, "status": "paid"},
            {"user": "bob", "amount": 5, "status": "paid"},
        ],
        "expected": [("alice", 10.0), ("bob", 5.0), ("zoe", 5.0)],
    },
    {
        "id": "empty_input",
        "group": "robustness",
        "records": [],
        "expected": [],
    },
    {
        "id": "ignore_malformed_rows",
        "group": "robustness",
        "records": [
            None,
            7,
            {},
            {"user": "alice"},
            {"amount": 3, "status": "paid"},
            {"user": "bob", "amount": 2, "status": "paid"},
        ],
        "expected": [("bob", 2.0)],
    },
)

# These cases are intentionally evaluated only by the post-run audit.  The live
# OpenEvolve evaluator never returns their ids, scores, or failure details to the
# proposer, which makes them a deterministic generalization check rather than
# another prompt-visible target.
HOLDOUT_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "holdout_permutation_invariance",
        "group": "aggregation",
        "records": [
            {"user": "bob", "amount": 2.0, "status": "paid"},
            {"user": "alice", "amount": 1.0, "status": "paid"},
            {"user": "bob", "amount": 3, "status": "paid"},
        ],
        "expected": [("bob", 5.0), ("alice", 1.0)],
    },
    {
        "id": "holdout_malformed_insertion_invariance",
        "group": "robustness",
        "records": [
            None,
            {"user": " Alice ", "amount": 2, "status": "paid"},
            ["not", "a", "mapping"],
            {"user": "alice", "amount": 3.0, "status": " PAID "},
            {"user": "noise", "amount": 999, "status": "pending"},
        ],
        "expected": [("alice", 5.0)],
    },
    {
        "id": "holdout_status_and_identity_normalization",
        "group": "normalization",
        "records": [
            {"user": " ZOE", "amount": 1, "status": "Paid"},
            {"user": "zoe ", "amount": 2, "status": " paid "},
            {"user": "   ", "amount": 20, "status": "paid"},
        ],
        "expected": [("zoe", 3.0)],
    },
    {
        "id": "holdout_numeric_equivalence_and_bool_rejection",
        "group": "validation",
        "records": [
            {"user": "a", "amount": 1, "status": "paid"},
            {"user": "a", "amount": 1.0, "status": "paid"},
            {"user": "a", "amount": True, "status": "paid"},
            {"user": "a", "amount": float("-inf"), "status": "paid"},
        ],
        "expected": [("a", 2.0)],
    },
    {
        "id": "holdout_split_merge_aggregation",
        "group": "aggregation",
        "records": [
            {"user": "kai", "amount": 0.333, "status": "paid"},
            {"user": "kai", "amount": 0.667, "status": "paid"},
            {"user": "lee", "amount": 0.999, "status": "paid"},
        ],
        "expected": [("kai", 1.0), ("lee", 1.0)],
    },
    {
        "id": "holdout_unique_finite_sorted_output",
        "group": "sorting",
        "records": [
            {"user": "b", "amount": 4.444, "status": "paid"},
            {"user": "a", "amount": 4.445, "status": "paid"},
            {"user": "c", "amount": float("nan"), "status": "paid"},
            {"user": "b", "amount": 0.001, "status": "paid"},
        ],
        "expected": [("a", 4.45), ("b", 4.45)],
    },
)

_ALLOWED_IMPORTS = {"math"}
_BLOCKED_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
_BLOCKED_ATTRIBUTE_ROOTS = {
    "os",
    "pathlib",
    "shutil",
    "socket",
    "subprocess",
    "sys",
}


def validate_candidate_source(source: str) -> tuple[bool, list[str]]:
    """Reject candidate features that could escape the pure-function task."""
    reasons: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, [f"syntax error: {exc.msg}"]

    solve_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    ]
    if len(solve_defs) != 1:
        reasons.append("candidate must define exactly one top-level solve function")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    reasons.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in _ALLOWED_IMPORTS:
                reasons.append(f"blocked import: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                reasons.append(f"blocked call: {node.func.id}")
        elif isinstance(node, ast.Attribute):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _BLOCKED_ATTRIBUTE_ROOTS:
                reasons.append(f"blocked attribute root: {root.id}")
            if node.attr.startswith("__"):
                reasons.append(f"blocked dunder attribute: {node.attr}")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            reasons.append(f"blocked dunder name: {node.id}")

    return not reasons, sorted(set(reasons))


def _valid_output(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        user, total = item
        if not isinstance(user, str):
            return False
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            return False
        if not math.isfinite(float(total)):
            return False
    return True


def _canonical_output(value: Any) -> list[tuple[str, float]] | None:
    if not _valid_output(value):
        return None
    return [(str(user), float(total)) for user, total in value]


def score_cases(
    solve: Callable[[list[Any]], Any],
    cases: tuple[dict[str, Any], ...],
    *,
    case_groups: tuple[str, ...] = CASE_GROUPS,
    group_weights: dict[str, float] = GROUP_WEIGHTS,
) -> dict[str, Any]:
    """Score a candidate against a supplied deterministic case partition."""
    case_results: list[dict[str, Any]] = []
    for case in cases:
        error = None
        actual = None
        try:
            actual = _canonical_output(solve(copy.deepcopy(case["records"])))
        except Exception as exc:  # noqa: BLE001 - candidate failures are evidence.
            error = f"{type(exc).__name__}: {exc}"
        passed = actual == case["expected"]
        case_results.append(
            {
                "id": case["id"],
                "group": case["group"],
                "passed": passed,
                "actual": actual,
                "expected": case["expected"],
                "error": error,
            }
        )

    metrics: dict[str, Any] = {}
    present_groups = tuple(
        group
        for group in case_groups
        if any(row["group"] == group for row in case_results)
    )
    for group in present_groups:
        group_rows = [row for row in case_results if row["group"] == group]
        metrics[group] = sum(row["passed"] for row in group_rows) / len(group_rows)
    weight_total = sum(group_weights[group] for group in present_groups)
    weighted_score = sum(
        metrics[group] * group_weights[group] for group in present_groups
    )
    if weight_total:
        weighted_score /= weight_total
    metrics["weighted_score"] = weighted_score
    metrics["passed_cases"] = sum(row["passed"] for row in case_results)
    metrics["total_cases"] = len(case_results)
    # Lexicographic scalarization: one additional passing case always dominates
    # any difference in the legacy group-weighted score, while a perfect result
    # remains exactly 1.0.
    metrics["combined_score"] = (metrics["passed_cases"] + weighted_score) / (
        len(case_results) + 1
    )
    metrics["case_results"] = case_results
    return metrics


def score_callable(solve: Callable[[list[Any]], Any]) -> dict[str, Any]:
    """Score a candidate on the prompt-visible development partition."""
    return score_cases(solve, CASES)


def score_holdout_callable(solve: Callable[[list[Any]], Any]) -> dict[str, Any]:
    """Score a candidate on the prompt-hidden deterministic partition."""
    result = score_cases(solve, HOLDOUT_CASES)
    result["holdout_pass_rate"] = result["passed_cases"] / result["total_cases"]
    return result


def ast_complexity(source: str) -> int:
    """Return a stable syntax-tree node count, excluding comments/formatting."""
    try:
        return sum(1 for _ in ast.walk(ast.parse(source)))
    except SyntaxError:
        return 0


def load_candidate(
    path: str | Path,
) -> tuple[Callable[[list[Any]], Any] | None, list[str]]:
    source = Path(path).read_text(encoding="utf-8")
    valid, reasons = validate_candidate_source(source)
    if not valid:
        return None, reasons

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "math" and level == 0:
            return math
        raise ImportError(f"import blocked by experiment sandbox: {name}")

    safe_builtins = {
        "__import__": safe_import,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, Any] = {"__builtins__": safe_builtins, "math": math}
    try:
        # The static screen and explicit builtins/import allowlist above are the
        # task boundary. Production use still requires an OS sandbox.
        exec(compile(source, str(path), "exec"), namespace, namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001 - candidate load failure is evidence.
        return None, [f"load error: {type(exc).__name__}: {exc}"]
    solve = namespace.get("solve")
    if not callable(solve):
        return None, ["solve is not callable"]
    return solve, []


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
