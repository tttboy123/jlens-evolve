"""Reachable, parent-relative convergence metrics for evolution rounds."""

from __future__ import annotations

import re
from statistics import fmean
from typing import Any

_EPSILON = 0.05
_K_CONSECUTIVE = 2
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _delta(metric: Any) -> float | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("native_score_delta_mean")
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _tasks(metric: Any) -> int | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("paired_tasks")
    return value if type(value) is int and value > 0 else None


def _pair_fingerprint(metric: Any) -> str | None:
    if not isinstance(metric, dict):
        return None
    value = metric.get("paired_task_fingerprint")
    return value if isinstance(value, str) and _SHA256.fullmatch(value) else None


def normalized_convergence_metrics(
    *,
    per_candidate: dict[str, dict[str, Any]],
    parent_vs_original: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize candidate movement against the already-accepted parent step.

    Raw candidate-vs-original remains audit evidence.  Convergence uses both
    candidate-vs-parent and the incremental original delta:
    (candidate-original) - (parent-original).  These are algebraically equal
    on identical paired tasks, but retaining both catches identity or pairing
    drift rather than hiding it behind one scalar.
    """

    parent_delta = _delta(parent_vs_original)
    parent_tasks = _tasks(parent_vs_original)
    normalized_values: list[float] = []
    raw_values: list[float] = []
    safety_regression = bool(
        isinstance(parent_vs_original, dict)
        and parent_vs_original.get("safety_regression")
    )
    complete = parent_delta is not None and parent_tasks is not None
    output: dict[str, Any] = {}
    for candidate, comparisons in sorted(per_candidate.items()):
        vs_original = comparisons.get("vs_original")
        vs_parent = comparisons.get("vs_parent")
        original_delta = _delta(vs_original)
        relative_delta = _delta(vs_parent)
        counts = (_tasks(vs_original), _tasks(vs_parent), parent_tasks)
        present_counts = [count for count in counts if count is not None]
        if len(present_counts) == 3 and len(set(present_counts)) != 1:
            raise ValueError("convergence paired task counts do not match")
        fingerprints = (
            _pair_fingerprint(vs_original),
            _pair_fingerprint(vs_parent),
            _pair_fingerprint(parent_vs_original),
        )
        present_fingerprints = [value for value in fingerprints if value is not None]
        if present_fingerprints and (
            len(present_fingerprints) != 3 or len(set(present_fingerprints)) != 1
        ):
            raise ValueError("convergence paired task fingerprints do not match")
        row_complete = (
            original_delta is not None
            and relative_delta is not None
            and parent_delta is not None
            and len(present_counts) == 3
            and len(present_fingerprints) == 3
        )
        incremental = original_delta - parent_delta if row_complete else None
        if original_delta is not None:
            raw_values.append(abs(original_delta))
        if relative_delta is not None:
            raw_values.append(abs(relative_delta))
        if row_complete:
            normalized_values.extend((abs(relative_delta), abs(incremental)))
        complete = complete and row_complete
        safety_regression = safety_regression or any(
            isinstance(metric, dict) and bool(metric.get("safety_regression"))
            for metric in (vs_original, vs_parent)
        )
        output[candidate] = {
            "vs_original": vs_original,
            "vs_parent": vs_parent,
            "normalized": {
                "candidate_vs_parent": (
                    round(relative_delta, 6) if relative_delta is not None else None
                ),
                "candidate_vs_original_incremental": (
                    round(incremental, 6) if incremental is not None else None
                ),
            },
        }
    if not per_candidate:
        complete = False
    return {
        "schema_version": 1,
        "normalization": "parent-relative-delta-of-deltas-v1",
        "per_candidate": output,
        "parent_vs_original": parent_vs_original,
        "raw_mean_abs_delta": (round(fmean(raw_values), 6) if raw_values else None),
        "normalized_mean_abs_delta": (
            round(fmean(normalized_values), 6) if normalized_values else None
        ),
        "safety_regression": safety_regression,
        "epsilon": _EPSILON,
        "k_consecutive": _K_CONSECUTIVE,
        "sample_gate": {
            "complete": complete,
            "candidate_count": len(per_candidate),
            "paired_tasks": parent_tasks,
        },
    }


def normalized_convergence_stop(history: tuple[dict[str, Any], ...]) -> bool:
    """Stop only after K complete, safe, below-epsilon normalized rounds."""

    if len(history) < _K_CONSECUTIVE:
        return False
    return all(
        row.get("sample_gate", {}).get("complete") is True
        and isinstance(row.get("normalized_mean_abs_delta"), (int, float))
        and row["normalized_mean_abs_delta"] < _EPSILON
        and row.get("safety_regression") is False
        for row in history[-_K_CONSECUTIVE:]
    )
