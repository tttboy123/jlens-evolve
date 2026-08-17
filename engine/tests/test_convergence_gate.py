from __future__ import annotations

import pytest

from skill_evolution_loop.convergence_gate import (
    normalized_convergence_metrics,
    normalized_convergence_stop,
)


def _metric(
    delta: float,
    *,
    safety: bool = False,
    tasks: int = 20,
    fingerprint: str | None = "a" * 64,
) -> dict[str, object]:
    metric: dict[str, object] = {
        "paired_tasks": tasks,
        "native_score_delta_mean": delta,
        "cost_delta_mean": 0.0,
        "safety_regression": safety,
    }
    if fingerprint is not None:
        metric["paired_task_fingerprint"] = fingerprint
    return metric


def test_normalization_removes_original_parent_distance_lower_bound() -> None:
    metrics = normalized_convergence_metrics(
        per_candidate={
            "candidate": {
                "vs_original": _metric(0.405),
                "vs_parent": _metric(0.005),
            }
        },
        parent_vs_original=_metric(0.4),
    )

    assert metrics["raw_mean_abs_delta"] == 0.205
    assert metrics["normalized_mean_abs_delta"] == 0.005
    assert metrics["per_candidate"]["candidate"]["normalized"] == {
        "candidate_vs_parent": 0.005,
        "candidate_vs_original_incremental": 0.005,
    }
    assert metrics["normalization"] == "parent-relative-delta-of-deltas-v1"


def test_normalization_is_fail_closed_on_inconsistent_pair_counts() -> None:
    with pytest.raises(ValueError, match="paired task counts"):
        normalized_convergence_metrics(
            per_candidate={
                "candidate": {
                    "vs_original": _metric(0.405, tasks=20),
                    "vs_parent": _metric(0.005, tasks=19),
                }
            },
            parent_vs_original=_metric(0.4, tasks=20),
        )


def test_normalization_is_fail_closed_without_pair_fingerprints() -> None:
    metrics = normalized_convergence_metrics(
        per_candidate={
            "candidate": {
                "vs_original": _metric(0.405, fingerprint=None),
                "vs_parent": _metric(0.005, fingerprint=None),
            }
        },
        parent_vs_original=_metric(0.4, fingerprint=None),
    )

    assert metrics["sample_gate"]["complete"] is False
    assert normalized_convergence_stop((metrics, metrics)) is False


def test_normalization_is_fail_closed_on_non_sha256_fingerprint() -> None:
    metrics = normalized_convergence_metrics(
        per_candidate={
            "candidate": {
                "vs_original": _metric(0.405, fingerprint="z" * 64),
                "vs_parent": _metric(0.005, fingerprint="z" * 64),
            }
        },
        parent_vs_original=_metric(0.4, fingerprint="z" * 64),
    )

    assert metrics["sample_gate"]["complete"] is False


def test_normalized_stop_requires_two_complete_safe_generations() -> None:
    safe = normalized_convergence_metrics(
        per_candidate={
            "candidate": {
                "vs_original": _metric(0.405),
                "vs_parent": _metric(0.005),
            }
        },
        parent_vs_original=_metric(0.4),
    )
    unsafe = normalized_convergence_metrics(
        per_candidate={
            "candidate": {
                "vs_original": _metric(0.405, safety=True),
                "vs_parent": _metric(0.005),
            }
        },
        parent_vs_original=_metric(0.4),
    )

    assert normalized_convergence_stop((safe,)) is False
    assert normalized_convergence_stop((safe, safe)) is True
    assert normalized_convergence_stop((safe, unsafe)) is False


def test_normalization_refuses_missing_comparisons() -> None:
    metrics = normalized_convergence_metrics(
        per_candidate={"candidate": {"vs_original": None, "vs_parent": None}},
        parent_vs_original=None,
    )

    assert metrics["normalized_mean_abs_delta"] is None
    assert metrics["sample_gate"]["complete"] is False
    assert normalized_convergence_stop((metrics, metrics)) is False
