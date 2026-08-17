from __future__ import annotations

from epistasis.experiments import (
    analyze_A,
    analyze_B,
    analyze_C,
    analyze_D,
    analyze_E,
    run_cell_plan,
)
from epistasis.tasks import generate_synthetic_task


def _matrix():
    return [
        generate_synthetic_task(
            task_id="s0", mechanisms=("status", "amount"), coupling=1, seed=1
        ),
        generate_synthetic_task(
            task_id="s1",
            mechanisms=("status", "amount", "identity", "empty"),
            coupling=2,
            seed=3,
        ),
    ]


def test_run_cell_plan_covers_a_to_e_specs():
    tasks = _matrix()
    results = {}
    for task in tasks:
        results.update(run_cell_plan(task, seed=0, budget=4))
    # 16 subsets + A orderings + C lineage
    assert len(results) >= 20


def test_experiment_a_composition_helps_on_coupled_tasks():
    tasks = _matrix()
    results = {}
    for task in tasks:
        results.update(run_cell_plan(task, seed=0, budget=4))
    report = analyze_A(tasks, results, [0], budget=4)
    assert report["wins"] > report["losses"]
    assert report["conclusion"] == "composition_helps"
    assert report["order_sensitive_cells"] == 0


def test_experiment_b_coverage_correlates_with_gain():
    tasks = _matrix()
    results = {}
    for task in tasks:
        results.update(run_cell_plan(task, seed=0, budget=4))
    report = analyze_B(tasks, results, [0], budget=4)
    assert report["cells"] >= 32
    assert report["spearman_best_delta"]["rho"] > 0.3
    # Ladder: mean best_delta at full coverage exceeds zero coverage.
    ladder = report["ladder"]
    assert ladder["1.0"]["mean_best_delta"] > ladder["0.0"]["mean_best_delta"]


def test_experiment_c_lineage_sufficient():
    tasks = _matrix()
    results = {}
    for task in tasks:
        results.update(run_cell_plan(task, seed=0, budget=8))
    report = analyze_C(tasks, results, [0], budget=8)
    assert report["composed_strictly_better_cells"] == 0
    assert report["conclusion"] == "lineage_sufficient"


def test_experiment_d_detects_coupled_pair_synergy():
    tasks = _matrix()
    results = {}
    for task in tasks:
        results.update(run_cell_plan(task, seed=0, budget=4))
    report = analyze_D(tasks, results, [0], budget=4)
    pair = report["pairs"].get("canonicalize_status|reject_invalid_amounts")
    assert pair is not None
    assert pair["mean_interaction"] > 0
    assert report["synergistic_count"] >= 1


def test_experiment_e_runs_and_reports_per_operator():
    tasks = _matrix()
    results = {}
    for task in tasks:
        results.update(run_cell_plan(task, seed=0, budget=4))
    report = analyze_E(tasks, results, [0], budget=4)
    assert set(report["operators"]) == {
        "canonicalize_status",
        "reject_invalid_amounts",
        "normalize_identity",
        "drop_empty_identity",
    }
    for value in report["operators"].values():
        assert len(value["cumulative_gain"]) == len(tasks)
