"""Experiment A-E cell plans, execution, and analysis.

One unified cell sweep per (task, seed) feeds all five experiments:

- B (coverage correlation): all 2^4 operator subsets.
- A (composition vs single): single subsets + both orderings of every
  mechanism pair, compared against the composed subset.
- C (lineage vs composed): a rotating single-operator lineage cell plus the
  full-mechanism composed cell.
- D (epistasis): pooled attempt events from every cell.
- E (cross-task threshold): pooled events across tasks per operator.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

from .analysis import (
    bootstrap_ci,
    bootstrap_correlation_ci,
    changepoint_cumulative,
    exact_binomial_p,
    mean,
    pearson,
    permutation_correlation_pvalue,
    spearman,
    threshold_vs_linear,
)
from .operators import OPERATOR_IDS, coverage_of
from .search import (
    CellResult,
    cell_key,
    run_cell,
    run_lineage_cell,
)
from .tasks import TaskSpec

EMPTY_SET: tuple[str, ...] = ()


def mechanisms_to_operators(mechanisms: Iterable[str]) -> tuple[str, ...]:
    mapping = {
        "status": "canonicalize_status",
        "amount": "reject_invalid_amounts",
        "identity": "normalize_identity",
        "empty": "drop_empty_identity",
    }
    return tuple(mapping[m] for m in mechanisms if m in mapping)


def plan_cell_specs(
    task: TaskSpec, seed: int, budget: int
) -> list[tuple[str, dict[str, Any]]]:
    """Return [(kind, kwargs)] for every cell needed by A-E on one (task, seed)."""
    specs: list[tuple[str, dict[str, Any]]] = []
    operators = OPERATOR_IDS
    # B: all subsets (includes empty, singletons, full set).
    for r in range(0, len(operators) + 1):
        for subset in itertools.combinations(operators, r):
            specs.append(
                (
                    "B",
                    {
                        "operator_ids": tuple(sorted(subset)),
                        "mode": "deterministic",
                        "lineage": False,
                    },
                )
            )
    # A: both orderings of each mechanism pair present in the task.
    task_ops = mechanisms_to_operators(task.mechanisms)
    for op_i, op_j in itertools.combinations(task_ops, 2):
        specs.append(
            (
                "A",
                {
                    "operator_ids": (op_i, op_j),
                    "mode": "deterministic",
                    "lineage": False,
                },
            )
        )
        specs.append(
            (
                "A",
                {
                    "operator_ids": (op_j, op_i),
                    "mode": "deterministic",
                    "lineage": False,
                },
            )
        )
    # C: lineage (rotating single operators) over the task's mechanisms.
    if task_ops:
        specs.append(
            ("C", {"operator_ids": task_ops, "mode": "lineage", "lineage": True})
        )
    return specs


def dedupe_specs(
    specs: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[tuple[str, dict[str, Any]]] = []
    for kind, spec in specs:
        key = (spec["operator_ids"], spec.get("lineage", False))
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, spec))
    return out


def run_cell_plan(
    task: TaskSpec,
    seed: int,
    budget: int,
    *,
    mode: str = "deterministic",
    model: Any = None,
    skip_existing: set[str] | None = None,
    llm_style: str = "scaffold",
) -> dict[str, CellResult]:
    specs = dedupe_specs(plan_cell_specs(task, seed, budget))
    results: dict[str, CellResult] = {}
    for kind, spec in specs:
        key = cell_key(
            task.task_id,
            seed,
            spec["operator_ids"],
            "lineage" if spec.get("lineage") else mode,
            budget,
        )
        if skip_existing is not None and key in skip_existing:
            continue
        if spec.get("lineage"):
            result = run_lineage_cell(
                task, spec["operator_ids"], seed=seed, budget=budget
            )
        else:
            result = run_cell(
                task,
                spec["operator_ids"],
                seed=seed,
                budget=budget,
                mode=mode,
                model=model,
                llm_style=llm_style,
            )
        results[key] = result
    return results


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def _cell_spec_from_key(key: str, results: dict[str, CellResult]) -> CellResult:
    return results[key]


def analyze_A(
    tasks: Sequence[TaskSpec],
    results_by_key: dict[str, CellResult],
    seeds: Sequence[int],
    *,
    budget: int,
    mode: str = "deterministic",
) -> dict[str, Any]:
    """Composition vs single: is the conjunction > sum of parts?"""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_ops = mechanisms_to_operators(task.mechanisms)
        for op_i, op_j in itertools.combinations(task_ops, 2):
            for seed in seeds:
                initial = _initial_passed(task)
                single_i = _best_passed(
                    results_by_key,
                    task.task_id,
                    seed,
                    (op_i,),
                    budget=budget,
                    mode=mode,
                )
                single_j = _best_passed(
                    results_by_key,
                    task.task_id,
                    seed,
                    (op_j,),
                    budget=budget,
                    mode=mode,
                )
                composed_ab = _best_passed(
                    results_by_key,
                    task.task_id,
                    seed,
                    tuple(sorted((op_i, op_j))),
                    budget=budget,
                    mode=mode,
                )
                composed_ba = _best_passed(
                    results_by_key,
                    task.task_id,
                    seed,
                    tuple(sorted((op_j, op_i))),
                    budget=budget,
                    mode=mode,
                )
                composed = max(composed_ab, composed_ba)
                additive_prediction = (
                    initial + (single_i - initial) + (single_j - initial)
                )
                rows.append(
                    {
                        "task_id": task.task_id,
                        "seed": seed,
                        "pair": [op_i, op_j],
                        "initial": initial,
                        "single_i": single_i,
                        "single_j": single_j,
                        "best_single": max(single_i, single_j),
                        "composed_ab": composed_ab,
                        "composed_ba": composed_ba,
                        "composed": composed,
                        "order_sensitive": composed_ab != composed_ba,
                        "delta_over_best": composed - max(single_i, single_j),
                        "delta_over_additive": composed - additive_prediction,
                    }
                )
    wins = sum(1 for r in rows if r["delta_over_best"] > 0)
    losses = sum(1 for r in rows if r["delta_over_best"] < 0)
    ties = sum(1 for r in rows if r["delta_over_best"] == 0)
    synergy = sum(1 for r in rows if r["delta_over_additive"] > 0)
    order_sensitive = sum(1 for r in rows if r["order_sensitive"])
    return {
        "rows": rows,
        "cells": len(rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p": exact_binomial_p(wins, losses),
        "synergy_cells": synergy,
        "order_sensitive_cells": order_sensitive,
        "mean_delta_over_best": mean([r["delta_over_best"] for r in rows]),
        "mean_delta_over_additive": mean([r["delta_over_additive"] for r in rows]),
        "conclusion": (
            "composition_helps"
            if wins > losses and exact_binomial_p(wins, losses) < 0.05
            else "no_evidence"
        ),
    }


def analyze_B(
    tasks: Sequence[TaskSpec],
    results_by_key: dict[str, CellResult],
    seeds: Sequence[int],
    *,
    budget: int,
    mode: str = "deterministic",
) -> dict[str, Any]:
    """Coverage -> yield correlation at scale (the model-vendor benchmark)."""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for seed in seeds:
            initial = _initial_passed(task)
            for op_ids in _all_subsets():
                key = cell_key(task.task_id, seed, op_ids, mode, budget)
                result = results_by_key.get(key)
                if result is None:
                    continue
                rows.append(
                    {
                        "task_id": task.task_id,
                        "seed": seed,
                        "operator_ids": list(op_ids),
                        "coverage": coverage_of(op_ids, task.mechanisms),
                        "yield": result.yield_,
                        "best_delta": result.best_passed - initial,
                        "best_passed": result.best_passed,
                        "initial": initial,
                    }
                )
    xs = [r["coverage"] for r in rows]
    ys_yield = [r["yield"] for r in rows]
    ys_delta = [r["best_delta"] for r in rows]
    sp_y, p_y = spearman(xs, ys_yield)
    sp_d, p_d = spearman(xs, ys_delta)
    pe_y, pe_p = pearson(xs, ys_yield)
    ladder: dict[str, dict[str, float]] = {}
    for coverage in sorted({r["coverage"] for r in rows}):
        sub = [r for r in rows if abs(r["coverage"] - coverage) < 1e-9]
        ladder[str(coverage)] = {
            "cells": len(sub),
            "mean_yield": mean([r["yield"] for r in sub]),
            "mean_best_delta": mean([r["best_delta"] for r in sub]),
            "p25_best_delta": _quantile([r["best_delta"] for r in sub], 0.25),
            "p75_best_delta": _quantile([r["best_delta"] for r in sub], 0.75),
        }
    return {
        "cells": len(rows),
        "spearman_yield": {"rho": sp_y, "p": p_y},
        "spearman_best_delta": {"rho": sp_d, "p": p_d},
        "pearson_yield": {"r": pe_y, "p": pe_p},
        "spearman_yield_ci": bootstrap_correlation_ci(xs, ys_yield, kind="spearman"),
        "permutation_p_yield": permutation_correlation_pvalue(
            xs, ys_yield, kind="spearman"
        ),
        "threshold_yield": threshold_vs_linear(xs, ys_yield),
        "ladder": ladder,
        "rows": rows,
    }


def analyze_C(
    tasks: Sequence[TaskSpec],
    results_by_key: dict[str, CellResult],
    seeds: Sequence[int],
    *,
    budget: int,
    mode: str = "deterministic",
) -> dict[str, Any]:
    """Lineage (sequential single ops) vs composed (simultaneous) best score."""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_ops = mechanisms_to_operators(task.mechanisms)
        if not task_ops:
            continue
        for seed in seeds:
            lineage = _best_passed(
                results_by_key,
                task.task_id,
                seed,
                task_ops,
                lineage=True,
                budget=budget,
            )
            composed = _best_passed(
                results_by_key,
                task.task_id,
                seed,
                tuple(sorted(task_ops)),
                budget=budget,
                mode=mode,
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    "seed": seed,
                    "operators": list(task_ops),
                    "lineage_best": lineage,
                    "composed_best": composed,
                    "lineage_reaches_composed": lineage >= composed,
                    "delta": lineage - composed,
                }
            )
    losses = sum(1 for r in rows if r["delta"] < 0)
    return {
        "rows": rows,
        "cells": len(rows),
        "lineage_reaches_composed_cells": sum(
            1 for r in rows if r["lineage_reaches_composed"]
        ),
        "mean_delta": mean([r["delta"] for r in rows]),
        "composed_strictly_better_cells": losses,
        "conclusion": (
            "lineage_sufficient"
            if losses == 0 and rows
            else "composition_required"
            if losses > 0
            else "insufficient_data"
        ),
    }


def analyze_D(
    tasks: Sequence[TaskSpec],
    results_by_key: dict[str, CellResult],
    seeds: Sequence[int],
    *,
    budget: int,
    mode: str = "deterministic",
) -> dict[str, Any]:
    """Epistasis via a 2^k factorial design at cell level.

    For each (task, seed, mechanism pair) the design points are the cells
    for empty, {i}, {j}, {i,j}.  The interaction is::

        gain(i,j) - gain(i) - gain(j) + gain(empty)

    where gain(S) = best_passed(S) - initial_passed.  A positive interaction
    means the conjunction improves more than the sum of the two main effects
    (the quantitative "emergence" signature).
    """
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_ops = mechanisms_to_operators(task.mechanisms)
        initial = _initial_passed(task)
        for op_i, op_j in itertools.combinations(task_ops, 2):
            for seed in seeds:
                gain_empty = (
                    _best_passed(
                        results_by_key,
                        task.task_id,
                        seed,
                        EMPTY_SET,
                        budget=budget,
                        mode=mode,
                    )
                    - initial
                )
                gain_i = (
                    _best_passed(
                        results_by_key,
                        task.task_id,
                        seed,
                        (op_i,),
                        budget=budget,
                        mode=mode,
                    )
                    - initial
                )
                gain_j = (
                    _best_passed(
                        results_by_key,
                        task.task_id,
                        seed,
                        (op_j,),
                        budget=budget,
                        mode=mode,
                    )
                    - initial
                )
                gain_ij = (
                    _best_passed(
                        results_by_key,
                        task.task_id,
                        seed,
                        tuple(sorted((op_i, op_j))),
                        budget=budget,
                        mode=mode,
                    )
                    - initial
                )
                interaction = gain_ij - gain_i - gain_j + gain_empty
                rows.append(
                    {
                        "task_id": task.task_id,
                        "seed": seed,
                        "pair": [op_i, op_j],
                        "gain_empty": gain_empty,
                        "gain_i": gain_i,
                        "gain_j": gain_j,
                        "gain_ij": gain_ij,
                        "interaction": interaction,
                    }
                )
    pairs_report: dict[str, Any] = {}
    for op_i, op_j in itertools.combinations(OPERATOR_IDS, 2):
        sub = [r for r in rows if set(r["pair"]) == {op_i, op_j}]
        if not sub:
            continue
        interactions = [r["interaction"] for r in sub]
        ci = bootstrap_ci(interactions, mean, seed=20260802)
        pairs_report[f"{op_i}|{op_j}"] = {
            "cells": len(sub),
            "mean_interaction": ci["estimate"],
            "ci_lower": ci["lower"],
            "ci_upper": ci["upper"],
            "positive_cells": sum(1 for v in interactions if v > 0),
            "synergy": bool(ci["lower"] > 0),
            "mean_gain_i": mean([r["gain_i"] for r in sub]),
            "mean_gain_j": mean([r["gain_j"] for r in sub]),
            "mean_gain_ij": mean([r["gain_ij"] for r in sub]),
        }
    synergistic = [k for k, v in pairs_report.items() if v["synergy"]]
    return {
        "design": "2k-factorial-cell-level",
        "rows": rows,
        "cells": len(rows),
        "pairs": pairs_report,
        "synergistic_pairs": synergistic,
        "synergistic_count": len(synergistic),
        "mean_interaction_all": mean([r["interaction"] for r in rows]),
        "conclusion": "epistasis_present" if synergistic else "additive_only",
    }


def analyze_E(
    tasks: Sequence[TaskSpec],
    results_by_key: dict[str, CellResult],
    seeds: Sequence[int],
    *,
    budget: int,
    mode: str = "deterministic",
) -> dict[str, Any]:
    """Cross-task validation curve: per-operator cumulative gain vs #tasks.

    For each operator the per-task marginal gain is the mean best-delta of the
    singleton cell over seeds.  The cumulative curve over the task-matrix order
    is tested for a jump (changepoint).  A jump would mean the operator only
    starts contributing after a critical number of tasks; a linear accumulation
    means per-task application is independent (no cross-task coupling in the
    deterministic search -- cross-task thresholds then live in the PSI/lessons
    layer, not in per-task search).
    """
    report: dict[str, Any] = {}
    for op in OPERATOR_IDS:
        per_task: list[dict[str, Any]] = []
        for task in tasks:
            gains = []
            for seed in seeds:
                best = _best_passed(
                    results_by_key, task.task_id, seed, (op,), budget=budget, mode=mode
                )
                if best >= 0:
                    gains.append(best - _initial_passed(task))
            if gains:
                per_task.append({"task_id": task.task_id, "mean_gain": mean(gains)})
        cumulative: list[float] = []
        running = 0.0
        for row in per_task:
            running += row["mean_gain"]
            cumulative.append(running)
        cp = changepoint_cumulative(cumulative)
        report[op] = {
            "tasks_observed": len(per_task),
            "per_task_gain": [round(row["mean_gain"], 3) for row in per_task],
            "cumulative_gain": [round(v, 3) for v in cumulative],
            "final_cumulative_gain": cumulative[-1] if cumulative else 0.0,
            "changepoint": cp,
        }
    thresholds = {
        op: v["changepoint"].get("max_acceleration_index")
        for op, v in report.items()
        if v["changepoint"].get("supported")
    }
    return {
        "operators": report,
        "cross_task_thresholds": thresholds,
        "conclusion": (
            "threshold_detected" if thresholds else "linear_accumulation_no_threshold"
        ),
    }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _all_subsets() -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for r in range(0, len(OPERATOR_IDS) + 1):
        for subset in itertools.combinations(OPERATOR_IDS, r):
            out.append(tuple(sorted(subset)))
    return out


def _initial_passed(task: TaskSpec) -> int:
    return int(task.score_source(task.initial_source).get("passed_cases", 0))


def _best_passed(
    results: dict[str, CellResult],
    task_id: str,
    seed: int,
    op_ids: tuple[str, ...],
    *,
    lineage: bool = False,
    budget: int,
    mode: str = "deterministic",
) -> int:
    key = cell_key(task_id, seed, op_ids, "lineage" if lineage else mode, budget)
    result = results.get(key)
    if result is not None:
        return result.best_passed
    return -1


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round(q * (len(ordered) - 1)))
    return float(ordered[index])


def serialize_events(results: Iterable[CellResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for result in results:
        for event in result.events:
            out.append(
                {
                    "cell_key": cell_key(
                        result.task_id,
                        result.seed,
                        result.operator_ids,
                        result.mode,
                        result.budget,
                    ),
                    "task_id": result.task_id,
                    "seed": result.seed,
                    "operator_ids": list(result.operator_ids),
                    "mode": result.mode,
                    "budget": result.budget,
                    **event.to_dict(),
                }
            )
    return out
