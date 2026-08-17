"""Leakage-gated, resumable native evaluation for Round 1 feedback cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .p1_native import (
    PatchEvaluator,
    _read_native_cell,
    evaluate_p1_feedback_cell_native,
    evaluate_p1_holdout_cell_native,
    summarize_native_cells,
)
from .round1_run import (
    _load_routes,
    _load_taskset,
    load_round1_feedback_gain_gate,
)


def run_round1_feedback_native(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    evidence_root: Path,
    pool_root: Path,
    evaluator: PatchEvaluator,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Evaluate only frozen Round 1 feedback attempts with official judges."""

    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ContractError("max_cells must be a positive integer")
    taskset = _load_taskset(taskset_path)
    routes = _load_routes(routes_path, taskset)
    feedback = tuple(task for task in taskset.tasks if task.cohort == "feedback")
    holdout = tuple(task for task in taskset.tasks if task.cohort == "holdout")
    if len(feedback) < 3 or len(holdout) < 3:
        raise ContractError("Round 1 native gate requires at least 3+3 tasks")

    executed = 0
    experiment = experiment_root.resolve()
    root = evidence_root.resolve()
    for task in feedback:
        mechanism = routes[task.task_id]
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            source = experiment / "cells" / task.task_id / condition_id
            if not source.is_dir():
                continue
            target = root / "cells" / task.task_id / condition_id
            if target.is_dir():
                _read_native_cell(target)
                continue
            if max_cells is not None and executed >= max_cells:
                continue
            evaluate_p1_feedback_cell_native(
                manifest_path=taskset_path,
                experiment_root=experiment,
                evidence_root=root,
                pool_root=pool_root,
                evaluator=evaluator,
                task_id=task.task_id,
                condition_id=condition_id,
            )
            executed += 1
            _write_summary(
                root,
                _summarize(
                    taskset_fingerprint=taskset.fingerprint,
                    routes=routes,
                    routes_path=routes_path,
                    experiment=experiment,
                    evidence=root,
                    feedback=feedback,
                ),
            )

    summary = _summarize(
        taskset_fingerprint=taskset.fingerprint,
        routes=routes,
        routes_path=routes_path,
        experiment=experiment,
        evidence=root,
        feedback=feedback,
    )
    _write_summary(root, summary)
    return summary


def run_round1_holdout_native(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    feedback_gain_path: Path,
    evidence_root: Path,
    pool_root: Path,
    evaluator: PatchEvaluator,
    max_cells: int | None = None,
    task_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate all frozen holdout A/B pairs after feedback unlocked them."""

    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ContractError("max_cells must be a positive integer")
    taskset = _load_taskset(taskset_path)
    routes = _load_routes(routes_path, taskset)
    feedback = tuple(task for task in taskset.tasks if task.cohort == "feedback")
    holdout = tuple(task for task in taskset.tasks if task.cohort == "holdout")
    if len(feedback) < 3 or len(holdout) < 3:
        raise ContractError("Round 1 holdout native gate requires at least 3+3 tasks")
    selected_ids = _select_holdout_task_ids(holdout, task_ids)
    selected_holdout = tuple(task for task in holdout if task.task_id in selected_ids)
    shard = task_ids is not None
    gain = load_round1_feedback_gain_gate(feedback_gain_path, taskset)
    experiment = experiment_root.resolve()
    holdout_summary = _load_holdout_experiment_summary(
        experiment / ("HOLDOUT-PROGRESS.json" if shard else "HOLDOUT-SUMMARY.json"),
        taskset_fingerprint=taskset.fingerprint,
        feedback_gain_summary_sha256=gain["summary_sha256"],
        planned_cells=len(holdout) * 2,
        allow_partial=shard,
        minimum_completed_cells=len(selected_holdout) * 2,
    )
    root = evidence_root.resolve()
    executed = 0
    for task in selected_holdout:
        mechanism = routes[task.task_id]
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            source = experiment / "cells" / task.task_id / condition_id
            if not source.is_dir():
                raise ContractError("Round 1 holdout experiment cell is missing")
            target = root / "cells" / task.task_id / condition_id
            if target.is_dir():
                _read_native_cell(target)
                continue
            if max_cells is not None and executed >= max_cells:
                continue
            evaluate_p1_holdout_cell_native(
                manifest_path=taskset_path,
                experiment_root=experiment,
                evidence_root=root,
                pool_root=pool_root,
                evaluator=evaluator,
                task_id=task.task_id,
                condition_id=condition_id,
                feedback_gain_summary_sha256=gain["summary_sha256"],
            )
            executed += 1
            _write_summary(
                root,
                _summarize_holdout(
                    taskset_fingerprint=taskset.fingerprint,
                    routes=routes,
                    routes_path=routes_path,
                    evidence=root,
                    holdout=selected_holdout,
                    feedback_gain=gain,
                    holdout_experiment_summary=holdout_summary,
                    full_capability=not shard,
                    selected_task_ids=(
                        tuple(task.task_id for task in selected_holdout)
                        if shard
                        else None
                    ),
                ),
            )
    summary = _summarize_holdout(
        taskset_fingerprint=taskset.fingerprint,
        routes=routes,
        routes_path=routes_path,
        evidence=root,
        holdout=selected_holdout,
        feedback_gain=gain,
        holdout_experiment_summary=holdout_summary,
        full_capability=not shard,
        selected_task_ids=(
            tuple(task.task_id for task in selected_holdout) if shard else None
        ),
    )
    _write_summary(root, summary)
    return summary


def project_round1_holdout_native_summary(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    feedback_gain_path: Path,
    evidence_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Recompute only the aggregation layer over immutable holdout native cells.

    This is the migration path for a frozen native evidence root when the summary
    contract changes. It validates the old summary against the current cell
    fingerprint and records that no native judge was rerun.
    """

    taskset = _load_taskset(taskset_path)
    routes = _load_routes(routes_path, taskset)
    holdout = tuple(task for task in taskset.tasks if task.cohort == "holdout")
    if len(holdout) < 3:
        raise ContractError("Round 1 holdout projection requires at least 3 tasks")
    feedback_gain = load_round1_feedback_gain_gate(feedback_gain_path, taskset)
    experiment = experiment_root.resolve()
    holdout_experiment_summary = _load_holdout_experiment_summary(
        experiment / "HOLDOUT-SUMMARY.json",
        taskset_fingerprint=taskset.fingerprint,
        feedback_gain_summary_sha256=feedback_gain["summary_sha256"],
        planned_cells=len(holdout) * 2,
    )
    evidence = evidence_root.resolve()
    projected = _summarize_holdout(
        taskset_fingerprint=taskset.fingerprint,
        routes=routes,
        routes_path=routes_path,
        evidence=evidence,
        holdout=holdout,
        feedback_gain=feedback_gain,
        holdout_experiment_summary=holdout_experiment_summary,
    )
    source_path = evidence / "SUMMARY.json"
    source = _load_native_summary(source_path)
    if (
        source.get("evaluation_scope") != "round1-full-capability"
        or source.get("taskset_fingerprint") != taskset.fingerprint
        or source.get("completed_holdout_cells") != projected["completed_holdout_cells"]
        or source.get("cell_evidence_fingerprint")
        != projected["cell_evidence_fingerprint"]
    ):
        raise ContractError("source native summary does not match immutable cells")

    content = {
        **{key: value for key, value in projected.items() if key != "summary_sha256"},
        "aggregation_projection": {
            "schema_version": 1,
            "contract": "holdout-safety-qualified-v1",
            "source_native_summary_sha256": source["summary_sha256"],
            "source_native_summary_file_sha256": hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            "source_native_evidence_root": str(evidence),
            "native_cells_reexecuted": False,
            "network_calls_performed": False,
        },
    }
    report = {**content, "summary_sha256": sha256_json(content)}
    _write_projected_summary(output_path.resolve(), report)
    return report


def retry_round1_holdout_native_failures(
    *,
    taskset_path: Path,
    routes_path: Path,
    experiment_root: Path,
    feedback_gain_path: Path,
    source_evidence_root: Path,
    retry_evidence_root: Path,
    pool_root: Path,
    evaluator: PatchEvaluator,
    output_path: Path,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Retry only infrastructure-failed holdout cells in a new evidence root.

    The original cells remain immutable.  The returned summary is an append-only
    projection that overlays retry attempts on the failed source cells and binds
    both evidence roots by SHA.
    """

    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ContractError("max_cells must be a positive integer")
    taskset = _load_taskset(taskset_path)
    routes = _load_routes(routes_path, taskset)
    holdout = tuple(task for task in taskset.tasks if task.cohort == "holdout")
    if len(holdout) < 3:
        raise ContractError("Round 1 holdout retry requires at least 3 tasks")
    feedback_gain = load_round1_feedback_gain_gate(feedback_gain_path, taskset)
    experiment = experiment_root.resolve()
    holdout_experiment_summary = _load_holdout_experiment_summary(
        experiment / "HOLDOUT-SUMMARY.json",
        taskset_fingerprint=taskset.fingerprint,
        feedback_gain_summary_sha256=feedback_gain["summary_sha256"],
        planned_cells=len(holdout) * 2,
    )
    source = source_evidence_root.resolve()
    source_summary_path = source / "SUMMARY.json"
    source_summary = _load_native_summary(source_summary_path)
    if (
        source_summary.get("evaluation_scope") != "round1-full-capability"
        or source_summary.get("status") != "complete"
        or source_summary.get("taskset_fingerprint") != taskset.fingerprint
        or source_summary.get("completed_holdout_cells") != len(holdout) * 2
    ):
        raise ContractError("source holdout native summary boundary is invalid")

    source_cells: dict[tuple[str, str], dict[str, Any]] = {}
    failed: list[tuple[Any, str]] = []
    for task in holdout:
        mechanism = routes[task.task_id]
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            cell = _read_native_cell(source / "cells" / task.task_id / condition_id)
            source_cells[(task.task_id, condition_id)] = cell
            if cell.get("native_evaluator_failure") is not None:
                failed.append((task, condition_id))
    if len(failed) != source_summary.get("native_evaluator_failure_count"):
        raise ContractError("source holdout evaluator failure count drifted")

    retry = retry_evidence_root.resolve()
    executed = 0
    for task, condition_id in failed:
        target = retry / "cells" / task.task_id / condition_id
        if target.is_dir():
            _read_native_cell(target)
            continue
        if max_cells is not None and executed >= max_cells:
            continue
        evaluate_p1_holdout_cell_native(
            manifest_path=taskset_path,
            experiment_root=experiment,
            evidence_root=retry,
            pool_root=pool_root,
            evaluator=evaluator,
            task_id=task.task_id,
            condition_id=condition_id,
            feedback_gain_summary_sha256=feedback_gain["summary_sha256"],
        )
        executed += 1

    overlaid = dict(source_cells)
    retry_cells: list[dict[str, Any]] = []
    for task, condition_id in failed:
        target = retry / "cells" / task.task_id / condition_id
        if target.is_dir():
            cell = _read_native_cell(target)
            overlaid[(task.task_id, condition_id)] = cell
            retry_cells.append(cell)
    projected = _summarize_holdout_cells(
        taskset_fingerprint=taskset.fingerprint,
        routes_path=routes_path,
        cells=list(overlaid.values()),
        planned_cells=len(holdout) * 2,
        feedback_gain=feedback_gain,
        holdout_experiment_summary=holdout_experiment_summary,
    )
    content = {
        **{key: value for key, value in projected.items() if key != "summary_sha256"},
        "aggregation_projection": {
            "schema_version": 1,
            "contract": "native-infrastructure-retry-overlay-v1",
            "source_native_summary_sha256": source_summary["summary_sha256"],
            "source_native_summary_file_sha256": hashlib.sha256(
                source_summary_path.read_bytes()
            ).hexdigest(),
            "source_native_evidence_root": str(source),
            "retry_native_evidence_root": str(retry),
            "source_failure_count": len(failed),
            "replaced_failure_count": len(retry_cells),
            "retry_cell_evidence_fingerprint": sha256_json(
                [
                    {
                        "task_id": row["task_id"],
                        "condition_id": row["condition_id"],
                        "native_cell_sha256": row["evidence_sha256"],
                    }
                    for row in sorted(
                        retry_cells,
                        key=lambda item: (item["task_id"], item["condition_id"]),
                    )
                ]
            ),
            "native_cells_reexecuted": bool(retry_cells),
            "network_calls_performed": False,
        },
    }
    report = {**content, "summary_sha256": sha256_json(content)}
    _write_projected_summary(output_path.resolve(), report)
    return report


def _summarize(
    *,
    taskset_fingerprint: str,
    routes: dict[str, str],
    routes_path: Path,
    experiment: Path,
    evidence: Path,
    feedback: tuple[Any, ...],
) -> dict[str, Any]:
    cells = []
    generated = 0
    for task in feedback:
        mechanism = routes[task.task_id]
        for teaching in ("baseline", "taught"):
            condition_id = f"{mechanism}-{teaching}"
            if (experiment / "cells" / task.task_id / condition_id).is_dir():
                generated += 1
            target = evidence / "cells" / task.task_id / condition_id
            if target.is_dir():
                cells.append(_read_native_cell(target))
    paired = summarize_native_cells(cells)
    evaluator_failures = sum(
        row.get("native_evaluator_failure") is not None for row in cells
    )
    planned = len(feedback) * 2
    cell_evidence_fingerprint = sha256_json(
        [
            {
                "task_id": row["task_id"],
                "condition_id": row["condition_id"],
                "experiment_cell_sha256": row["experiment_cell_sha256"],
                "native_cell_sha256": row["evidence_sha256"],
            }
            for row in sorted(
                cells, key=lambda item: (item["task_id"], item["condition_id"])
            )
        ]
    )
    content = {
        "schema_version": 1,
        "evaluation_scope": "round1-feedback-only",
        "status": "complete" if len(cells) == planned else "partial",
        "taskset_fingerprint": taskset_fingerprint,
        "routes_file_sha256": hashlib.sha256(
            routes_path.resolve().read_bytes()
        ).hexdigest(),
        "planned_cells": planned,
        "generated_feedback_cells": generated,
        "completed_cells": len(cells),
        "cell_evidence_fingerprint": cell_evidence_fingerprint,
        "native_invocations": sum(row["native_report"] is not None for row in cells),
        "native_evaluator_failure_count": evaluator_failures,
        "native_admitted_pair_count": paired["native_admitted_pair_count"],
        "paired_structural_invalid_pair_count": paired[
            "paired_structural_invalid_pair_count"
        ],
        "native_invocations_avoided_by_pair_gate": paired[
            "native_invocations_avoided_by_pair_gate"
        ],
        "pairs": paired["pairs"],
        "feedback_gain_count": paired["feedback_gain_count"],
        "feedback_gain_gate_passed": (
            paired["feedback_gain_count"] > 0 and evaluator_failures == 0
        ),
        "full_capability_gate_evaluated": False,
        "holdout_cells_opened": False,
        "network_calls_performed": False,
    }
    return {**content, "summary_sha256": sha256_json(content)}


def _summarize_holdout(
    *,
    taskset_fingerprint: str,
    routes: dict[str, str],
    routes_path: Path,
    evidence: Path,
    holdout: tuple[Any, ...],
    feedback_gain: dict[str, Any],
    holdout_experiment_summary: dict[str, Any],
    full_capability: bool = True,
    selected_task_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    cells = []
    for task in holdout:
        mechanism = routes[task.task_id]
        for teaching in ("baseline", "taught"):
            target = evidence / "cells" / task.task_id / f"{mechanism}-{teaching}"
            if target.is_dir():
                cells.append(_read_native_cell(target))
    planned_cells = len(holdout) * 2
    return _summarize_holdout_cells(
        taskset_fingerprint=taskset_fingerprint,
        routes_path=routes_path,
        cells=cells,
        planned_cells=planned_cells,
        feedback_gain=feedback_gain,
        holdout_experiment_summary=holdout_experiment_summary,
        full_capability=full_capability,
        selected_task_ids=selected_task_ids,
    )


def _summarize_holdout_cells(
    *,
    taskset_fingerprint: str,
    routes_path: Path,
    cells: list[dict[str, Any]],
    planned_cells: int,
    feedback_gain: dict[str, Any],
    holdout_experiment_summary: dict[str, Any],
    full_capability: bool = True,
    selected_task_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    paired = summarize_native_cells(cells)
    complete = len(cells) == planned_cells
    regression_count = paired["holdout_regression_count"]
    evaluable_pair_count = paired["holdout_evaluable_pair_count"]
    safety_qualified_pair_count = paired["holdout_safety_qualified_pair_count"]
    safety_failure_count = paired["holdout_safety_failure_count"]
    evaluator_failures = sum(
        row.get("native_evaluator_failure") is not None for row in cells
    )
    cell_evidence_fingerprint = sha256_json(
        [
            {
                "task_id": row["task_id"],
                "condition_id": row["condition_id"],
                "experiment_cell_sha256": row["experiment_cell_sha256"],
                "native_cell_sha256": row["evidence_sha256"],
            }
            for row in sorted(
                cells, key=lambda item: (item["task_id"], item["condition_id"])
            )
        ]
    )
    content = {
        "schema_version": 1,
        "evaluation_scope": (
            "round1-full-capability" if full_capability else "round1-holdout-shard"
        ),
        "status": "complete" if complete else "partial",
        "taskset_fingerprint": taskset_fingerprint,
        "routes_file_sha256": hashlib.sha256(
            routes_path.resolve().read_bytes()
        ).hexdigest(),
        "planned_holdout_cells": planned_cells,
        "completed_holdout_cells": len(cells),
        "cell_evidence_fingerprint": cell_evidence_fingerprint,
        "native_invocations": sum(row["native_report"] is not None for row in cells),
        "native_evaluator_failure_count": evaluator_failures,
        "native_admitted_pair_count": paired["native_admitted_pair_count"],
        "paired_structural_invalid_pair_count": paired[
            "paired_structural_invalid_pair_count"
        ],
        "native_invocations_avoided_by_pair_gate": paired[
            "native_invocations_avoided_by_pair_gate"
        ],
        "pairs": paired["pairs"],
        "feedback_gain_count": feedback_gain["feedback_gain_count"],
        "feedback_cell_evidence_fingerprint": feedback_gain[
            "cell_evidence_fingerprint"
        ],
        "feedback_gain_summary_sha256": feedback_gain["summary_sha256"],
        "holdout_experiment_summary_sha256": holdout_experiment_summary[
            "summary_sha256"
        ],
        "holdout_regression_count": regression_count,
        "holdout_evaluable_pair_count": evaluable_pair_count,
        "holdout_safety_qualified_pair_count": safety_qualified_pair_count,
        "minimum_holdout_evaluable_pairs": 3,
        "minimum_holdout_safety_qualified_pairs": 3,
        "holdout_safety_failure_count": safety_failure_count,
        "holdout_safety_failures": paired["holdout_safety_failures"],
        "capability_gate_passed": (
            full_capability
            and complete
            and safety_qualified_pair_count >= 3
            and regression_count == 0
            and safety_failure_count == 0
            and evaluator_failures == 0
        ),
        "full_capability_gate_evaluated": full_capability and complete,
        "holdout_cells_opened": True,
        "network_calls_performed": False,
    }
    if selected_task_ids is not None:
        content["selected_task_ids"] = list(selected_task_ids)
    return {**content, "summary_sha256": sha256_json(content)}


def _load_holdout_experiment_summary(
    path: Path,
    *,
    taskset_fingerprint: str,
    feedback_gain_summary_sha256: str,
    planned_cells: int,
    allow_partial: bool = False,
    minimum_completed_cells: int | None = None,
) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("Round 1 holdout experiment summary is unreadable") from exc
    if not isinstance(summary, dict):
        raise ContractError("Round 1 holdout experiment summary must be an object")
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != sha256_json(content):
        raise ContractError("Round 1 holdout experiment summary sha256 mismatch")
    completed_cells = summary.get("completed_cells")
    completed_boundary_valid = (
        type(completed_cells) is int
        and completed_cells <= planned_cells
        and (
            completed_cells == planned_cells
            if not allow_partial
            else completed_cells >= (minimum_completed_cells or 0)
        )
    )
    if (
        summary.get("evaluation_scope") != "round1-holdout-only"
        or summary.get("status")
        not in ({"complete", "partial"} if allow_partial else {"complete"})
        or summary.get("taskset_fingerprint") != taskset_fingerprint
        or summary.get("planned_cells") != planned_cells
        or not completed_boundary_valid
        or summary.get("feedback_gain_summary_sha256") != feedback_gain_summary_sha256
        or summary.get("holdout_cells_opened") is not True
        or summary.get("network_calls_performed") is not False
    ):
        raise ContractError("Round 1 holdout experiment summary boundary is invalid")
    return summary


def _select_holdout_task_ids(
    holdout: tuple[Any, ...], requested: tuple[str, ...] | None
) -> set[str]:
    available = {task.task_id for task in holdout}
    if requested is None:
        return available
    selected = set(requested)
    if not selected or len(selected) != len(requested):
        raise ContractError("explicit Round 1 holdout native tasks must be unique")
    outside = sorted(selected - available)
    if outside:
        raise ContractError(
            "explicit native task is outside holdout cohort: " + ", ".join(outside)
        )
    return selected


def _load_native_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("source native summary is unreadable") from exc
    if not isinstance(summary, dict):
        raise ContractError("source native summary must be an object")
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != sha256_json(content):
        raise ContractError("source native summary sha256 mismatch")
    return summary


def _write_projected_summary(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("holdout native projection is unreadable") from exc
        if existing != report:
            raise ContractError("frozen holdout native projection changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report) + "\n", encoding="utf-8")


def _write_summary(root: Path, summary: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    name = "SUMMARY.json" if summary["status"] == "complete" else "PROGRESS.json"
    target = root / name
    if target.exists() and name == "SUMMARY.json":
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                "Round 1 feedback native summary is unreadable"
            ) from exc
        if existing != summary:
            raise ContractError("frozen Round 1 feedback native summary changed")
        return
    target.write_text(canonical_json(summary) + "\n", encoding="utf-8")
