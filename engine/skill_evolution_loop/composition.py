"""Compose immutable experiment cells across TaskSet revisions without rerunning them."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .eval_manifest import EvaluationTaskSet
from .target_selection import TargetSelectionManifest


@dataclass(frozen=True)
class ExperimentEvidenceSource:
    """One cohort's frozen experiment evidence and its original qualification."""

    cohort: str
    experiment_root: Path
    taskset_path: Path
    target_selection_path: Path


def compose_experiment_evidence(
    *,
    taskset_path: Path,
    target_selection_path: Path,
    sources: list[ExperimentEvidenceSource],
    output_path: Path,
) -> dict[str, Any]:
    """Validate and index cells from compatible source runs without copying evidence."""
    taskset = _load_taskset(taskset_path)
    selection = _load_selection(target_selection_path, taskset)
    expected_cohorts = set(taskset.cohort_counts)
    source_cohorts = [source.cohort for source in sources]
    if len(source_cohorts) != len(set(source_cohorts)):
        raise ContractError("composition evidence cohorts must be unique")
    if set(source_cohorts) != expected_cohorts:
        raise ContractError("composition requires exactly one source per cohort")

    final_tasks = {task.task_id: task for task in taskset.tasks}
    final_records = {record.task_id: record for record in selection.records}
    condition_fingerprints: dict[str, str] = {}
    cells: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for source in sorted(sources, key=lambda row: row.cohort):
        source_taskset = _load_taskset(source.taskset_path)
        source_selection = _load_selection(source.target_selection_path, source_taskset)
        source_tasks = {task.task_id: task for task in source_taskset.tasks}
        source_records = {record.task_id: record for record in source_selection.records}
        cohort_tasks = sorted(
            task.task_id for task in taskset.tasks if task.cohort == source.cohort
        )
        for task_id in cohort_tasks:
            if task_id not in source_tasks or task_id not in source_records:
                raise ContractError("composition source is missing a qualified task")
            if source_tasks[task_id].fingerprint != final_tasks[task_id].fingerprint:
                raise ContractError("composition source task fingerprint mismatch")
            if (
                source_records[task_id].fingerprint
                != final_records[task_id].fingerprint
            ):
                raise ContractError(
                    "composition source target selection fingerprint mismatch"
                )

        root = source.experiment_root.resolve()
        source_cells: list[dict[str, Any]] = []
        for task_id in cohort_tasks:
            task_root = root / "cells" / task_id
            if not task_root.is_dir():
                raise ContractError("composition source task evidence is missing")
            attempts = sorted(task_root.glob("*/ATTEMPT.json"))
            if not attempts:
                raise ContractError("composition source has no condition evidence")
            for attempt_path in attempts:
                report = load_frozen_attempt(attempt_path)
                condition = report.get("condition", {})
                condition_id = condition.get("condition_id")
                condition_fingerprint = condition.get("fingerprint")
                if not isinstance(condition_id, str) or not isinstance(
                    condition_fingerprint, str
                ):
                    raise ContractError("composition condition fields are invalid")
                existing = condition_fingerprints.setdefault(
                    condition_id, condition_fingerprint
                )
                if existing != condition_fingerprint:
                    raise ContractError("composition condition fingerprint mismatch")
                if report.get("taskset_fingerprint") != source_taskset.fingerprint:
                    raise ContractError(
                        "composition source taskset fingerprint mismatch"
                    )
                if (
                    report.get("qualification_fingerprint")
                    != source_selection.fingerprint
                ):
                    raise ContractError(
                        "composition source qualification fingerprint mismatch"
                    )
                if (
                    report.get("task", {}).get("fingerprint")
                    != final_tasks[task_id].fingerprint
                ):
                    raise ContractError("composition attempt task fingerprint mismatch")
                if report.get("network_calls_performed") is not False:
                    raise ContractError("composition accepts offline evidence only")
                source_cells.append(
                    {
                        "task_id": task_id,
                        "condition_id": condition_id,
                        "evidence_sha256": report["evidence_sha256"],
                        "attempt_path": str(attempt_path.resolve()),
                        "structural_valid": report.get("attempt", {}).get(
                            "structural_valid"
                        ),
                        "failure_reason": report.get("attempt", {}).get(
                            "failure_reason"
                        ),
                    }
                )
        cells.extend(source_cells)
        source_rows.append(
            {
                "cohort": source.cohort,
                "experiment_root": str(root),
                "taskset_id": source_taskset.taskset_id,
                "taskset_fingerprint": source_taskset.fingerprint,
                "qualification_fingerprint": source_selection.fingerprint,
                "cell_count": len(source_cells),
            }
        )

    condition_ids = set(condition_fingerprints)
    expected_cells = {
        (task.task_id, condition_id)
        for task in taskset.tasks
        for condition_id in condition_ids
    }
    actual_cells = [(cell["task_id"], cell["condition_id"]) for cell in cells]
    if (
        len(actual_cells) != len(set(actual_cells))
        or set(actual_cells) != expected_cells
    ):
        raise ContractError("composition cells are incomplete or duplicated")

    metrics: dict[str, Any] = {}
    for condition_id in sorted(condition_ids):
        rows = [cell for cell in cells if cell["condition_id"] == condition_id]
        reasons = Counter(
            str(cell["failure_reason"])
            for cell in rows
            if cell["failure_reason"] is not None
        )
        valid = sum(cell["structural_valid"] is True for cell in rows)
        metrics[condition_id] = {
            "completed": len(rows),
            "structural_valid": valid,
            "structural_rate": valid / len(rows),
            "reason_counts": dict(sorted(reasons.items())),
        }

    content = {
        "schema_version": 1,
        "status": "complete",
        "taskset_id": taskset.taskset_id,
        "taskset_fingerprint": taskset.fingerprint,
        "qualification_fingerprint": selection.fingerprint,
        "cohort_counts": taskset.cohort_counts,
        "condition_fingerprints": dict(sorted(condition_fingerprints.items())),
        "planned_cells": len(expected_cells),
        "completed_cells": len(cells),
        "condition_metrics": metrics,
        "sources": source_rows,
        "cells": sorted(cells, key=lambda row: (row["task_id"], row["condition_id"])),
        "network_calls_performed": False,
    }
    report = {**content, "composition_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError("composition output is unreadable") from exc
        if existing != report:
            raise ContractError("frozen composition output does not match evidence")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def _load_taskset(path: Path) -> EvaluationTaskSet:
    try:
        return EvaluationTaskSet.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("composition taskset is unreadable") from exc


def _load_selection(path: Path, taskset: EvaluationTaskSet) -> TargetSelectionManifest:
    try:
        return TargetSelectionManifest.from_dict(
            json.loads(path.read_text(encoding="utf-8")), taskset=taskset
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("composition target selection is unreadable") from exc


def load_frozen_attempt(path: Path) -> dict[str, Any]:
    """Load one cell only after its report and artifact hashes validate."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("composition attempt evidence is unreadable") from exc
    if not isinstance(report, dict) or "evidence_sha256" not in report:
        raise ContractError("composition attempt evidence fields are invalid")
    content = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if report["evidence_sha256"] != sha256_json(content):
        raise ContractError("composition attempt evidence sha256 mismatch")
    cell_root = path.parent
    artifacts = report.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise ContractError("composition artifact hashes are invalid")
    for name, digest in artifacts.items():
        artifact = cell_root / name
        if not artifact.is_file() or _file_sha256(artifact) != digest:
            raise ContractError("composition artifact sha256 mismatch")
    return report


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
