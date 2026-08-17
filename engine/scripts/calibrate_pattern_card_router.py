#!/usr/bin/env python3
"""Build immutable feedback-only PatternCard router calibration evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skill_evolution_loop.contracts import ContractError, canonical_json, sha256_json
from skill_evolution_loop.pattern_card_calibration import (
    PatternCardLabel,
    calibrate_pattern_card_router,
)


def _load_object(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{name} is unreadable") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value, raw


def _build_report(
    *, taskset_path: Path, skill_path: Path, labels_path: Path
) -> dict[str, Any]:
    taskset, taskset_raw = _load_object(taskset_path, name="taskset")
    wrapper, skill_raw = _load_object(skill_path, name="Skill revision")
    label_set, labels_raw = _load_object(labels_path, name="calibration labels")
    if label_set.get("holdout_task_ids_included") is not False:
        raise ContractError("calibration labels must exclude holdout task IDs")
    tasks = taskset.get("tasks")
    rows = label_set.get("labels")
    if not isinstance(tasks, list) or not isinstance(rows, list):
        raise ContractError("taskset or calibration labels are malformed")
    feedback = {
        row.get("task_id"): row
        for row in tasks
        if isinstance(row, dict) and row.get("cohort") == "feedback"
    }
    if len(feedback) != 30:
        raise ContractError("calibration requires the frozen 30-task feedback cohort")
    label_ids = [row.get("task_id") for row in rows if isinstance(row, dict)]
    if len(rows) != 30 or set(label_ids) != set(feedback):
        raise ContractError("labels must cover every feedback task exactly once")
    revision = wrapper.get("next_revision")
    if not isinstance(revision, dict) or not isinstance(
        revision.get("skill_text"), str
    ):
        raise ContractError("Skill revision has no frozen skill_text")
    labels = tuple(
        PatternCardLabel(
            task_id=row["task_id"],
            instruction=feedback[row["task_id"]]["instruction"],
            applicable_card_numbers=tuple(row["applicable_card_numbers"]),
        )
        for row in rows
    )
    calibration = calibrate_pattern_card_router(
        skill_text=revision["skill_text"], labels=labels
    )
    content = {
        "schema_version": 1,
        "run_id": "round3-r080-pattern-card-fpr",
        "calibration": calibration,
        "source_sha256": {
            "taskset": hashlib.sha256(taskset_raw).hexdigest(),
            "skill_revision": hashlib.sha256(skill_raw).hexdigest(),
            "labels": hashlib.sha256(labels_raw).hexdigest(),
        },
        "holdout_task_ids_included": False,
        "network_calls_performed": False,
        "cloud_resources_started": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taskset", required=True, type=Path)
    parser.add_argument("--skill-revision", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = _build_report(
        taskset_path=args.taskset,
        skill_path=args.skill_revision,
        labels_path=args.labels,
    )
    if args.output.exists():
        existing, _raw = _load_object(args.output, name="existing calibration report")
        if existing != report:
            raise ContractError("existing calibration report does not match inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
