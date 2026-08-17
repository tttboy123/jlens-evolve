"""Combine independently frozen feedback and holdout selections."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .round1_selection import _verified


def freeze_round1_mixed_selection(
    *,
    feedback_selection_path: Path,
    holdout_selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze one 3+3-or-larger selection without crossing cohort identities."""

    feedback_raw, feedback = _verified(
        feedback_selection_path, label="Round 1 feedback selection"
    )
    holdout_raw, holdout = _verified(
        holdout_selection_path, label="Round 1 holdout selection"
    )
    feedback_tasks = feedback.get("tasks")
    holdout_tasks = holdout.get("tasks")
    if (
        feedback.get("status") != "frozen"
        or holdout.get("status") != "frozen"
        or not isinstance(feedback_tasks, list)
        or not isinstance(holdout_tasks, list)
        or len(feedback_tasks) < 3
        or len(holdout_tasks) < 3
        or any(row.get("cohort") != "feedback" for row in feedback_tasks)
        or any(row.get("cohort") != "holdout" for row in holdout_tasks)
    ):
        raise ContractError("Round 1 mixed selection cohorts are invalid")
    tasks = [*feedback_tasks, *holdout_tasks]
    identities = [(row.get("task_uid"), row.get("instance_id")) for row in tasks]
    if any(
        not all(isinstance(value, str) and value for value in row) for row in identities
    ):
        raise ContractError("Round 1 mixed selection identity is invalid")
    if len({row[0] for row in identities}) != len(tasks) or len(
        {row[1] for row in identities}
    ) != len(tasks):
        raise ContractError("Round 1 mixed selection cohorts overlap")

    content = {
        "schema_version": 1,
        "status": "frozen",
        "selection_id": (
            f"round1-mixed-{len(feedback_tasks)}-feedback-"
            f"{len(holdout_tasks)}-holdout-v1"
        ),
        "task_count": len(tasks),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in tasks).items())),
        "benchmark_counts": dict(
            sorted(Counter(row["benchmark_id"] for row in tasks).items())
        ),
        "language_counts": dict(
            sorted(Counter(row["language"] for row in tasks).items())
        ),
        "mechanism_counts": dict(
            sorted(Counter(row["mechanism"] for row in tasks).items())
        ),
        "tasks": tasks,
        "source_feedback_selection_file_sha256": hashlib.sha256(
            feedback_raw
        ).hexdigest(),
        "source_feedback_selection_evidence_sha256": feedback["evidence_sha256"],
        "source_holdout_selection_file_sha256": hashlib.sha256(holdout_raw).hexdigest(),
        "source_holdout_selection_evidence_sha256": holdout["evidence_sha256"],
        "qualification_used_for_admission_only": True,
        "gold_fields_included": False,
        "reference_paths_included": False,
        "promotion_partition_opened": bool(
            feedback.get("promotion_partition_opened")
            or holdout.get("promotion_partition_opened")
        ),
        "final_sealed_partition_opened": bool(
            feedback.get("final_sealed_partition_opened")
            or holdout.get("final_sealed_partition_opened")
        ),
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = output.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ContractError("Round 1 mixed selection is unreadable") from exc
        if existing != canonical_json(report) + "\n":
            raise ContractError("frozen Round 1 mixed selection does not match replay")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def freeze_round1_sharded_selection(
    *,
    selection_paths: tuple[Path, ...],
    output_path: Path,
    expected_feedback_tasks: int = 30,
    expected_holdout_tasks: int = 30,
) -> dict[str, Any]:
    """Combine disjoint qualified shards into one predeclared A/B cohort split."""

    if (
        len(selection_paths) < 2
        or type(expected_feedback_tasks) is not int
        or type(expected_holdout_tasks) is not int
        or expected_feedback_tasks < 3
        or expected_holdout_tasks < 3
    ):
        raise ContractError("Round 1 sharded selection policy is invalid")
    sources: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for path in sorted(row.resolve() for row in selection_paths):
        raw, selection = _verified(path, label="Round 1 qualified shard")
        shard_tasks = selection.get("tasks")
        if (
            selection.get("status") != "frozen"
            or not isinstance(shard_tasks, list)
            or selection.get("task_count") != len(shard_tasks)
            or not shard_tasks
            or selection.get("final_sealed_partition_opened") is not False
            or any(
                not isinstance(row, dict)
                or row.get("cohort") not in {"feedback", "holdout"}
                for row in shard_tasks
            )
        ):
            raise ContractError("Round 1 qualified shard boundary is invalid")
        tasks.extend(shard_tasks)
        sources.append(
            {
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "evidence_sha256": selection["evidence_sha256"],
                "task_count": len(shard_tasks),
                "promotion_partition_opened": bool(
                    selection.get("promotion_partition_opened")
                ),
            }
        )

    identities = [(row.get("task_uid"), row.get("instance_id")) for row in tasks]
    if any(
        not all(isinstance(value, str) and value for value in identity)
        for identity in identities
    ):
        raise ContractError("Round 1 sharded selection identity is invalid")
    if len({row[0] for row in identities}) != len(tasks) or len(
        {row[1] for row in identities}
    ) != len(tasks):
        raise ContractError("Round 1 sharded selection identities overlap")
    cohort_counts = Counter(row["cohort"] for row in tasks)
    if cohort_counts != Counter(
        feedback=expected_feedback_tasks,
        holdout=expected_holdout_tasks,
    ):
        raise ContractError("Round 1 sharded selection cohort counts do not match")

    tasks.sort(key=lambda row: (row["cohort"], row["task_uid"]))
    content = {
        "schema_version": 1,
        "status": "frozen",
        "selection_id": (
            f"round1-sharded-{expected_feedback_tasks}-feedback-"
            f"{expected_holdout_tasks}-holdout-v1"
        ),
        "task_count": len(tasks),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "benchmark_counts": dict(
            sorted(Counter(row["benchmark_id"] for row in tasks).items())
        ),
        "language_counts": dict(
            sorted(Counter(row["language"] for row in tasks).items())
        ),
        "mechanism_counts": dict(
            sorted(Counter(row["mechanism"] for row in tasks).items())
        ),
        "tasks": tasks,
        "source_selections": sources,
        "qualification_used_for_admission_only": True,
        "cohorts_frozen_before_student_execution": True,
        "gold_fields_included": False,
        "reference_paths_included": False,
        "promotion_partition_opened": any(
            row["promotion_partition_opened"] for row in sources
        ),
        "final_sealed_partition_opened": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = output.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ContractError("Round 1 sharded selection is unreadable") from exc
        if existing != canonical_json(report) + "\n":
            raise ContractError(
                "frozen Round 1 sharded selection does not match replay"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
