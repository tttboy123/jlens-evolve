from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError, canonical_json, sha256_json
from skill_evolution_loop.round1_mixed_selection import (
    freeze_round1_mixed_selection,
    freeze_round1_sharded_selection,
)


def _selection(path: Path, cohort: str, offset: int) -> None:
    tasks = [
        {
            "task_uid": f"{offset + index:064x}",
            "instance_id": f"org__repo-{offset + index}",
            "benchmark_id": "swe-bench-verified",
            "language": "python",
            "mechanism": "operator",
            "cohort": cohort,
        }
        for index in range(3)
    ]
    content = {
        "status": "frozen",
        "task_count": 3,
        "tasks": tasks,
        "promotion_partition_opened": cohort == "holdout",
        "final_sealed_partition_opened": False,
    }
    path.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )


def test_mixed_selection_preserves_disjoint_cohorts(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.json"
    holdout = tmp_path / "holdout.json"
    _selection(feedback, "feedback", 1)
    _selection(holdout, "holdout", 10)

    report = freeze_round1_mixed_selection(
        feedback_selection_path=feedback,
        holdout_selection_path=holdout,
        output_path=tmp_path / "mixed.json",
    )

    assert report["task_count"] == 6
    assert report["cohort_counts"] == {"feedback": 3, "holdout": 3}
    assert report["promotion_partition_opened"] is True
    assert report["final_sealed_partition_opened"] is False


def test_mixed_selection_rejects_overlapping_identity(tmp_path: Path) -> None:
    feedback = tmp_path / "feedback.json"
    holdout = tmp_path / "holdout.json"
    _selection(feedback, "feedback", 1)
    _selection(holdout, "holdout", 10)
    data = json.loads(holdout.read_text(encoding="utf-8"))
    data["tasks"][0]["instance_id"] = "org__repo-1"
    content = {key: value for key, value in data.items() if key != "evidence_sha256"}
    holdout.write_text(
        canonical_json({**content, "evidence_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match="overlap"):
        freeze_round1_mixed_selection(
            feedback_selection_path=feedback,
            holdout_selection_path=holdout,
            output_path=tmp_path / "mixed.json",
        )


def test_sharded_selection_freezes_exact_counts_before_student_execution(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _selection(first, "feedback", 1)
    _selection(second, "holdout", 10)

    report = freeze_round1_sharded_selection(
        selection_paths=(second, first),
        output_path=tmp_path / "sharded.json",
        expected_feedback_tasks=3,
        expected_holdout_tasks=3,
    )

    assert report["cohort_counts"] == {"feedback": 3, "holdout": 3}
    assert report["cohorts_frozen_before_student_execution"] is True
    assert report["final_sealed_partition_opened"] is False
    assert [row["path"] for row in report["source_selections"]] == sorted(
        [str(first.resolve()), str(second.resolve())]
    )
