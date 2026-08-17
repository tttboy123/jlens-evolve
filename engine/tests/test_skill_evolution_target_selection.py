from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.eval_manifest import EvaluationTask, EvaluationTaskSet
from skill_evolution_loop.target_selection import (
    TargetSelectionManifest,
    TargetSelectionRecord,
)


def _taskset(tmp_path: Path) -> EvaluationTaskSet:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "src").mkdir()
    (repository / "src" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tasks = [
        EvaluationTask.create(
            task_id=f"task-{index}",
            instance_id=f"owner__repo-{index}",
            benchmark_id="pilot",
            benchmark_base_commit=revision,
            repo="owner/repo",
            source_repository=repository,
            source_revision=revision,
            instruction=f"Fix Core.value for task {index}",
            allowed_targets=["src/core.py"],
            cohort="feedback" if index < 3 else "holdout",
        )
        for index in range(6)
    ]
    return EvaluationTaskSet.create(taskset_id="selection", tasks=tasks)


def _record(task: EvaluationTask) -> TargetSelectionRecord:
    return TargetSelectionRecord.create(
        task=task,
        selector_id="issue-symbol-dependency-v1",
        evidence=["issue-symbol:Core.value", "lexical-hit:src/core.py"],
    )


def test_target_selection_manifest_binds_every_task_without_gold(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path)
    manifest = TargetSelectionManifest.create(
        taskset=taskset,
        records=[_record(task) for task in taskset.tasks],
    )

    serialized = manifest.to_dict()
    restored = TargetSelectionManifest.from_dict(serialized, taskset=taskset)

    assert restored.fingerprint == manifest.fingerprint
    assert restored.taskset_fingerprint == taskset.fingerprint
    assert restored.gold_fields_included is False
    assert len(restored.records) == 6
    assert all(
        record.selected_targets == ("src/core.py",) for record in restored.records
    )


def test_target_selection_rejects_missing_task_or_target_mismatch(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path)
    records = [_record(task) for task in taskset.tasks]

    with pytest.raises(ContractError, match="exactly one record"):
        TargetSelectionManifest.create(taskset=taskset, records=records[:-1])

    bad = TargetSelectionRecord(
        schema_version=1,
        task_id=taskset.tasks[0].task_id,
        instruction_sha256=taskset.tasks[0].instruction_sha256,
        source_revision=taskset.tasks[0].source_revision,
        selector_id="issue-symbol-dependency-v1",
        selected_targets=("src/missing.py",),
        evidence=("manual",),
    )
    with pytest.raises(ContractError, match="selected targets"):
        TargetSelectionManifest.create(
            taskset=taskset,
            records=[bad, *records[1:]],
        )


def test_target_selection_rejects_gold_evidence_or_fingerprint_tampering(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path)
    with pytest.raises(ContractError, match="gold"):
        TargetSelectionRecord.create(
            task=taskset.tasks[0],
            selector_id="issue-symbol-dependency-v1",
            evidence=["gold-patch:src/core.py"],
        )

    manifest = TargetSelectionManifest.create(
        taskset=taskset,
        records=[_record(task) for task in taskset.tasks],
    )
    serialized = manifest.to_dict()
    serialized["fingerprint"] = "0" * 64
    with pytest.raises(ContractError, match="fingerprint"):
        TargetSelectionManifest.from_dict(serialized, taskset=taskset)


def test_target_selection_allows_legitimate_source_path_containing_gold(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path)

    record = TargetSelectionRecord.create(
        task=taskset.tasks[0],
        selector_id="issue-symbol-dependency-v1",
        evidence=["ranked_source_path=markup/goldmark/parser.go"],
    )

    assert record.evidence == ("ranked_source_path=markup/goldmark/parser.go",)
