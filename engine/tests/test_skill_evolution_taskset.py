"""Capability evals for the fixed, gold-free P1 TaskSet."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skill_evolution_loop import (
    ContractError,
    EvaluationTask,
    EvaluationTaskSet,
    materialize_evaluation_task,
)


def _source_repo(path: Path) -> tuple[Path, str]:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    source = path / "src/example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TaskSet Test",
            "-c",
            "user.email=taskset@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    return path, revision


def _task(
    source: Path,
    revision: str,
    number: int,
    cohort: str,
) -> EvaluationTask:
    return EvaluationTask.create(
        task_id=f"eval-{number:03d}",
        instance_id=f"project__repo-{number}",
        benchmark_id="fixture-benchmark",
        benchmark_base_commit=revision,
        repo="project/repo",
        source_repository=source,
        source_revision=revision,
        instruction=f"Change value for task {number}.",
        allowed_targets=["src/example.py"],
        cohort=cohort,
    )


def _taskset(source: Path, revision: str) -> EvaluationTaskSet:
    tasks = [
        _task(source, revision, number, "feedback" if number <= 3 else "holdout")
        for number in range(1, 7)
    ]
    return EvaluationTaskSet.create(taskset_id="p1-fixture", tasks=tasks)


def test_taskset_is_fingerprinted_round_trippable_and_requires_three_plus_three(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)

    assert EvaluationTaskSet.from_dict(taskset.to_dict()) == taskset
    assert len(taskset.fingerprint) == 64
    assert taskset.cohort_counts == {"feedback": 3, "holdout": 3}

    with pytest.raises(ContractError, match="at least 3 feedback and 3 holdout"):
        EvaluationTaskSet.create(
            taskset_id="too-small",
            tasks=list(taskset.tasks[:5]),
        )


def test_taskset_rejects_cross_cohort_identity_leakage_and_gold_fields(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)
    duplicate = EvaluationTask.create(
        **{
            **taskset.tasks[-1].creation_fields(),
            "task_id": "eval-duplicate",
            "instance_id": taskset.tasks[0].instance_id,
        }
    )

    with pytest.raises(ContractError, match="instance ids must be unique"):
        EvaluationTaskSet.create(
            taskset_id="leaky",
            tasks=[*taskset.tasks[:-1], duplicate],
        )

    malformed = taskset.to_dict()
    malformed["tasks"][0]["gold_patch"] = "hidden answer"
    with pytest.raises(ContractError, match="fields"):
        EvaluationTaskSet.from_dict(malformed)


def test_preflight_checks_pinned_revision_and_targets_without_mutating_source(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    taskset = _taskset(source, revision)
    before_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    before_status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=source, text=True
    )

    report = taskset.preflight()

    assert report.ready is True
    assert report.ready_tasks == 6
    assert report.errors == ()
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        == before_head
    )
    assert (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
        == before_status
    )


def test_materializer_creates_an_isolated_clean_checkout_at_pinned_revision(
    tmp_path: Path,
) -> None:
    source, revision = _source_repo(tmp_path / "source")
    task = _task(source, revision, 1, "feedback")

    checkout = materialize_evaluation_task(task, tmp_path / "checkout")

    assert (checkout / "src/example.py").read_text() == "value = 1\n"
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        == revision
    )
    assert (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=checkout, text=True
        )
        == ""
    )
    assert (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
        == ""
    )
