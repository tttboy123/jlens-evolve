from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.eval_manifest import EvaluationTask, EvaluationTaskSet
from skill_evolution_loop.round1_target_audit import audit_round1_targets


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "src").mkdir()
    (repository / "src" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    tasks = []
    rows = []
    for index, cohort in enumerate(["feedback"] * 3 + ["holdout"] * 3):
        instance_id = f"owner__repo-{index}"
        tasks.append(
            EvaluationTask.create(
                task_id=f"task-{index}",
                instance_id=instance_id,
                benchmark_id="swe-bench-verified",
                benchmark_base_commit=revision,
                repo="owner/repo",
                source_repository=repository,
                source_revision=revision,
                instruction="Fix VALUE",
                allowed_targets=["src/core.py"],
                cohort=cohort,
            )
        )
        rows.append(
            {
                "instance_id": instance_id,
                "patch": "--- a/src/core.py\n+++ b/src/core.py\n"
                "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            }
        )
    taskset = EvaluationTaskSet.create(taskset_id="round1-test", tasks=tasks)
    taskset_path = tmp_path / "TASKSET.json"
    taskset_path.write_text(json.dumps(taskset.to_dict()), encoding="utf-8")

    pool = tmp_path / "pool"
    (pool / "harness-inputs").mkdir(parents=True)
    (pool / "inputs/multi-swe-bench-flash").mkdir(parents=True)
    (pool / "harness-inputs/swe-bench-verified.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (pool / "harness-inputs/swe-bench-multilingual.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "unused-multilingual",
                "patch": "--- a/src/core.py\n+++ b/src/core.py\n"
                "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (pool / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl").write_text(
        json.dumps(
            {
                "instance_id": "unused",
                "fix_patch": "--- a/src/core.py\n+++ b/src/core.py\n"
                "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return taskset_path, pool


def test_round1_target_audit_is_append_only_and_gold_free(tmp_path: Path) -> None:
    taskset_path, pool = _fixture(tmp_path)
    output = tmp_path / "TARGET-COVERAGE.json"

    first = audit_round1_targets(
        taskset_path=taskset_path,
        pool_root=pool,
        output_path=output,
    )
    second = audit_round1_targets(
        taskset_path=taskset_path,
        pool_root=pool,
        output_path=output,
    )

    assert first == second
    assert first["audit"]["ready"] is True
    assert first["audit"]["ready_tasks"] == 6
    assert first["answer_content_serialized"] is False
    assert "VALUE = 2" not in output.read_text(encoding="utf-8")


def test_round1_target_audit_rejects_frozen_mismatch(tmp_path: Path) -> None:
    taskset_path, pool = _fixture(tmp_path)
    output = tmp_path / "TARGET-COVERAGE.json"
    audit_round1_targets(
        taskset_path=taskset_path,
        pool_root=pool,
        output_path=output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["audit"]["ready"] = False
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ContractError, match="does not match replay"):
        audit_round1_targets(
            taskset_path=taskset_path,
            pool_root=pool,
            output_path=output,
        )
