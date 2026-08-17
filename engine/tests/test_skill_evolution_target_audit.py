from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.eval_manifest import EvaluationTask, EvaluationTaskSet
from skill_evolution_loop.target_audit import (
    GoldPatchReference,
    MechanismCapacityPolicy,
    audit_mechanism_capacity,
    audit_target_coverage,
)


def _repository(tmp_path: Path) -> tuple[Path, str]:
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
    (repository / "tests").mkdir()
    (repository / "src" / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "src" / "helper.py").write_text("HELPER = 1\n", encoding="utf-8")
    (repository / "tests" / "test_core.py").write_text(
        "def test_core(): pass\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, revision


def _taskset(tmp_path: Path, allowed_targets: list[str]) -> EvaluationTaskSet:
    repository, revision = _repository(tmp_path)
    tasks = []
    for index, cohort in enumerate(["feedback"] * 3 + ["holdout"] * 3):
        tasks.append(
            EvaluationTask.create(
                task_id=f"task-{index}",
                instance_id=f"owner__repo-{index}",
                benchmark_id="pilot",
                benchmark_base_commit=revision,
                repo="owner/repo",
                source_repository=repository,
                source_revision=revision,
                instruction=f"Fix task {index}",
                allowed_targets=allowed_targets,
                cohort=cohort,
            )
        )
    return EvaluationTaskSet.create(taskset_id="target-audit", tasks=tasks)


def _patch(path: Path) -> None:
    path.write_text(
        """diff --git a/CHANGES b/CHANGES
--- a/CHANGES
+++ b/CHANGES
@@ -1 +1 @@
-old
+new
diff --git a/src/core.py b/src/core.py
--- a/src/core.py
+++ b/src/core.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1 +1 @@
-def test_core(): pass
+def test_core(): assert True
""",
        encoding="utf-8",
    )


def test_target_audit_is_evaluator_only_and_ignores_test_patch_paths(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path, ["src/core.py"])
    patch = tmp_path / "fix.patch"
    _patch(patch)

    audit = audit_target_coverage(
        taskset,
        [GoldPatchReference(task_id="task-3", patch_path=patch)],
    )

    assert audit.ready is True
    assert audit.audited_tasks == 1
    assert audit.rows[0].implementation_targets == ("src/core.py",)
    assert audit.rows[0].missing_targets == ()
    assert audit.network_calls_performed is False
    serialized = audit.to_dict()
    assert "VALUE = 2" not in json.dumps(serialized)
    assert serialized["audit_sha256"] == audit.audit_sha256


def test_target_audit_fails_when_implementation_target_is_not_allowed(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path, ["src/helper.py"])
    patch = tmp_path / "fix.patch"
    _patch(patch)

    audit = audit_target_coverage(
        taskset,
        [GoldPatchReference(task_id="task-3", patch_path=patch)],
    )

    assert audit.ready is False
    assert audit.rows[0].missing_targets == ("src/core.py",)


def test_target_audit_rejects_tasks_beyond_single_file_mechanism_capacity(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path, ["src/core.py", "src/helper.py"])
    patch = tmp_path / "multi-file.patch"
    patch.write_text(
        """--- a/src/core.py
+++ b/src/core.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
--- a/src/helper.py
+++ b/src/helper.py
@@ -1 +1 @@
-HELPER = 1
+HELPER = 2
""",
        encoding="utf-8",
    )

    audit = audit_target_coverage(
        taskset,
        [GoldPatchReference(task_id="task-3", patch_path=patch)],
    )

    assert audit.ready is False
    assert audit.rows[0].missing_targets == ()
    assert audit.rows[0].within_mechanism_capacity is False
    assert audit.rows[0].mechanism_max_files == 1


def test_target_audit_rejects_unknown_duplicate_or_missing_patch(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path, ["src/core.py"])
    patch = tmp_path / "fix.patch"
    _patch(patch)

    with pytest.raises(ContractError, match="unknown task"):
        audit_target_coverage(
            taskset,
            [GoldPatchReference(task_id="missing", patch_path=patch)],
        )
    with pytest.raises(ContractError, match="unique"):
        audit_target_coverage(
            taskset,
            [
                GoldPatchReference(task_id="task-3", patch_path=patch),
                GoldPatchReference(task_id="task-3", patch_path=patch),
            ],
        )
    with pytest.raises(ContractError, match="unavailable"):
        audit_target_coverage(
            taskset,
            [GoldPatchReference(task_id="task-3", patch_path=tmp_path / "none")],
        )


def _tokenizer_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "models/Qwen3.5-4B-mlx-4bit/tokenizer.json"
    )


def _capacity_policy(*, structured_max_tokens: int = 384) -> MechanismCapacityPolicy:
    return MechanismCapacityPolicy(
        structured_max_tokens=structured_max_tokens,
        tokenizer_path=_tokenizer_path(),
    )


def test_mechanism_capacity_accepts_one_small_unique_edit(tmp_path: Path) -> None:
    taskset = _taskset(tmp_path, ["src/core.py"])
    patch = tmp_path / "fix.patch"
    _patch(patch)

    audit = audit_mechanism_capacity(
        taskset,
        [GoldPatchReference(task_id="task-3", patch_path=patch)],
        policy=_capacity_policy(),
    )

    row = audit.rows[0]
    assert audit.ready is True
    assert row.implementation_hunks == 1
    assert row.structured_output_tokens is not None
    assert row.structured_output_tokens <= 384
    assert row.structured_unique_search is True
    assert row.structured_ready is True
    assert row.hunk_ready is True
    serialized = json.dumps(audit.to_dict())
    assert "VALUE = 2" not in serialized
    assert str(_tokenizer_path()) not in serialized


def test_mechanism_capacity_rejects_distant_edit_over_token_budget(
    tmp_path: Path,
) -> None:
    taskset = _taskset(tmp_path, ["src/core.py"])
    repository = taskset.tasks[0].source_repository
    source = "FIRST = 1\n" + "\n".join(f"LINE_{i} = '{'x' * 40}'" for i in range(60))
    source += "\nLAST = 1\n"
    (repository / "src/core.py").write_text(source, encoding="utf-8")
    patch = tmp_path / "distant.patch"
    patch.write_text(
        """--- a/src/core.py
+++ b/src/core.py
@@ -1 +1 @@
-FIRST = 1
+FIRST = 2
@@ -62 +62 @@
-LAST = 1
+LAST = 2
""",
        encoding="utf-8",
    )

    audit = audit_mechanism_capacity(
        taskset,
        [GoldPatchReference(task_id="task-3", patch_path=patch)],
        policy=_capacity_policy(structured_max_tokens=64),
    )

    row = audit.rows[0]
    assert row.structured_ready is False
    assert row.hunk_ready is True
    assert "structured-token-budget" in row.reasons
    assert row.ready is False


def test_mechanism_capacity_rejects_hunk_protocol_overflow(tmp_path: Path) -> None:
    taskset = _taskset(tmp_path, ["src/core.py"])
    repository = taskset.tasks[0].source_repository
    original = "".join(f"VALUE_{index} = 1\n" for index in range(30))
    (repository / "src/core.py").write_text(original, encoding="utf-8")
    body = "".join(f"-VALUE_{index} = 1\n+VALUE_{index} = 2\n" for index in range(30))
    patch = tmp_path / "large.patch"
    patch.write_text(
        f"--- a/src/core.py\n+++ b/src/core.py\n@@ -1,30 +1,30 @@\n{body}",
        encoding="utf-8",
    )

    audit = audit_mechanism_capacity(
        taskset,
        [GoldPatchReference(task_id="task-3", patch_path=patch)],
        policy=_capacity_policy(structured_max_tokens=2_000),
    )

    row = audit.rows[0]
    assert row.structured_ready is True
    assert row.hunk_ready is False
    assert "hunk-changed-lines" in row.reasons
    assert row.ready is False
