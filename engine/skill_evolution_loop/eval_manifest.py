"""Strict, gold-free task manifests for paired Skill capability evals."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, sha256_json

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_COHORTS = {"feedback", "holdout"}


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_test_path(relative: str) -> bool:
    path = Path(relative)
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return bool(
        parts.intersection({"test", "tests", "__tests__"})
        or name.startswith("test_")
        or ".test." in name
        or ".spec." in name
    )


@dataclass(frozen=True)
class EvaluationTask:
    """One pinned, answer-free task in a feedback or evaluator-only cohort."""

    schema_version: int
    task_id: str
    instance_id: str
    benchmark_id: str
    benchmark_base_commit: str
    repo: str
    source_repository: Path
    source_revision: str
    instruction: str
    instruction_sha256: str
    allowed_targets: tuple[str, ...]
    cohort: str

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "task_id",
            "instance_id",
            "benchmark_id",
            "benchmark_base_commit",
            "repo",
            "source_repository",
            "source_revision",
            "instruction",
            "instruction_sha256",
            "allowed_targets",
            "cohort",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        instance_id: str,
        benchmark_id: str,
        benchmark_base_commit: str,
        repo: str,
        source_repository: Path,
        source_revision: str,
        instruction: str,
        allowed_targets: list[str],
        cohort: str,
    ) -> EvaluationTask:
        task = cls(
            schema_version=1,
            task_id=task_id,
            instance_id=instance_id,
            benchmark_id=benchmark_id,
            benchmark_base_commit=benchmark_base_commit,
            repo=repo,
            source_repository=source_repository.resolve(),
            source_revision=source_revision,
            instruction=instruction,
            instruction_sha256=_text_sha256(instruction),
            allowed_targets=tuple(allowed_targets),
            cohort=cohort,
        )
        task.validate()
        return task

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationTask:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid evaluation task fields")
        task = cls(
            schema_version=data["schema_version"],
            task_id=str(data["task_id"]),
            instance_id=str(data["instance_id"]),
            benchmark_id=str(data["benchmark_id"]),
            benchmark_base_commit=str(data["benchmark_base_commit"]),
            repo=str(data["repo"]),
            source_repository=Path(str(data["source_repository"])).resolve(),
            source_revision=str(data["source_revision"]),
            instruction=str(data["instruction"]),
            instruction_sha256=str(data["instruction_sha256"]),
            allowed_targets=tuple(str(row) for row in data["allowed_targets"]),
            cohort=str(data["cohort"]),
        )
        task.validate()
        if data["fingerprint"] != task.fingerprint:
            raise ContractError("evaluation task fingerprint mismatch")
        return task

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported evaluation task schema")
        for label, value in (
            ("task_id", self.task_id),
            ("instance_id", self.instance_id),
            ("benchmark_id", self.benchmark_id),
        ):
            if _IDENTIFIER.fullmatch(value) is None:
                raise ContractError(f"invalid {label}")
        if not re.fullmatch(r"[^/\s]+/[^/\s]+", self.repo):
            raise ContractError("invalid repository identity")
        if _COMMIT.fullmatch(self.benchmark_base_commit) is None:
            raise ContractError("invalid benchmark base commit")
        if _COMMIT.fullmatch(self.source_revision) is None:
            raise ContractError("invalid source revision")
        if not self.instruction.strip():
            raise ContractError("evaluation instruction must be non-empty")
        if self.instruction_sha256 != _text_sha256(self.instruction):
            raise ContractError("evaluation instruction sha256 mismatch")
        if self.cohort not in _COHORTS:
            raise ContractError("evaluation cohort must be feedback or holdout")
        if not self.allowed_targets or len(set(self.allowed_targets)) != len(
            self.allowed_targets
        ):
            raise ContractError("allowed targets must be non-empty and unique")
        for relative in self.allowed_targets:
            path = Path(relative)
            if (
                path.is_absolute()
                or not path.parts
                or ".." in path.parts
                or _is_test_path(relative)
            ):
                raise ContractError("allowed target must be a bounded non-test path")

    def creation_fields(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_base_commit": self.benchmark_base_commit,
            "repo": self.repo,
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "instruction": self.instruction,
            "allowed_targets": list(self.allowed_targets),
            "cohort": self.cohort,
        }

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_base_commit": self.benchmark_base_commit,
            "repo": self.repo,
            "source_repository": str(self.source_repository),
            "source_revision": self.source_revision,
            "instruction": self.instruction,
            "instruction_sha256": self.instruction_sha256,
            "allowed_targets": list(self.allowed_targets),
            "cohort": self.cohort,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class TaskSetPreflight:
    ready: bool
    ready_tasks: int
    total_tasks: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "ready_tasks": self.ready_tasks,
            "total_tasks": self.total_tasks,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class EvaluationTaskSet:
    """Fixed 3+3 minimum split with evaluator-only hold-out visibility."""

    schema_version: int
    taskset_id: str
    tasks: tuple[EvaluationTask, ...]
    holdout_visibility: str
    gold_fields_included: bool

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "taskset_id",
            "tasks",
            "holdout_visibility",
            "gold_fields_included",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(
        cls, *, taskset_id: str, tasks: list[EvaluationTask]
    ) -> EvaluationTaskSet:
        taskset = cls(
            schema_version=1,
            taskset_id=taskset_id,
            tasks=tuple(tasks),
            holdout_visibility="evaluator-only",
            gold_fields_included=False,
        )
        taskset.validate()
        return taskset

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationTaskSet:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid evaluation taskset fields")
        taskset = cls(
            schema_version=data["schema_version"],
            taskset_id=str(data["taskset_id"]),
            tasks=tuple(EvaluationTask.from_dict(row) for row in data["tasks"]),
            holdout_visibility=str(data["holdout_visibility"]),
            gold_fields_included=data["gold_fields_included"],
        )
        taskset.validate()
        if data["fingerprint"] != taskset.fingerprint:
            raise ContractError("evaluation taskset fingerprint mismatch")
        return taskset

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported evaluation taskset schema")
        if _IDENTIFIER.fullmatch(self.taskset_id) is None:
            raise ContractError("invalid taskset id")
        if self.holdout_visibility != "evaluator-only":
            raise ContractError("holdout visibility must be evaluator-only")
        if self.gold_fields_included is not False:
            raise ContractError("gold fields are prohibited")
        for task in self.tasks:
            task.validate()
        counts = self.cohort_counts
        if counts.get("feedback", 0) < 3 or counts.get("holdout", 0) < 3:
            raise ContractError(
                "taskset requires at least 3 feedback and 3 holdout tasks"
            )
        if len({row.task_id for row in self.tasks}) != len(self.tasks):
            raise ContractError("evaluation task ids must be unique")
        if len({row.instance_id for row in self.tasks}) != len(self.tasks):
            raise ContractError("evaluation instance ids must be unique")

    @property
    def cohort_counts(self) -> dict[str, int]:
        return dict(Counter(row.cohort for row in self.tasks))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taskset_id": self.taskset_id,
            "tasks": [row.to_dict() for row in self.tasks],
            "holdout_visibility": self.holdout_visibility,
            "gold_fields_included": self.gold_fields_included,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}

    def preflight(self) -> TaskSetPreflight:
        errors: list[str] = []
        ready_tasks = 0
        for task in self.tasks:
            task_errors = self._preflight_task(task)
            if task_errors:
                errors.extend(task_errors)
            else:
                ready_tasks += 1
        return TaskSetPreflight(
            ready=not errors,
            ready_tasks=ready_tasks,
            total_tasks=len(self.tasks),
            errors=tuple(errors),
        )

    @staticmethod
    def _preflight_task(task: EvaluationTask) -> list[str]:
        prefix = task.task_id
        if not task.source_repository.is_dir():
            return [f"{prefix}: source repository is missing"]
        commit = subprocess.run(
            [
                "git",
                "-C",
                str(task.source_repository),
                "cat-file",
                "-e",
                f"{task.source_revision}^{{commit}}",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if commit.returncode != 0:
            return [f"{prefix}: pinned source revision is missing"]
        errors: list[str] = []
        for target in task.allowed_targets:
            exists = subprocess.run(
                [
                    "git",
                    "-C",
                    str(task.source_repository),
                    "cat-file",
                    "-e",
                    f"{task.source_revision}:{target}",
                ],
                capture_output=True,
                check=False,
                text=True,
            )
            if exists.returncode != 0:
                errors.append(f"{prefix}: target missing at revision: {target}")
        return errors


def materialize_evaluation_task(task: EvaluationTask, destination: Path) -> Path:
    """Clone a local pinned source into an isolated detached checkout."""
    task.validate()
    target = destination.resolve()
    source = task.source_repository.resolve()
    if target.exists():
        raise ContractError("evaluation checkout destination already exists")
    if target == source or target.is_relative_to(source):
        raise ContractError(
            "evaluation checkout cannot be inside its source repository"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--shared",
            "--no-checkout",
            str(source),
            str(target),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractError(
            f"evaluation source clone failed: {completed.stderr.strip()}"
        )
    completed = subprocess.run(
        ["git", "checkout", "--quiet", "--detach", task.source_revision],
        cwd=target,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ContractError(
            f"evaluation source checkout failed: {completed.stderr.strip()}"
        )
    for relative in task.allowed_targets:
        if not (target / relative).is_file():
            raise ContractError(f"materialized target is missing: {relative}")
    return target
