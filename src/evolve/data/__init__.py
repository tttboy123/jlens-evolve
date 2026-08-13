"""Leakage-gated, content-addressed task data factory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from evolve.contracts import Cohort, ContractViolation, TaskRevision, canonical_json


class TaskRegistry:
    """Import feedback tasks into an append-only local registry.

    This first v3 slice deliberately has no method for opening holdout.  A future
    independent release-audit module must own that authorization seam.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "tasks.jsonl"

    def import_task(
        self,
        *,
        task_id: str,
        revision_id: str,
        project: str,
        cohort: Cohort,
        source: Path,
        evaluator_id: str,
    ) -> TaskRevision:
        if cohort is not Cohort.FEEDBACK:
            raise ContractViolation("this registry imports feedback tasks only")
        source = source.resolve()
        if not source.is_file():
            raise ContractViolation("task source must be a regular file")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        task = TaskRevision(
            task_id=task_id,
            revision_id=revision_id,
            project=project,
            cohort=cohort,
            source_sha256=digest,
            evaluator_id=evaluator_id,
            source_uri=str(source),
        )
        existing = {row.revision_id: row for row in self.all()}
        prior = existing.get(revision_id)
        if prior is not None:
            if prior != task:
                raise ContractViolation("task revision conflicts with immutable record")
            return prior
        line = canonical_json(
            {
                "task_id": task.task_id,
                "revision_id": task.revision_id,
                "project": task.project,
                "cohort": str(task.cohort),
                "source_sha256": task.source_sha256,
                "evaluator_id": task.evaluator_id,
                "source_uri": task.source_uri,
                "content_sha256": task.content_sha256,
            }
        )
        descriptor = os.open(self.index, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, (line + "\n").encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return task

    def all(self) -> tuple[TaskRevision, ...]:
        if not self.index.exists():
            return ()
        rows: list[TaskRevision] = []
        for line in self.index.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            task = TaskRevision(
                task_id=data["task_id"],
                revision_id=data["revision_id"],
                project=data["project"],
                cohort=Cohort(data["cohort"]),
                source_sha256=data["source_sha256"],
                evaluator_id=data["evaluator_id"],
                source_uri=data.get("source_uri"),
            )
            if data.get("content_sha256") != task.content_sha256:
                raise ContractViolation("task registry hash mismatch")
            rows.append(task)
        return tuple(rows)


__all__ = ["TaskRegistry"]
