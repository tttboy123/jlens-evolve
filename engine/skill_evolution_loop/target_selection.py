"""Gold-free provenance binding for every bounded Student target scope."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .contracts import ContractError, sha256_json
from .eval_manifest import EvaluationTask, EvaluationTaskSet

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_FORBIDDEN_EVIDENCE = (
    "gold-patch",
    "gold_patch",
    "gold patch",
    "gold evidence",
    "gold target",
    "test patch",
    "fix.patch",
    "reference_patch",
)


@dataclass(frozen=True)
class TargetSelectionRecord:
    schema_version: int
    task_id: str
    instruction_sha256: str
    source_revision: str
    selector_id: str
    selected_targets: tuple[str, ...]
    evidence: tuple[str, ...]

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "task_id",
            "instruction_sha256",
            "source_revision",
            "selector_id",
            "selected_targets",
            "evidence",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(
        cls,
        *,
        task: EvaluationTask,
        selector_id: str,
        evidence: list[str],
    ) -> TargetSelectionRecord:
        record = cls(
            schema_version=1,
            task_id=task.task_id,
            instruction_sha256=task.instruction_sha256,
            source_revision=task.source_revision,
            selector_id=selector_id,
            selected_targets=task.allowed_targets,
            evidence=tuple(evidence),
        )
        record.validate(task)
        return record

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, task: EvaluationTask
    ) -> TargetSelectionRecord:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid target selection record fields")
        record = cls(
            schema_version=data["schema_version"],
            task_id=str(data["task_id"]),
            instruction_sha256=str(data["instruction_sha256"]),
            source_revision=str(data["source_revision"]),
            selector_id=str(data["selector_id"]),
            selected_targets=tuple(str(row) for row in data["selected_targets"]),
            evidence=tuple(str(row) for row in data["evidence"]),
        )
        record.validate(task)
        if data["fingerprint"] != record.fingerprint:
            raise ContractError("target selection record fingerprint mismatch")
        return record

    def validate(self, task: EvaluationTask) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported target selection record schema")
        if self.task_id != task.task_id:
            raise ContractError("target selection task mismatch")
        if self.instruction_sha256 != task.instruction_sha256:
            raise ContractError("target selection instruction mismatch")
        if self.source_revision != task.source_revision:
            raise ContractError("target selection source revision mismatch")
        if _IDENTIFIER.fullmatch(self.selector_id) is None:
            raise ContractError("invalid target selector id")
        if self.selected_targets != task.allowed_targets:
            raise ContractError("target selection selected targets mismatch")
        if not self.evidence or any(not row.strip() for row in self.evidence):
            raise ContractError("target selection evidence must be non-empty")
        normalized = "\n".join(self.evidence).lower()
        if any(token in normalized for token in _FORBIDDEN_EVIDENCE):
            raise ContractError("gold evidence is prohibited from target selection")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction_sha256": self.instruction_sha256,
            "source_revision": self.source_revision,
            "selector_id": self.selector_id,
            "selected_targets": list(self.selected_targets),
            "evidence": list(self.evidence),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class TargetSelectionManifest:
    schema_version: int
    taskset_id: str
    taskset_fingerprint: str
    gold_fields_included: bool
    records: tuple[TargetSelectionRecord, ...]

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "taskset_id",
            "taskset_fingerprint",
            "gold_fields_included",
            "records",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(
        cls,
        *,
        taskset: EvaluationTaskSet,
        records: list[TargetSelectionRecord],
    ) -> TargetSelectionManifest:
        manifest = cls(
            schema_version=1,
            taskset_id=taskset.taskset_id,
            taskset_fingerprint=taskset.fingerprint,
            gold_fields_included=False,
            records=tuple(sorted(records, key=lambda row: row.task_id)),
        )
        manifest.validate(taskset)
        return manifest

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, taskset: EvaluationTaskSet
    ) -> TargetSelectionManifest:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid target selection manifest fields")
        tasks = {task.task_id: task for task in taskset.tasks}
        raw_records = data["records"]
        if not isinstance(raw_records, list):
            raise ContractError("target selection records must be a list")
        records: list[TargetSelectionRecord] = []
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise ContractError("invalid target selection record")
            task_id = str(raw.get("task_id", ""))
            if task_id not in tasks:
                raise ContractError("target selection has unknown task")
            records.append(TargetSelectionRecord.from_dict(raw, task=tasks[task_id]))
        manifest = cls(
            schema_version=data["schema_version"],
            taskset_id=str(data["taskset_id"]),
            taskset_fingerprint=str(data["taskset_fingerprint"]),
            gold_fields_included=data["gold_fields_included"],
            records=tuple(records),
        )
        manifest.validate(taskset)
        if data["fingerprint"] != manifest.fingerprint:
            raise ContractError("target selection manifest fingerprint mismatch")
        return manifest

    def validate(self, taskset: EvaluationTaskSet) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported target selection manifest schema")
        if self.taskset_id != taskset.taskset_id:
            raise ContractError("target selection taskset id mismatch")
        if self.taskset_fingerprint != taskset.fingerprint:
            raise ContractError("target selection taskset fingerprint mismatch")
        if self.gold_fields_included is not False:
            raise ContractError("gold fields are prohibited from target selection")
        tasks = {task.task_id: task for task in taskset.tasks}
        record_ids = [record.task_id for record in self.records]
        if len(record_ids) != len(set(record_ids)) or set(record_ids) != set(tasks):
            raise ContractError("target selection requires exactly one record per task")
        for record in self.records:
            record.validate(tasks[record.task_id])

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taskset_id": self.taskset_id,
            "taskset_fingerprint": self.taskset_fingerprint,
            "gold_fields_included": self.gold_fields_included,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}
