"""Pinned benchmark adapters and cross-dataset task lifecycle authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

TASK_STATES = frozenset({"unopened", "search", "promotion", "final_sealed", "retired"})
PARTITIONS = frozenset({"search", "promotion", "final_sealed"})
MOVING_REVISIONS = frozenset({"latest", "main", "master", "head", "test"})


class BenchmarkContractError(ValueError):
    """Raised when a benchmark or task pool crosses a frozen contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BenchmarkContractError(f"{field} must be a lowercase SHA-256")


def _require_pinned_revision(revision: str) -> None:
    if not revision or revision.lower() in MOVING_REVISIONS or len(revision) < 8:
        raise BenchmarkContractError("benchmark revision must be pinned")


@dataclass(frozen=True)
class BenchmarkTask:
    """Normalized task identity without exposing task content to the scheduler."""

    benchmark_id: str
    benchmark_revision: str
    instance_id: str
    task_family: str
    language: str
    repo: str
    base_commit: str
    environment_ref: str
    grader_ref: str
    instruction_ref: str
    source_url: str
    license_id: str
    overlap_keys: tuple[str, ...]
    content_sha256: str
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise BenchmarkContractError("unsupported benchmark task schema")
        _require_pinned_revision(self.benchmark_revision)
        _require_sha256(self.content_sha256, field="content_sha256")
        required = (
            self.benchmark_id,
            self.instance_id,
            self.task_family,
            self.language,
            self.repo,
            self.base_commit,
            self.environment_ref,
            self.grader_ref,
            self.instruction_ref,
            self.source_url,
            self.license_id,
        )
        if not all(isinstance(value, str) and value.strip() for value in required):
            raise BenchmarkContractError(
                "benchmark task fields must be non-empty strings"
            )
        if not self.overlap_keys or any(not key for key in self.overlap_keys):
            raise BenchmarkContractError("overlap_keys must be non-empty")
        if len(set(self.overlap_keys)) != len(self.overlap_keys):
            raise BenchmarkContractError("overlap_keys must be unique")

    @property
    def task_uid(self) -> str:
        return _sha256_json(
            {
                "benchmark_id": self.benchmark_id,
                "benchmark_revision": self.benchmark_revision,
                "instance_id": self.instance_id,
            }
        )

    @property
    def identity_fingerprint(self) -> str:
        return _sha256_json(
            {
                "repo": self.repo.lower(),
                "base_commit": self.base_commit,
                "overlap_keys": sorted(key.lower() for key in self.overlap_keys),
                "grader_ref": self.grader_ref,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["overlap_keys"] = list(self.overlap_keys)
        payload["task_uid"] = self.task_uid
        payload["identity_fingerprint"] = self.identity_fingerprint
        return payload


@dataclass(frozen=True)
class StaticBenchmarkAdapter:
    """Offline adapter backed by an already-frozen normalized manifest."""

    adapter_id: str
    revision: str
    executable: bool
    tasks: tuple[BenchmarkTask, ...]

    def __post_init__(self) -> None:
        if not self.adapter_id:
            raise BenchmarkContractError("adapter_id must be non-empty")
        _require_pinned_revision(self.revision)
        if not self.executable:
            raise BenchmarkContractError("benchmark adapter must be executable")
        if not self.tasks:
            raise BenchmarkContractError("benchmark adapter must contain tasks")

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(
            replace(
                task,
                benchmark_id=self.adapter_id,
                benchmark_revision=self.revision,
            )
            for task in self.tasks
        )


class BenchmarkRegistry:
    """Version-pinned adapter registry with no implicit network access."""

    def __init__(self) -> None:
        self._adapters: dict[str, StaticBenchmarkAdapter] = {}

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def register(self, adapter: StaticBenchmarkAdapter) -> None:
        _require_pinned_revision(adapter.revision)
        if adapter.adapter_id in self._adapters:
            raise BenchmarkContractError(
                f"adapter already registered: {adapter.adapter_id}"
            )
        if not adapter.executable:
            raise BenchmarkContractError("benchmark adapter must be executable")
        self._adapters[adapter.adapter_id] = adapter

    def tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(
            task
            for adapter_id in sorted(self._adapters)
            for task in self._adapters[adapter_id].load_tasks()
        )


@dataclass
class TaskPoolRecord:
    task_uid: str
    identity_fingerprint: str
    benchmark_id: str
    instance_id: str
    assigned_partition: str
    task_contract: dict[str, Any]
    state: str = "unopened"

    def __post_init__(self) -> None:
        _require_sha256(self.task_uid, field="task_uid")
        _require_sha256(self.identity_fingerprint, field="identity_fingerprint")
        if self.assigned_partition not in PARTITIONS:
            raise BenchmarkContractError("invalid assigned partition")
        if self.state not in TASK_STATES:
            raise BenchmarkContractError("invalid task state")
        if self.task_contract.get("task_uid") != self.task_uid:
            raise BenchmarkContractError("task contract uid mismatch")
        if self.task_contract.get("identity_fingerprint") != self.identity_fingerprint:
            raise BenchmarkContractError("task contract identity mismatch")
        if self.task_contract.get("benchmark_id") != self.benchmark_id:
            raise BenchmarkContractError("task contract benchmark mismatch")


class TaskPool:
    """Single authority for task deduplication, partition opening, and retirement."""

    def __init__(
        self,
        *,
        seed_material: str,
        adapters: tuple[str, ...],
        records: list[TaskPoolRecord],
        duplicates: list[dict[str, str]],
        retired_exclusions: list[dict[str, str]] | None = None,
    ) -> None:
        if not seed_material:
            raise BenchmarkContractError("seed_material must be non-empty")
        self.seed_material = seed_material
        self.adapters = adapters
        self.records = records
        self.duplicates = duplicates
        self.retired_exclusions = retired_exclusions or []
        self._validate()

    @classmethod
    def build(
        cls,
        *,
        registry: BenchmarkRegistry,
        seed_material: str,
        target_count: int,
        promotion_count: int,
        final_sealed_count: int,
        partition_quotas: dict[str, dict[str, int]] | None = None,
        retired_instance_ids: frozenset[str] = frozenset(),
    ) -> TaskPool:
        if target_count < 1:
            raise BenchmarkContractError("target_count must be positive")
        if promotion_count < 0 or final_sealed_count < 0:
            raise BenchmarkContractError("partition counts cannot be negative")
        if promotion_count + final_sealed_count > target_count:
            raise BenchmarkContractError("partition counts exceed target_count")

        unique: dict[str, BenchmarkTask] = {}
        duplicates: list[dict[str, str]] = []
        retired_exclusions: list[dict[str, str]] = []
        seen_overlap_keys: dict[str, BenchmarkTask] = {}
        for task in registry.tasks():
            if task.instance_id in retired_instance_ids:
                retired_exclusions.append(
                    {
                        "benchmark_id": task.benchmark_id,
                        "instance_id": task.instance_id,
                        "reason": "prior_use_retired",
                    }
                )
                continue
            conflicting = next(
                (
                    seen_overlap_keys[key.lower()]
                    for key in task.overlap_keys
                    if key.lower() in seen_overlap_keys
                ),
                None,
            )
            if conflicting is not None:
                duplicates.append(
                    {
                        "dropped_task_uid": task.task_uid,
                        "kept_task_uid": conflicting.task_uid,
                        "reason": "overlap_key",
                    }
                )
                continue
            if task.identity_fingerprint in unique:
                duplicates.append(
                    {
                        "dropped_task_uid": task.task_uid,
                        "kept_task_uid": unique[task.identity_fingerprint].task_uid,
                        "reason": "identity_fingerprint",
                    }
                )
                continue
            unique[task.identity_fingerprint] = task
            for key in task.overlap_keys:
                seen_overlap_keys[key.lower()] = task

        if len(unique) < target_count:
            raise BenchmarkContractError(
                f"not enough unique tasks: required={target_count} available={len(unique)}"
            )

        def rank(task: BenchmarkTask) -> str:
            return hashlib.sha256(
                f"{seed_material}\0{task.task_uid}".encode()
            ).hexdigest()

        selected_with_partitions: list[tuple[BenchmarkTask, str]] = []
        if partition_quotas is not None:
            if set(partition_quotas) != set(registry.adapter_ids):
                raise BenchmarkContractError(
                    "partition quota adapters must match the registry"
                )
            if any(
                set(quotas) != PARTITIONS
                or any(
                    not isinstance(count, int) or count < 0 for count in quotas.values()
                )
                for quotas in partition_quotas.values()
            ):
                raise BenchmarkContractError("invalid partition quota shape")
            expected_partition_totals = {
                "search": target_count - promotion_count - final_sealed_count,
                "promotion": promotion_count,
                "final_sealed": final_sealed_count,
            }
            actual_partition_totals = {
                partition: sum(
                    quotas[partition] for quotas in partition_quotas.values()
                )
                for partition in PARTITIONS
            }
            if actual_partition_totals != expected_partition_totals:
                raise BenchmarkContractError(
                    "partition quotas do not match declared partition counts"
                )
            by_adapter: dict[str, list[BenchmarkTask]] = {
                adapter_id: [] for adapter_id in registry.adapter_ids
            }
            partition_tasks: dict[str, dict[str, list[BenchmarkTask]]] = {
                partition: {} for partition in PARTITIONS
            }
            for task in unique.values():
                by_adapter[task.benchmark_id].append(task)
            for adapter_id in registry.adapter_ids:
                quotas = partition_quotas[adapter_id]
                required = sum(quotas.values())
                available = sorted(
                    by_adapter[adapter_id],
                    key=lambda task: (rank(task), task.task_uid),
                )
                if len(available) < required:
                    raise BenchmarkContractError(
                        f"not enough unique tasks for {adapter_id}: "
                        f"required={required} available={len(available)}"
                    )
                selected = available[:required]
                cursor = 0
                for partition in ("final_sealed", "promotion", "search"):
                    next_cursor = cursor + quotas[partition]
                    partition_tasks[partition][adapter_id] = selected[
                        cursor:next_cursor
                    ]
                    cursor = next_cursor
            for partition in ("final_sealed", "promotion", "search"):
                maximum = max(
                    len(tasks) for tasks in partition_tasks[partition].values()
                )
                for position in range(maximum):
                    for adapter_id in registry.adapter_ids:
                        tasks = partition_tasks[partition][adapter_id]
                        if position < len(tasks):
                            selected_with_partitions.append(
                                (tasks[position], partition)
                            )
        else:
            selected = sorted(
                unique.values(), key=lambda task: (rank(task), task.task_uid)
            )[:target_count]
            for index, task in enumerate(selected):
                if index < final_sealed_count:
                    partition = "final_sealed"
                elif index < final_sealed_count + promotion_count:
                    partition = "promotion"
                else:
                    partition = "search"
                selected_with_partitions.append((task, partition))

        records: list[TaskPoolRecord] = []
        for task, partition in selected_with_partitions:
            records.append(
                TaskPoolRecord(
                    task_uid=task.task_uid,
                    identity_fingerprint=task.identity_fingerprint,
                    benchmark_id=task.benchmark_id,
                    instance_id=task.instance_id,
                    assigned_partition=partition,
                    task_contract=task.to_dict(),
                )
            )
        return cls(
            seed_material=seed_material,
            adapters=registry.adapter_ids,
            records=records,
            duplicates=duplicates,
            retired_exclusions=retired_exclusions,
        )

    def _validate(self) -> None:
        task_uids = [record.task_uid for record in self.records]
        identities = [record.identity_fingerprint for record in self.records]
        if len(task_uids) != len(set(task_uids)):
            raise BenchmarkContractError("duplicate task_uid in task pool")
        if len(identities) != len(set(identities)):
            raise BenchmarkContractError("duplicate identity in task pool")
        for record in self.records:
            if record.state not in {"unopened", record.assigned_partition, "retired"}:
                raise BenchmarkContractError("task state violates assigned partition")
        retired_ids = {
            exclusion["instance_id"] for exclusion in self.retired_exclusions
        }
        if any(record.instance_id in retired_ids for record in self.records):
            raise BenchmarkContractError("retired task leaked into task pool")

    def _record(self, task_uid: str) -> TaskPoolRecord:
        matches = [record for record in self.records if record.task_uid == task_uid]
        if len(matches) != 1:
            raise BenchmarkContractError(f"unknown task_uid: {task_uid}")
        return matches[0]

    def claim(self, task_uid: str, partition: str) -> None:
        record = self._record(task_uid)
        if record.state == "retired":
            raise BenchmarkContractError("retired task cannot be reopened")
        if partition != record.assigned_partition:
            raise BenchmarkContractError("partition does not match assigned partition")
        if record.state != "unopened":
            raise BenchmarkContractError("task has already been opened")
        record.state = partition

    def retire(self, task_uid: str) -> None:
        record = self._record(task_uid)
        if record.state not in PARTITIONS:
            raise BenchmarkContractError("only an opened task can be retired")
        record.state = "retired"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "1.0",
            "seed_material": self.seed_material,
            "adapters": list(self.adapters),
            "records": [asdict(record) for record in self.records],
            "duplicates": self.duplicates,
            "retired_exclusions": self.retired_exclusions,
        }
        payload["semantic_fingerprint"] = _sha256_json(payload)
        return payload

    def save(self, path: Path) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> TaskPool:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0":
            raise BenchmarkContractError("unsupported task pool schema")
        semantic_fingerprint = payload.pop("semantic_fingerprint", None)
        if semantic_fingerprint != _sha256_json(payload):
            raise BenchmarkContractError("task pool semantic fingerprint mismatch")
        return cls(
            seed_material=payload["seed_material"],
            adapters=tuple(payload["adapters"]),
            records=[TaskPoolRecord(**row) for row in payload["records"]],
            duplicates=list(payload["duplicates"]),
            retired_exclusions=list(payload.get("retired_exclusions", [])),
        )
