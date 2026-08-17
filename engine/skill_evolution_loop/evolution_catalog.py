"""Append-only, model-aware knowledge catalog for evolution assets."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import ContractError, canonical_json, sha256_json

RECORD_TYPES = frozenset(
    {
        "skills",
        "failure_clusters",
        "mechanisms",
        "infrastructure_gaps",
        "experiments",
    }
)
STATUSES = frozenset(
    {"candidate", "implemented", "validated", "disproven", "retired", "pending"}
)


class CatalogConflict(ContractError):
    """Raised when immutable catalog identity is reused with new content."""


def _strings(label: str, values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ContractError(f"{label} must contain non-empty strings")
    return normalized


@dataclass(frozen=True)
class EvolutionRecord:
    record_type: str
    record_id: str
    title: str
    status: str
    capability_tags: tuple[str, ...]
    task_tags: tuple[str, ...]
    failure_mode_tags: tuple[str, ...]
    source_model: str
    source_runtime: str
    payload: dict[str, Any]
    evidence_refs: tuple[dict[str, Any], ...]
    cross_model_validations: tuple[dict[str, Any], ...]

    @classmethod
    def create(
        cls,
        *,
        record_type: str,
        record_id: str,
        title: str,
        status: str,
        capability_tags: Iterable[str],
        task_tags: Iterable[str],
        failure_mode_tags: Iterable[str],
        source_model: str,
        source_runtime: str,
        payload: dict[str, Any],
        evidence_refs: Iterable[dict[str, Any]],
        cross_model_validations: Iterable[dict[str, Any]],
    ) -> EvolutionRecord:
        record = cls(
            record_type=record_type,
            record_id=record_id,
            title=title,
            status=status,
            capability_tags=_strings("capability_tags", capability_tags),
            task_tags=_strings("task_tags", task_tags),
            failure_mode_tags=_strings("failure_mode_tags", failure_mode_tags),
            source_model=source_model,
            source_runtime=source_runtime,
            payload=payload,
            evidence_refs=tuple(evidence_refs),
            cross_model_validations=tuple(cross_model_validations),
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.record_type not in RECORD_TYPES:
            raise ContractError("unsupported evolution record_type")
        if self.status not in STATUSES:
            raise ContractError("unsupported evolution record status")
        for label, value in (
            ("record_id", self.record_id),
            ("title", self.title),
            ("source_model", self.source_model),
            ("source_runtime", self.source_runtime),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{label} must be non-empty")
        if not isinstance(self.payload, dict):
            raise ContractError("payload must be an object")
        for ref in self.evidence_refs:
            if set(ref) != {"path", "sha256"} or not _is_sha(ref["sha256"]):
                raise ContractError("invalid evidence reference")
        required = {
            "target_model",
            "target_runtime",
            "outcome",
            "evidence_sha256",
        }
        for validation in self.cross_model_validations:
            if set(validation) != required:
                raise ContractError("invalid cross-model validation")
            evidence_sha = validation["evidence_sha256"]
            if evidence_sha is not None and not _is_sha(evidence_sha):
                raise ContractError("invalid cross-model evidence SHA")

    def content_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "record_id": self.record_id,
            "title": self.title,
            "status": self.status,
            "capability_tags": self.capability_tags,
            "task_tags": self.task_tags,
            "failure_mode_tags": self.failure_mode_tags,
            "source_model": self.source_model,
            "source_runtime": self.source_runtime,
            "payload": self.payload,
            "evidence_refs": self.evidence_refs,
            "cross_model_validations": self.cross_model_validations,
        }

    @property
    def fingerprint(self) -> str:
        semantic = self.content_dict()
        semantic.pop("record_id")
        return sha256_json(semantic)

    def to_dict(self) -> dict[str, Any]:
        content = {
            "schema_version": 1,
            **self.content_dict(),
            "fingerprint": self.fingerprint,
        }
        return {**content, "evidence_sha256": sha256_json(content)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionRecord:
        content = dict(data)
        evidence_sha = content.pop("evidence_sha256", None)
        if evidence_sha != sha256_json(content):
            raise ContractError("evolution record evidence SHA mismatch")
        if content.pop("schema_version", None) != 1:
            raise ContractError("unsupported evolution record schema")
        fingerprint = content.pop("fingerprint", None)
        record = cls.create(**content)
        if fingerprint != record.fingerprint:
            raise ContractError("evolution record fingerprint mismatch")
        return record


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class AppendResult:
    created: bool
    path: Path
    record: EvolutionRecord


class EvolutionCatalog:
    """Immutable records plus a deterministic, rebuildable search index."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock_path = self.root / ".writer.lock"

    @contextmanager
    def _writer_lease(self) -> Any:
        """Exclusive append lock; fail fast instead of racing the catalog."""

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self._lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            os.close(descriptor)
        except FileExistsError as exc:
            raise ContractError(
                "evolution catalog writer lease is held by another writer"
            ) from exc
        try:
            yield
        finally:
            try:
                self._lock_path.unlink()
            except FileNotFoundError:
                pass

    def append(self, record: EvolutionRecord) -> AppendResult:
        with self._writer_lease():
            record.validate()
            records = self._load_all()
            for existing_path, existing in records:
                if existing.record_id == record.record_id:
                    if existing.fingerprint != record.fingerprint:
                        raise CatalogConflict("record_id already has different content")
                    return AppendResult(False, existing_path, existing)
                if existing.fingerprint == record.fingerprint:
                    return AppendResult(False, existing_path, existing)
            destination = self.root / record.record_type / f"{record.record_id}.json"
            if destination.exists():
                raise CatalogConflict("record path already exists")
            self._atomic_write(destination, record.to_dict(), exclusive=True)
            self.rebuild_index()
            return AppendResult(True, destination, record)

    def search(
        self,
        *,
        record_types: Iterable[str] = (),
        capability_tags: Iterable[str] = (),
        task_tags: Iterable[str] = (),
        failure_mode_tags: Iterable[str] = (),
        statuses: Iterable[str] = (),
        source_models: Iterable[str] = (),
    ) -> tuple[EvolutionRecord, ...]:
        filters = {
            "record_type": set(record_types),
            "capability_tags": set(capability_tags),
            "task_tags": set(task_tags),
            "failure_mode_tags": set(failure_mode_tags),
            "status": set(statuses),
            "source_model": set(source_models),
        }
        found: list[EvolutionRecord] = []
        for _path, record in self._load_all():
            if (
                filters["record_type"]
                and record.record_type not in filters["record_type"]
            ):
                continue
            if filters["status"] and record.status not in filters["status"]:
                continue
            if (
                filters["source_model"]
                and record.source_model not in filters["source_model"]
            ):
                continue
            if filters["capability_tags"] and not filters[
                "capability_tags"
            ].intersection(record.capability_tags):
                continue
            if filters["task_tags"] and not filters["task_tags"].intersection(
                record.task_tags
            ):
                continue
            if filters["failure_mode_tags"] and not filters[
                "failure_mode_tags"
            ].intersection(record.failure_mode_tags):
                continue
            found.append(record)
        return tuple(sorted(found, key=lambda item: (item.record_type, item.record_id)))

    def proposal_context(
        self,
        *,
        capability_tags: Iterable[str] = (),
        task_tags: Iterable[str] = (),
        failure_mode_tags: Iterable[str] = (),
        limit: int = 25,
    ) -> dict[str, Any]:
        if limit < 1:
            raise ContractError("proposal context limit must be positive")
        query = {
            "capability_tags": tuple(sorted(set(capability_tags))),
            "task_tags": tuple(sorted(set(task_tags))),
            "failure_mode_tags": tuple(sorted(set(failure_mode_tags))),
        }
        records = self.search(**query)

        def select(kind: str, statuses: set[str] | None = None) -> list[dict[str, Any]]:
            return [
                record.to_dict()
                for record in records
                if record.record_type == kind
                and (statuses is None or record.status in statuses)
            ][:limit]

        return {
            "schema_version": 1,
            "query": query,
            "query_fingerprint": sha256_json(query),
            "implemented_mechanisms": select(
                "mechanisms", {"implemented", "validated"}
            ),
            "disproven_mechanisms": select("mechanisms", {"disproven", "retired"}),
            "relevant_skills": select("skills"),
            "failure_clusters": select("failure_clusters"),
            "infrastructure_gaps": select("infrastructure_gaps"),
            "experiments": select("experiments"),
        }

    def rebuild_index(self) -> Path:
        entries = [
            {
                "record_id": record.record_id,
                "record_type": record.record_type,
                "fingerprint": record.fingerprint,
                "path": path.relative_to(self.root).as_posix(),
                "status": record.status,
                "capability_tags": record.capability_tags,
                "task_tags": record.task_tags,
                "failure_mode_tags": record.failure_mode_tags,
                "source_model": record.source_model,
                "source_runtime": record.source_runtime,
            }
            for path, record in self._load_all()
        ]
        entries.sort(key=lambda item: (item["record_type"], item["record_id"]))
        content = {
            "schema_version": 1,
            "record_count": len(entries),
            "entries": entries,
        }
        document = {**content, "evidence_sha256": sha256_json(content)}
        destination = self.root / "indexes" / "CATALOG.json"
        self._atomic_write(destination, document)
        return destination

    def _load_all(self) -> list[tuple[Path, EvolutionRecord]]:
        loaded: list[tuple[Path, EvolutionRecord]] = []
        for record_type in sorted(RECORD_TYPES):
            folder = self.root / record_type
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.json")):
                loaded.append(
                    (path, EvolutionRecord.from_dict(json.loads(path.read_text())))
                )
        return loaded

    @staticmethod
    def _atomic_write(
        path: Path, value: dict[str, Any], *, exclusive: bool = False
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical_json(value) + "\n"
        if exclusive:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(data)
            return
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}."
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
