"""Append-only JSONL registries with an exclusive single-writer lease."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Generic, TypeVar

from evolve.contracts import canonical_json

from .records import (
    AgentProgramRecord,
    CandidateRecord,
    CapabilityRecord,
    RegistryViolation,
)

RecordT = TypeVar("RecordT", CandidateRecord, CapabilityRecord, AgentProgramRecord)


class RegistryConflict(RegistryViolation):
    """The same immutable revision identity was supplied with other content."""


class RegistryBusy(RegistryViolation):
    """Another writer currently owns the registry lease."""


class _AppendOnlyRegistry(Generic[RecordT]):
    record_type: type[RecordT]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")

    def append(self, record: RecordT) -> bool:
        if not isinstance(record, self.record_type):
            raise RegistryViolation(f"expected {self.record_type.__name__}")
        existing = self._by_key().get(record.key)
        if existing is not None:
            if existing == record:
                return False
            raise RegistryConflict(f"conflicting immutable revision {record.key!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lease_fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RegistryBusy(f"writer lease already held for {self.path}") from error
        try:
            os.write(lease_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(lease_fd)
            # Recheck after acquiring the lease so concurrent callers cannot append
            # two values with the same revision identity.
            existing = self._by_key().get(record.key)
            if existing is not None:
                if existing == record:
                    return False
                raise RegistryConflict(f"conflicting immutable revision {record.key!r}")
            payload = dataclasses.asdict(record)
            payload["record_sha256"] = record.content_sha256
            line = (canonical_json(payload) + "\n").encode("utf-8")
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
            return True
        finally:
            os.close(lease_fd)
            self.lease_path.unlink(missing_ok=True)

    def get(self, asset_id: str, revision_id: str) -> RecordT | None:
        return self._by_key().get((asset_id, revision_id))

    def all(self) -> tuple[RecordT, ...]:
        return tuple(self._read())

    def _by_key(self) -> dict[tuple[str, str], RecordT]:
        return {record.key: record for record in self._read()}

    def _read(self) -> list[RecordT]:
        if not self.path.exists():
            return []
        records: list[RecordT] = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            try:
                payload = json.loads(line)
                expected_hash = payload.pop("record_sha256")
                for field in dataclasses.fields(self.record_type):
                    if field.name in payload and field.name.endswith("_ids"):
                        payload[field.name] = tuple(payload[field.name])
                record = self.record_type(**payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RegistryViolation(
                    f"invalid registry entry at {self.path}:{line_number}"
                ) from error
            if record.content_sha256 != expected_hash:
                raise RegistryViolation(
                    f"registry hash mismatch at {self.path}:{line_number}"
                )
            records.append(record)
        return records


class CandidateRegistry(_AppendOnlyRegistry[CandidateRecord]):
    record_type = CandidateRecord


class CapabilityRegistry(_AppendOnlyRegistry[CapabilityRecord]):
    record_type = CapabilityRecord


class AgentProgramRegistry(_AppendOnlyRegistry[AgentProgramRecord]):
    record_type = AgentProgramRecord
