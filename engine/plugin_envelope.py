"""Unified, authority-bounded envelopes for integrated evolution plugins."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,191}")
_AUTHORITIES = {"execute", "observe", "propose", "persist", "admit"}
_STATUSES = {"completed", "failed", "skipped"}


class EnvelopeContractError(ValueError):
    """Raised when a plugin envelope violates authority or evidence boundaries."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str
    role: str

    @classmethod
    def from_path(cls, path: Path, *, role: str) -> ArtifactRef:
        resolved = path.resolve()
        return cls(path=str(resolved), sha256=_sha256_file(resolved), role=role)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        if not isinstance(data, dict) or set(data) != {"path", "sha256", "role"}:
            raise EnvelopeContractError("invalid artifact ref fields")
        ref = cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            role=str(data["role"]),
        )
        ref.validate()
        return ref

    def validate(self) -> None:
        path = Path(self.path)
        if not path.is_absolute() or not path.is_file():
            raise EnvelopeContractError(
                f"artifact ref is not an absolute file: {self.path}"
            )
        if _SHA256.fullmatch(self.sha256) is None:
            raise EnvelopeContractError("invalid artifact ref sha256")
        if _sha256_file(path) != self.sha256:
            raise EnvelopeContractError(f"artifact ref sha256 mismatch: {self.path}")
        if not self.role:
            raise EnvelopeContractError("artifact ref role must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "role": self.role}


@dataclass(frozen=True)
class PluginEnvelope:
    schema_version: int
    operation_id: str
    plugin_id: str
    plugin_version: str
    authority: str
    status: str
    input_hashes: dict[str, str]
    config_hashes: dict[str, str]
    candidate_ref: ArtifactRef | None
    active_ref: ArtifactRef | None
    result_refs: tuple[ArtifactRef, ...]
    evidence_refs: tuple[ArtifactRef, ...]
    used_for_admission: bool
    error: str | None

    _FIELDS = frozenset(
        {
            "schema_version",
            "operation_id",
            "plugin_id",
            "plugin_version",
            "authority",
            "status",
            "input_hashes",
            "config_hashes",
            "candidate_ref",
            "active_ref",
            "result_refs",
            "evidence_refs",
            "used_for_admission",
            "error",
            "envelope_fingerprint",
        }
    )

    @classmethod
    def create(cls, **fields: Any) -> PluginEnvelope:
        envelope = cls(schema_version=1, **fields)
        envelope.validate()
        return envelope

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginEnvelope:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise EnvelopeContractError("invalid PluginEnvelope fields")
        envelope = cls(
            schema_version=data["schema_version"],
            operation_id=str(data["operation_id"]),
            plugin_id=str(data["plugin_id"]),
            plugin_version=str(data["plugin_version"]),
            authority=str(data["authority"]),
            status=str(data["status"]),
            input_hashes=dict(data["input_hashes"]),
            config_hashes=dict(data["config_hashes"]),
            candidate_ref=(
                ArtifactRef.from_dict(data["candidate_ref"])
                if data["candidate_ref"] is not None
                else None
            ),
            active_ref=(
                ArtifactRef.from_dict(data["active_ref"])
                if data["active_ref"] is not None
                else None
            ),
            result_refs=tuple(
                ArtifactRef.from_dict(row) for row in data["result_refs"]
            ),
            evidence_refs=tuple(
                ArtifactRef.from_dict(row) for row in data["evidence_refs"]
            ),
            used_for_admission=data["used_for_admission"],
            error=data["error"],
        )
        envelope.validate()
        if data["envelope_fingerprint"] != envelope.envelope_fingerprint:
            raise EnvelopeContractError("envelope fingerprint mismatch")
        return envelope

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "authority": self.authority,
            "status": self.status,
            "input_hashes": self.input_hashes,
            "config_hashes": self.config_hashes,
            "candidate_ref": (
                self.candidate_ref.to_dict() if self.candidate_ref else None
            ),
            "active_ref": self.active_ref.to_dict() if self.active_ref else None,
            "result_refs": [ref.to_dict() for ref in self.result_refs],
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "used_for_admission": self.used_for_admission,
            "error": self.error,
        }

    @property
    def envelope_fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._content_dict()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "envelope_fingerprint": self.envelope_fingerprint,
        }

    def validate(self) -> None:
        if self.schema_version != 1:
            raise EnvelopeContractError("unsupported envelope schema")
        for label, value in (
            ("operation_id", self.operation_id),
            ("plugin_id", self.plugin_id),
            ("plugin_version", self.plugin_version),
        ):
            if _IDENTIFIER.fullmatch(value) is None:
                raise EnvelopeContractError(f"invalid {label}")
        if self.authority not in _AUTHORITIES:
            raise EnvelopeContractError("unknown envelope authority")
        if self.status not in _STATUSES:
            raise EnvelopeContractError("unknown envelope status")
        for label, values in (
            ("input_hashes", self.input_hashes),
            ("config_hashes", self.config_hashes),
        ):
            if not values or any(
                _SHA256.fullmatch(value) is None for value in values.values()
            ):
                raise EnvelopeContractError(f"invalid {label}")
        refs = [*self.result_refs, *self.evidence_refs]
        if self.candidate_ref:
            refs.append(self.candidate_ref)
        if self.active_ref:
            refs.append(self.active_ref)
        for ref in refs:
            ref.validate()
        if self.status == "completed" and (
            self.error is not None or not self.result_refs
        ):
            raise EnvelopeContractError(
                "completed envelope requires result and no error"
            )
        if self.status == "failed" and not self.error:
            raise EnvelopeContractError("failed envelope requires error")
        if self.authority in {"observe", "propose", "persist", "execute"} and (
            self.active_ref is not None or self.used_for_admission
        ):
            raise EnvelopeContractError(
                f"{self.authority} authority cannot publish active or admission"
            )
        if self.authority == "observe" and self.used_for_admission is not False:
            raise EnvelopeContractError("observe authority must remain read-only")
        if self.authority == "admit" and (
            self.active_ref is None
            or self.candidate_ref is None
            or not self.evidence_refs
            or self.used_for_admission is not True
        ):
            raise EnvelopeContractError(
                "admit authority requires candidate, active and evidence"
            )


class EnvelopeLog:
    """Append-only envelope log with operation/plugin conflict detection."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[PluginEnvelope]:
        if not self.path.exists():
            return []
        return [
            PluginEnvelope.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, envelope: PluginEnvelope) -> bool:
        envelope.validate()
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            rows = [json.loads(line) for line in handle if line.strip()]
            matching = [
                row
                for row in rows
                if row["operation_id"] == envelope.operation_id
                and row["plugin_id"] == envelope.plugin_id
            ]
            if matching:
                if (
                    matching[-1]["envelope_fingerprint"]
                    == envelope.envelope_fingerprint
                ):
                    return False
                raise EnvelopeContractError(
                    f"operation/plugin envelope conflict: {envelope.operation_id}/{envelope.plugin_id}"
                )
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(envelope.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True
