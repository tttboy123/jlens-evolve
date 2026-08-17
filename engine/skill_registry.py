"""Strict, append-only registry for project-local skill candidates."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_STATUSES = {"candidate", "transfer_verified", "rejected"}
_TERMINAL_STATUSES = {"transfer_verified", "rejected"}


class SkillContractError(ValueError):
    """Raised when a skill candidate violates the project-local contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class SkillEvidenceRef:
    path: str
    sha256: str
    role: str

    @classmethod
    def from_path(cls, path: Path, *, role: str) -> SkillEvidenceRef:
        resolved = path.resolve()
        return cls(
            path=str(resolved),
            sha256=_sha256_bytes(resolved.read_bytes()),
            role=role,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillEvidenceRef:
        if not isinstance(data, dict) or set(data) != {"path", "sha256", "role"}:
            raise SkillContractError("invalid evidence ref fields")
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
            raise SkillContractError(
                f"evidence path is not an absolute file: {self.path}"
            )
        if _SHA256.fullmatch(self.sha256) is None:
            raise SkillContractError("invalid evidence sha256")
        if _sha256_bytes(path.read_bytes()) != self.sha256:
            raise SkillContractError(f"evidence sha256 mismatch: {self.path}")
        if not self.role:
            raise SkillContractError("evidence role must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "role": self.role}


@dataclass(frozen=True)
class SkillCandidate:
    schema_version: int
    skill_id: str
    revision_id: str
    parent_revision_id: str | None
    status: str
    status_reason: str
    task_family: str
    content: tuple[str, ...]
    source_task_ids: tuple[str, ...]
    applicability: dict[str, Any]
    counterexamples: tuple[str, ...]
    known_failure_modes: tuple[str, ...]
    evidence_refs: tuple[SkillEvidenceRef, ...]
    project_local_only: bool
    auto_install: bool
    active: bool

    _SERIALIZED_FIELDS = frozenset(
        {
            "schema_version",
            "skill_id",
            "revision_id",
            "parent_revision_id",
            "status",
            "status_reason",
            "task_family",
            "content",
            "source_task_ids",
            "applicability",
            "counterexamples",
            "known_failure_modes",
            "evidence_refs",
            "project_local_only",
            "auto_install",
            "active",
            "content_sha256",
            "candidate_fingerprint",
        }
    )

    @classmethod
    def create(cls, **fields: Any) -> SkillCandidate:
        candidate = cls(schema_version=1, **fields)
        candidate.validate()
        return candidate

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillCandidate:
        if not isinstance(data, dict) or set(data) != cls._SERIALIZED_FIELDS:
            raise SkillContractError("invalid SkillCandidate fields")
        candidate = cls(
            schema_version=data["schema_version"],
            skill_id=str(data["skill_id"]),
            revision_id=str(data["revision_id"]),
            parent_revision_id=data["parent_revision_id"],
            status=str(data["status"]),
            status_reason=str(data["status_reason"]),
            task_family=str(data["task_family"]),
            content=tuple(str(row) for row in data["content"]),
            source_task_ids=tuple(str(row) for row in data["source_task_ids"]),
            applicability=dict(data["applicability"]),
            counterexamples=tuple(str(row) for row in data["counterexamples"]),
            known_failure_modes=tuple(str(row) for row in data["known_failure_modes"]),
            evidence_refs=tuple(
                SkillEvidenceRef.from_dict(row) for row in data["evidence_refs"]
            ),
            project_local_only=data["project_local_only"],
            auto_install=data["auto_install"],
            active=data["active"],
        )
        candidate.validate()
        if data["content_sha256"] != candidate.content_sha256:
            raise SkillContractError("content sha256 mismatch")
        if data["candidate_fingerprint"] != candidate.candidate_fingerprint:
            raise SkillContractError("candidate fingerprint mismatch")
        return candidate

    @property
    def content_sha256(self) -> str:
        return _sha256_bytes("\n".join(self.content).encode("utf-8"))

    def constructor_fields(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "task_family": self.task_family,
            "content": self.content,
            "source_task_ids": self.source_task_ids,
            "applicability": self.applicability,
            "counterexamples": self.counterexamples,
            "known_failure_modes": self.known_failure_modes,
            "evidence_refs": self.evidence_refs,
            "project_local_only": self.project_local_only,
            "auto_install": self.auto_install,
            "active": self.active,
        }

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "status": self.status,
            "status_reason": self.status_reason,
            "task_family": self.task_family,
            "content": list(self.content),
            "source_task_ids": list(self.source_task_ids),
            "applicability": self.applicability,
            "counterexamples": list(self.counterexamples),
            "known_failure_modes": list(self.known_failure_modes),
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "project_local_only": self.project_local_only,
            "auto_install": self.auto_install,
            "active": self.active,
            "content_sha256": self.content_sha256,
        }

    @property
    def candidate_fingerprint(self) -> str:
        stable = {
            **self._content_dict(),
            "evidence_refs": [
                {"sha256": ref.sha256, "role": ref.role} for ref in self.evidence_refs
            ],
        }
        return _sha256_bytes(_canonical_json(stable).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "candidate_fingerprint": self.candidate_fingerprint,
        }

    def validate(self) -> None:
        if self.schema_version != 1:
            raise SkillContractError("unsupported skill schema")
        for label, value in (
            ("skill_id", self.skill_id),
            ("revision_id", self.revision_id),
        ):
            if _IDENTIFIER.fullmatch(value) is None:
                raise SkillContractError(f"invalid {label}")
        if self.parent_revision_id is not None and (
            _IDENTIFIER.fullmatch(self.parent_revision_id) is None
        ):
            raise SkillContractError("invalid parent_revision_id")
        if self.status not in _STATUSES:
            raise SkillContractError("invalid skill status")
        if not self.status_reason.strip():
            raise SkillContractError("status reason must be non-empty")
        if not self.task_family or not self.content or not self.source_task_ids:
            raise SkillContractError(
                "task family, content and source tasks are required"
            )
        required = self.applicability.get("required_semantics")
        if not isinstance(required, list) or not required:
            raise SkillContractError("applicability requires semantic predicates")
        if not self.counterexamples or not self.known_failure_modes:
            raise SkillContractError("counterexamples and failure modes are required")
        if not self.evidence_refs:
            raise SkillContractError("skill evidence is required")
        for ref in self.evidence_refs:
            ref.validate()
        if self.project_local_only is not True:
            raise SkillContractError("skill must be project-local")
        if self.auto_install is not False:
            raise SkillContractError("skill cannot request automatic install")
        if self.active is not False:
            raise SkillContractError("candidate skill cannot be active")


class SkillRegistry:
    """Append-only candidate revisions stored under an explicit project path."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.jsonl"

    def read_revisions(self) -> list[SkillCandidate]:
        if not self.path.exists():
            return []
        return [
            SkillCandidate.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def latest(self, skill_id: str) -> SkillCandidate | None:
        matches = [row for row in self.read_revisions() if row.skill_id == skill_id]
        return matches[-1] if matches else None

    def append(self, candidate: SkillCandidate) -> bool:
        candidate.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            existing = {
                json.loads(line)["candidate_fingerprint"]
                for line in handle
                if line.strip()
            }
            if candidate.candidate_fingerprint in existing:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(candidate.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True

    def retrieve(
        self,
        *,
        task_family: str,
        semantics: set[str],
        limit: int = 5,
    ) -> list[SkillCandidate]:
        latest = {row.skill_id: row for row in self.read_revisions()}
        selected = []
        for candidate in latest.values():
            required = set(candidate.applicability["required_semantics"])
            if (
                candidate.task_family == task_family
                and candidate.status != "rejected"
                and required.issubset(semantics)
            ):
                selected.append(candidate)
        selected.sort(key=lambda row: (row.status != "transfer_verified", row.skill_id))
        return selected[:limit]

    def transition(
        self,
        *,
        skill_id: str,
        new_status: str,
        reason: str,
        evidence_refs: tuple[SkillEvidenceRef, ...],
    ) -> SkillCandidate:
        current = self.latest(skill_id)
        if current is None:
            raise SkillContractError(f"unknown skill: {skill_id}")
        if current.status != "candidate" or new_status not in _TERMINAL_STATUSES:
            raise SkillContractError(
                f"illegal skill transition: {current.status}->{new_status}"
            )
        evidence = tuple(dict.fromkeys((*current.evidence_refs, *evidence_refs)))
        revision_seed = _canonical_json(
            {
                "parent": current.revision_id,
                "status": new_status,
                "reason": reason,
                "evidence": [ref.sha256 for ref in evidence],
            }
        )
        revision = replace(
            current,
            revision_id=f"{current.skill_id}-{_sha256_bytes(revision_seed.encode())[:12]}",
            parent_revision_id=current.revision_id,
            status=new_status,
            status_reason=reason,
            evidence_refs=evidence,
        )
        revision.validate()
        self.append(revision)
        return revision

    def render_for_review(self, skill_id: str) -> Path:
        candidate = self.latest(skill_id)
        if candidate is None:
            raise SkillContractError(f"unknown skill: {skill_id}")
        directory = self.root / "skills" / candidate.skill_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "SKILL.md"
        lines = [
            "---",
            f"name: {candidate.skill_id}",
            f"status: {candidate.status}",
            "project_local_only: true",
            "auto_install: false",
            "active: false",
            "---",
            "",
            f"# Candidate skill: {candidate.skill_id}",
            "",
            *[f"- {line}" for line in candidate.content],
            "",
            "## Counterexamples",
            "",
            *[f"- {line}" for line in candidate.counterexamples],
            "",
            "This artifact requires human review and is not installed globally.",
            "",
        ]
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, delete=False
        ) as handle:
            handle.write("\n".join(lines))
            temporary = Path(handle.name)
        temporary.replace(path)
        return path
