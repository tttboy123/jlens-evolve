"""Immutable, hash-verified AgentProgram revision bundles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from evolve.contracts import canonical_json


class AgentProgramViolation(ValueError):
    """An AgentProgram artifact or tournament invariant was violated."""


_ARTIFACTS = (
    "CAPABILITIES.json",
    "CONTEXT.json",
    "PROGRAM-PROMPT.txt",
    "TOOL-POLICY.json",
)
_ALLOWED_TOOLS = frozenset({"inspect_workspace", "emit_patch"})


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentProgramViolation(f"{name} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class AgentProgramRevision:
    root: Path
    program_id: str
    revision_id: str
    parent_revision_id: str | None
    program_prompt: str
    context: Mapping[str, Any]
    tool_policy: tuple[str, ...]
    capability_revision_ids: tuple[str, ...]
    artifact_sha256: tuple[tuple[str, str], ...]
    manifest_path: Path
    bundle_sha256: str

    @classmethod
    def freeze(
        cls,
        root: str | Path,
        *,
        program_id: str,
        revision_id: str,
        parent_revision_id: str | None,
        program_prompt: str,
        context: Mapping[str, Any],
        tool_policy: tuple[str, ...],
        capability_revision_ids: tuple[str, ...],
    ) -> AgentProgramRevision:
        bundle_root = Path(root).resolve()
        if bundle_root.exists():
            raise AgentProgramViolation("AgentProgram revision root already exists")
        _validate_projection(
            program_id=program_id,
            revision_id=revision_id,
            parent_revision_id=parent_revision_id,
            program_prompt=program_prompt,
            context=context,
            tool_policy=tool_policy,
            capability_revision_ids=capability_revision_ids,
        )
        bundle_root.mkdir(parents=True)
        artifacts = {
            "PROGRAM-PROMPT.txt": program_prompt.encode("utf-8"),
            "CONTEXT.json": (canonical_json(context) + "\n").encode("utf-8"),
            "TOOL-POLICY.json": (
                canonical_json(list(tool_policy)) + "\n"
            ).encode("utf-8"),
            "CAPABILITIES.json": (
                canonical_json(list(capability_revision_ids)) + "\n"
            ).encode("utf-8"),
        }
        artifact_rows = []
        for name in _ARTIFACTS:
            content = artifacts[name]
            (bundle_root / name).write_bytes(content)
            artifact_rows.append({"path": name, "sha256": _sha256(content)})
        manifest = {
            "schema_version": 1,
            "program_id": program_id,
            "revision_id": revision_id,
            "parent_revision_id": parent_revision_id,
            "artifacts": artifact_rows,
        }
        (bundle_root / "AGENT-PROGRAM.json").write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        return cls.load(bundle_root)

    @classmethod
    def load(cls, root: str | Path) -> AgentProgramRevision:
        bundle_root = Path(root).resolve()
        manifest_path = bundle_root / "AGENT-PROGRAM.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AgentProgramViolation(
                "AgentProgram revision manifest is unreadable"
            ) from error
        required = {
            "schema_version",
            "program_id",
            "revision_id",
            "parent_revision_id",
            "artifacts",
        }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != required
            or manifest.get("schema_version") != 1
        ):
            raise AgentProgramViolation("AgentProgram revision manifest is invalid")
        rows = manifest.get("artifacts")
        if not isinstance(rows, list) or len(rows) != len(_ARTIFACTS):
            raise AgentProgramViolation("AgentProgram artifacts are incomplete")
        hashes: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise AgentProgramViolation("AgentProgram artifact entry is invalid")
            name = row["path"]
            digest = row["sha256"]
            if name not in _ARTIFACTS or name in hashes:
                raise AgentProgramViolation("AgentProgram artifact set is invalid")
            path = bundle_root / name
            if path.is_symlink() or not path.is_file():
                raise AgentProgramViolation("AgentProgram artifact is missing")
            if not isinstance(digest, str) or _sha256(path.read_bytes()) != digest:
                raise AgentProgramViolation("AgentProgram artifact hash mismatch")
            hashes[name] = digest
        if set(hashes) != set(_ARTIFACTS):
            raise AgentProgramViolation("AgentProgram artifacts are incomplete")
        try:
            prompt = (bundle_root / "PROGRAM-PROMPT.txt").read_text(encoding="utf-8")
            context = json.loads((bundle_root / "CONTEXT.json").read_bytes())
            tools = json.loads((bundle_root / "TOOL-POLICY.json").read_bytes())
            capabilities = json.loads((bundle_root / "CAPABILITIES.json").read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AgentProgramViolation("AgentProgram artifact payload is invalid") from error
        if not isinstance(context, dict):
            raise AgentProgramViolation("AgentProgram context must be an object")
        if not isinstance(tools, list) or not isinstance(capabilities, list):
            raise AgentProgramViolation("AgentProgram policy artifacts must be arrays")
        parent = manifest["parent_revision_id"]
        if parent is not None and not isinstance(parent, str):
            raise AgentProgramViolation("parent_revision_id must be text or null")
        normalized_tools = tuple(tools)
        normalized_capabilities = tuple(capabilities)
        _validate_projection(
            program_id=manifest["program_id"],
            revision_id=manifest["revision_id"],
            parent_revision_id=parent,
            program_prompt=prompt,
            context=context,
            tool_policy=normalized_tools,
            capability_revision_ids=normalized_capabilities,
        )
        return cls(
            root=bundle_root,
            program_id=manifest["program_id"],
            revision_id=manifest["revision_id"],
            parent_revision_id=parent,
            program_prompt=prompt,
            context=MappingProxyType(dict(context)),
            tool_policy=normalized_tools,
            capability_revision_ids=normalized_capabilities,
            artifact_sha256=tuple((name, hashes[name]) for name in _ARTIFACTS),
            manifest_path=manifest_path,
            bundle_sha256=_sha256(manifest_bytes),
        )

    def artifact_hash(self, name: str) -> str:
        try:
            return dict(self.artifact_sha256)[name]
        except KeyError as error:
            raise AgentProgramViolation(f"unknown AgentProgram artifact: {name}") from error


def _validate_projection(
    *,
    program_id: object,
    revision_id: object,
    parent_revision_id: object,
    program_prompt: object,
    context: object,
    tool_policy: object,
    capability_revision_ids: object,
) -> None:
    _text("program_id", program_id)
    revision = _text("revision_id", revision_id)
    if parent_revision_id is not None:
        parent = _text("parent_revision_id", parent_revision_id)
        if parent == revision:
            raise AgentProgramViolation("AgentProgram revision cannot parent itself")
    _text("program_prompt", program_prompt)
    if not isinstance(context, Mapping) or not context:
        raise AgentProgramViolation("AgentProgram context must be a non-empty object")
    try:
        canonical_json(context)
    except (TypeError, ValueError) as error:
        raise AgentProgramViolation("AgentProgram context must be canonical JSON") from error
    if (
        not isinstance(tool_policy, tuple)
        or not tool_policy
        or not all(isinstance(item, str) and item in _ALLOWED_TOOLS for item in tool_policy)
        or len(set(tool_policy)) != len(tool_policy)
    ):
        raise AgentProgramViolation("AgentProgram tool policy is not allowlisted")
    if (
        not isinstance(capability_revision_ids, tuple)
        or not capability_revision_ids
        or not all(isinstance(item, str) and item.strip() for item in capability_revision_ids)
        or len(set(capability_revision_ids)) != len(capability_revision_ids)
    ):
        raise AgentProgramViolation("AgentProgram capabilities are invalid")


__all__ = ["AgentProgramRevision", "AgentProgramViolation"]
