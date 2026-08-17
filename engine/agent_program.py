"""Versioned AgentProgram and replay-only mutation contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ContractError(ValueError):
    """Raised when an AgentProgram or proposal crosses the frozen contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ComponentRegistry:
    prompts: dict[str, dict[str, Any]]
    demonstrations: dict[str, dict[str, Any]]
    skills: dict[str, dict[str, Any]]
    tool_policies: dict[str, dict[str, Any]]
    context_policies: dict[str, dict[str, Any]]
    retry_policies: dict[str, dict[str, Any]]
    routing_policies: dict[str, dict[str, Any]]
    harnesses: dict[str, dict[str, Any]]

    @classmethod
    def from_path(cls, path: Path) -> ComponentRegistry:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ContractError("unsupported component registry schema")
        required = {
            "prompts",
            "skills",
            "tool_policies",
            "context_policies",
            "retry_policies",
            "routing_policies",
            "harnesses",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ContractError(f"component registry missing sections: {missing}")
        sections = {}
        for name in required | {"demonstrations"}:
            value = data.get(name, {})
            if not isinstance(value, dict):
                raise ContractError(f"registry section must be a mapping: {name}")
            sections[name] = value
        return cls(**sections)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentProgram:
    schema_version: int
    program_id: str
    parent_program_hash: str | None
    system_prompt_ref: str
    demonstration_refs: tuple[str, ...]
    skill_refs: tuple[str, ...]
    tool_policy_ref: str
    context_policy_ref: str
    retry_policy_ref: str
    routing_policy_ref: str
    harness_code_ref: str

    _FIELDS = frozenset(
        {
            "schema_version",
            "program_id",
            "parent_program_hash",
            "system_prompt_ref",
            "demonstration_refs",
            "skill_refs",
            "tool_policy_ref",
            "context_policy_ref",
            "retry_policy_ref",
            "routing_policy_ref",
            "harness_code_ref",
        }
    )

    @classmethod
    def from_path(cls, path: Path) -> AgentProgram:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentProgram:
        if not isinstance(data, dict):
            raise ContractError("AgentProgram must be a mapping")
        unknown = sorted(data.keys() - cls._FIELDS)
        missing = sorted(cls._FIELDS - data.keys())
        if unknown:
            raise ContractError(f"unknown AgentProgram fields: {unknown}")
        if missing:
            raise ContractError(f"missing AgentProgram fields: {missing}")
        for field in ("demonstration_refs", "skill_refs"):
            if not isinstance(data[field], list) or not all(
                isinstance(item, str) for item in data[field]
            ):
                raise ContractError(f"{field} must be a string list")
        values = dict(data)
        values["demonstration_refs"] = tuple(values["demonstration_refs"])
        values["skill_refs"] = tuple(values["skill_refs"])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "program_id": self.program_id,
            "parent_program_hash": self.parent_program_hash,
            "system_prompt_ref": self.system_prompt_ref,
            "demonstration_refs": list(self.demonstration_refs),
            "skill_refs": list(self.skill_refs),
            "tool_policy_ref": self.tool_policy_ref,
            "context_policy_ref": self.context_policy_ref,
            "retry_policy_ref": self.retry_policy_ref,
            "routing_policy_ref": self.routing_policy_ref,
            "harness_code_ref": self.harness_code_ref,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    def validate(self, registry: ComponentRegistry) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported AgentProgram schema")
        if _IDENTIFIER.fullmatch(self.program_id) is None:
            raise ContractError("invalid program_id")
        if (
            self.parent_program_hash is not None
            and _SHA256.fullmatch(self.parent_program_hash) is None
        ):
            raise ContractError("invalid parent_program_hash")
        if len(set(self.demonstration_refs)) != len(self.demonstration_refs):
            raise ContractError("duplicate demonstration_refs")
        if len(set(self.skill_refs)) != len(self.skill_refs):
            raise ContractError("duplicate skill_refs")
        references = (
            ("system_prompt_ref", self.system_prompt_ref, registry.prompts),
            ("tool_policy_ref", self.tool_policy_ref, registry.tool_policies),
            ("context_policy_ref", self.context_policy_ref, registry.context_policies),
            ("retry_policy_ref", self.retry_policy_ref, registry.retry_policies),
            ("routing_policy_ref", self.routing_policy_ref, registry.routing_policies),
            ("harness_code_ref", self.harness_code_ref, registry.harnesses),
        )
        for name, reference, section in references:
            if reference not in section:
                raise ContractError(f"unknown {name}: {reference}")
        unknown_demonstrations = sorted(
            set(self.demonstration_refs) - registry.demonstrations.keys()
        )
        if unknown_demonstrations:
            raise ContractError(f"unknown demonstration_refs: {unknown_demonstrations}")
        unknown_skills = sorted(set(self.skill_refs) - registry.skills.keys())
        if unknown_skills:
            raise ContractError(f"unknown skill_refs: {unknown_skills}")


_MUTATION_FIELDS = {
    "prompt_instruction": "system_prompt_ref",
    "skill_composition": "skill_refs",
    "retry_policy": "retry_policy_ref",
}


@dataclass(frozen=True)
class MutationProposal:
    schema_version: int
    proposal_id: str
    parent_program_hash: str
    program_id: str
    mutation_type: str
    changes: dict[str, Any]
    reason: str

    _FIELDS = frozenset(
        {
            "schema_version",
            "proposal_id",
            "parent_program_hash",
            "program_id",
            "mutation_type",
            "changes",
            "reason",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MutationProposal:
        if not isinstance(data, dict):
            raise ContractError("MutationProposal must be a mapping")
        unknown = sorted(data.keys() - cls._FIELDS)
        missing = sorted(cls._FIELDS - data.keys())
        if unknown:
            raise ContractError(f"unknown MutationProposal fields: {unknown}")
        if missing:
            raise ContractError(f"missing MutationProposal fields: {missing}")
        if data["schema_version"] != 1:
            raise ContractError("unsupported MutationProposal schema")
        mutation_type = data["mutation_type"]
        if mutation_type not in _MUTATION_FIELDS:
            raise ContractError(f"unknown mutation_type: {mutation_type}")
        changes = data["changes"]
        expected_field = _MUTATION_FIELDS[mutation_type]
        if not isinstance(changes, dict) or set(changes) != {expected_field}:
            raise ContractError(
                f"changes does not match mutation_type {mutation_type}: "
                f"expected only {expected_field}"
            )
        if expected_field == "skill_refs":
            value = changes[expected_field]
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ContractError("skill_refs change must be a string list")
        elif not isinstance(changes[expected_field], str):
            raise ContractError(f"{expected_field} change must be a string")
        for field in ("proposal_id", "program_id"):
            if (
                not isinstance(data[field], str)
                or _IDENTIFIER.fullmatch(data[field]) is None
            ):
                raise ContractError(f"invalid {field}")
        if (
            not isinstance(data["parent_program_hash"], str)
            or _SHA256.fullmatch(data["parent_program_hash"]) is None
        ):
            raise ContractError("invalid proposal parent_program_hash")
        if not isinstance(data["reason"], str) or not data["reason"].strip():
            raise ContractError("proposal reason must be non-empty")
        return cls(
            schema_version=1,
            proposal_id=data["proposal_id"],
            parent_program_hash=data["parent_program_hash"],
            program_id=data["program_id"],
            mutation_type=mutation_type,
            changes=dict(changes),
            reason=data["reason"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "parent_program_hash": self.parent_program_hash,
            "program_id": self.program_id,
            "mutation_type": self.mutation_type,
            "changes": self.changes,
            "reason": self.reason,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplaySupervisor:
    supervisor_id: str
    proposals: tuple[MutationProposal, ...]

    @classmethod
    def from_path(cls, path: Path) -> ReplaySupervisor:
        data = json.loads(path.read_text(encoding="utf-8"))
        required = {"schema_version", "supervisor_id", "proposals"}
        if not isinstance(data, dict) or set(data) != required:
            raise ContractError("invalid ReplaySupervisor config fields")
        if data["schema_version"] != 1:
            raise ContractError("unsupported ReplaySupervisor schema")
        if (
            not isinstance(data["supervisor_id"], str)
            or _IDENTIFIER.fullmatch(data["supervisor_id"]) is None
        ):
            raise ContractError("invalid supervisor_id")
        if not isinstance(data["proposals"], list):
            raise ContractError("ReplaySupervisor proposals must be a list")
        proposals = tuple(
            MutationProposal.from_dict(item) for item in data["proposals"]
        )
        ids = [proposal.proposal_id for proposal in proposals]
        if len(set(ids)) != len(ids):
            raise ContractError("duplicate proposal_id")
        return cls(supervisor_id=data["supervisor_id"], proposals=proposals)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(
                {
                    "schema_version": 1,
                    "supervisor_id": self.supervisor_id,
                    "proposals": [proposal.to_dict() for proposal in self.proposals],
                }
            ).encode("utf-8")
        ).hexdigest()


def apply_proposal(
    parent: AgentProgram,
    proposal: MutationProposal,
    registry: ComponentRegistry,
) -> AgentProgram:
    """Create one child while keeping all non-proposed application axes frozen."""
    parent.validate(registry)
    if proposal.parent_program_hash != parent.sha256:
        raise ContractError(
            "parent hash mismatch: "
            f"expected {parent.sha256}, got {proposal.parent_program_hash}"
        )
    field = _MUTATION_FIELDS[proposal.mutation_type]
    value = proposal.changes[field]
    if field == "skill_refs":
        value = tuple(value)
    child = replace(
        parent,
        program_id=proposal.program_id,
        parent_program_hash=parent.sha256,
        **{field: value},
    )
    child.validate(registry)
    changed_axes = {
        name
        for name in _MUTATION_FIELDS.values()
        if getattr(parent, name) != getattr(child, name)
    }
    if changed_axes != {field}:
        raise ContractError(
            f"proposal must change exactly one allowed axis: {sorted(changed_axes)}"
        )
    for frozen_field in (
        "demonstration_refs",
        "tool_policy_ref",
        "context_policy_ref",
        "routing_policy_ref",
        "harness_code_ref",
    ):
        if getattr(parent, frozen_field) != getattr(child, frozen_field):
            raise ContractError(f"proposal changed frozen field: {frozen_field}")
    return child
