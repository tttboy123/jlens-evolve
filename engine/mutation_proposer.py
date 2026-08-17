"""Provider-neutral, inactive Agent application-layer mutation proposals."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pattern_miner import PatternCard

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
_ALLOWED_SURFACES = (
    "prompt",
    "skills",
    "policy",
    "router",
    "memory_policy",
    "constrained_harness_code",
)
_PATH_RULES = {
    "prompt": ("AGENTS.md",),
    "skills": (".agents/skills/",),
    "policy": (".codex/evolution-policy.json",),
    "router": (".codex/evolution-policy.json",),
    "memory_policy": (".codex/evolution-policy.json",),
    "constrained_harness_code": (".codex/harness/",),
}


class MutationContractError(ValueError):
    """Raised when a mutation crosses the frozen application-layer boundary."""


@dataclass(frozen=True)
class MutationRequest:
    request_id: str
    status: str
    parent_agent_program_sha256: str
    native_evaluator_epoch: str
    surface: str
    pattern_cards: tuple[PatternCard, ...]
    admission_gate_allowed: bool = False

    @property
    def hypothesis_ids(self) -> tuple[str, ...]:
        return tuple(card.pattern_id for card in self.pattern_cards)

    def prompt(self, provider: dict[str, str]) -> str:
        contract = {
            "request_id": self.request_id,
            "status": "inactive",
            "parent_agent_program_sha256": self.parent_agent_program_sha256,
            "native_evaluator_epoch": self.native_evaluator_epoch,
            "surface": self.surface,
            "hypotheses": [card.to_dict() for card in self.pattern_cards],
            "provider": provider,
            "constraints": {
                "model_weights_frozen": True,
                "native_evaluator_external_fixed": True,
                "auto_apply": False,
                "production_promotion_allowed": False,
                "rollback_required": True,
            },
        }
        return json.dumps(contract, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class InactiveChangeSet:
    schema_version: int
    changeset_id: str
    status: str
    parent_agent_program_sha256: str
    candidate_agent_program_sha256: str
    hypothesis_ids: tuple[str, ...]
    surface: str
    operations: tuple[dict[str, Any], ...]
    rollback_operations: tuple[dict[str, Any], ...]
    proposer: dict[str, str]
    native_evaluator_epoch: str
    native_evaluator_authority: str
    auto_apply: bool
    production_promotion_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "changeset_id": self.changeset_id,
            "status": self.status,
            "parent_agent_program_sha256": self.parent_agent_program_sha256,
            "candidate_agent_program_sha256": self.candidate_agent_program_sha256,
            "hypothesis_ids": list(self.hypothesis_ids),
            "surface": self.surface,
            "operations": list(self.operations),
            "rollback_operations": list(self.rollback_operations),
            "proposer": self.proposer,
            "native_evaluator_epoch": self.native_evaluator_epoch,
            "native_evaluator_authority": self.native_evaluator_authority,
            "auto_apply": self.auto_apply,
            "production_promotion_allowed": self.production_promotion_allowed,
        }


@dataclass(frozen=True)
class ProposalResult:
    changeset: InactiveChangeSet
    repairs_used: int
    raw_responses: tuple[str, ...]


class MutationProposer:
    _FIELDS = frozenset(
        {
            "schema_version",
            "changeset_id",
            "status",
            "parent_agent_program_sha256",
            "candidate_agent_program_sha256",
            "hypothesis_ids",
            "surface",
            "operations",
            "rollback_operations",
            "proposer",
            "native_evaluator_epoch",
            "native_evaluator_authority",
            "auto_apply",
            "production_promotion_allowed",
        }
    )

    def build_requests(
        self,
        cards: Iterable[PatternCard],
        *,
        parent_agent_program_sha256: str,
        native_evaluator_epoch: str,
        maximum_candidates: int,
    ) -> tuple[MutationRequest, ...]:
        if _SHA256.fullmatch(parent_agent_program_sha256) is None:
            raise MutationContractError("invalid parent AgentProgram hash")
        if maximum_candidates < 1:
            raise MutationContractError("maximum_candidates must be positive")
        grouped: dict[str, list[PatternCard]] = {}
        surface_order: list[str] = []
        for card in cards:
            if card.causal_boundary != "observational_not_causal":
                raise MutationContractError("PatternCard lost observation boundary")
            if card.admission_gate_allowed is not False:
                raise MutationContractError(
                    "PatternCard cannot claim admission authority"
                )
            for surface in card.expected_surfaces:
                if surface not in _ALLOWED_SURFACES:
                    raise MutationContractError(
                        f"unsupported mutation surface: {surface}"
                    )
                if surface not in grouped:
                    grouped[surface] = []
                    surface_order.append(surface)
                grouped[surface].append(card)
        requests = []
        for ordinal, surface in enumerate(surface_order[:maximum_candidates], start=1):
            requests.append(
                MutationRequest(
                    request_id=f"mutation-{ordinal:02d}-{surface}",
                    status="inactive",
                    parent_agent_program_sha256=parent_agent_program_sha256,
                    native_evaluator_epoch=native_evaluator_epoch,
                    surface=surface,
                    pattern_cards=tuple(grouped[surface]),
                )
            )
        return tuple(requests)

    @staticmethod
    def _validate_operation(
        value: Any, *, surface: str, rollback: bool = False
    ) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"op", "path", "after"}:
            raise MutationContractError("invalid mutation operation fields")
        allowed_ops = {"replace", "delete"} if rollback else {"create", "replace"}
        if value["op"] not in allowed_ops:
            raise MutationContractError(
                f"mutation operation must be one of {sorted(allowed_ops)}"
            )
        path = value["path"]
        if not isinstance(path, str) or path.startswith(("/", "../")) or "/../" in path:
            raise MutationContractError("mutation path leaves project profile")
        rules = _PATH_RULES[surface]
        if not any(path == prefix or path.startswith(prefix) for prefix in rules):
            raise MutationContractError(f"mutation path is not allowed for {surface}")
        if surface == "skills" and not path.endswith("/SKILL.md"):
            raise MutationContractError("skill mutation path must end with SKILL.md")
        if value["op"] == "delete":
            if value["after"] is not None:
                raise MutationContractError("delete rollback after must be null")
        elif not isinstance(value["after"], str):
            raise MutationContractError("mutation operation after must be text")
        return {"op": value["op"], "path": path, "after": value["after"]}

    def validate_response(
        self,
        response: str,
        *,
        expected_parent_sha256: str,
        allowed_hypothesis_ids: set[str],
        frozen_native_evaluator_epoch: str,
    ) -> InactiveChangeSet:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError) as error:
            raise MutationContractError(f"invalid ChangeSet JSON: {error}") from error
        if not isinstance(data, dict):
            raise MutationContractError("ChangeSet must be a mapping")
        unknown = sorted(data.keys() - self._FIELDS)
        missing = sorted(self._FIELDS - data.keys())
        if unknown or missing:
            raise MutationContractError(
                f"ChangeSet fields mismatch: unknown={unknown}, missing={missing}"
            )
        if data["schema_version"] != 1:
            raise MutationContractError("unsupported ChangeSet schema")
        if data["status"] != "inactive":
            raise MutationContractError("ChangeSet must remain inactive")
        if data["auto_apply"] is not False:
            raise MutationContractError("auto_apply must be false")
        if data["production_promotion_allowed"] is not False:
            raise MutationContractError("production promotion must be disabled")
        if data["parent_agent_program_sha256"] != expected_parent_sha256:
            raise MutationContractError("parent AgentProgram hash mismatch")
        if (
            not isinstance(data["candidate_agent_program_sha256"], str)
            or _SHA256.fullmatch(data["candidate_agent_program_sha256"]) is None
        ):
            raise MutationContractError("invalid candidate AgentProgram hash")
        if data["native_evaluator_epoch"] != frozen_native_evaluator_epoch:
            raise MutationContractError("native evaluator epoch mutation is forbidden")
        if data["native_evaluator_authority"] != "external_fixed":
            raise MutationContractError("native evaluator authority must stay external")
        surface = data["surface"]
        if surface not in _ALLOWED_SURFACES:
            raise MutationContractError(f"unsupported mutation surface: {surface}")
        if (
            not isinstance(data["changeset_id"], str)
            or _IDENTIFIER.fullmatch(data["changeset_id"]) is None
        ):
            raise MutationContractError("invalid changeset_id")
        hypothesis_ids = data["hypothesis_ids"]
        if (
            not isinstance(hypothesis_ids, list)
            or not hypothesis_ids
            or not all(isinstance(item, str) for item in hypothesis_ids)
            or len(set(hypothesis_ids)) != len(hypothesis_ids)
            or not set(hypothesis_ids).issubset(allowed_hypothesis_ids)
        ):
            raise MutationContractError("ChangeSet hypothesis IDs are not authorized")
        proposer = data["proposer"]
        if (
            not isinstance(proposer, dict)
            or set(proposer) != {"platform", "model"}
            or not all(
                isinstance(value, str) and value.strip() for value in proposer.values()
            )
        ):
            raise MutationContractError("invalid proposer identity")
        operations = data["operations"]
        rollback = data["rollback_operations"]
        if not isinstance(operations, list) or not operations:
            raise MutationContractError("ChangeSet operations cannot be empty")
        if not isinstance(rollback, list) or len(rollback) != len(operations):
            raise MutationContractError(
                "rollback operations must match forward operations"
            )
        parsed_operations = tuple(
            self._validate_operation(item, surface=surface) for item in operations
        )
        parsed_rollback = tuple(
            self._validate_operation(item, surface=surface, rollback=True)
            for item in rollback
        )
        if [item["path"] for item in parsed_operations] != [
            item["path"] for item in parsed_rollback
        ]:
            raise MutationContractError("rollback paths must match forward paths")
        for forward, reverse in zip(parsed_operations, parsed_rollback, strict=True):
            expected_reverse = "delete" if forward["op"] == "create" else "replace"
            if reverse["op"] != expected_reverse:
                raise MutationContractError(
                    "rollback operation does not reverse the forward operation"
                )
        return InactiveChangeSet(
            schema_version=1,
            changeset_id=data["changeset_id"],
            status="inactive",
            parent_agent_program_sha256=expected_parent_sha256,
            candidate_agent_program_sha256=data["candidate_agent_program_sha256"],
            hypothesis_ids=tuple(hypothesis_ids),
            surface=surface,
            operations=parsed_operations,
            rollback_operations=parsed_rollback,
            proposer={"platform": proposer["platform"], "model": proposer["model"]},
            native_evaluator_epoch=frozen_native_evaluator_epoch,
            native_evaluator_authority="external_fixed",
            auto_apply=False,
            production_promotion_allowed=False,
        )

    def propose(
        self,
        *,
        request: MutationRequest,
        provider: dict[str, str],
        propose: Callable[[str], str],
        repair: Callable[[str], str] | None,
    ) -> ProposalResult:
        prompt = request.prompt(provider)
        first = propose(prompt)
        raw = [first]
        try:
            changeset = self.validate_response(
                first,
                expected_parent_sha256=request.parent_agent_program_sha256,
                allowed_hypothesis_ids=set(request.hypothesis_ids),
                frozen_native_evaluator_epoch=request.native_evaluator_epoch,
            )
            repairs_used = 0
        except MutationContractError as first_error:
            if repair is None:
                raise
            repair_prompt = json.dumps(
                {
                    "original_request": json.loads(prompt),
                    "invalid_response": first,
                    "validation_error": str(first_error),
                    "instruction": "Return one corrected JSON object only; keep the frozen constraints.",
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            second = repair(repair_prompt)
            raw.append(second)
            changeset = self.validate_response(
                second,
                expected_parent_sha256=request.parent_agent_program_sha256,
                allowed_hypothesis_ids=set(request.hypothesis_ids),
                frozen_native_evaluator_epoch=request.native_evaluator_epoch,
            )
            repairs_used = 1
        if changeset.proposer != provider:
            raise MutationContractError("response proposer identity mismatch")
        return ProposalResult(
            changeset=changeset,
            repairs_used=repairs_used,
            raw_responses=tuple(raw),
        )
