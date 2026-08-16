"""Fail-closed compilation of a real Teacher receipt into runtime assets."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from evolve.contracts import (
    Cohort,
    ContractViolation,
    canonical_json,
    content_sha256,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{name} must be non-empty text")
    return value


def _require_identifier(name: str, value: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ContractViolation(f"{name} is not a safe immutable identifier")


@dataclass(frozen=True, slots=True)
class CompileSpec:
    """Immutable revision identity plus the legacy v1 executable bridge."""

    candidate_id: str
    revision_id: str
    parent_revision_id: str
    cohort: Cohort
    operator_id: str = ""
    operator_instruction: str = ""
    routes: tuple[tuple[str, str], ...] = ()
    required_route_task_ids: tuple[str, ...] = dataclasses.field(
        default=(), metadata={"omit_if_empty": True}
    )

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "revision_id",
            "parent_revision_id",
        ):
            value = _require_text(name, getattr(self, name))
            _require_identifier(name, value)
        if self.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("candidate compilation is feedback-only")
        required: set[str] = set()
        for task_id in self.required_route_task_ids:
            _require_identifier("required route task_id", task_id)
            if task_id in required:
                raise ContractViolation("required route task IDs must be unique")
            required.add(task_id)
        if not self.operator_id and not self.operator_instruction and not self.routes:
            return
        _require_identifier(
            "operator_id", _require_text("operator_id", self.operator_id)
        )
        _require_text("operator_instruction", self.operator_instruction)
        if not self.routes:
            raise ContractViolation("legacy candidate router must contain routes")
        task_ids: set[str] = set()
        for task_id, operator_id in self.routes:
            _require_identifier("route task_id", task_id)
            if task_id in task_ids:
                raise ContractViolation("candidate router task IDs must be unique")
            if operator_id != self.operator_id:
                raise ContractViolation("candidate router references another operator")
            task_ids.add(task_id)

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class CandidateChangeSet:
    candidate_id: str
    revision_id: str
    parent_revision_id: str
    source_candidate_sha256: str
    compile_spec_sha256: str
    protocol: str
    prompt_template: str
    skill_text: str
    eval_note: str
    operator_id: str
    operator_instruction: str
    routes: tuple[tuple[str, str], ...]
    active: bool = False
    operator_kind: str = "zero-arg"
    operator_arguments: tuple[str, ...] = ()
    memory_policy: dict[str, Any] | None = None
    preconditions: tuple[object, ...] = ()
    expected_external_effect: object | None = None
    expected_internal_effect: object | None = None
    falsification: object | None = None
    synthesized_task_ids: tuple[str, ...] = dataclasses.field(
        default=(), metadata={"omit_if_empty": True}
    )
    source_request_sha256: str = ""
    source_response_sha256: str = ""
    lineage_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "revision_id",
            "parent_revision_id",
            "operator_id",
        ):
            value = _require_text(name, getattr(self, name))
            _require_identifier(name, value)
        for name in (
            "protocol",
            "prompt_template",
            "skill_text",
            "eval_note",
            "operator_instruction",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "source_candidate_sha256",
            "compile_spec_sha256",
            "source_request_sha256",
            "source_response_sha256",
            "lineage_sha256",
        ):
            value = getattr(self, name)
            if value and _SHA256.fullmatch(value) is None:
                raise ContractViolation(f"{name} must be a literal SHA-256")
        if self.active:
            raise ContractViolation("compiled candidate must remain inactive")
        if not self.routes or any(
            operator_id != self.operator_id for _, operator_id in self.routes
        ):
            raise ContractViolation("candidate routes must select its Operator")

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class CompiledSkill:
    candidate_id: str
    revision_id: str
    parent_revision_id: str
    protocol: str
    prompt_template: str
    skill_text: str
    preconditions: tuple[object, ...] = ()
    expected_external_effect: object | None = None
    expected_internal_effect: object | None = None
    falsification: object | None = None
    lineage_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CompiledOperator:
    candidate_id: str
    revision_id: str
    operator_id: str
    kind: str
    arguments: tuple[str, ...]
    instruction: str
    parent_revision_id: str = ""
    lineage_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CompiledRouter:
    candidate_id: str
    revision_id: str
    routes: tuple[tuple[str, str], ...]
    parent_revision_id: str = ""
    lineage_sha256: str = ""


@dataclass(frozen=True, slots=True)
class CompiledMemoryPolicy:
    candidate_id: str
    revision_id: str
    parent_revision_id: str
    policy: dict[str, Any]
    lineage_sha256: str


@dataclass(frozen=True, slots=True)
class CompiledRevision:
    root: Path
    change_set: CandidateChangeSet
    skill: CompiledSkill
    operator: CompiledOperator
    router: CompiledRouter
    provider: str
    model: str
    cost_cny: float
    artifact_sha256: tuple[tuple[str, str], ...]
    manifest_path: Path
    bundle_sha256: str
    memory_policy: CompiledMemoryPolicy | None = None
    synthesized_task_ids: tuple[str, ...] = ()
    lineage_sha256: str = ""

    @classmethod
    def load(cls, root: str | Path) -> CompiledRevision:
        bundle_root = Path(root).resolve()
        manifest_path = bundle_root / "COMPILED-REVISION.json"
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractViolation(
                "compiled revision manifest is unreadable"
            ) from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") not in {
            1,
            2,
        }:
            raise ContractViolation("compiled revision manifest is invalid")
        manifest_schema = manifest["schema_version"]
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ContractViolation("compiled revision artifacts are missing")
        artifact_hashes: list[tuple[str, str]] = []
        for entry in artifacts:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ContractViolation("compiled artifact entry is invalid")
            name = entry["path"]
            digest = entry["sha256"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not isinstance(digest, str)
                or _SHA256.fullmatch(digest) is None
            ):
                raise ContractViolation("compiled artifact identity is invalid")
            try:
                content = (bundle_root / name).read_bytes()
            except OSError as error:
                raise ContractViolation(
                    f"compiled artifact is missing: {name}"
                ) from error
            if _sha256(content) != digest:
                raise ContractViolation(f"compiled artifact hash mismatch: {name}")
            artifact_hashes.append((name, digest))

        expected_names = {
            "TEACHER-REQUEST.json",
            "TEACHER-RESPONSE.json",
            "COMPILE-SPEC.json",
            "MODEL-RECEIPT.json",
            "COST-RECEIPT.json",
            "CANDIDATE-CHANGESET.json",
            "COMPILED-SKILL.json",
            "COMPILED-OPERATOR.json",
            "COMPILED-ROUTER.json",
        }
        if manifest.get("memory_policy_present") is True:
            expected_names.add("COMPILED-MEMORY-POLICY.json")
        if (
            len(artifact_hashes) != len(expected_names)
            or {name for name, _ in artifact_hashes} != expected_names
        ):
            raise ContractViolation("compiled revision artifact set is incomplete")

        request_bytes = _artifact_bytes(
            bundle_root, artifact_hashes, "TEACHER-REQUEST.json"
        )
        response_bytes = _artifact_bytes(
            bundle_root, artifact_hashes, "TEACHER-RESPONSE.json"
        )
        if manifest.get("request_sha256") != _sha256(request_bytes) or manifest.get(
            "response_sha256"
        ) != _sha256(response_bytes):
            raise ContractViolation("compiled Teacher lineage hash mismatch")
        try:
            request = json.loads(request_bytes)
            response = json.loads(response_bytes)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ContractViolation(
                "compiled Teacher lineage is invalid JSON"
            ) from error
        teacher = _parse_teacher_exchange(request, response, request_bytes)
        if (
            manifest.get("provider") != teacher["provider"]
            or manifest.get("model") != teacher["model"]
            or manifest.get("cost_cny") != teacher["cost_cny"]
        ):
            raise ContractViolation("compiled Teacher projection identity mismatch")
        model_receipt = _artifact_json(
            bundle_root, artifact_hashes, "MODEL-RECEIPT.json"
        )
        if model_receipt != {
            "schema_version": 1,
            "provider": teacher["provider"],
            "model": teacher["model"],
            "usage": teacher["usage"],
            "network_calls": teacher["network_calls"],
            "request_sha256": _sha256(request_bytes),
            "response_sha256": _sha256(response_bytes),
        }:
            raise ContractViolation("compiled model receipt identity mismatch")
        cost_receipt = _artifact_json(bundle_root, artifact_hashes, "COST-RECEIPT.json")
        if cost_receipt != {
            "schema_version": 1,
            "currency": "CNY",
            "cost_kind": "estimated",
            "cost_cny": teacher["cost_cny"],
            "pricing_cny_per_million": teacher["pricing"],
            "source": "TEACHER-RESPONSE.json:estimated_cost_cny",
        }:
            raise ContractViolation("compiled cost receipt identity mismatch")

        compile_spec_payload = _artifact_json(
            bundle_root, artifact_hashes, "COMPILE-SPEC.json"
        )
        compile_spec = CompileSpec(
            candidate_id=compile_spec_payload["candidate_id"],
            revision_id=compile_spec_payload["revision_id"],
            parent_revision_id=compile_spec_payload["parent_revision_id"],
            cohort=Cohort(compile_spec_payload["cohort"]),
            operator_id=compile_spec_payload.get("operator_id", ""),
            operator_instruction=compile_spec_payload.get("operator_instruction", ""),
            routes=_routes(compile_spec_payload.get("routes", ())),
            required_route_task_ids=tuple(
                compile_spec_payload.get("required_route_task_ids", ())
            ),
        )
        if manifest.get("compile_spec_sha256") != compile_spec.content_sha256:
            raise ContractViolation("compiled CompileSpec identity mismatch")

        change_set_payload = _artifact_json(
            bundle_root, artifact_hashes, "CANDIDATE-CHANGESET.json"
        )
        change_set = CandidateChangeSet(
            **{
                **change_set_payload,
                "routes": _routes(change_set_payload["routes"]),
                "operator_arguments": tuple(
                    change_set_payload.get("operator_arguments", ())
                ),
                "preconditions": tuple(change_set_payload.get("preconditions", ())),
                "synthesized_task_ids": tuple(
                    change_set_payload.get("synthesized_task_ids", ())
                ),
            }
        )
        _validate_change_set_source(change_set, teacher, compile_spec)
        if manifest.get("revision_sha256") != change_set.content_sha256:
            raise ContractViolation("compiled candidate revision identity mismatch")
        if (
            manifest.get("candidate_id") != change_set.candidate_id
            or manifest.get("revision_id") != change_set.revision_id
            or manifest.get("parent_revision_id") != change_set.parent_revision_id
            or compile_spec.candidate_id != change_set.candidate_id
            or compile_spec.revision_id != change_set.revision_id
            or compile_spec.parent_revision_id != change_set.parent_revision_id
        ):
            raise ContractViolation("compiled revision lineage identity mismatch")
        skill_payload = _without_metadata(
            _artifact_json(bundle_root, artifact_hashes, "COMPILED-SKILL.json"),
            "skill",
        )
        skill = CompiledSkill(
            **{
                **skill_payload,
                "preconditions": tuple(skill_payload.get("preconditions", ())),
            }
        )
        operator_payload = _without_metadata(
            _artifact_json(bundle_root, artifact_hashes, "COMPILED-OPERATOR.json"),
            "operator",
        )
        operator = CompiledOperator(
            **{**operator_payload, "arguments": tuple(operator_payload["arguments"])}
        )
        router_payload = _without_metadata(
            _artifact_json(bundle_root, artifact_hashes, "COMPILED-ROUTER.json"),
            "router",
        )
        router = CompiledRouter(
            **{**router_payload, "routes": _routes(router_payload["routes"])}
        )
        memory_policy: CompiledMemoryPolicy | None = None
        if "COMPILED-MEMORY-POLICY.json" in expected_names:
            memory_payload = _without_metadata(
                _artifact_json(
                    bundle_root,
                    artifact_hashes,
                    "COMPILED-MEMORY-POLICY.json",
                ),
                "memory_policy",
            )
            memory_policy = CompiledMemoryPolicy(**memory_payload)
        _validate_compiled_projection(
            change_set, skill, operator, router, memory_policy=memory_policy
        )
        lineage_sha256 = ""
        if manifest_schema == 2:
            lineage = {
                "parent_revision_id": compile_spec.parent_revision_id,
                "source_candidate_sha256": content_sha256(teacher["candidate"]),
                "compile_spec_sha256": compile_spec.content_sha256,
                "request_sha256": _sha256(request_bytes),
                "response_sha256": _sha256(response_bytes),
            }
            lineage_sha256 = content_sha256(lineage)
            if (
                manifest.get("lineage") != lineage
                or manifest.get("lineage_sha256") != lineage_sha256
            ):
                raise ContractViolation("compiled candidate lineage hash mismatch")
            if (
                change_set.source_candidate_sha256 != lineage["source_candidate_sha256"]
                or change_set.compile_spec_sha256 != lineage["compile_spec_sha256"]
                or change_set.parent_revision_id != lineage["parent_revision_id"]
                or manifest.get("parent_revision_id") != lineage["parent_revision_id"]
                or change_set.source_request_sha256 != lineage["request_sha256"]
                or change_set.source_response_sha256 != lineage["response_sha256"]
                or change_set.lineage_sha256 != lineage_sha256
                or skill.lineage_sha256 != lineage_sha256
                or operator.lineage_sha256 != lineage_sha256
                or router.lineage_sha256 != lineage_sha256
                or (
                    memory_policy is not None
                    and memory_policy.lineage_sha256 != lineage_sha256
                )
            ):
                raise ContractViolation("compiled artifact lineage is incomplete")
        return cls(
            root=bundle_root,
            change_set=change_set,
            skill=skill,
            operator=operator,
            router=router,
            provider=_require_text("manifest provider", manifest.get("provider")),
            model=_require_text("manifest model", manifest.get("model")),
            cost_cny=_cost(manifest.get("cost_cny")),
            artifact_sha256=tuple(artifact_hashes),
            manifest_path=manifest_path,
            bundle_sha256=_sha256(manifest_bytes),
            memory_policy=memory_policy,
            synthesized_task_ids=change_set.synthesized_task_ids,
            lineage_sha256=lineage_sha256,
        )


class CandidateCompiler:
    """Compile a verified Teacher receipt without inventing missing v2 assets."""

    def compile(
        self,
        *,
        request_path: str | Path,
        response_path: str | Path,
        compile_spec: CompileSpec,
        output_root: str | Path,
    ) -> CompiledRevision:
        request_source = Path(request_path).resolve()
        response_source = Path(response_path).resolve()
        try:
            request_bytes = request_source.read_bytes()
            response_bytes = response_source.read_bytes()
            request = json.loads(request_bytes)
            response = json.loads(response_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractViolation("Teacher request/response is unreadable") from error
        parsed = _parse_teacher_exchange(request, response, request_bytes)
        candidate = parsed["candidate"]
        if parsed["candidate_schema_version"] == 1:
            if (
                not compile_spec.operator_id
                or not compile_spec.operator_instruction
                or not compile_spec.routes
            ):
                raise ContractViolation(
                    "v1 Teacher candidate requires the legacy CompileSpec bridge"
                )
            operator_id = compile_spec.operator_id
            operator_kind = "zero-arg"
            operator_arguments: tuple[str, ...] = ()
            operator_instruction = compile_spec.operator_instruction
            routes = compile_spec.routes
            memory_policy_payload: dict[str, Any] | None = None
            preconditions: tuple[object, ...] = ()
            expected_external_effect: object | None = None
            expected_internal_effect: object | None = None
            falsification: object | None = None
        else:
            (
                operator_id,
                operator_kind,
                operator_arguments,
                operator_instruction,
            ) = _operator(candidate["operator"])
            routes = _candidate_routes(
                candidate["router"],
                operator_id,
                required_task_ids=compile_spec.required_route_task_ids,
            )
            synthesized_task_ids = tuple(
                task_id
                for task_id in compile_spec.required_route_task_ids
                if task_id not in {task_id for task_id, _ in routes}
            )
            routes = _repair_routes(
                routes, compile_spec.required_route_task_ids, operator_id
            )
            memory_value = candidate["memory_policy"]
            memory_policy_payload = None if memory_value is None else dict(memory_value)
            preconditions = tuple(candidate["preconditions"])
            expected_external_effect = candidate["expected_external_effect"]
            expected_internal_effect = candidate["expected_internal_effect"]
            falsification = candidate["falsification"]

        request_sha256 = _sha256(request_bytes)
        response_sha256 = _sha256(response_bytes)
        source_candidate_sha256 = content_sha256(candidate)
        lineage = {
            "parent_revision_id": compile_spec.parent_revision_id,
            "source_candidate_sha256": source_candidate_sha256,
            "compile_spec_sha256": compile_spec.content_sha256,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
        }
        lineage_sha256 = content_sha256(lineage)
        change_set = CandidateChangeSet(
            candidate_id=compile_spec.candidate_id,
            revision_id=compile_spec.revision_id,
            parent_revision_id=compile_spec.parent_revision_id,
            source_candidate_sha256=source_candidate_sha256,
            compile_spec_sha256=compile_spec.content_sha256,
            protocol=candidate["protocol"],
            prompt_template=candidate["prompt_template"],
            skill_text=candidate["skill_text"],
            eval_note=candidate["eval_note"],
            operator_id=operator_id,
            operator_instruction=operator_instruction,
            routes=routes,
            operator_kind=operator_kind,
            operator_arguments=operator_arguments,
            memory_policy=memory_policy_payload,
            preconditions=preconditions,
            expected_external_effect=expected_external_effect,
            expected_internal_effect=expected_internal_effect,
            falsification=falsification,
            synthesized_task_ids=(
                synthesized_task_ids if parsed["candidate_schema_version"] == 2 else ()
            ),
            source_request_sha256=request_sha256,
            source_response_sha256=response_sha256,
            lineage_sha256=lineage_sha256,
        )
        skill = CompiledSkill(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            parent_revision_id=change_set.parent_revision_id,
            protocol=change_set.protocol,
            prompt_template=change_set.prompt_template,
            skill_text=change_set.skill_text,
            preconditions=change_set.preconditions,
            expected_external_effect=change_set.expected_external_effect,
            expected_internal_effect=change_set.expected_internal_effect,
            falsification=change_set.falsification,
            lineage_sha256=lineage_sha256,
        )
        operator = CompiledOperator(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            operator_id=change_set.operator_id,
            kind=change_set.operator_kind,
            arguments=change_set.operator_arguments,
            instruction=change_set.operator_instruction,
            parent_revision_id=change_set.parent_revision_id,
            lineage_sha256=lineage_sha256,
        )
        router = CompiledRouter(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            routes=change_set.routes,
            parent_revision_id=change_set.parent_revision_id,
            lineage_sha256=lineage_sha256,
        )
        memory_policy = (
            None
            if memory_policy_payload is None
            else CompiledMemoryPolicy(
                candidate_id=change_set.candidate_id,
                revision_id=change_set.revision_id,
                parent_revision_id=change_set.parent_revision_id,
                policy=memory_policy_payload,
                lineage_sha256=lineage_sha256,
            )
        )
        _validate_compiled_projection(
            change_set, skill, operator, router, memory_policy=memory_policy
        )

        root = Path(output_root).resolve() / compile_spec.revision_id
        payloads: dict[str, bytes] = {
            "TEACHER-REQUEST.json": request_bytes,
            "TEACHER-RESPONSE.json": response_bytes,
            "COMPILE-SPEC.json": _encoded(compile_spec),
            "MODEL-RECEIPT.json": _encoded(
                {
                    "schema_version": 1,
                    "provider": parsed["provider"],
                    "model": parsed["model"],
                    "usage": parsed["usage"],
                    "network_calls": parsed["network_calls"],
                    "request_sha256": _sha256(request_bytes),
                    "response_sha256": _sha256(response_bytes),
                }
            ),
            "COST-RECEIPT.json": _encoded(
                {
                    "schema_version": 1,
                    "currency": "CNY",
                    "cost_kind": "estimated",
                    "cost_cny": parsed["cost_cny"],
                    "pricing_cny_per_million": parsed["pricing"],
                    "source": "TEACHER-RESPONSE.json:estimated_cost_cny",
                }
            ),
            "CANDIDATE-CHANGESET.json": _encoded(change_set),
            "COMPILED-SKILL.json": _compiled_encoded("skill", skill),
            "COMPILED-OPERATOR.json": _compiled_encoded("operator", operator),
            "COMPILED-ROUTER.json": _compiled_encoded("router", router),
        }
        if memory_policy is not None:
            payloads["COMPILED-MEMORY-POLICY.json"] = _compiled_encoded(
                "memory_policy", memory_policy
            )
        artifacts = [
            {"path": name, "sha256": _sha256(content)}
            for name, content in payloads.items()
        ]
        manifest = {
            "schema_version": 2,
            "candidate_id": change_set.candidate_id,
            "revision_id": change_set.revision_id,
            "parent_revision_id": change_set.parent_revision_id,
            "revision_sha256": change_set.content_sha256,
            "compile_spec_sha256": compile_spec.content_sha256,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "lineage": lineage,
            "lineage_sha256": lineage_sha256,
            "memory_policy_present": memory_policy is not None,
            "provider": parsed["provider"],
            "model": parsed["model"],
            "cost_cny": parsed["cost_cny"],
            "artifacts": artifacts,
        }
        for name, content in payloads.items():
            _freeze(root / name, content)
        _freeze(root / "COMPILED-REVISION.json", _encoded(manifest))
        return CompiledRevision.load(root)


def _parse_teacher_exchange(
    request: object, response: object, request_bytes: bytes
) -> dict[str, Any]:
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ContractViolation("Teacher request/response must be JSON objects")
    receipt_sha256 = response.get("receipt_sha256")
    if receipt_sha256 is not None:
        unsigned = {
            key: value for key, value in response.items() if key != "receipt_sha256"
        }
        if receipt_sha256 != content_sha256(unsigned):
            raise ContractViolation("Teacher response hash mismatch")
    if response.get("request_sha256") != _sha256(request_bytes):
        raise ContractViolation("Teacher response request hash mismatch")
    if (
        response.get("candidate_status") != "inactive"
        or response.get("auto_activate") is not False
    ):
        raise ContractViolation("Teacher candidate must be inactive")
    provider = _require_text("Teacher provider", response.get("provider"))
    model = _require_text("Teacher model", response.get("model"))
    if request.get("model") != model:
        raise ContractViolation("Teacher model identity mismatch")
    candidate = response.get("candidate")
    v1_fields = {"protocol", "prompt_template", "skill_text", "eval_note"}
    v2_fields = v1_fields | {
        "operator",
        "router",
        "memory_policy",
        "preconditions",
        "expected_external_effect",
        "expected_internal_effect",
        "falsification",
    }
    if not isinstance(candidate, dict):
        raise ContractViolation("Teacher candidate fields are invalid")
    candidate_fields = frozenset(candidate)
    if candidate_fields not in {frozenset(v1_fields), frozenset(v2_fields)}:
        raise ContractViolation("Teacher candidate fields are invalid")
    candidate_schema_version = 1 if set(candidate) == v1_fields else 2
    candidate = dict(candidate)
    candidate_sha256 = response.get("candidate_sha256")
    if candidate_sha256 is not None and candidate_sha256 != content_sha256(candidate):
        raise ContractViolation("Teacher candidate hash mismatch")
    for name in v1_fields:
        candidate[name] = _require_text(f"candidate {name}", candidate.get(name))
    if candidate_schema_version == 2:
        for name in ("operator", "router"):
            value = candidate.get(name)
            if not isinstance(value, dict) or not value:
                raise ContractViolation(f"candidate {name} must be an object")
        memory_policy = candidate.get("memory_policy")
        if memory_policy is not None and (
            not isinstance(memory_policy, dict) or not memory_policy
        ):
            raise ContractViolation("candidate memory_policy must be an object or null")
        preconditions = candidate.get("preconditions")
        if not isinstance(preconditions, list) or not preconditions:
            raise ContractViolation("candidate preconditions are invalid")
        for name in (
            "expected_external_effect",
            "expected_internal_effect",
            "falsification",
        ):
            value = candidate.get(name)
            if value is None or value == "" or value == [] or value == {}:
                raise ContractViolation(f"candidate {name} must be non-empty")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ContractViolation("Teacher usage is invalid")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractViolation(f"Teacher usage {name} is invalid")
    network_calls = response.get("network_calls")
    if network_calls != 1:
        raise ContractViolation("Teacher network call count is invalid")
    pricing = response.get("pricing_cny_per_million")
    if (
        not isinstance(pricing, dict)
        or not pricing
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0
            or not math.isfinite(float(value))
            for value in pricing.values()
        )
    ):
        raise ContractViolation("Teacher pricing is invalid")
    return {
        "candidate": candidate,
        "candidate_schema_version": candidate_schema_version,
        "provider": provider,
        "model": model,
        "usage": usage,
        "network_calls": network_calls,
        "pricing": pricing,
        "cost_cny": _cost(response.get("estimated_cost_cny")),
    }


# Kept for callers that imported the former private parser while v1 receipts
# remain supported. New compilation always uses the schema-aware parser above.
_parse_v1_exchange = _parse_teacher_exchange


def _operator(value: object) -> tuple[str, str, tuple[str, ...], str]:
    if not isinstance(value, dict):
        raise ContractViolation("candidate Operator is invalid")
    operator_id = value.get("id", value.get("operator_id"))
    operator_id = _require_text("candidate Operator id", operator_id)
    _require_identifier("candidate Operator id", operator_id)
    kind = _require_text("candidate Operator kind", value.get("kind", "zero-arg"))
    arguments = value.get("arguments", [])
    if not isinstance(arguments, list) or any(
        not isinstance(argument, str) or not argument.strip() for argument in arguments
    ):
        raise ContractViolation("candidate Operator arguments are invalid")
    instruction = _require_text(
        "candidate Operator instruction", value.get("instruction")
    )
    return operator_id, kind, tuple(arguments), instruction


def _repair_routes(
    routes: tuple[tuple[str, str], ...],
    required_task_ids: Sequence[str],
    operator_id: str,
) -> tuple[tuple[str, str], ...]:
    """Deterministically extend a valid Router to cover required feedback tasks.

    The Teacher's paid response is never mutated: the repair is a pure function
    of the parsed routes, the authoritative selected-task list and the
    Candidate's own Operator. It is bound to the revision through the
    CompileSpec content hash and re-derived identically on load.
    """

    provided = {task_id for task_id, _ in routes}
    missing = tuple(task_id for task_id in required_task_ids if task_id not in provided)
    if not missing:
        return routes
    return routes + tuple((task_id, operator_id) for task_id in missing)


def _normalize_route_key(key: str, required_task_ids: Sequence[str]) -> str:
    """Map a Teacher Router key onto the authoritative selected instance_id.

    The Teacher contract demands exact instance_id keys, but weak models
    occasionally emit derived forms such as ``feedback-<id>@<short-commit>``
    or ``round1-<id>``.  When the key contains exactly one required task id as
    a substring, it is deterministically normalized here (a pure function,
    re-derived identically on load).  Unrecognizable or ambiguous keys stay
    fail-closed so a malformed Router is never silently accepted.
    """
    if key in required_task_ids:
        return key
    candidates = [task_id for task_id in required_task_ids if task_id in key]
    if len(candidates) == 1:
        return candidates[0]
    raise ContractViolation(
        "candidate Router task_id is not a safe immutable identifier"
    )


def _candidate_routes(
    value: object, operator_id: str, required_task_ids: Sequence[str] = ()
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or not value:
        raise ContractViolation("candidate Router is invalid")
    raw_routes = value["routes"] if set(value) == {"routes"} else value
    routes: list[tuple[str, str]] = []
    if isinstance(raw_routes, dict):
        rows: object = list(raw_routes.items())
    else:
        rows = raw_routes
    if not isinstance(rows, list) or not rows:
        raise ContractViolation("candidate Router routes are invalid")
    for row in rows:
        if isinstance(row, dict) and set(row) == {"task_id", "operator_id"}:
            task_id = row["task_id"]
            routed_operator = row["operator_id"]
        elif isinstance(row, (tuple, list)) and len(row) == 2:
            task_id, routed_operator = row
        else:
            raise ContractViolation("candidate Router route is invalid")
        task_id = _require_text("candidate Router task_id", task_id)
        if required_task_ids:
            task_id = _normalize_route_key(task_id, required_task_ids)
        else:
            _require_identifier("candidate Router task_id", task_id)
        if routed_operator != operator_id:
            raise ContractViolation("candidate Router references another Operator")
        routes.append((task_id, operator_id))
    if len({task_id for task_id, _ in routes}) != len(routes):
        raise ContractViolation("candidate Router task IDs must be unique")
    return tuple(routes)


def _validate_compiled_projection(
    change_set: CandidateChangeSet,
    skill: CompiledSkill,
    operator: CompiledOperator,
    router: CompiledRouter,
    *,
    memory_policy: CompiledMemoryPolicy | None = None,
) -> None:
    identity = (change_set.candidate_id, change_set.revision_id)
    if any(
        (record.candidate_id, record.revision_id) != identity
        for record in (skill, operator, router)
    ):
        raise ContractViolation("compiled candidate lineage identity mismatch")
    if (
        skill.parent_revision_id != change_set.parent_revision_id
        or skill.protocol != change_set.protocol
        or skill.prompt_template != change_set.prompt_template
        or skill.skill_text != change_set.skill_text
        or skill.preconditions != change_set.preconditions
        or skill.expected_external_effect != change_set.expected_external_effect
        or skill.expected_internal_effect != change_set.expected_internal_effect
        or skill.falsification != change_set.falsification
    ):
        raise ContractViolation("compiled Skill is not derived from the change set")
    if (
        operator.kind != change_set.operator_kind
        or operator.arguments != change_set.operator_arguments
        or operator.operator_id != change_set.operator_id
        or operator.instruction != change_set.operator_instruction
        or operator.parent_revision_id not in {"", change_set.parent_revision_id}
    ):
        raise ContractViolation("compiled Operator is not derived from the change set")
    if (
        router.routes != change_set.routes
        or router.parent_revision_id not in {"", change_set.parent_revision_id}
        or any(operator_id != operator.operator_id for _, operator_id in router.routes)
    ):
        raise ContractViolation("compiled Router is not derived from the change set")
    if (memory_policy is None) != (change_set.memory_policy is None):
        raise ContractViolation("compiled Memory Policy presence is invalid")
    if memory_policy is not None:
        if (
            (memory_policy.candidate_id, memory_policy.revision_id) != identity
            or memory_policy.parent_revision_id != change_set.parent_revision_id
            or memory_policy.policy != change_set.memory_policy
        ):
            raise ContractViolation(
                "compiled Memory Policy is not derived from the change set"
            )


def _validate_change_set_source(
    change_set: CandidateChangeSet,
    teacher: dict[str, Any],
    compile_spec: CompileSpec,
) -> None:
    candidate = teacher["candidate"]
    if (
        change_set.protocol != candidate["protocol"]
        or change_set.prompt_template != candidate["prompt_template"]
        or change_set.skill_text != candidate["skill_text"]
        or change_set.eval_note != candidate["eval_note"]
        or change_set.source_candidate_sha256 != content_sha256(candidate)
        or change_set.compile_spec_sha256 != compile_spec.content_sha256
    ):
        raise ContractViolation("compiled change set source projection mismatch")
    expected: tuple[object, ...]
    if teacher["candidate_schema_version"] == 1:
        expected = (
            compile_spec.operator_id,
            "zero-arg",
            (),
            compile_spec.operator_instruction,
            compile_spec.routes,
            None,
            (),
            None,
            None,
            None,
        )
    else:
        operator_id, kind, arguments, instruction = _operator(candidate["operator"])
        memory = candidate["memory_policy"]
        expected = (
            operator_id,
            kind,
            arguments,
            instruction,
            _repair_routes(
                _candidate_routes(candidate["router"], operator_id),
                compile_spec.required_route_task_ids,
                operator_id,
            ),
            None if memory is None else dict(memory),
            tuple(candidate["preconditions"]),
            candidate["expected_external_effect"],
            candidate["expected_internal_effect"],
            candidate["falsification"],
        )
    if teacher["candidate_schema_version"] == 2:
        provided_routes = _candidate_routes(candidate["router"], operator_id)
        expected_synthesized = tuple(
            task_id
            for task_id in compile_spec.required_route_task_ids
            if task_id not in {task_id for task_id, _ in provided_routes}
        )
        if change_set.synthesized_task_ids != expected_synthesized:
            raise ContractViolation("compiled Router repair projection mismatch")
    actual = (
        change_set.operator_id,
        change_set.operator_kind,
        change_set.operator_arguments,
        change_set.operator_instruction,
        change_set.routes,
        change_set.memory_policy,
        change_set.preconditions,
        change_set.expected_external_effect,
        change_set.expected_internal_effect,
        change_set.falsification,
    )
    if actual != expected:
        raise ContractViolation("compiled Harness source projection mismatch")


def _cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation("Teacher cost is invalid")
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ContractViolation("Teacher cost is invalid")
    return result


def _routes(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise ContractViolation("compiled routes are invalid")
    routes: list[tuple[str, str]] = []
    for row in value:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ContractViolation("compiled route entry is invalid")
        routes.append((str(row[0]), str(row[1])))
    return tuple(routes)


def _without_metadata(value: dict[str, Any], expected_kind: str) -> dict[str, Any]:
    if value.pop("schema_version", None) != 1:
        raise ContractViolation("compiled artifact schema is invalid")
    if value.pop("artifact_kind", None) != expected_kind:
        raise ContractViolation("compiled artifact kind is invalid")
    return value


def _artifact_bytes(root: Path, artifacts: list[tuple[str, str]], name: str) -> bytes:
    if name not in dict(artifacts):
        raise ContractViolation(f"compiled artifact manifest omits {name}")
    return (root / name).read_bytes()


def _artifact_json(
    root: Path, artifacts: list[tuple[str, str]], name: str
) -> dict[str, Any]:
    try:
        value = json.loads(_artifact_bytes(root, artifacts, name))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation(f"compiled artifact is invalid JSON: {name}") from error
    if not isinstance(value, dict):
        raise ContractViolation(f"compiled artifact must be an object: {name}")
    return value


def _encoded(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _compiled_encoded(kind: str, value: Any) -> bytes:
    return _encoded(
        {"schema_version": 1, "artifact_kind": kind, **dataclasses.asdict(value)}
    )


def _freeze(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        if path.read_bytes() != content:
            raise ContractViolation(
                f"immutable compiled artifact conflict: {path.name}"
            ) from error
        return
    try:
        written = os.write(descriptor, content)
        if written != len(content):
            raise ContractViolation(f"partial compiled artifact write: {path.name}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CandidateChangeSet",
    "CandidateCompiler",
    "CompileSpec",
    "CompiledOperator",
    "CompiledMemoryPolicy",
    "CompiledRevision",
    "CompiledRouter",
    "CompiledSkill",
]
