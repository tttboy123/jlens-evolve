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
from typing import Any

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
    """Human-authored, immutable bridge from a v1 candidate to executable form."""

    candidate_id: str
    revision_id: str
    parent_revision_id: str
    cohort: Cohort
    operator_id: str
    operator_instruction: str
    routes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "revision_id",
            "parent_revision_id",
            "operator_id",
        ):
            value = _require_text(name, getattr(self, name))
            _require_identifier(name, value)
        _require_text("operator_instruction", self.operator_instruction)
        if self.cohort is not Cohort.FEEDBACK:
            raise ContractViolation("candidate compilation is feedback-only")
        if not self.routes:
            raise ContractViolation("candidate router must contain feedback routes")
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
        for name in ("source_candidate_sha256", "compile_spec_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
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


@dataclass(frozen=True, slots=True)
class CompiledOperator:
    candidate_id: str
    revision_id: str
    operator_id: str
    kind: str
    arguments: tuple[str, ...]
    instruction: str


@dataclass(frozen=True, slots=True)
class CompiledRouter:
    candidate_id: str
    revision_id: str
    routes: tuple[tuple[str, str], ...]


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
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ContractViolation("compiled revision manifest is invalid")
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
        teacher = _parse_v1_exchange(request, response, request_bytes)
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
            operator_id=compile_spec_payload["operator_id"],
            operator_instruction=compile_spec_payload["operator_instruction"],
            routes=_routes(compile_spec_payload["routes"]),
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
            }
        )
        if manifest.get("revision_sha256") != change_set.content_sha256:
            raise ContractViolation("compiled candidate revision identity mismatch")
        skill = CompiledSkill(
            **_without_metadata(
                _artifact_json(bundle_root, artifact_hashes, "COMPILED-SKILL.json"),
                "skill",
            )
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
        _validate_compiled_projection(change_set, skill, operator, router)
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
        )


class CandidateCompiler:
    """Compile only verified v1 Teacher receipts; never fall back to old assets."""

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
        parsed = _parse_v1_exchange(request, response, request_bytes)
        candidate = parsed["candidate"]
        change_set = CandidateChangeSet(
            candidate_id=compile_spec.candidate_id,
            revision_id=compile_spec.revision_id,
            parent_revision_id=compile_spec.parent_revision_id,
            source_candidate_sha256=content_sha256(candidate),
            compile_spec_sha256=compile_spec.content_sha256,
            protocol=candidate["protocol"],
            prompt_template=candidate["prompt_template"],
            skill_text=candidate["skill_text"],
            eval_note=candidate["eval_note"],
            operator_id=compile_spec.operator_id,
            operator_instruction=compile_spec.operator_instruction,
            routes=compile_spec.routes,
        )
        skill = CompiledSkill(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            parent_revision_id=change_set.parent_revision_id,
            protocol=change_set.protocol,
            prompt_template=change_set.prompt_template,
            skill_text=change_set.skill_text,
        )
        operator = CompiledOperator(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            operator_id=change_set.operator_id,
            kind="zero-arg",
            arguments=(),
            instruction=change_set.operator_instruction,
        )
        router = CompiledRouter(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            routes=change_set.routes,
        )
        _validate_compiled_projection(change_set, skill, operator, router)

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
        artifacts = [
            {"path": name, "sha256": _sha256(content)}
            for name, content in payloads.items()
        ]
        manifest = {
            "schema_version": 1,
            "candidate_id": change_set.candidate_id,
            "revision_id": change_set.revision_id,
            "parent_revision_id": change_set.parent_revision_id,
            "revision_sha256": change_set.content_sha256,
            "compile_spec_sha256": compile_spec.content_sha256,
            "request_sha256": _sha256(request_bytes),
            "response_sha256": _sha256(response_bytes),
            "provider": parsed["provider"],
            "model": parsed["model"],
            "cost_cny": parsed["cost_cny"],
            "artifacts": artifacts,
        }
        for name, content in payloads.items():
            _freeze(root / name, content)
        _freeze(root / "COMPILED-REVISION.json", _encoded(manifest))
        return CompiledRevision.load(root)


def _parse_v1_exchange(
    request: object, response: object, request_bytes: bytes
) -> dict[str, Any]:
    if not isinstance(request, dict) or not isinstance(response, dict):
        raise ContractViolation("Teacher request/response must be JSON objects")
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
    required_candidate = {"protocol", "prompt_template", "skill_text", "eval_note"}
    if not isinstance(candidate, dict) or set(candidate) != required_candidate:
        raise ContractViolation("Teacher candidate fields are invalid")
    candidate = {
        name: _require_text(f"candidate {name}", candidate.get(name))
        for name in sorted(required_candidate)
    }
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
        "provider": provider,
        "model": model,
        "usage": usage,
        "network_calls": network_calls,
        "pricing": pricing,
        "cost_cny": _cost(response.get("estimated_cost_cny")),
    }


def _validate_compiled_projection(
    change_set: CandidateChangeSet,
    skill: CompiledSkill,
    operator: CompiledOperator,
    router: CompiledRouter,
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
    ):
        raise ContractViolation("compiled Skill is not derived from the change set")
    if (
        operator.kind != "zero-arg"
        or operator.arguments
        or operator.operator_id != change_set.operator_id
        or operator.instruction != change_set.operator_instruction
    ):
        raise ContractViolation("compiled Operator is not a zero-argument projection")
    if router.routes != change_set.routes or any(
        operator_id != operator.operator_id for _, operator_id in router.routes
    ):
        raise ContractViolation("compiled Router is not derived from the change set")


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
    "CompiledRevision",
    "CompiledRouter",
    "CompiledSkill",
]
