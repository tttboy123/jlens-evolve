"""Budgeted Teacher proposal boundary; Teacher creates inactive candidates only."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evolve.contracts import ContractViolation, canonical_json
from evolve.kernel import DurableCostLedger
from evolve.teachers import TeacherTransport

from .candidate_chain import (
    CandidateChangeSet,
    CandidateCompiler,
    CompiledMemoryPolicy,
    CompiledOperator,
    CompiledRevision,
    CompiledRouter,
    CompiledSkill,
    CompileSpec,
)


@dataclass(frozen=True, slots=True)
class PricingCnyPerMillionTokens:
    input: float
    output: float


@dataclass(frozen=True, slots=True)
class ProposalCandidate:
    candidate_id: str
    protocol: str
    prompt_template: str
    skill_text: str
    eval_note: str
    active: bool = False
    operator: Mapping[str, object] | None = None
    router: Mapping[str, object] | None = None
    memory_policy: Mapping[str, object] | None = None
    preconditions: tuple[object, ...] = ()
    expected_external_effect: object | None = None
    expected_internal_effect: object | None = None
    falsification: object | None = None


@dataclass(frozen=True, slots=True)
class ProposalUsage:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_cny: float


@dataclass(frozen=True, slots=True)
class ProposalResult:
    candidate: ProposalCandidate
    usage: ProposalUsage
    request_path: Path
    response_path: Path


Transport = TeacherTransport

_V1_CANDIDATE_FIELDS = {
    "protocol",
    "prompt_template",
    "skill_text",
    "eval_note",
}
_V2_CANDIDATE_FIELDS = _V1_CANDIDATE_FIELDS | {
    "operator",
    "router",
    "memory_policy",
    "preconditions",
    "expected_external_effect",
    "expected_internal_effect",
    "falsification",
}


def _validate_teacher_candidate(content: object) -> tuple[dict[str, object], int]:
    if not isinstance(content, dict):
        raise ContractViolation("teacher candidate fields are invalid")
    fields = set(content)
    if fields == _V1_CANDIDATE_FIELDS:
        schema_version = 1
    elif fields == _V2_CANDIDATE_FIELDS:
        schema_version = 2
    else:
        raise ContractViolation("teacher candidate fields are invalid")
    for name in _V1_CANDIDATE_FIELDS:
        value = content.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ContractViolation(f"teacher candidate {name} must be non-empty text")
    if schema_version == 2:
        for name in ("operator", "router"):
            value = content.get(name)
            if not isinstance(value, Mapping) or not value:
                raise ContractViolation(f"teacher candidate {name} must be an object")
        memory_policy = content.get("memory_policy")
        if memory_policy is not None and (
            not isinstance(memory_policy, Mapping) or not memory_policy
        ):
            raise ContractViolation(
                "teacher candidate memory_policy must be an object or null"
            )
        preconditions = content.get("preconditions")
        if not isinstance(preconditions, list) or not preconditions:
            raise ContractViolation("teacher candidate preconditions must be a list")
        for name in (
            "expected_external_effect",
            "expected_internal_effect",
            "falsification",
        ):
            value = content.get(name)
            if value is None or value == "" or value == [] or value == {}:
                raise ContractViolation(f"teacher candidate {name} must be non-empty")
    return dict(content), schema_version


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _freeze_request(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json(payload) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ContractViolation("frozen teacher request identity mismatch")
        return
    _atomic_write(path, payload)


def _contains_prohibited_teacher_data(value: object) -> bool:
    """Reject protected fields and path leaks without scanning safe goal prose."""

    protected = ("gold", "reference_patch", "holdout")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if any(term in str(key).casefold() for term in protected):
                return True
            if _contains_prohibited_teacher_data(nested):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_prohibited_teacher_data(item) for item in value)
    if isinstance(value, str):
        segments = {
            segment
            for segment in value.casefold().replace("\\", "/").split("/")
            if segment
        }
        return bool(set(protected) & segments)
    return False


class CandidateProposer:
    """Reserve cost, dispatch once, and freeze request/response/usage."""

    def __init__(
        self,
        *,
        root: Path,
        provider: str,
        model: str,
        transport: Transport,
        pricing: PricingCnyPerMillionTokens,
        hard_budget_cny: float,
        cost_ledger: DurableCostLedger | None = None,
    ) -> None:
        self.root = root.resolve()
        self.provider = provider
        self.model = model
        self.transport = transport
        self.pricing = pricing
        self.hard_budget_cny = hard_budget_cny
        self.cost_ledger = cost_ledger or DurableCostLedger(
            self.root / "COST-LEDGER.jsonl",
            campaign_id=f"teacher:{provider}:{model}",
            max_cost_cny=hard_budget_cny,
            max_model_calls=1_000_000,
        )
        if self.cost_ledger.snapshot().max_cost_cny != hard_budget_cny:
            raise ContractViolation("Teacher ledger budget identity mismatch")

    def propose(
        self,
        *,
        request_id: str,
        failure_package: dict[str, object],
        max_output_tokens: int,
    ) -> ProposalResult:
        if _contains_prohibited_teacher_data(failure_package):
            raise ContractViolation(
                "teacher package contains prohibited leakage fields"
            )
        request = {
            "request_id": request_id,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": max_output_tokens,
            "failure_package": failure_package,
            "constraints": [
                "feedback-only",
                "candidate inactive",
                "do not modify evaluator or model weights",
                "router must cover every selected feedback task",
            ],
        }
        # A byte-level bound plus a fixed envelope allowance is deliberately
        # conservative for tokenizer/provider framing. It is an authorization
        # reservation, not a prediction of the eventual bill.
        estimated_input = max(1, len(canonical_json(request).encode("utf-8")) + 4096)
        reservation = (
            estimated_input * self.pricing.input
            + max_output_tokens * self.pricing.output
        ) / 1_000_000
        reservation_id = f"teacher-reservation:{request_id}"
        result_id = f"teacher-result:{request_id}"
        request_path = self.root / request_id / "TEACHER-REQUEST.json"
        response_path = self.root / request_id / "TEACHER-RESPONSE.json"
        raw_response_path = (
            self.root / request_id / "TEACHER-RAW-RESPONSE.json"
        )
        if response_path.is_file():
            expected_request = canonical_json(request) + "\n"
            try:
                frozen_request = request_path.read_text(encoding="utf-8")
            except OSError as error:
                raise ContractViolation("frozen teacher request is missing") from error
            if frozen_request != expected_request:
                raise ContractViolation("replayed teacher request identity mismatch")
            result = self._load_result(request_path, response_path)
            self.cost_ledger.record(
                reservation_id,
                result_id=result_id,
                actual_cost_cny=result.usage.estimated_cost_cny,
                actual_model_calls=1,
            )
            return result
        _freeze_request(request_path, request)
        if raw_response_path.is_file():
            try:
                raw = json.loads(raw_response_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ContractViolation(
                    "frozen raw Teacher response is unreadable"
                ) from error
            if not isinstance(raw, dict):
                raise ContractViolation(
                    "frozen raw Teacher response must be an object"
                )
        else:
            dispatched, raw = self.cost_ledger.dispatch_once(
                reservation_id,
                cost_cny=reservation,
                model_calls=1,
                dispatch=lambda: self.transport(request),
            )
            if not dispatched:
                raise ContractViolation(
                    "paid Teacher reservation has no response; manual reconcile required"
                )
            if raw is None:
                raise ContractViolation("paid Teacher dispatch returned no response")
            _atomic_write(raw_response_path, raw)
        try:
            choices = raw["choices"]
            usage = raw["usage"]
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], Mapping)
                or not isinstance(choices[0].get("message"), Mapping)
                or not isinstance(usage, Mapping)
            ):
                raise TypeError("Teacher response shape is invalid")
            message = choices[0]["message"]
            assert isinstance(message, Mapping)
            content_text = message.get("content")
            if not isinstance(content_text, str):
                raise TypeError("Teacher response content is invalid")
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            self.cost_ledger.record(
                reservation_id,
                result_id=result_id,
                actual_cost_cny=reservation,
                actual_model_calls=1,
            )
            raise ContractViolation("teacher response contract is invalid") from exc
        cost = round(
            (input_tokens * self.pricing.input + output_tokens * self.pricing.output)
            / 1_000_000,
            8,
        )
        self.cost_ledger.record(
            reservation_id,
            result_id=result_id,
            actual_cost_cny=cost,
            actual_model_calls=1,
        )
        try:
            content = json.loads(content_text)
        except json.JSONDecodeError as exc:
            raise ContractViolation("teacher response contract is invalid") from exc
        content, schema_version = _validate_teacher_candidate(content)
        payload = {
            "schema_version": schema_version,
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "provider": self.provider,
            "model": self.model,
            "provider_response_model": str(raw.get("model", self.model)),
            "candidate": content,
            "candidate_status": "inactive",
            "auto_activate": False,
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "network_calls": 1,
            "pricing_cny_per_million": {
                "input": self.pricing.input,
                "output": self.pricing.output,
            },
            "estimated_cost_cny": cost,
            "raw_response_sha256": hashlib.sha256(
                raw_response_path.read_bytes()
            ).hexdigest(),
        }
        payload["candidate_sha256"] = hashlib.sha256(
            canonical_json(content).encode("utf-8")
        ).hexdigest()
        payload["receipt_sha256"] = hashlib.sha256(
            canonical_json(payload).encode("utf-8")
        ).hexdigest()
        _atomic_write(response_path, payload)
        result = self._load_result(request_path, response_path)
        return result

    def _load_result(self, request_path: Path, response_path: Path) -> ProposalResult:
        try:
            payload = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractViolation("frozen teacher response is unreadable") from error
        if not isinstance(payload, dict):
            raise ContractViolation("frozen teacher response must be an object")
        if (
            "candidate_status" in payload
            and payload.get("candidate_status") != "inactive"
        ) or ("auto_activate" in payload and payload.get("auto_activate") is not False):
            raise ContractViolation("Teacher candidate must remain inactive")
        receipt_sha256 = payload.get("receipt_sha256")
        if receipt_sha256 is not None:
            unsigned = {
                key: value for key, value in payload.items() if key != "receipt_sha256"
            }
            if receipt_sha256 != hashlib.sha256(
                canonical_json(unsigned).encode("utf-8")
            ).hexdigest():
                raise ContractViolation("frozen teacher response hash mismatch")
        if (
            payload.get("request_sha256")
            != hashlib.sha256(request_path.read_bytes()).hexdigest()
        ):
            raise ContractViolation("frozen teacher request hash mismatch")
        raw_response_sha256 = payload.get("raw_response_sha256")
        if raw_response_sha256 is not None:
            raw_response_path = response_path.with_name(
                "TEACHER-RAW-RESPONSE.json"
            )
            if (
                not raw_response_path.is_file()
                or raw_response_sha256
                != hashlib.sha256(raw_response_path.read_bytes()).hexdigest()
            ):
                raise ContractViolation("frozen raw teacher response hash mismatch")
        try:
            candidate_payload = dict(payload["candidate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ContractViolation("frozen teacher candidate is invalid") from error
        if {"candidate_id", "active"} <= candidate_payload.keys():
            try:
                candidate = ProposalCandidate(**candidate_payload)
                usage_payload = payload["usage"]
                usage = ProposalUsage(
                    provider=payload["provider"],
                    model=payload["model"],
                    input_tokens=usage_payload["input_tokens"],
                    output_tokens=usage_payload["output_tokens"],
                    estimated_cost_cny=usage_payload["estimated_cost_cny"],
                )
            except (KeyError, TypeError) as error:
                raise ContractViolation("frozen v1 candidate is invalid") from error
        else:
            candidate_payload, _schema_version = _validate_teacher_candidate(
                candidate_payload
            )
            candidate_sha256 = payload.get("candidate_sha256")
            if candidate_sha256 is not None and candidate_sha256 != hashlib.sha256(
                canonical_json(candidate_payload).encode("utf-8")
            ).hexdigest():
                raise ContractViolation("frozen teacher candidate hash mismatch")
            candidate_id = (
                "candidate-"
                + hashlib.sha256(
                    canonical_json(candidate_payload).encode()
                ).hexdigest()[:16]
            )
            operator = candidate_payload.get("operator")
            router = candidate_payload.get("router")
            memory_policy = candidate_payload.get("memory_policy")
            preconditions = candidate_payload.get("preconditions", ())
            assert isinstance(candidate_payload["protocol"], str)
            assert isinstance(candidate_payload["prompt_template"], str)
            assert isinstance(candidate_payload["skill_text"], str)
            assert isinstance(candidate_payload["eval_note"], str)
            assert isinstance(preconditions, (tuple, list))
            candidate = ProposalCandidate(
                candidate_id=candidate_id,
                protocol=candidate_payload["protocol"],
                prompt_template=candidate_payload["prompt_template"],
                skill_text=candidate_payload["skill_text"],
                eval_note=candidate_payload["eval_note"],
                active=False,
                operator=dict(operator) if isinstance(operator, Mapping) else None,
                router=dict(router) if isinstance(router, Mapping) else None,
                memory_policy=(
                    dict(memory_policy)
                    if isinstance(memory_policy, Mapping)
                    else None
                ),
                preconditions=tuple(preconditions),
                expected_external_effect=candidate_payload.get(
                    "expected_external_effect"
                ),
                expected_internal_effect=candidate_payload.get(
                    "expected_internal_effect"
                ),
                falsification=candidate_payload.get("falsification"),
            )
            try:
                usage_payload = payload["usage"]
                usage = ProposalUsage(
                    provider=payload["provider"],
                    model=payload["model"],
                    input_tokens=usage_payload["prompt_tokens"],
                    output_tokens=usage_payload["completion_tokens"],
                    estimated_cost_cny=payload["estimated_cost_cny"],
                )
            except (KeyError, TypeError) as error:
                raise ContractViolation("frozen teacher usage is invalid") from error
        if candidate.active:
            raise ContractViolation("Teacher candidate must remain inactive")
        return ProposalResult(candidate, usage, request_path, response_path)


__all__ = [
    "CandidateChangeSet",
    "CandidateCompiler",
    "CandidateProposer",
    "CompileSpec",
    "CompiledOperator",
    "CompiledMemoryPolicy",
    "CompiledRevision",
    "CompiledRouter",
    "CompiledSkill",
    "PricingCnyPerMillionTokens",
    "ProposalCandidate",
    "ProposalResult",
    "ProposalUsage",
    "TeacherTransport",
    "Transport",
]
