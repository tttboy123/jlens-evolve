"""Budgeted Teacher proposal boundary; Teacher creates inactive candidates only."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from evolve.contracts import ContractViolation, canonical_json
from evolve.kernel import DurableCostLedger

from .candidate_chain import (
    CandidateChangeSet,
    CandidateCompiler,
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


Transport = Callable[[dict[str, object]], dict[str, object]]


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
        serialized = canonical_json(failure_package)
        lowered = serialized.casefold()
        if any(term in lowered for term in ('"gold"', "reference_patch", "holdout")):
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
            content = json.loads(content_text)
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise ContractViolation("teacher response contract is invalid") from exc
        fields = {"protocol", "prompt_template", "skill_text", "eval_note"}
        if not isinstance(content, dict) or set(content) != fields:
            raise ContractViolation("teacher candidate fields are invalid")
        cost = round(
            (input_tokens * self.pricing.input + output_tokens * self.pricing.output)
            / 1_000_000,
            8,
        )
        payload = {
            "schema_version": 1,
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
        }
        _atomic_write(response_path, payload)
        result = self._load_result(request_path, response_path)
        self.cost_ledger.record(
            reservation_id,
            result_id=result_id,
            actual_cost_cny=result.usage.estimated_cost_cny,
            actual_model_calls=1,
        )
        return result

    def _load_result(self, request_path: Path, response_path: Path) -> ProposalResult:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        if (
            payload["request_sha256"]
            != hashlib.sha256(request_path.read_bytes()).hexdigest()
        ):
            raise ContractViolation("frozen teacher request hash mismatch")
        candidate_payload = dict(payload["candidate"])
        if {"candidate_id", "active"} <= candidate_payload.keys():
            candidate = ProposalCandidate(**candidate_payload)
            usage_payload = payload["usage"]
            usage = ProposalUsage(
                provider=payload["provider"],
                model=payload["model"],
                input_tokens=usage_payload["input_tokens"],
                output_tokens=usage_payload["output_tokens"],
                estimated_cost_cny=usage_payload["estimated_cost_cny"],
            )
        else:
            candidate_id = (
                "candidate-"
                + hashlib.sha256(
                    canonical_json(candidate_payload).encode()
                ).hexdigest()[:16]
            )
            candidate = ProposalCandidate(
                candidate_id=candidate_id,
                **candidate_payload,
                active=False,
            )
            usage_payload = payload["usage"]
            usage = ProposalUsage(
                provider=payload["provider"],
                model=payload["model"],
                input_tokens=usage_payload["prompt_tokens"],
                output_tokens=usage_payload["completion_tokens"],
                estimated_cost_cny=payload["estimated_cost_cny"],
            )
        return ProposalResult(candidate, usage, request_path, response_path)


__all__ = [
    "CandidateChangeSet",
    "CandidateCompiler",
    "CandidateProposer",
    "CompileSpec",
    "CompiledOperator",
    "CompiledRevision",
    "CompiledRouter",
    "CompiledSkill",
    "PricingCnyPerMillionTokens",
    "ProposalCandidate",
    "ProposalResult",
    "ProposalUsage",
]
