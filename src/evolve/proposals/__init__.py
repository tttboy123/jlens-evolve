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
    ) -> None:
        self.root = root.resolve()
        self.provider = provider
        self.model = model
        self.transport = transport
        self.pricing = pricing
        self.hard_budget_cny = hard_budget_cny
        self._spent_cny = 0.0

    def propose(
        self,
        *,
        request_id: str,
        failure_package: dict[str, object],
        max_output_tokens: int,
    ) -> ProposalResult:
        request_path = self.root / request_id / "TEACHER-REQUEST.json"
        response_path = self.root / request_id / "TEACHER-RESPONSE.json"
        if response_path.is_file():
            return self._load_result(request_path, response_path)
        serialized = canonical_json(failure_package)
        lowered = serialized.casefold()
        if any(term in lowered for term in ('"gold"', "reference_patch", "holdout")):
            raise ContractViolation(
                "teacher package contains prohibited leakage fields"
            )
        estimated_input = max(1, (len(serialized) + 3) // 4)
        reservation = (
            estimated_input * self.pricing.input
            + max_output_tokens * self.pricing.output
        ) / 1_000_000
        if self._spent_cny + reservation > self.hard_budget_cny:
            raise ContractViolation("teacher budget reservation exceeded")
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
        _atomic_write(request_path, request)
        raw = self.transport(request)
        try:
            content = json.loads(raw["choices"][0]["message"]["content"])
            usage = raw["usage"]
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
        if self._spent_cny + cost > self.hard_budget_cny:
            raise ContractViolation("teacher actual cost exceeded hard budget")
        self._spent_cny += cost
        candidate_id = (
            "candidate-"
            + hashlib.sha256(canonical_json(content).encode()).hexdigest()[:16]
        )
        payload = {
            "schema_version": 1,
            "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "provider": self.provider,
            "model": str(raw.get("model", self.model)),
            "candidate": {**content, "candidate_id": candidate_id, "active": False},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_cny": cost,
            },
            "network_call_count": 1,
        }
        _atomic_write(response_path, payload)
        return self._load_result(request_path, response_path)

    def _load_result(self, request_path: Path, response_path: Path) -> ProposalResult:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        if (
            payload["request_sha256"]
            != hashlib.sha256(request_path.read_bytes()).hexdigest()
        ):
            raise ContractViolation("frozen teacher request hash mismatch")
        candidate = ProposalCandidate(**payload["candidate"])
        usage = ProposalUsage(
            provider=payload["provider"],
            model=payload["model"],
            **payload["usage"],
        )
        return ProposalResult(candidate, usage, request_path, response_path)


__all__ = [
    "CandidateProposer",
    "PricingCnyPerMillionTokens",
    "ProposalCandidate",
    "ProposalResult",
    "ProposalUsage",
]
