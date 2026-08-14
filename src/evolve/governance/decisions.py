"""Immutable promotion decisions and their append-only replay log."""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from evolve.contracts import (
    ClaimGrade,
    ContractViolation,
    canonical_json,
    content_sha256,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DecisionLogError(ContractViolation):
    """Promotion decision evidence is corrupt, conflicting, or locked."""


class DecisionLogBusy(DecisionLogError):
    """Another process owns the promotion decision writer lease."""


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    decision_id: str
    candidate_id: str
    candidate_revision_id: str
    gate_decision: object
    evidence_grade: ClaimGrade
    claim_ids: tuple[str, ...]
    prediction_evidence_ids: tuple[str, ...]
    human_approval: bool
    decided_at: str
    rationale: str
    evidence_state_sha256: str | None = None
    authority_key_id: str | None = None
    authority_signature_hmac_sha256: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.decision_id,
            self.candidate_id,
            self.candidate_revision_id,
            self.decided_at,
            self.rationale,
        )
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ContractViolation("promotion decision identities must be non-empty")
        if not self.claim_ids:
            raise ContractViolation("promotion decision must reference claims")
        for name, value in (
            ("evidence_state_sha256", self.evidence_state_sha256),
            ("authority_signature_hmac_sha256", self.authority_signature_hmac_sha256),
        ):
            if value is not None and _SHA256.fullmatch(value) is None:
                raise ContractViolation(f"{name} must be a literal SHA-256")
        if self.authority_key_id is not None and not self.authority_key_id.strip():
            raise ContractViolation("authority_key_id must be non-empty")

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


class PromotionDecisionLog:
    """Single-writer append log with idempotent exact replay."""

    def __init__(
        self,
        path: str | Path,
        *,
        authority: GovernanceDecisionAuthority | None = None,
    ) -> None:
        self.path = Path(path)
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")
        self._authority = authority

    def append(self, decision: PromotionDecision) -> bool:
        self._verify_authoritative_decision(decision)
        existing = {item.decision_id: item for item in self.all()}.get(
            decision.decision_id
        )
        if existing is not None:
            if existing == decision:
                return False
            raise DecisionLogError("conflicting immutable promotion decision")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lease_fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise DecisionLogBusy("promotion decision writer lease is held") from error
        try:
            os.write(lease_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(lease_fd)
            existing = {item.decision_id: item for item in self.all()}.get(
                decision.decision_id
            )
            if existing is not None:
                if existing == decision:
                    return False
                raise DecisionLogError("conflicting immutable promotion decision")
            payload = dataclasses.asdict(decision)
            payload["gate_decision"] = str(decision.gate_decision)
            payload["record_sha256"] = decision.content_sha256
            line = (canonical_json(payload) + "\n").encode()
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
            try:
                os.write(fd, line)
                os.fsync(fd)
            finally:
                os.close(fd)
            return True
        finally:
            os.close(lease_fd)
            self.lease_path.unlink(missing_ok=True)

    def all(self) -> tuple[PromotionDecision, ...]:
        if not self.path.exists():
            return ()
        from . import GateDecision

        decisions = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            try:
                payload = json.loads(line)
                expected_sha256 = payload.pop("record_sha256")
                original_payload = dict(payload)
                decision = PromotionDecision(
                    decision_id=payload["decision_id"],
                    candidate_id=payload["candidate_id"],
                    candidate_revision_id=payload["candidate_revision_id"],
                    gate_decision=GateDecision(payload["gate_decision"]),
                    evidence_grade=ClaimGrade(payload["evidence_grade"]),
                    claim_ids=tuple(payload["claim_ids"]),
                    prediction_evidence_ids=tuple(
                        payload.get("prediction_evidence_ids", ())
                    ),
                    human_approval=payload["human_approval"],
                    decided_at=payload["decided_at"],
                    rationale=payload["rationale"],
                    evidence_state_sha256=payload.get("evidence_state_sha256"),
                    authority_key_id=payload.get("authority_key_id"),
                    authority_signature_hmac_sha256=payload.get(
                        "authority_signature_hmac_sha256"
                    ),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise DecisionLogError(
                    f"invalid promotion decision at line {line_number}"
                ) from error
            valid_hashes = {decision.content_sha256}
            if any(
                name not in original_payload
                for name in (
                    "prediction_evidence_ids",
                    "evidence_state_sha256",
                    "authority_key_id",
                    "authority_signature_hmac_sha256",
                )
            ):
                valid_hashes.add(content_sha256(original_payload))
            if (
                not isinstance(expected_sha256, str)
                or _SHA256.fullmatch(expected_sha256) is None
                or expected_sha256 not in valid_hashes
            ):
                raise DecisionLogError(
                    f"promotion decision hash mismatch at line {line_number}"
                )
            self._verify_authoritative_decision(decision)
            decisions.append(decision)
        if len({item.decision_id for item in decisions}) != len(decisions):
            raise DecisionLogError("duplicate promotion decision identity")
        return tuple(decisions)

    def verified_approved(self, decision_id: str) -> PromotionDecision:
        matches = [item for item in self.all() if item.decision_id == decision_id]
        if len(matches) != 1:
            raise DecisionLogError("approved promotion decision is missing or ambiguous")
        decision = matches[0]
        if str(decision.gate_decision) != "approved":
            raise DecisionLogError("promotion decision is not approved")
        self._verify_authoritative_decision(decision)
        return decision

    def _verify_authoritative_decision(self, decision: PromotionDecision) -> None:
        if str(decision.gate_decision) != "approved":
            return
        if (
            self._authority is None
            or decision.evidence_state_sha256 is None
            or not self._authority.verify(decision)
        ):
            raise DecisionLogError(
                "approved promotion decision lacks valid governance authority"
            )


@dataclass(frozen=True, slots=True)
class GovernanceDecisionAuthority:
    """Process-local signing authority for capability-eligible decisions."""

    key_id: str
    secret_key: bytes = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise ContractViolation("governance authority key_id must be non-empty")
        if not isinstance(self.secret_key, bytes) or len(self.secret_key) < 32:
            raise ContractViolation(
                "governance authority secret_key must contain at least 32 bytes"
            )

    def sign(self, decision: PromotionDecision) -> PromotionDecision:
        unsigned = dataclasses.replace(
            decision,
            authority_key_id=self.key_id,
            authority_signature_hmac_sha256=None,
        )
        signature = hmac.new(
            self.secret_key,
            canonical_json(_decision_signing_payload(unsigned)).encode(),
            hashlib.sha256,
        ).hexdigest()
        return dataclasses.replace(
            unsigned,
            authority_signature_hmac_sha256=signature,
        )

    def verify(self, decision: PromotionDecision) -> bool:
        signature = decision.authority_signature_hmac_sha256
        if decision.authority_key_id != self.key_id or signature is None:
            return False
        unsigned = dataclasses.replace(
            decision,
            authority_signature_hmac_sha256=None,
        )
        expected = hmac.new(
            self.secret_key,
            canonical_json(_decision_signing_payload(unsigned)).encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)


def _decision_signing_payload(decision: PromotionDecision) -> dict[str, object]:
    payload = dataclasses.asdict(decision)
    payload["gate_decision"] = str(decision.gate_decision)
    payload.pop("authority_signature_hmac_sha256", None)
    return payload


def decision_identity(payload: object) -> str:
    return "decision-" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
