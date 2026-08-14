"""Immutable promotion decisions and their append-only replay log."""

from __future__ import annotations

import dataclasses
import hashlib
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

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


class PromotionDecisionLog:
    """Single-writer append log with idempotent exact replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")

    def append(self, decision: PromotionDecision) -> bool:
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
                legacy_payload = (
                    dict(payload) if "prediction_evidence_ids" not in payload else None
                )
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
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise DecisionLogError(
                    f"invalid promotion decision at line {line_number}"
                ) from error
            valid_hashes = {decision.content_sha256}
            if legacy_payload is not None:
                valid_hashes.add(content_sha256(legacy_payload))
            if (
                not isinstance(expected_sha256, str)
                or _SHA256.fullmatch(expected_sha256) is None
                or expected_sha256 not in valid_hashes
            ):
                raise DecisionLogError(
                    f"promotion decision hash mismatch at line {line_number}"
                )
            decisions.append(decision)
        if len({item.decision_id for item in decisions}) != len(decisions):
            raise DecisionLogError("duplicate promotion decision identity")
        return tuple(decisions)


def decision_identity(payload: object) -> str:
    return "decision-" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()
