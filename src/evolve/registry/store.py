"""Append-only JSONL registries with an exclusive single-writer lease."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from evolve.contracts import (
    ClaimGrade,
    ContractViolation,
    canonical_json,
    content_sha256,
)

from .records import (
    AgentProgramRecord,
    CandidateRecord,
    CapabilityRecord,
    RegistryViolation,
    RejectedRecord,
)

RecordT = TypeVar(
    "RecordT", CandidateRecord, CapabilityRecord, RejectedRecord, AgentProgramRecord
)


class RegistryConflict(RegistryViolation):
    """The same immutable revision identity was supplied with other content."""


class RegistryBusy(RegistryViolation):
    """Another writer currently owns the registry lease."""


class DecisionReader(Protocol):
    def all(self) -> tuple[Any, ...]: ...


class _AppendOnlyRegistry(Generic[RecordT]):
    record_type: type[RecordT]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lease_path = self.path.with_suffix(self.path.suffix + ".writer.lock")

    def append(self, record: RecordT) -> bool:
        if not isinstance(record, self.record_type):
            raise RegistryViolation(f"expected {self.record_type.__name__}")
        existing = self._by_key().get(record.key)
        if existing is not None:
            if existing == record:
                return False
            raise RegistryConflict(f"conflicting immutable revision {record.key!r}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lease_fd = os.open(self.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RegistryBusy(f"writer lease already held for {self.path}") from error
        try:
            os.write(lease_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(lease_fd)
            # Recheck after acquiring the lease so concurrent callers cannot append
            # two values with the same revision identity.
            existing = self._by_key().get(record.key)
            if existing is not None:
                if existing == record:
                    return False
                raise RegistryConflict(f"conflicting immutable revision {record.key!r}")
            payload = dataclasses.asdict(record)
            payload["record_sha256"] = record.content_sha256
            line = (canonical_json(payload) + "\n").encode("utf-8")
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

    def get(self, asset_id: str, revision_id: str) -> RecordT | None:
        return self._by_key().get((asset_id, revision_id))

    def all(self) -> tuple[RecordT, ...]:
        return tuple(self._read())

    def _by_key(self) -> dict[tuple[str, str], RecordT]:
        return {record.key: record for record in self._read()}

    def _read(self) -> list[RecordT]:
        if not self.path.exists():
            return []
        records: list[RecordT] = []
        for line_number, line in enumerate(self.path.read_text().splitlines(), 1):
            try:
                payload = json.loads(line)
                expected_hash = payload.pop("record_sha256")
                legacy_capability = (
                    self.record_type is CapabilityRecord
                    and "promotion_decision_id" not in payload
                )
                legacy_payload = dict(payload) if legacy_capability else None
                for field in dataclasses.fields(self.record_type):
                    if field.name in payload and field.name.endswith("_ids"):
                        payload[field.name] = tuple(payload[field.name])
                record = self.record_type(**payload)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise RegistryViolation(
                    f"invalid registry entry at {self.path}:{line_number}"
                ) from error
            valid_hashes = {record.content_sha256}
            if legacy_payload is not None:
                valid_hashes.add(content_sha256(legacy_payload))
            if expected_hash not in valid_hashes:
                raise RegistryViolation(
                    f"registry hash mismatch at {self.path}:{line_number}"
                )
            records.append(record)
        return records


class CandidateRegistry(_AppendOnlyRegistry[CandidateRecord]):
    record_type = CandidateRecord


class CapabilityRegistry(_AppendOnlyRegistry[CapabilityRecord]):
    record_type = CapabilityRecord

    def __init__(
        self, path: str | Path, *, decision_log: DecisionReader | None = None
    ) -> None:
        super().__init__(path)
        self._decision_log = decision_log

    def append(self, record: CapabilityRecord) -> bool:
        if record.promotion_decision_id is None or record.source_candidate_id is None:
            raise RegistryViolation(
                "authoritative capability writes require promotion decision identity"
            )
        decision = _approved_decision_for(
            self._decision_log, record.promotion_decision_id
        )
        if (
            str(decision.gate_decision) != "approved"
            or decision.evidence_grade is not ClaimGrade.E3
            or not decision.prediction_evidence_ids
            or not decision.human_approval
            or decision.candidate_id != record.source_candidate_id
            or decision.candidate_revision_id != record.revision_id
            or set(decision.claim_ids) != set(record.evidence_claim_ids)
        ):
            raise RegistryViolation("capability promotion decision identity mismatch")
        return super().append(record)


class RejectedRegistry(_AppendOnlyRegistry[RejectedRecord]):
    record_type = RejectedRecord

    def __init__(
        self, path: str | Path, *, decision_log: DecisionReader | None = None
    ) -> None:
        super().__init__(path)
        self._decision_log = decision_log

    def append(self, record: RejectedRecord) -> bool:
        decision = _decision_for(self._decision_log, record.promotion_decision_id)
        if (
            str(decision.gate_decision) != "rejected"
            or decision.candidate_id != record.candidate_id
            or decision.candidate_revision_id != record.revision_id
            or set(decision.claim_ids) != set(record.evidence_claim_ids)
        ):
            raise RegistryViolation("rejected promotion decision identity mismatch")
        return super().append(record)


class AgentProgramRegistry(_AppendOnlyRegistry[AgentProgramRecord]):
    record_type = AgentProgramRecord


def _decision_for(reader: DecisionReader | None, decision_id: str) -> Any:
    if reader is None:
        raise RegistryViolation("authoritative registry requires a decision log")
    matches = [item for item in reader.all() if item.decision_id == decision_id]
    if len(matches) != 1:
        raise RegistryViolation("promotion decision is missing or ambiguous")
    return matches[0]


def _approved_decision_for(
    reader: DecisionReader | None, decision_id: str
) -> Any:
    # Capability is a product authority projection, so a duck-typed reader is
    # insufficient: it could simply return a hand-built APPROVED object.  The
    # concrete log re-verifies the Governance HMAC before yielding a decision.
    from evolve.governance.decisions import PromotionDecisionLog

    if type(reader) is not PromotionDecisionLog:
        raise RegistryViolation(
            "capability registry requires authoritative promotion decision log"
        )
    try:
        return reader.verified_approved(decision_id)
    except ContractViolation as error:
        raise RegistryViolation(
            "capability promotion decision authority verification failed"
        ) from error
