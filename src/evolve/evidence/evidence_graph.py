"""File-backed relationship projection for evidence and claims."""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, TypeVar

from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    EvidenceEnvelope,
    canonical_json,
)

from .receipt_store import IntegrityError, ReceiptConflict, ReceiptStore

_Record = TypeVar("_Record", EvidenceEnvelope, Claim)


class EvidenceGraph:
    """Append-only facts plus a projection that is rebuilt on every open."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence_path = self.root / "evidence.jsonl"
        self.claims_path = self.root / "claims.jsonl"

    def append_evidence(self, envelope: EvidenceEnvelope) -> EvidenceEnvelope:
        self._append(
            self.evidence_path,
            identifier=envelope.evidence_id,
            value=envelope,
            existing=self.list_evidence(),
            id_of=lambda item: item.evidence_id,
        )
        return envelope

    def append_claim(self, claim: Claim) -> Claim:
        existing = self.list_claims()
        if claim.supersedes_claim_id is not None:
            prior = next(
                (
                    item
                    for item in existing
                    if item.claim_id == claim.supersedes_claim_id
                ),
                None,
            )
            if prior is None:
                raise IntegrityError(
                    f"superseded claim does not exist: {claim.supersedes_claim_id}"
                )
            if prior.candidate_id != claim.candidate_id:
                raise IntegrityError("a claim can only supersede the same candidate")
        self._append(
            self.claims_path,
            identifier=claim.claim_id,
            value=claim,
            existing=existing,
            id_of=lambda item: item.claim_id,
        )
        return claim

    def list_evidence(self) -> tuple[EvidenceEnvelope, ...]:
        return self._read(
            self.evidence_path,
            constructor=lambda value: EvidenceEnvelope(
                evidence_id=value["evidence_id"],
                receipt_ids=tuple(value["receipt_ids"]),
                observer_id=value["observer_id"],
                grade=ClaimGrade(value["grade"]),
                payload=value["payload"],
                artifact_sha256=value["artifact_sha256"],
            ),
            id_of=lambda item: item.evidence_id,
        )

    def list_claims(self) -> tuple[Claim, ...]:
        return self._read(
            self.claims_path,
            constructor=lambda value: Claim(
                claim_id=value["claim_id"],
                candidate_id=value["candidate_id"],
                grade=ClaimGrade(value["grade"]),
                classification=ClaimClassification(value["classification"]),
                evidence_ids=tuple(value["evidence_ids"]),
                rationale=value["rationale"],
                supersedes_claim_id=value["supersedes_claim_id"],
            ),
            id_of=lambda item: item.claim_id,
        )

    def evidence_for_receipt(self, receipt_id: str) -> tuple[EvidenceEnvelope, ...]:
        return tuple(
            item for item in self.list_evidence() if receipt_id in item.receipt_ids
        )

    def evidence_by_observer(self, observer_id: str) -> tuple[EvidenceEnvelope, ...]:
        return tuple(
            item for item in self.list_evidence() if item.observer_id == observer_id
        )

    def evidence_for_plan(self, plan_id: str) -> tuple[EvidenceEnvelope, ...]:
        """Return the aligned observer window for one immutable execution plan."""

        return tuple(
            item
            for item in self.list_evidence()
            if item.payload.get("plan_id") == plan_id
        )

    def claims_for_candidate(self, candidate_id: str) -> tuple[Claim, ...]:
        return tuple(
            item for item in self.list_claims() if item.candidate_id == candidate_id
        )

    def latest_claims(self) -> tuple[Claim, ...]:
        claims = self.list_claims()
        superseded = {
            claim.supersedes_claim_id
            for claim in claims
            if claim.supersedes_claim_id is not None
        }
        return tuple(claim for claim in claims if claim.claim_id not in superseded)

    def classification_counts(self) -> dict[str, int]:
        return dict(
            Counter(str(claim.classification) for claim in self.latest_claims())
        )

    @classmethod
    def rebuild(cls, root: str | Path, receipt_store: ReceiptStore) -> EvidenceGraph:
        graph = cls(root)
        receipts = {item.receipt_id: item for item in receipt_store.list_receipts()}
        evidence = graph.list_evidence()
        evidence_ids = {item.evidence_id for item in evidence}
        for envelope in evidence:
            for receipt_id in envelope.receipt_ids:
                if receipt_id not in receipts:
                    raise IntegrityError(
                        f"evidence {envelope.evidence_id} references missing receipt "
                        f"{receipt_id}"
                    )
            if not any(
                receipts[receipt_id].artifact_sha256 == envelope.artifact_sha256
                for receipt_id in envelope.receipt_ids
            ):
                raise IntegrityError(
                    f"evidence {envelope.evidence_id} artifact is not backed by receipt"
                )
            receipt_store.read_artifact(envelope.artifact_sha256)
        claims = graph.list_claims()
        claims_by_id = {claim.claim_id: claim for claim in claims}
        for claim in claims:
            missing = set(claim.evidence_ids) - evidence_ids
            if missing:
                raise IntegrityError(
                    f"claim {claim.claim_id} references missing evidence: "
                    f"{sorted(missing)}"
                )
            if claim.supersedes_claim_id is not None:
                prior = claims_by_id.get(claim.supersedes_claim_id)
                if prior is None or prior.candidate_id != claim.candidate_id:
                    raise IntegrityError(
                        f"claim {claim.claim_id} has invalid supersession target"
                    )
        return graph

    @staticmethod
    def _append(
        path: Path,
        *,
        identifier: str,
        value: _Record,
        existing: tuple[_Record, ...],
        id_of: Callable[[_Record], str],
    ) -> None:
        prior = next((item for item in existing if id_of(item) == identifier), None)
        if prior is not None:
            if prior != value:
                raise ReceiptConflict(f"immutable record conflict: {identifier}")
            return
        record = {
            "content_sha256": value.content_sha256,
            "value": dataclasses.asdict(value),
        }
        encoded = (canonical_json(record) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise IntegrityError(f"partial graph append: {identifier}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read(
        path: Path,
        *,
        constructor: Callable[[dict[str, Any]], _Record],
        id_of: Callable[[_Record], str],
    ) -> tuple[_Record, ...]:
        if not path.exists():
            return ()
        output: list[_Record] = []
        seen: dict[str, _Record] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    raw = json.loads(line)
                    item = constructor(raw["value"])
                    expected = raw["content_sha256"]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise IntegrityError(
                        f"invalid graph record at {path.name}:{line_number}"
                    ) from error
                if item.content_sha256 != expected:
                    raise IntegrityError(
                        f"graph record hash mismatch at {path.name}:{line_number}"
                    )
                identifier = id_of(item)
                if prior := seen.get(identifier):
                    if prior != item:
                        raise ReceiptConflict(
                            f"immutable graph record conflict: {identifier}"
                        )
                    continue
                seen[identifier] = item
                output.append(item)
        return tuple(output)
