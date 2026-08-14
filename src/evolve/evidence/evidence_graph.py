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
    ContractViolation,
    EvidenceEnvelope,
    MatchedCounterfactualPair,
    Receipt,
    canonical_json,
)

from .counterfactual import build_matched_counterfactual_pair
from .receipt_store import IntegrityError, ReceiptConflict, ReceiptStore

_Record = TypeVar("_Record", EvidenceEnvelope, Claim)


class EvidenceGraph:
    """Append-only facts plus a projection that is rebuilt on every open."""

    def __init__(
        self,
        root: str | Path,
        *,
        allow_legacy_unbound_claims: bool = False,
    ) -> None:
        self.root = Path(root)
        self.allow_legacy_unbound_claims = allow_legacy_unbound_claims
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

    def append_claim(
        self,
        claim: Claim,
        *,
        counterfactual_pair: MatchedCounterfactualPair | None = None,
    ) -> Claim:
        existing = self.list_claims()
        if claim.grade in {ClaimGrade.E2, ClaimGrade.E3}:
            if counterfactual_pair is None:
                raise IntegrityError(
                    "new E2/E3 claim requires the complete counterfactual pair"
                )
            _validate_appended_counterfactual(
                claim=claim,
                pair=counterfactual_pair,
                evidence_by_id={row.evidence_id: row for row in self.list_evidence()},
            )
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
                counterfactual_pair_sha256=value.get(
                    "counterfactual_pair_sha256"
                ),
                counterfactual_receipt_ids=tuple(
                    value.get("counterfactual_receipt_ids", ())
                ),
                _legacy_read=(
                    self.allow_legacy_unbound_claims
                    and "counterfactual_pair_sha256" not in value
                ),
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
    def rebuild(
        cls,
        root: str | Path,
        receipt_store: ReceiptStore,
        *,
        allow_legacy_unbound_claims: bool = False,
    ) -> EvidenceGraph:
        graph = cls(
            root,
            allow_legacy_unbound_claims=allow_legacy_unbound_claims,
        )
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
        evidence_by_id = {item.evidence_id: item for item in evidence}
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
            if claim.counterfactual_pair_sha256 is not None:
                _rebuild_counterfactual_binding(
                    claim=claim,
                    evidence_by_id=evidence_by_id,
                    receipts_by_id=receipts,
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


def _rebuild_counterfactual_binding(
    *,
    claim: Claim,
    evidence_by_id: dict[str, EvidenceEnvelope],
    receipts_by_id: dict[str, Receipt],
) -> None:
    selected = tuple(evidence_by_id[row] for row in claim.evidence_ids)

    def only(arm: str, observer_id: str) -> EvidenceEnvelope:
        matches = tuple(
            row
            for row in selected
            if row.observer_id == observer_id and row.payload.get("arm") == arm
        )
        if len(matches) != 1:
            raise IntegrityError(
                f"claim {claim.claim_id} has incomplete counterfactual evidence"
            )
        return matches[0]

    baseline_external = only("baseline", "external-trace-v1")
    baseline_native = only("baseline", "native-v1")
    taught_external = only("taught", "external-trace-v1")
    taught_native = only("taught", "native-v1")
    try:
        for arm, external, native in (
            ("baseline", baseline_external, baseline_native),
            ("taught", taught_external, taught_native),
        ):
            external_receipt = _only_typed_receipt(
                claim=claim,
                envelope=external,
                receipts_by_id=receipts_by_id,
                expected_kind="external_trace",
                label=f"{arm} external",
            )
            native_receipt = _only_typed_receipt(
                claim=claim,
                envelope=native,
                receipts_by_id=receipts_by_id,
                expected_kind="native_evaluation",
                label=f"{arm} native",
            )
            if external_receipt.plan_id != native_receipt.plan_id:
                raise IntegrityError(
                    f"claim {claim.claim_id} {arm} receipt plan mismatch"
                )
        baseline_model = receipts_by_id[
            str(baseline_external.payload["model_receipt_id"])
        ]
        taught_model = receipts_by_id[str(taught_external.payload["model_receipt_id"])]
        if baseline_model.kind != "model" or taught_model.kind != "model":
            raise IntegrityError(
                f"claim {claim.claim_id} model receipt kind mismatch"
            )
        candidate_revision_id = taught_model.payload["candidate_revision_id"]
        candidate_bundle_sha256 = taught_model.payload["candidate_bundle_sha256"]
        pair = build_matched_counterfactual_pair(
            candidate_id=claim.candidate_id,
            candidate_revision_id=candidate_revision_id,
            candidate_bundle_sha256=candidate_bundle_sha256,
            baseline_model_receipt=baseline_model,
            baseline_external_evidence=baseline_external,
            baseline_native_evidence=baseline_native,
            taught_model_receipt=taught_model,
            taught_external_evidence=taught_external,
            taught_native_evidence=taught_native,
        )
    except (KeyError, TypeError, ContractViolation) as error:
        raise IntegrityError(
            f"claim {claim.claim_id} counterfactual lineage is invalid"
        ) from error
    if pair.evidence_ids != claim.evidence_ids:
        raise IntegrityError(
            f"claim {claim.claim_id} counterfactual evidence order mismatch"
        )
    if pair.receipt_ids != claim.counterfactual_receipt_ids:
        raise IntegrityError(
            f"claim {claim.claim_id} counterfactual receipt binding mismatch"
        )
    if pair.content_sha256 != claim.counterfactual_pair_sha256:
        raise IntegrityError(
            f"claim {claim.claim_id} counterfactual pair hash mismatch"
        )
    baseline_outcome = baseline_native.payload
    taught_outcome = taught_native.payload
    if any(
        payload.get("evaluator_error") not in (None, "")
        for payload in (baseline_outcome, taught_outcome)
    ):
        raise IntegrityError(
            f"claim {claim.claim_id} binds infrastructure failure as E2"
        )
    baseline_resolved = baseline_outcome.get("resolved")
    taught_resolved = taught_outcome.get("resolved")
    if not isinstance(baseline_resolved, bool) or not isinstance(
        taught_resolved, bool
    ):
        raise IntegrityError(
            f"claim {claim.claim_id} has invalid native outcome types"
        )
    if not baseline_resolved and taught_resolved:
        expected_classification = ClaimClassification.GAIN
    elif baseline_resolved and not taught_resolved:
        expected_classification = ClaimClassification.REGRESSION
    else:
        expected_classification = ClaimClassification.NEUTRAL
    if claim.classification is not expected_classification:
        raise IntegrityError(
            f"claim {claim.claim_id} classification does not match native outcomes"
        )


def _validate_appended_counterfactual(
    *,
    claim: Claim,
    pair: MatchedCounterfactualPair,
    evidence_by_id: dict[str, EvidenceEnvelope],
) -> None:
    if (
        pair.candidate_id != claim.candidate_id
        or pair.content_sha256 != claim.counterfactual_pair_sha256
        or pair.evidence_ids != claim.evidence_ids
        or pair.receipt_ids != claim.counterfactual_receipt_ids
    ):
        raise IntegrityError("claim does not bind the supplied counterfactual pair")
    expected = (
        (
            pair.baseline.external_trace_evidence_id,
            pair.baseline.external_trace_evidence_sha256,
        ),
        (
            pair.baseline.native_outcome_evidence_id,
            pair.baseline.native_outcome_evidence_sha256,
        ),
        (
            pair.taught.external_trace_evidence_id,
            pair.taught.external_trace_evidence_sha256,
        ),
        (
            pair.taught.native_outcome_evidence_id,
            pair.taught.native_outcome_evidence_sha256,
        ),
    )
    if any(
        evidence_by_id.get(evidence_id) is None
        or evidence_by_id[evidence_id].content_sha256 != evidence_sha256
        for evidence_id, evidence_sha256 in expected
    ):
        raise IntegrityError("counterfactual pair evidence is not frozen in graph")


def _only_typed_receipt(
    *,
    claim: Claim,
    envelope: EvidenceEnvelope,
    receipts_by_id: dict[str, Receipt],
    expected_kind: str,
    label: str,
) -> Receipt:
    if len(envelope.receipt_ids) != 1:
        raise IntegrityError(
            f"claim {claim.claim_id} {label} evidence must bind one receipt"
        )
    receipt_id = envelope.receipt_ids[0]
    try:
        receipt = receipts_by_id[receipt_id]
    except KeyError as error:
        raise IntegrityError(
            f"claim {claim.claim_id} {label} receipt is missing"
        ) from error
    if receipt.kind != expected_kind:
        raise IntegrityError(
            f"claim {claim.claim_id} {label} receipt kind mismatch"
        )
    if receipt.artifact_sha256 != envelope.artifact_sha256:
        raise IntegrityError(
            f"claim {claim.claim_id} {label} artifact binding mismatch"
        )
    if receipt.plan_id != envelope.payload.get("plan_id"):
        raise IntegrityError(
            f"claim {claim.claim_id} {label} plan binding mismatch"
        )
    return receipt
