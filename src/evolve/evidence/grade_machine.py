"""Deterministic evidence-grade projection across independently matched tasks."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    ContractViolation,
    EvidenceEnvelope,
    Receipt,
)

from .evidence_graph import EvidenceGraph
from .receipt_store import IntegrityError, ReceiptStore


@dataclass(frozen=True, slots=True)
class CandidateEvidenceState:
    candidate_id: str
    grade: ClaimGrade
    classification: str
    claim_ids: tuple[str, ...]
    task_revision_ids: tuple[str, ...]
    task_count: int
    project_count: int
    gain_count: int
    neutral_count: int
    regression_count: int
    infra_failure_count: int
    mechanism_id: str | None
    prediction_consistent_task_count: int
    prediction_evidence_ids: tuple[str, ...]
    e3_eligible: bool
    rationale: str

    @property
    def positive_count(self) -> int:
        return self.gain_count

    @property
    def negative_count(self) -> int:
        return self.regression_count


class TrustedObserverVerifier(Protocol):
    """Process-local trust root for independently produced observations."""

    def verify_evidence(self, envelope: EvidenceEnvelope) -> bool: ...

    def verify_receipt_lineage(
        self,
        *,
        envelope: EvidenceEnvelope,
        observation_receipt: Receipt,
        prediction_receipt: Receipt,
        model_receipt: Receipt,
        artifact_reader: Callable[[str], bytes],
    ) -> bool: ...


class EvidenceGradeMachine:
    """Rebuild E0/E1/E2/E3 from current, non-superseded native claims."""

    def __init__(
        self,
        graph: EvidenceGraph,
        *,
        trusted_observer_verifier: TrustedObserverVerifier | None = None,
        receipt_store: ReceiptStore | None = None,
    ) -> None:
        self._graph = graph
        self._trusted_observer_verifier = trusted_observer_verifier
        self._receipt_store = receipt_store

    def aggregate(
        self,
        candidate_id: str,
        *,
        task_projects: Mapping[str, str],
        mechanism_id: str | None = None,
    ) -> CandidateEvidenceState:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ContractViolation("candidate_id must be non-empty")
        claims = tuple(
            claim
            for claim in self._graph.latest_claims()
            if claim.candidate_id == candidate_id
        )
        if not claims:
            return CandidateEvidenceState(
                candidate_id=candidate_id,
                grade=ClaimGrade.E0,
                classification="insufficient_evidence",
                claim_ids=(),
                task_revision_ids=(),
                task_count=0,
                project_count=0,
                gain_count=0,
                neutral_count=0,
                regression_count=0,
                infra_failure_count=0,
                mechanism_id=mechanism_id,
                prediction_consistent_task_count=0,
                prediction_evidence_ids=(),
                e3_eligible=False,
                rationale="no matched native claims",
            )

        evidence = {
            envelope.evidence_id: envelope for envelope in self._graph.list_evidence()
        }
        claims_by_task: dict[str, Claim] = {}
        for claim in claims:
            task_ids = {
                str(evidence[evidence_id].payload.get("task_revision_id", ""))
                for evidence_id in claim.evidence_ids
                if evidence_id in evidence
            }
            task_ids.discard("")
            if len(task_ids) != 1:
                raise IntegrityError(
                    f"claim {claim.claim_id} must bind exactly one task revision"
                )
            task_id = next(iter(task_ids))
            prior = claims_by_task.get(task_id)
            if prior is not None and prior.claim_id != claim.claim_id:
                raise IntegrityError(
                    f"multiple latest claims bind task revision {task_id}"
                )
            claims_by_task[task_id] = claim

        missing_projects = set(claims_by_task) - set(task_projects)
        if missing_projects:
            raise ContractViolation(
                "task project identity missing for: "
                + ", ".join(sorted(missing_projects))
            )
        projects = {task_projects[task_id] for task_id in claims_by_task}
        if any(
            not isinstance(project, str) or not project.strip() for project in projects
        ):
            raise ContractViolation("task project identities must be non-empty")

        counts = Counter(claim.classification for claim in claims_by_task.values())
        classification = _aggregate_classification(counts)
        task_count = len(claims_by_task)
        project_count = len(projects)
        receipts_by_id = (
            {
                receipt.receipt_id: receipt
                for receipt in self._receipt_store.list_receipts()
            }
            if self._receipt_store is not None
            else {}
        )
        prediction_evidence_by_task = {
            task_id: _task_prediction_evidence(
                tuple(evidence.values()),
                task_revision_id=task_id,
                mechanism_id=mechanism_id,
                claim_evidence_ids=claims_by_task[task_id].evidence_ids,
                trusted_observer_verifier=self._trusted_observer_verifier,
                receipts_by_id=receipts_by_id,
                artifact_reader=(
                    self._receipt_store.read_artifact
                    if self._receipt_store is not None
                    else None
                ),
            )
            for task_id in claims_by_task
        }
        prediction_consistent = sum(
            bool(rows) for rows in prediction_evidence_by_task.values()
        )
        prediction_evidence_ids = tuple(
            sorted(
                evidence_id
                for rows in prediction_evidence_by_task.values()
                for evidence_id in rows
            )
        )
        has_infra = bool(counts[ClaimClassification.INFRA_FAILURE])
        valid_native_claims = all(
            claim.grade >= ClaimGrade.E2 for claim in claims_by_task.values()
        )
        counterfactual_lineage_rebuilt = False
        if self._receipt_store is not None and valid_native_claims:
            # E3 is a fresh projection, not a property that a stored Claim may
            # self-assert.  Rebuild from the immutable Receipt Store so a
            # forged pair digest, renamed receipt, or missing model artifact
            # fails closed before trusted-observer evidence is considered.
            EvidenceGraph.rebuild(
                self._graph.root,
                self._receipt_store,
                allow_legacy_unbound_claims=False,
            )
            counterfactual_lineage_rebuilt = True
        e3_eligible = (
            isinstance(mechanism_id, str)
            and bool(mechanism_id.strip())
            and task_count >= 3
            and project_count >= 2
            and counts[ClaimClassification.GAIN] >= 2
            and not counts[ClaimClassification.REGRESSION]
            and not has_infra
            and valid_native_claims
            and counterfactual_lineage_rebuilt
            and prediction_consistent == task_count
        )
        if has_infra:
            grade = ClaimGrade.E1
        elif e3_eligible:
            grade = ClaimGrade.E3
        elif valid_native_claims:
            grade = ClaimGrade.E2
        else:
            grade = ClaimGrade.E1
        rationale = (
            f"{task_count} matched native task(s) across {project_count} project(s); "
            f"gain={counts[ClaimClassification.GAIN]}, "
            f"neutral={counts[ClaimClassification.NEUTRAL]}, "
            f"regression={counts[ClaimClassification.REGRESSION]}, "
            f"infra={counts[ClaimClassification.INFRA_FAILURE]}, "
            f"counterfactual_rebuilt={counterfactual_lineage_rebuilt}, "
            f"prediction_consistent={prediction_consistent}/{task_count}"
        )
        return CandidateEvidenceState(
            candidate_id=candidate_id,
            grade=grade,
            classification=classification,
            claim_ids=tuple(
                sorted(claim.claim_id for claim in claims_by_task.values())
            ),
            task_revision_ids=tuple(sorted(claims_by_task)),
            task_count=task_count,
            project_count=project_count,
            gain_count=counts[ClaimClassification.GAIN],
            neutral_count=counts[ClaimClassification.NEUTRAL],
            regression_count=counts[ClaimClassification.REGRESSION],
            infra_failure_count=counts[ClaimClassification.INFRA_FAILURE],
            mechanism_id=mechanism_id,
            prediction_consistent_task_count=prediction_consistent,
            prediction_evidence_ids=prediction_evidence_ids,
            e3_eligible=e3_eligible,
            rationale=rationale,
        )


def _aggregate_classification(counts: Counter[ClaimClassification]) -> str:
    if counts[ClaimClassification.INFRA_FAILURE]:
        return str(ClaimClassification.INFRA_FAILURE)
    if counts[ClaimClassification.REGRESSION]:
        return str(ClaimClassification.REGRESSION)
    if counts[ClaimClassification.GAIN]:
        return str(ClaimClassification.GAIN)
    return str(ClaimClassification.NEUTRAL)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _task_prediction_evidence(
    evidence: tuple[EvidenceEnvelope, ...],
    *,
    task_revision_id: str,
    mechanism_id: str | None,
    claim_evidence_ids: tuple[str, ...],
    trusted_observer_verifier: TrustedObserverVerifier | None,
    receipts_by_id: Mapping[str, Receipt],
    artifact_reader: Callable[[str], bytes] | None,
) -> tuple[str, ...]:
    if (
        not isinstance(mechanism_id, str)
        or not mechanism_id.strip()
        or trusted_observer_verifier is None
        or artifact_reader is None
    ):
        return ()
    native_rows = [
        envelope
        for envelope in evidence
        if envelope.evidence_id in claim_evidence_ids
        and getattr(envelope, "observer_id", None) == "native-v1"
        and envelope.payload.get("task_revision_id") == task_revision_id
        and _valid_sha256(envelope.payload.get("prediction_sha256"))
    ]
    raw_plan_ids = [row.payload.get("plan_id") for row in native_rows]
    if any(not isinstance(plan_id, str) or not plan_id for plan_id in raw_plan_ids):
        return ()
    plan_ids = {
        plan_id for plan_id in raw_plan_ids if isinstance(plan_id, str) and plan_id
    }
    if len(plan_ids) != 2:
        return ()
    if {row.payload.get("arm") for row in native_rows} != {"baseline", "taught"}:
        return ()
    evidence_ids: list[str] = []
    expected_effect_sha256: str | None = None
    prediction_candidate_revision_id: str | None = None
    for plan_id in sorted(plan_ids):
        native = [row for row in native_rows if row.payload.get("plan_id") == plan_id]
        external = [
            envelope
            for envelope in evidence
            if envelope.evidence_id in claim_evidence_ids
            and getattr(envelope, "observer_id", None) == "external-trace-v1"
            and envelope.payload.get("task_revision_id") == task_revision_id
            and envelope.payload.get("plan_id") == plan_id
            and envelope.payload.get("mechanism_id") == mechanism_id
        ]
        internal = [
            envelope
            for envelope in evidence
            if getattr(envelope, "observer_id", None) == "trusted-jlens-v1"
            and envelope.payload.get("task_revision_id") == task_revision_id
            and envelope.payload.get("plan_id") == plan_id
            and envelope.payload.get("mechanism_id") == mechanism_id
            and trusted_observer_verifier.verify_evidence(envelope)
        ]
        if len(native) != 1 or len(external) != 1 or len(internal) != 1:
            return ()
        target = native[0].payload["prediction_sha256"]
        subject = _model_subject(native[0])
        if subject is None:
            return ()
        if any(_model_subject(row) != subject for row in (*external, *internal)):
            return ()
        if any(
            row.payload.get("prediction_sha256") != target
            for row in (*external, *internal)
        ):
            return ()
        internal_row = internal[0]
        if len(internal_row.receipt_ids) != 1:
            return ()
        observation_receipt = receipts_by_id.get(internal_row.receipt_ids[0])
        prediction_receipt_id = internal_row.payload.get(
            "mechanism_prediction_receipt_id"
        )
        model_receipt_id = internal_row.payload.get("model_receipt_id")
        if (
            observation_receipt is None
            or not isinstance(prediction_receipt_id, str)
            or not isinstance(model_receipt_id, str)
        ):
            return ()
        prediction_receipt = receipts_by_id.get(prediction_receipt_id)
        model_receipt = receipts_by_id.get(model_receipt_id)
        if (
            prediction_receipt is None
            or model_receipt is None
            or not trusted_observer_verifier.verify_receipt_lineage(
                envelope=internal_row,
                observation_receipt=observation_receipt,
                prediction_receipt=prediction_receipt,
                model_receipt=model_receipt,
                artifact_reader=artifact_reader,
            )
        ):
            return ()
        try:
            prediction_candidate = str(
                prediction_receipt.payload["candidate_revision_id"]
            )
        except KeyError:
            return ()
        if prediction_candidate_revision_id is None:
            prediction_candidate_revision_id = prediction_candidate
        elif prediction_candidate_revision_id != prediction_candidate:
            return ()
        if (
            native[0].payload.get("arm") == "taught"
            and external[0].payload.get("candidate_revision_id") != prediction_candidate
        ):
            return ()
        current_expected_effect_sha256 = internal_row.payload.get(
            "expected_internal_effect_sha256"
        )
        if (
            not _valid_sha256(current_expected_effect_sha256)
            or not _valid_sha256(
                internal_row.payload.get("observed_internal_effect_sha256")
            )
            or not _valid_sha256(internal_row.payload.get("raw_trace_sha256"))
            or not isinstance(internal_row.payload.get("locations"), list)
            or not internal_row.payload.get("locations")
        ):
            return ()
        if expected_effect_sha256 is None:
            expected_effect_sha256 = current_expected_effect_sha256
        elif expected_effect_sha256 != current_expected_effect_sha256:
            return ()
        expected_consistency = native[0].payload.get("arm") == "taught"
        if internal_row.payload.get("effect_consistent") is not expected_consistency:
            return ()
        evidence_ids.extend(
            (native[0].evidence_id, external[0].evidence_id, internal[0].evidence_id)
        )
    return tuple(evidence_ids)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _model_subject(envelope: EvidenceEnvelope) -> tuple[str, str] | None:
    receipt_id = envelope.payload.get("model_receipt_id")
    artifact_sha256 = envelope.payload.get("model_artifact_sha256")
    if (
        not isinstance(receipt_id, str)
        or not receipt_id.strip()
        or not _valid_sha256(artifact_sha256)
    ):
        return None
    return receipt_id, artifact_sha256  # type: ignore[return-value]
