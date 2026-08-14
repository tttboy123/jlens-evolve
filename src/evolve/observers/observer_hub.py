"""Evidence-only observation boundary for execution receipts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from evolve.contracts import ClaimGrade, EvidenceEnvelope, Receipt
from evolve.evidence.evidence_graph import EvidenceGraph


class Observer(Protocol):
    @property
    def observer_id(self) -> str: ...

    def observe(self, receipt: Receipt) -> EvidenceEnvelope | None: ...


@dataclass(frozen=True, slots=True)
class ReceiptObserver:
    observer_id: str
    receipt_kinds: tuple[str, ...]
    grade: ClaimGrade = ClaimGrade.E0

    def observe(self, receipt: Receipt) -> EvidenceEnvelope | None:
        if receipt.kind not in self.receipt_kinds:
            return None
        identity = hashlib.sha256(
            f"{self.observer_id}\0{receipt.content_sha256}".encode()
        ).hexdigest()
        return EvidenceEnvelope(
            evidence_id=f"evidence-{identity}",
            receipt_ids=(receipt.receipt_id,),
            observer_id=self.observer_id,
            grade=self.grade,
            payload={
                "campaign_id": receipt.campaign_id,
                "plan_id": receipt.plan_id,
                "receipt_kind": receipt.kind,
                **receipt.payload,
            },
            artifact_sha256=receipt.artifact_sha256,
        )


class ExternalTraceObserver(ReceiptObserver):
    def __init__(self) -> None:
        super().__init__("external-trace-v1", ("external_trace",), ClaimGrade.E0)


class JacobianLensObserver(ReceiptObserver):
    """Legacy/untrusted trace projection; never sufficient for E3.

    This adapter intentionally preserves historical replay compatibility.  E3
    uses ``TrustedJacobianLensObserver`` and a process-local trust verifier.
    """

    def __init__(self) -> None:
        super().__init__("jlens-v1", ("jlens", "internal_trace"), ClaimGrade.E0)


class NativeOutcomeObserver(ReceiptObserver):
    def __init__(self) -> None:
        super().__init__("native-v1", ("native_evaluation",), ClaimGrade.E1)


class CostObserver(ReceiptObserver):
    def __init__(self) -> None:
        super().__init__("cost-v1", ("cost", "model_usage"), ClaimGrade.E0)


class SafetyObserver(ReceiptObserver):
    def __init__(self) -> None:
        super().__init__("safety-v1", ("safety",), ClaimGrade.E0)


class ObserverHub:
    """Fans receipts out to configured observers and appends resulting evidence."""

    def __init__(
        self,
        observers: tuple[Observer, ...],
        *,
        graph: EvidenceGraph | None = None,
    ) -> None:
        self._observers = observers
        self._graph = graph

    def observe(self, receipt: Receipt) -> tuple[EvidenceEnvelope, ...]:
        emitted = tuple(
            envelope
            for observer in self._observers
            if (envelope := observer.observe(receipt)) is not None
        )
        if self._graph is not None:
            for envelope in emitted:
                self._graph.append_evidence(envelope)
        return emitted
