"""Ports owned by the single Execution Runtime entry point."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from evolve.contracts import (
    Authorization,
    EvidenceEnvelope,
    ExecutionPlan,
    MechanismPrediction,
    Receipt,
)


class ModelTransport(Protocol):
    """A frozen local or remote inference transport."""

    remote: bool

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class WorkspaceManager(Protocol):
    def materialize(self, plan: ExecutionPlan) -> Mapping[str, Any]: ...


class NativeEvaluator(Protocol):
    evaluator_id: str

    def evaluate(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        model_output: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class ObserverHub(Protocol):
    def observe(self, receipt: Receipt) -> tuple[EvidenceEnvelope, ...]: ...


class ReceiptSink(Protocol):
    def append(self, receipt: Receipt, artifact: bytes) -> Receipt: ...

    def receipts_for(self, plan_id: str) -> tuple[Receipt, ...]: ...


class RuntimeEntry(Protocol):
    """Strategy-facing seam: plans enter here, never through adapters directly."""

    @property
    def remote(self) -> bool: ...

    def execute(
        self,
        plan: ExecutionPlan,
        authorization: Authorization,
        *,
        mechanism_prediction: MechanismPrediction | None = None,
    ) -> object: ...
