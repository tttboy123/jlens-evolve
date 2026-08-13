"""The only component allowed to dispatch model, workspace, and evaluator I/O."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping, Sequence

from evolve.contracts import (
    Authorization,
    ContractViolation,
    EvidenceEnvelope,
    ExecutionPlan,
    Receipt,
    canonical_json,
)

from .adapters import (
    ModelTransport,
    NativeEvaluator,
    ObserverHub,
    ReceiptSink,
    WorkspaceManager,
)


class ExecutionInterrupted(RuntimeError):
    """A safe interruption boundary, including a translated SIGTERM."""


class EvaluatorInfrastructureError(RuntimeError):
    """The evaluator failed to produce a native outcome."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    status: str
    receipts: tuple[Receipt, ...]
    evidence: tuple[EvidenceEnvelope, ...]
    replayed: bool


class ExecutionRuntime:
    """Execute a frozen plan and preserve every stage as append-only facts."""

    def __init__(
        self,
        *,
        model_transport: ModelTransport,
        workspace_manager: WorkspaceManager,
        native_evaluator: NativeEvaluator,
        observer_hub: ObserverHub,
        receipt_sink: ReceiptSink,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._model_transport = model_transport
        self._workspace_manager = workspace_manager
        self._native_evaluator = native_evaluator
        self._observer_hub = observer_hub
        self._receipt_sink = receipt_sink
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(
        self, plan: ExecutionPlan, authorization: Authorization
    ) -> ExecutionResult:
        existing = tuple(
            sorted(
                self._receipt_sink.receipts_for(plan.plan_id),
                key=lambda item: item.sequence,
            )
        )
        for singleton_kind in (
            "workspace",
            "model",
            "external_trace",
            "cost",
            "native_evaluation",
            "execution_terminal",
        ):
            if sum(item.kind == singleton_kind for item in existing) > 1:
                raise ContractViolation(
                    f"resume contains duplicate {singleton_kind} receipts"
                )
        self._validate_admission(plan, authorization)
        terminal = next(
            (item for item in reversed(existing) if item.kind == "execution_terminal"),
            None,
        )
        if terminal is not None:
            workspace_receipt = next(
                (item for item in existing if item.kind == "workspace"), None
            )
            if (
                workspace_receipt is None
                or workspace_receipt.payload.get("plan_sha256")
                != plan.content_sha256
            ):
                raise ContractViolation("terminal replay plan identity mismatch")
            return ExecutionResult(
                status=str(terminal.payload["status"]),
                receipts=existing,
                evidence=(),
                replayed=True,
            )
        appender = _ReceiptAppender(
            plan=plan,
            sink=self._receipt_sink,
            observer_hub=self._observer_hub,
            clock=self._clock,
            existing=existing,
        )
        existing_by_kind = {receipt.kind: receipt for receipt in existing}
        workspace: Mapping[str, Any]
        model_output: Mapping[str, Any]

        try:
            workspace_receipt = existing_by_kind.get("workspace")
            if workspace_receipt is None:
                materialized = dict(self._workspace_manager.materialize(plan))
                materialized["plan_sha256"] = plan.content_sha256
                workspace_receipt = appender.append_fact("workspace", materialized)
            elif workspace_receipt.payload.get("plan_sha256") != plan.content_sha256:
                raise ContractViolation("resume plan does not match workspace receipt")
            workspace = workspace_receipt.payload

            model_receipt = existing_by_kind.get("model")
            if model_receipt is None:
                model_payload = dict(self._model_transport.infer(plan, workspace))
                model_payload.update(
                    {
                        "provider": plan.model.provider,
                        "model": plan.model.model,
                        "revision": plan.model.revision,
                    }
                )
                model_receipt = appender.append_fact("model", model_payload)
            model_output = model_receipt.payload

            trace_receipt = existing_by_kind.get("external_trace")
            if trace_receipt is None:
                trace_payload = {
                    "model_receipt_id": model_receipt.receipt_id,
                    "model_artifact_sha256": model_receipt.artifact_sha256,
                    "arm": plan.arm,
                    "task_revision_id": plan.task.revision_id,
                }
                for name in (
                    "raw_output_path",
                    "raw_output_sha256",
                    "prompt_paths",
                    "prompt_sha256",
                    "structural_valid",
                    "failure_reason",
                    "mechanism",
                    "condition_id",
                ):
                    if name in model_output:
                        trace_payload[name] = model_output[name]
                appender.append_fact("external_trace", trace_payload)
            elif (
                trace_receipt.payload.get("model_receipt_id")
                != model_receipt.receipt_id
            ):
                raise ContractViolation("resume external trace identity mismatch")

            cost_receipt = existing_by_kind.get("cost")
            if cost_receipt is None:
                actual_cost = _validated_cost(model_output.get("cost_cny", 0))
                cost_receipt = appender.append_fact(
                    "cost",
                    {
                        "cost_cny": actual_cost,
                        "input_tokens": int(model_output.get("input_tokens", 0)),
                        "output_tokens": int(model_output.get("output_tokens", 0)),
                        "provider": plan.model.provider,
                        "model": plan.model.model,
                        "revision": plan.model.revision,
                    },
                )
            if float(cost_receipt.payload["cost_cny"]) > plan.limits.max_cost_cny:
                appender.append_control(
                    "execution_terminal",
                    {
                        "status": "budget_exhausted",
                        "reason": "actual model cost exceeded the plan limit",
                    },
                )
                return appender.result("budget_exhausted")

            native_receipt = existing_by_kind.get("native_evaluation")
            if native_receipt is None:
                native_payload = dict(
                    self._native_evaluator.evaluate(plan, workspace, model_output)
                )
                native_payload.update(_native_identity(plan))
                native_payload["evaluator_error"] = None
                appender.append_fact("native_evaluation", native_payload)

            appender.append_control("execution_terminal", {"status": "completed"})
            return appender.result("completed")
        except ExecutionInterrupted as error:
            appender.append_control(
                "execution_partial",
                {"status": "partial", "reason": str(error), "retryable": True},
            )
            return appender.result("partial")
        except EvaluatorInfrastructureError as error:
            appender.append_fact(
                "native_evaluation",
                {
                    **_native_identity(plan),
                    "resolved": False,
                    "evaluator_error": f"{type(error).__name__}: {error}",
                },
            )
            appender.append_control(
                "execution_terminal",
                {"status": "infra_failure", "reason": str(error)},
            )
            return appender.result("infra_failure")

    def _validate_admission(
        self, plan: ExecutionPlan, authorization: Authorization
    ) -> None:
        if plan.campaign_id != authorization.campaign_id:
            raise ContractViolation(
                "authorization campaign does not match execution plan"
            )
        if plan.native_evaluator_id != plan.task.evaluator_id:
            raise ContractViolation(
                "plan evaluator does not match frozen task evaluator"
            )
        if self._native_evaluator.evaluator_id != plan.native_evaluator_id:
            raise ContractViolation(
                "runtime evaluator identity does not match execution plan"
            )
        authorization.assert_allows(
            cohort=plan.task.cohort,
            reserved_cost_cny=plan.limits.max_cost_cny,
            reserved_model_calls=1,
            remote=self._model_transport.remote,
            now=self._clock(),
        )


class _ReceiptAppender:
    def __init__(
        self,
        *,
        plan: ExecutionPlan,
        sink: ReceiptSink,
        observer_hub: ObserverHub,
        clock: Callable[[], datetime],
        existing: Sequence[Receipt],
    ) -> None:
        self._plan = plan
        self._sink = sink
        self._observer_hub = observer_hub
        self._clock = clock
        self._receipts = list(existing)
        self._evidence: list[EvidenceEnvelope] = []
        self._next_sequence = max((item.sequence for item in existing), default=0) + 1

    def append_fact(self, kind: str, payload: Mapping[str, Any]) -> Receipt:
        receipt = self._append(kind, payload)
        self._evidence.extend(self._observer_hub.observe(receipt))
        return receipt

    def append_control(self, kind: str, payload: Mapping[str, Any]) -> Receipt:
        return self._append(kind, payload)

    def result(self, status: str) -> ExecutionResult:
        return ExecutionResult(
            status=status,
            receipts=tuple(self._receipts),
            evidence=tuple(self._evidence),
            replayed=False,
        )

    def _append(self, kind: str, payload: Mapping[str, Any]) -> Receipt:
        artifact = canonical_json(payload).encode("utf-8")
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        sequence = self._next_sequence
        identity = canonical_json(
            {
                "campaign_id": self._plan.campaign_id,
                "plan_id": self._plan.plan_id,
                "sequence": sequence,
                "kind": kind,
                "artifact_sha256": artifact_sha256,
            }
        ).encode("utf-8")
        receipt = Receipt(
            receipt_id="receipt-" + hashlib.sha256(identity).hexdigest(),
            campaign_id=self._plan.campaign_id,
            plan_id=self._plan.plan_id,
            sequence=sequence,
            kind=kind,
            created_at=_isoformat(self._clock()),
            payload=dict(payload),
            artifact_sha256=artifact_sha256,
        )
        stored = self._sink.append(receipt, artifact)
        self._receipts.append(stored)
        self._next_sequence += 1
        return stored


def _validated_cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation("model cost must be numeric")
    numeric = float(value)
    if numeric < 0 or not math.isfinite(numeric):
        raise ContractViolation("model cost must be finite and non-negative")
    return numeric


def _native_identity(plan: ExecutionPlan) -> dict[str, Any]:
    """Project the exact matched A/B identity into each native fact."""

    config = {
        "context_policy_id": plan.context_policy_id,
        "tool_policy_id": plan.tool_policy_id,
        "observer_policy_ids": list(plan.observer_policy_ids),
        "limits": {
            "max_tokens": plan.limits.max_tokens,
            "max_seconds": plan.limits.max_seconds,
            "max_cost_cny": plan.limits.max_cost_cny,
        },
        "holdout_scope": plan.holdout_scope,
    }
    return {
        "arm": plan.arm,
        "task_revision_id": plan.task.revision_id,
        "task_source_sha256": plan.task.source_sha256,
        "model_identity": (
            f"{plan.model.provider}/{plan.model.model}@{plan.model.revision}"
        ),
        "native_evaluator_id": plan.native_evaluator_id,
        "execution_config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
    }


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ContractViolation("runtime clock must return a timezone-aware value")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
