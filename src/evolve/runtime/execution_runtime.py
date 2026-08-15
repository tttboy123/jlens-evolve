"""The only component allowed to dispatch model, workspace, and evaluator I/O."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evolve.contracts import (
    Authorization,
    ContractViolation,
    EvidenceEnvelope,
    ExecutionPlan,
    MechanismPrediction,
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

    @property
    def remote(self) -> bool:
        """Expose transport locality to the Campaign budget authority."""

        return self._model_transport.remote

    def execute(
        self,
        plan: ExecutionPlan,
        authorization: Authorization,
        *,
        mechanism_prediction: MechanismPrediction | None = None,
    ) -> ExecutionResult:
        existing = tuple(
            sorted(
                self._receipt_sink.receipts_for(plan.plan_id),
                key=lambda item: item.sequence,
            )
        )
        for singleton_kind in (
            "workspace",
            "mechanism_prediction",
            "model",
            "external_trace",
            "internal_trace",
            "cost",
            "native_evaluation",
            "execution_terminal",
        ):
            if sum(item.kind == singleton_kind for item in existing) > 1:
                raise ContractViolation(
                    f"resume contains duplicate {singleton_kind} receipts"
                )
        self._validate_admission(plan, authorization)
        existing_prediction = next(
            (item for item in existing if item.kind == "mechanism_prediction"),
            None,
        )
        if mechanism_prediction is None and existing_prediction is not None:
            raise ContractViolation("resume mechanism prediction identity mismatch")
        if (
            mechanism_prediction is not None
            and existing_prediction is not None
            and existing_prediction.payload != mechanism_prediction.as_payload()
        ):
            raise ContractViolation("mechanism prediction identity drift")
        terminal = next(
            (item for item in reversed(existing) if item.kind == "execution_terminal"),
            None,
        )
        if terminal is not None:
            if mechanism_prediction is not None and existing_prediction is None:
                raise ContractViolation(
                    "terminal replay is missing mechanism prediction receipt"
                )
            workspace_receipt = next(
                (item for item in existing if item.kind == "workspace"), None
            )
            if (
                workspace_receipt is None
                or workspace_receipt.payload.get("plan_sha256") != plan.content_sha256
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
        model_receipt: Receipt | None = existing_by_kind.get("model")

        try:
            workspace_receipt = existing_by_kind.get("workspace")
            if workspace_receipt is None:
                materialized = dict(self._workspace_manager.materialize(plan))
                materialized["plan_sha256"] = plan.content_sha256
                workspace_receipt = appender.append_fact("workspace", materialized)
            elif workspace_receipt.payload.get("plan_sha256") != plan.content_sha256:
                raise ContractViolation("resume plan does not match workspace receipt")
            workspace = workspace_receipt.payload

            prediction_receipt = existing_by_kind.get("mechanism_prediction")
            if mechanism_prediction is not None:
                if prediction_receipt is None:
                    if model_receipt is not None:
                        raise ContractViolation(
                            "mechanism prediction was not frozen before model dispatch"
                        )
                    prediction_receipt = appender.append_fact(
                        "mechanism_prediction", mechanism_prediction.as_payload()
                    )

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
                prediction_sha256 = _model_prediction_sha256(model_output)
                mechanism_id = model_output.get(
                    "mechanism_id", plan.metadata.get("mechanism_id")
                )
                trace_payload = {
                    "model_receipt_id": model_receipt.receipt_id,
                    "model_artifact_sha256": model_receipt.artifact_sha256,
                    "arm": plan.arm,
                    "task_revision_id": plan.task.revision_id,
                }
                if mechanism_id is not None:
                    trace_payload["mechanism_id"] = mechanism_id
                if prediction_sha256 is not None:
                    _require_sha256(prediction_sha256, "external prediction")
                    trace_payload["prediction_sha256"] = prediction_sha256
                    trace_payload["observation_sha256"] = prediction_sha256
                for name in (
                    "raw_output_path",
                    "raw_output_sha256",
                    "prompt_paths",
                    "prompt_texts",
                    "prompt_sha256",
                    "patch",
                    "patch_sha256",
                    "structural_valid",
                    "failure_reason",
                    "mechanism",
                    "condition_id",
                    "candidate_consumed",
                    "candidate_bundle_sha256",
                    "candidate_revision_id",
                    "candidate_prompt",
                    "candidate_prompt_sha256",
                    "compiled_artifact_sha256",
                    "model_identity_sha256",
                    "parent_harness_revision_id",
                    "parent_harness_bundle_sha256",
                    "parent_harness_prompt",
                    "parent_harness_prompt_sha256",
                ):
                    if name in model_output:
                        trace_payload[name] = model_output[name]
                appender.append_fact("external_trace", trace_payload)
            elif (
                trace_receipt.payload.get("model_receipt_id")
                != model_receipt.receipt_id
            ):
                raise ContractViolation("resume external trace identity mismatch")

            internal_receipt = existing_by_kind.get("internal_trace")
            internal_payload = model_output.get("internal_trace")
            if internal_receipt is None and internal_payload is not None:
                if not isinstance(internal_payload, Mapping):
                    raise ContractViolation("internal trace must be a mapping")
                internal_payload = dict(internal_payload)
                trace_path = internal_payload.get("artifact_path")
                trace_sha256 = internal_payload.get("artifact_sha256")
                if not isinstance(trace_path, str) or not trace_path:
                    raise ContractViolation("internal trace artifact path is missing")
                trace = Path(trace_path).resolve()
                if not trace.is_file():
                    raise ContractViolation("internal trace artifact is missing")
                _require_sha256(trace_sha256, "internal trace artifact")
                if _file_sha256(trace) != trace_sha256:
                    raise ContractViolation("internal trace artifact hash mismatch")
                try:
                    trace_payload = json.loads(trace.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    raise ContractViolation(
                        "internal trace artifact is invalid"
                    ) from error
                if not isinstance(trace_payload, Mapping):
                    raise ContractViolation("internal trace artifact must be an object")
                internal_prediction = trace_payload.get("prediction_sha256")
                _require_sha256(internal_prediction, "internal prediction")
                mechanism = internal_payload.get(
                    "mechanism_id", plan.metadata.get("mechanism_id")
                )
                if not isinstance(mechanism, str) or not mechanism.strip():
                    raise ContractViolation("internal trace mechanism is missing")
                internal_payload.update(
                    {
                        "arm": plan.arm,
                        "task_revision_id": plan.task.revision_id,
                        "mechanism_id": mechanism,
                        "model_receipt_id": model_receipt.receipt_id,
                        "artifact_path": str(trace),
                        "artifact_sha256": trace_sha256,
                        "prediction_sha256": internal_prediction,
                        "observation_sha256": internal_prediction,
                    }
                )
                appender.append_fact("internal_trace", internal_payload)
            elif internal_receipt is not None and internal_payload is None:
                raise ContractViolation("resume internal trace identity mismatch")

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

            execution_mode = plan.metadata.get("execution_mode", "full")
            if execution_mode == "model-only-prescreen":
                appender.append_control(
                    "execution_terminal",
                    {
                        "status": "completed",
                        "execution_mode": "model-only-prescreen",
                    },
                )
                return appender.result("completed")
            if execution_mode != "full":
                raise ContractViolation("unsupported execution mode")

            native_receipt = existing_by_kind.get("native_evaluation")
            if native_receipt is None:
                native_payload = dict(
                    self._native_evaluator.evaluate(plan, workspace, model_output)
                )
                native_prediction = native_payload.get(
                    "prediction_sha256", native_payload.get("patch_sha256")
                )
                if native_prediction is not None:
                    _require_sha256(native_prediction, "native prediction")
                    external_prediction = _model_prediction_sha256(model_output)
                    if (
                        external_prediction is None
                        or native_prediction != external_prediction
                    ):
                        raise ContractViolation(
                            "native prediction does not match model artifact"
                        )
                    native_payload["prediction_sha256"] = native_prediction
                native_payload.update(native_execution_identity(plan))
                native_payload.update(
                    {
                        "model_receipt_id": model_receipt.receipt_id,
                        "model_artifact_sha256": model_receipt.artifact_sha256,
                    }
                )
                for name in (
                    "parent_harness_revision_id",
                    "parent_harness_bundle_sha256",
                    "parent_harness_prompt_sha256",
                ):
                    if name in model_output:
                        native_payload[name] = model_output[name]
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
            if model_receipt is None:
                raise ContractViolation(
                    "evaluator infrastructure error occurred before model receipt"
                ) from error
            appender.append_fact(
                "native_evaluation",
                {
                    **native_execution_identity(plan),
                    "model_receipt_id": model_receipt.receipt_id,
                    "model_artifact_sha256": model_receipt.artifact_sha256,
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
        sequence = self._next_sequence
        artifact, artifact_sha256, receipt_id = runtime_receipt_identity(
            campaign_id=self._plan.campaign_id,
            plan_id=self._plan.plan_id,
            sequence=sequence,
            kind=kind,
            payload=payload,
        )
        receipt = Receipt(
            receipt_id=receipt_id,
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


def runtime_receipt_identity(
    *,
    campaign_id: str,
    plan_id: str,
    sequence: int,
    kind: str,
    payload: Mapping[str, Any],
) -> tuple[bytes, str, str]:
    """Return the canonical artifact bytes, digest, and Runtime receipt ID."""

    artifact = canonical_json(payload).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    identity = canonical_json(
        {
            "campaign_id": campaign_id,
            "plan_id": plan_id,
            "sequence": sequence,
            "kind": kind,
            "artifact_sha256": artifact_sha256,
        }
    ).encode("utf-8")
    return (
        artifact,
        artifact_sha256,
        "receipt-" + hashlib.sha256(identity).hexdigest(),
    )


def _validated_cost(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractViolation("model cost must be numeric")
    numeric = float(value)
    if numeric < 0 or not math.isfinite(numeric):
        raise ContractViolation("model cost must be finite and non-negative")
    return numeric


def _require_sha256(value: object, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractViolation(f"{field} must be a literal SHA-256")


def _model_prediction_sha256(model_output: Mapping[str, Any]) -> str | None:
    patch = model_output.get("patch")
    patch_sha256 = model_output.get("patch_sha256")
    if patch is None and patch_sha256 is None:
        return None
    if not isinstance(patch, str):
        raise ContractViolation("model prediction patch must be literal text")
    _require_sha256(patch_sha256, "model prediction")
    derived = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    if derived != patch_sha256:
        raise ContractViolation("model prediction artifact hash mismatch")
    return derived


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_execution_identity(plan: ExecutionPlan) -> dict[str, Any]:
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
