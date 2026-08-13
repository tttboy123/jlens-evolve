"""Runtime-backed orchestration for a bounded feedback Skill campaign.

This module composes existing v3 authorities.  It does not dispatch adapters,
manufacture native facts, interpret evaluator results, or activate assets on its
own: plans go through :class:`ExecutionRuntime`, outcomes go through the native
Observer and ClaimEngine, and the resulting Capability remains inactive.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evolve.alignment import align_native_pair
from evolve.contracts import (
    Authorization,
    Claim,
    Cohort,
    ContractViolation,
    EvidenceEnvelope,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    TaskRevision,
)
from evolve.evidence import ClaimEngine, ReceiptStore
from evolve.kernel import (
    CampaignController,
    CampaignSnapshot,
    CampaignStatus,
    WorkItemSnapshot,
)
from evolve.observers import ObserverHub
from evolve.registry import CapabilityRecord, CapabilityRegistry
from evolve.reporting import CampaignReportProjector, ReportPaths
from evolve.runtime import (
    ExecutionResult,
    ExecutionRuntime,
    ModelTransport,
    NativeEvaluator,
    WorkspaceManager,
)
from evolve.strategies import SkillPairedStrategy


@dataclass(frozen=True, slots=True)
class LiveCampaignSpec:
    """Frozen inputs shared by all six matched executions."""

    campaign_id: str
    baseline_revision_id: str
    candidate_id: str
    candidate_revision_id: str
    candidate_kind: str
    candidate_artifact_sha256: str
    model: ModelIdentity
    context_policy_id: str
    tool_policy_id: str
    observer_policy_ids: tuple[str, ...]
    limits: ExecutionLimits
    final_commit_sha: str
    generation_config: Mapping[str, Any] = field(default_factory=dict)
    task_execution_metadata: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    report_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LiveCampaignResult:
    """Rebuildable product projection returned by a completed orchestration."""

    plans: tuple[ExecutionPlan, ...]
    executions: tuple[ExecutionResult, ...]
    claims: tuple[Claim, ...]
    capability: CapabilityRecord
    snapshot: CampaignSnapshot
    report: Mapping[str, Any]
    report_paths: ReportPaths


def run_skill_paired_campaign(
    *,
    spec: LiveCampaignSpec,
    tasks: Iterable[TaskRevision],
    strategy: SkillPairedStrategy,
    controller: CampaignController,
    authorization: Authorization,
    model_transport: ModelTransport,
    workspace_manager: WorkspaceManager,
    native_evaluator: NativeEvaluator,
    receipt_store: ReceiptStore,
    observer_hub: ObserverHub,
    claim_engine: ClaimEngine,
    capability_registry: CapabilityRegistry,
    report_root: str | Path,
    clock: Callable[[], datetime] | None = None,
) -> LiveCampaignResult:
    """Run or idempotently replay one three-task feedback paired campaign.

    The function deliberately accepts the shared adapters rather than invoking
    them.  ``ExecutionRuntime`` remains the only I/O dispatch boundary.
    """

    task_rows = tuple(tasks)
    _validate_admission(
        spec=spec,
        tasks=task_rows,
        controller=controller,
        authorization=authorization,
        native_evaluator=native_evaluator,
    )
    plan_pairs = tuple(
        strategy.build_plans(
            campaign_id=spec.campaign_id,
            task=task,
            baseline_revision_id=spec.baseline_revision_id,
            taught_revision_id=spec.candidate_revision_id,
            model=spec.model,
            context_policy_id=spec.context_policy_id,
            tool_policy_id=spec.tool_policy_id,
            observer_policy_ids=spec.observer_policy_ids,
            limits=spec.limits,
            generation_config=spec.generation_config,
            plan_metadata=spec.task_execution_metadata.get(task.revision_id, {}),
        )
        for task in task_rows
    )
    for baseline, taught in plan_pairs:
        strategy.validate_matched_pair(baseline, taught)
    plans = tuple(plan for pair in plan_pairs for plan in pair)

    starting_snapshot = controller.snapshot()
    terminal_replay = starting_snapshot.status is CampaignStatus.COMPLETED
    if starting_snapshot.status in {
        CampaignStatus.FAILED,
        CampaignStatus.BLOCKED,
        CampaignStatus.CANCELLED,
        CampaignStatus.PARTIAL,
    }:
        raise ContractViolation(
            f"cannot resume campaign from terminal state {starting_snapshot.status}"
        )
    if terminal_replay:
        _validate_terminal_replay(starting_snapshot, plans, receipt_store)
    else:
        if starting_snapshot.status in {
            CampaignStatus.CREATED,
            CampaignStatus.PAUSED,
        }:
            controller.start()
        for plan in plans:
            controller.submit(
                plan,
                reserved_model_calls=1,
                remote=model_transport.remote,
            )

    runtime = ExecutionRuntime(
        model_transport=model_transport,
        workspace_manager=workspace_manager,
        native_evaluator=native_evaluator,
        observer_hub=observer_hub,
        receipt_sink=receipt_store,
        clock=clock,
    )
    executions: list[ExecutionResult] = []
    for plan in plans:
        execution = runtime.execute(plan, authorization)
        executions.append(execution)
        if not terminal_replay:
            item = _work_item(controller.snapshot(), plan.plan_id)
            if item.status == "pending":
                controller.record_result(
                    plan.plan_id,
                    actual_cost_cny=_execution_cost(execution),
                    actual_model_calls=_execution_model_calls(execution),
                    succeeded=execution.status == "completed",
                )

    if not terminal_replay:
        statuses = {execution.status for execution in executions}
        if statuses == {"completed"}:
            controller.finalize(CampaignStatus.COMPLETED)
        else:
            controller.mark_partial(
                "one or more Runtime executions did not complete: "
                + ", ".join(sorted(statuses))
            )

    claims = tuple(
        claim_engine.classify_pair(
            spec.candidate_id,
            align_native_pair(
                _native_evidence(baseline, receipt_store, observer_hub),
                _native_evidence(taught, receipt_store, observer_hub),
            ),
        )
        for baseline, taught in plan_pairs
    )
    capability = CapabilityRecord(
        capability_id=spec.candidate_id,
        revision_id=spec.candidate_revision_id,
        capability_kind=spec.candidate_kind,
        evidence_claim_ids=tuple(claim.claim_id for claim in claims),
        artifact_sha256=spec.candidate_artifact_sha256,
    )
    capability_registry.append(capability)

    snapshot = controller.snapshot()
    campaign_receipts = tuple(
        receipt
        for receipt in receipt_store.list_receipts()
        if receipt.campaign_id == spec.campaign_id
    )
    projector = CampaignReportProjector()
    report = projector.project(
        campaign_id=spec.campaign_id,
        receipts=campaign_receipts,
        claims=claims,
        final_commit_sha=spec.final_commit_sha,
        metadata={
            **dict(spec.report_metadata),
            "campaign_status": str(snapshot.status),
            "task_revision_ids": [task.revision_id for task in task_rows],
            "capability_id": capability.capability_id,
            "capability_revision_id": capability.revision_id,
            "capability_active": capability.active,
        },
    )
    report_paths = projector.write(report, Path(report_root))
    return LiveCampaignResult(
        plans=plans,
        executions=tuple(executions),
        claims=claims,
        capability=capability,
        snapshot=snapshot,
        report=report,
        report_paths=report_paths,
    )


def _validate_admission(
    *,
    spec: LiveCampaignSpec,
    tasks: tuple[TaskRevision, ...],
    controller: CampaignController,
    authorization: Authorization,
    native_evaluator: NativeEvaluator,
) -> None:
    if len(tasks) != 3 or any(task.cohort is not Cohort.FEEDBACK for task in tasks):
        raise ContractViolation(
            "live Skill campaign requires exactly three feedback tasks"
        )
    if len({task.revision_id for task in tasks}) != len(tasks):
        raise ContractViolation("live Skill campaign task revisions must be unique")
    if set(spec.task_execution_metadata) != {
        task.revision_id for task in tasks
    }:
        raise ContractViolation("live Skill campaign metadata must cover every task")
    snapshot = controller.snapshot()
    if snapshot.campaign_id != spec.campaign_id:
        raise ContractViolation("campaign spec does not match controller")
    if authorization.campaign_id != spec.campaign_id:
        raise ContractViolation("campaign spec does not match authorization")
    if snapshot.authorization_id != authorization.authorization_id:
        raise ContractViolation("controller authorization identity mismatch")
    if any(task.evaluator_id != native_evaluator.evaluator_id for task in tasks):
        raise ContractViolation("feedback task evaluator identity mismatch")


def _validate_terminal_replay(
    snapshot: CampaignSnapshot,
    plans: tuple[ExecutionPlan, ...],
    receipt_store: ReceiptStore,
) -> None:
    expected = {plan.plan_id: plan for plan in plans}
    actual = {item.plan_id: item for item in snapshot.work_items}
    if set(actual) != set(expected):
        raise ContractViolation("terminal checkpoint plan set does not match campaign")
    for plan_id, plan in expected.items():
        item = actual[plan_id]
        if item.status != "completed" or item.plan_sha256 != plan.content_sha256:
            raise ContractViolation("terminal checkpoint execution identity mismatch")
        if not any(
            receipt.kind == "execution_terminal"
            and receipt.payload.get("status") == "completed"
            for receipt in receipt_store.receipts_for(plan_id)
        ):
            raise ContractViolation("terminal checkpoint is missing Runtime receipts")


def _work_item(snapshot: CampaignSnapshot, plan_id: str) -> WorkItemSnapshot:
    try:
        return next(item for item in snapshot.work_items if item.plan_id == plan_id)
    except StopIteration as error:
        raise ContractViolation(
            f"campaign checkpoint is missing plan {plan_id}"
        ) from error


def _execution_cost(execution: ExecutionResult) -> float:
    cost_receipts = tuple(
        receipt for receipt in execution.receipts if receipt.kind == "cost"
    )
    if len(cost_receipts) > 1:
        raise ContractViolation("Runtime produced duplicate cost receipts")
    return (
        float(cost_receipts[0].payload["cost_cny"])
        if cost_receipts
        else 0.0
    )


def _execution_model_calls(execution: ExecutionResult) -> int:
    model_receipts = tuple(
        receipt for receipt in execution.receipts if receipt.kind == "model"
    )
    if len(model_receipts) > 1:
        raise ContractViolation("Runtime produced duplicate model receipts")
    return len(model_receipts)


def _native_evidence(
    plan: ExecutionPlan,
    receipt_store: ReceiptStore,
    observer_hub: ObserverHub,
) -> EvidenceEnvelope:
    native_receipts = tuple(
        receipt
        for receipt in receipt_store.receipts_for(plan.plan_id)
        if receipt.kind == "native_evaluation"
    )
    if len(native_receipts) != 1:
        raise ContractViolation(
            f"plan {plan.plan_id} requires exactly one native Runtime receipt"
        )
    native_evidence = tuple(
        envelope
        for envelope in observer_hub.observe(native_receipts[0])
        if envelope.observer_id == "native-v1"
    )
    if len(native_evidence) != 1:
        raise ContractViolation(
            f"plan {plan.plan_id} requires exactly one native-v1 observation"
        )
    return native_evidence[0]


__all__ = [
    "LiveCampaignResult",
    "LiveCampaignSpec",
    "run_skill_paired_campaign",
]
