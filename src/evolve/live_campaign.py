"""Runtime-backed orchestration for a bounded feedback Skill campaign.

This module composes existing v3 authorities.  It does not dispatch adapters,
manufacture native facts, interpret evaluator results, or activate assets on its
own: plans go through :class:`ExecutionRuntime`, outcomes go through the native
Observer and ClaimEngine, and any approved Capability remains inactive.
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
    MechanismPrediction,
    ModelIdentity,
    Receipt,
    TaskRevision,
)
from evolve.evidence import (
    CandidateEvidenceState,
    ClaimEngine,
    EvidenceGradeMachine,
    ReceiptStore,
    build_matched_counterfactual_pair,
)
from evolve.governance import (
    GateDecision,
    GovernanceService,
    PromotionDecision,
    PromotionDecisionLog,
)
from evolve.kernel import (
    CampaignController,
    CampaignSnapshot,
    CampaignStatus,
    DurableCostLedger,
    WorkItemSnapshot,
)
from evolve.observers import ObserverHub, TrustedJLensReceiptIssuer
from evolve.registry import (
    CandidateRecord,
    CandidateRegistry,
    CapabilityRecord,
    CapabilityRegistry,
    RejectedRecord,
    RejectedRegistry,
)
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
    mechanism_id: str | None = None
    mechanism_prediction: MechanismPrediction | None = None
    human_approval: bool = False
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
    candidate: CandidateRecord
    evidence_state: CandidateEvidenceState
    promotion_decision: PromotionDecision
    capability: CapabilityRecord | None
    rejected: RejectedRecord | None
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
    evidence_grade_machine: EvidenceGradeMachine,
    governance_service: GovernanceService,
    promotion_decision_log: PromotionDecisionLog,
    candidate_registry: CandidateRegistry,
    capability_registry: CapabilityRegistry,
    rejected_registry: RejectedRegistry,
    report_root: str | Path,
    trusted_observation_issuer: TrustedJLensReceiptIssuer | None = None,
    budget_ledger: DurableCostLedger | None = None,
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
            plan_metadata=dict(spec.task_execution_metadata.get(task.revision_id, {})),
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
        execution = runtime.execute(
            plan,
            authorization,
            mechanism_prediction=spec.mechanism_prediction,
        )
        executions.append(execution)
        _validate_candidate_isolation(
            plans=(plan,),
            executions=(execution,),
            candidate_revision_id=spec.candidate_revision_id,
            candidate_bundle_sha256=spec.candidate_artifact_sha256,
        )
        if not terminal_replay:
            item = _work_item(controller.snapshot(), plan.plan_id)
            if item.status == "pending":
                controller.record_result(
                    plan.plan_id,
                    actual_cost_cny=_execution_cost(execution),
                    actual_model_calls=_execution_model_calls(execution),
                    succeeded=execution.status == "completed",
                )

    if trusted_observation_issuer is not None:
        if spec.mechanism_prediction is None:
            raise ContractViolation(
                "trusted observation issuer requires a mechanism prediction"
            )
        for plan, execution in zip(plans, executions, strict=True):
            if execution.status != "completed":
                raise ContractViolation(
                    "trusted observation requires a completed model execution"
                )
            plan_receipts = receipt_store.receipts_for(plan.plan_id)
            prior_trusted = tuple(
                receipt
                for receipt in plan_receipts
                if receipt.kind == "trusted_jlens_observation"
            )
            if len(prior_trusted) > 1:
                raise ContractViolation(
                    "plan has multiple trusted JLens observation receipts"
                )
            if prior_trusted:
                stored = prior_trusted[0]
            else:
                receipt, artifact = trusted_observation_issuer.issue(
                    plan,
                    plan_receipts,
                )
                stored = receipt_store.append(receipt, artifact)
            trusted = tuple(
                envelope
                for envelope in observer_hub.observe(stored)
                if envelope.observer_id == "trusted-jlens-v1"
            )
            if len(trusted) != 1:
                raise ContractViolation(
                    "trusted observation issuer requires trusted-jlens-v1 observer"
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

    claim_rows: list[Claim] = []
    for baseline, taught in plan_pairs:
        baseline_native = _native_evidence(baseline, receipt_store, observer_hub)
        taught_native = _native_evidence(taught, receipt_store, observer_hub)
        native_pair = align_native_pair(baseline_native, taught_native)
        evaluator_failed = any(
            envelope.payload.get("evaluator_error") not in (None, "")
            for envelope in (baseline_native, taught_native)
        )
        counterfactual_pair = None
        if not evaluator_failed:
            counterfactual_pair = build_matched_counterfactual_pair(
                candidate_id=spec.candidate_id,
                candidate_revision_id=spec.candidate_revision_id,
                candidate_bundle_sha256=spec.candidate_artifact_sha256,
                baseline_model_receipt=_only_plan_receipt(
                    baseline, receipt_store, "model"
                ),
                baseline_external_evidence=_external_evidence(
                    baseline, receipt_store, observer_hub
                ),
                baseline_native_evidence=baseline_native,
                taught_model_receipt=_only_plan_receipt(taught, receipt_store, "model"),
                taught_external_evidence=_external_evidence(
                    taught, receipt_store, observer_hub
                ),
                taught_native_evidence=taught_native,
            )
        claim_rows.append(
            claim_engine.classify_pair(
                spec.candidate_id,
                native_pair,
                counterfactual_pair=counterfactual_pair,
            )
        )
    claims = tuple(claim_rows)
    candidate = CandidateRecord(
        candidate_id=spec.candidate_id,
        revision_id=spec.candidate_revision_id,
        candidate_kind=spec.candidate_kind,
        source_claim_ids=tuple(claim.claim_id for claim in claims),
        artifact_sha256=spec.candidate_artifact_sha256,
    )
    candidate_registry.append(candidate)
    evidence_state = evidence_grade_machine.aggregate(
        spec.candidate_id,
        task_projects={task.revision_id: task.project for task in task_rows},
        mechanism_id=spec.mechanism_id,
    )

    snapshot = controller.snapshot()
    campaign_receipts = tuple(
        receipt
        for receipt in receipt_store.list_receipts()
        if receipt.campaign_id == spec.campaign_id
    )
    if not campaign_receipts:
        raise ContractViolation("completed campaign has no receipts")
    runtime_spend_cny = round(
        sum(
            float(receipt.payload.get("cost_cny", 0))
            for receipt in campaign_receipts
            if receipt.kind == "cost"
        ),
        8,
    )
    budget_ledger_event_count = 0
    budget_ledger_head_sha256: str | None = None
    budget_limit: float | None = None
    if budget_ledger is not None:
        ledger_events = budget_ledger.events()
        ledger_snapshot = budget_ledger.snapshot()
        budget_spent_cny = ledger_snapshot.spent_cost_cny
        budget_limit = ledger_snapshot.max_cost_cny
        budget_ledger_event_count = len(ledger_events)
        budget_ledger_head_sha256 = str(ledger_events[-1]["event_sha256"])
        budget_integrity_status = (
            "validated" if 0 <= budget_spent_cny <= budget_limit else "exceeded"
        )
    else:
        budget_spent_cny = runtime_spend_cny
        budget_integrity_status = "recorded"
    promotion_decision = governance_service.decide(
        candidate=candidate,
        evidence=evidence_state,
        claims=claims,
        human_approval=spec.human_approval,
        decided_at=max(receipt.created_at for receipt in campaign_receipts),
        log=promotion_decision_log,
    )
    capability: CapabilityRecord | None = None
    rejected: RejectedRecord | None = None
    if promotion_decision.gate_decision is GateDecision.APPROVED:
        capability = governance_service.to_capability(
            candidate=candidate,
            decision=promotion_decision,
            capability_id=spec.candidate_id,
        )
        capability_registry.append(capability)
    elif promotion_decision.gate_decision is GateDecision.REJECTED:
        rejected = governance_service.to_rejected(
            candidate=candidate,
            decision=promotion_decision,
        )
        rejected_registry.append(rejected)

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
            "candidate_id": candidate.candidate_id,
            "candidate_revision_id": candidate.revision_id,
            "evidence_grade_reached": str(evidence_state.grade),
            "e3_eligible": evidence_state.e3_eligible,
            "e2_lineage_status": (
                "validated"
                if claims
                and all(
                    claim.counterfactual_pair_sha256 is not None
                    and len(claim.counterfactual_receipt_ids) == 6
                    for claim in claims
                )
                else "incomplete"
            ),
            "e2_counterfactual_receipt_count": sum(
                len(claim.counterfactual_receipt_ids) for claim in claims
            ),
            "e3_trust_status": (
                "validated_observed"
                if evidence_state.e3_eligible
                else "guarded_not_observed"
            ),
            "e3_trusted_evidence_count": len(evidence_state.prediction_evidence_ids)
            // 3,
            "promotion_status": str(promotion_decision.gate_decision),
            "governance_status": str(promotion_decision.gate_decision),
            "promotion_decision_id": promotion_decision.decision_id,
            "capability_id": (
                capability.capability_id if capability is not None else None
            ),
            "capability_revision_id": (
                capability.revision_id if capability is not None else None
            ),
            "capability_active": (
                capability.active if capability is not None else False
            ),
            "rejected_revision_id": (
                rejected.revision_id if rejected is not None else None
            ),
            "implementation_status": "complete",
            "causal_pipeline_status": (
                "validated"
                if all(execution.status == "completed" for execution in executions)
                else "not_validated"
            ),
            "empirical_gain_status": evidence_state.classification,
            "actual_api_spend_cny": budget_spent_cny,
            "api_budget_limit_cny": budget_limit,
            "api_budget_remaining_cny": (
                round(budget_limit - budget_spent_cny, 8)
                if budget_limit is not None
                else None
            ),
            "budget_spent_cny": budget_spent_cny,
            "budget_integrity_status": budget_integrity_status,
            "budget_ledger_event_count": budget_ledger_event_count,
            "budget_ledger_head_sha256": budget_ledger_head_sha256,
            "full_v3_release_status": "incomplete",
            "unresolved_findings": [
                *(
                    []
                    if evidence_state.grade.value == "E3"
                    else ["E3 evidence was not reached by this feedback campaign"]
                ),
                "holdout/final release gates were intentionally not opened",
            ],
        },
    )
    report_paths = projector.write(report, Path(report_root))
    return LiveCampaignResult(
        plans=plans,
        executions=tuple(executions),
        claims=claims,
        candidate=candidate,
        evidence_state=evidence_state,
        promotion_decision=promotion_decision,
        capability=capability,
        rejected=rejected,
        snapshot=snapshot,
        report=report,
        report_paths=report_paths,
    )


def _validate_candidate_isolation(
    *,
    plans: tuple[ExecutionPlan, ...],
    executions: tuple[ExecutionResult, ...],
    candidate_revision_id: str,
    candidate_bundle_sha256: str,
) -> None:
    """Prove baseline isolation and taught consumption from immutable receipts."""

    if len(plans) != len(executions):
        raise ContractViolation("causal execution result count mismatch")
    for plan, execution in zip(plans, executions, strict=True):
        model_receipts = [row for row in execution.receipts if row.kind == "model"]
        if len(model_receipts) != 1:
            raise ContractViolation("causal execution requires one model receipt")
        payload = model_receipts[0].payload
        if plan.arm == "baseline":
            if (
                payload.get("candidate_consumed") is not False
                or payload.get("candidate_bundle_sha256") is not None
                or payload.get("candidate_revision_id") is not None
            ):
                raise ContractViolation("baseline consumed or exposed candidate state")
        elif plan.arm == "taught":
            if (
                payload.get("candidate_consumed") is not True
                or payload.get("candidate_bundle_sha256") != candidate_bundle_sha256
                or payload.get("candidate_revision_id") != candidate_revision_id
            ):
                raise ContractViolation("taught arm did not consume compiled candidate")
        else:
            raise ContractViolation("causal campaign requires paired arms")


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
    if set(spec.task_execution_metadata) != {task.revision_id for task in tasks}:
        raise ContractViolation("live Skill campaign metadata must cover every task")
    if spec.mechanism_prediction is not None and (
        spec.mechanism_prediction.candidate_revision_id != spec.candidate_revision_id
        or spec.mechanism_prediction.mechanism_id != spec.mechanism_id
    ):
        raise ContractViolation(
            "mechanism prediction does not match campaign candidate"
        )
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
    return float(cost_receipts[0].payload["cost_cny"]) if cost_receipts else 0.0


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


def _external_evidence(
    plan: ExecutionPlan,
    receipt_store: ReceiptStore,
    observer_hub: ObserverHub,
) -> EvidenceEnvelope:
    receipt = _only_plan_receipt(plan, receipt_store, "external_trace")
    evidence = tuple(
        envelope
        for envelope in observer_hub.observe(receipt)
        if envelope.observer_id == "external-trace-v1"
    )
    if len(evidence) != 1:
        raise ContractViolation(
            f"plan {plan.plan_id} requires exactly one external-trace-v1 observation"
        )
    return evidence[0]


def _only_plan_receipt(
    plan: ExecutionPlan,
    receipt_store: ReceiptStore,
    kind: str,
) -> Receipt:
    receipts = tuple(
        receipt
        for receipt in receipt_store.receipts_for(plan.plan_id)
        if receipt.kind == kind
    )
    if len(receipts) != 1:
        raise ContractViolation(
            f"plan {plan.plan_id} requires exactly one {kind} Runtime receipt"
        )
    return receipts[0]


__all__ = [
    "LiveCampaignResult",
    "LiveCampaignSpec",
    "run_skill_paired_campaign",
]
