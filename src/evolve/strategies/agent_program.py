"""Candidate-DAG tournament planning for complete AgentProgram revisions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolve.agent_program import (
    AgentProgramRevision,
    AgentProgramViolation,
    TournamentDecision,
)
from evolve.contracts import (
    Claim,
    ClaimClassification,
    ClaimGrade,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
    content_sha256,
)
from evolve.evidence import EvidenceGraph, IntegrityError, ReceiptConflict, ReceiptStore

from .base import (
    StrategyContext,
    StrategyDecision,
    StrategyResult,
    StrategyStatus,
    StrategyViolation,
    advisory_decision,
    interpretation_inputs,
)


class AgentProgramSearchStrategy:
    strategy_id = "agent-program-search-v3"
    status = StrategyStatus.NOT_YET_LIVE

    def __init__(self, *, execution_profile: str | None = None) -> None:
        if execution_profile not in {None, "fixture", "live"}:
            raise StrategyViolation("unsupported AgentProgram execution profile")
        self.execution_profile = execution_profile
        self.status = (
            StrategyStatus.LIVE
            if execution_profile in {"fixture", "live"}
            else StrategyStatus.NOT_YET_LIVE
        )

    def plan(self, context: StrategyContext) -> tuple[ExecutionPlan, ...]:
        required = {
            "parent_revision_id",
            "candidate_revision_ids",
            "tournament_id",
        }
        optional = {"generation_config"}
        if self.execution_profile in {"fixture", "live"}:
            required.update({"execution_profile", "revision_roots"})
        if self.execution_profile == "live":
            required.update(
                {"claim_evidence_graph_root", "claim_receipt_store_root"}
            )
        unknown = set(context.inputs) - required - optional
        missing = required - set(context.inputs)
        if missing or unknown:
            raise StrategyViolation(
                f"agent search inputs invalid; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        candidate_ids = context.inputs["candidate_revision_ids"]
        if not isinstance(candidate_ids, tuple) or not all(
            isinstance(item, str) for item in candidate_ids
        ):
            raise StrategyViolation("candidate_revision_ids must be a tuple of text")
        generation = context.inputs.get("generation_config")
        if generation is not None and not isinstance(generation, Mapping):
            raise StrategyViolation("generation_config must be a mapping")
        revisions: Mapping[str, AgentProgramRevision] | None = None
        if self.execution_profile in {"fixture", "live"}:
            if context.inputs.get("execution_profile") != self.execution_profile:
                raise StrategyViolation("AgentProgram execution profile identity mismatch")
            roots = context.inputs["revision_roots"]
            if not isinstance(roots, Mapping):
                raise StrategyViolation("revision_roots must be a mapping")
            expected_revisions = (
                str(context.inputs["parent_revision_id"]),
                *candidate_ids,
            )
            if set(roots) != set(expected_revisions):
                raise StrategyViolation("AgentProgram revision roots are incomplete")
            try:
                revisions = {
                    revision_id: AgentProgramRevision.load(
                        Path(str(roots[revision_id])).resolve()
                    )
                    for revision_id in expected_revisions
                }
            except AgentProgramViolation as error:
                raise StrategyViolation("AgentProgram revision is invalid") from error
            if any(
                revision.revision_id != revision_id
                for revision_id, revision in revisions.items()
            ):
                raise StrategyViolation("AgentProgram revision root identity mismatch")
            program_ids = {revision.program_id for revision in revisions.values()}
            if len(program_ids) != 1:
                raise StrategyViolation("tournament mixes AgentPrograms")
            parent_revision_id = expected_revisions[0]
            if any(
                revisions[revision_id].parent_revision_id != parent_revision_id
                for revision_id in candidate_ids
            ):
                raise StrategyViolation("candidate parent lineage mismatch")
        return self._plans(
            campaign_id=context.campaign_id,
            task=context.task,
            parent_revision_id=str(context.inputs["parent_revision_id"]),
            candidate_revision_ids=candidate_ids,
            tournament_id=str(context.inputs["tournament_id"]),
            model=context.model,
            context_policy_id=context.context_policy_id,
            tool_policy_id=context.tool_policy_id,
            observer_policy_ids=context.observer_policy_ids,
            limits=context.limits,
            generation_config=generation,
            revisions=revisions,
            execution_profile=self.execution_profile,
        )

    def build_plans(
        self,
        *,
        campaign_id: str,
        task: TaskRevision,
        parent_revision_id: str,
        candidate_revision_ids: tuple[str, ...],
        tournament_id: str,
        model: ModelIdentity,
        context_policy_id: str,
        tool_policy_id: str,
        observer_policy_ids: tuple[str, ...],
        limits: ExecutionLimits,
        generation_config: Mapping[str, Any] | None = None,
        execution_profile: str | None = None,
        revision_roots: Mapping[str, str] | None = None,
    ) -> tuple[ExecutionPlan, ...]:
        return self.plan(
            StrategyContext(
                campaign_id=campaign_id,
                task=task,
                model=model,
                context_policy_id=context_policy_id,
                tool_policy_id=tool_policy_id,
                observer_policy_ids=observer_policy_ids,
                limits=limits,
                inputs={
                    "parent_revision_id": parent_revision_id,
                    "candidate_revision_ids": candidate_revision_ids,
                    "tournament_id": tournament_id,
                    "generation_config": generation_config or {},
                    **(
                        {
                            "execution_profile": execution_profile,
                            "revision_roots": dict(revision_roots or {}),
                        }
                        if execution_profile is not None
                        else {}
                    ),
                },
            )
        )

    def _plans(
        self,
        *,
        campaign_id: str,
        task: TaskRevision,
        parent_revision_id: str,
        candidate_revision_ids: tuple[str, ...],
        tournament_id: str,
        model: ModelIdentity,
        context_policy_id: str,
        tool_policy_id: str,
        observer_policy_ids: tuple[str, ...],
        limits: ExecutionLimits,
        generation_config: Mapping[str, Any] | None = None,
        revisions: Mapping[str, AgentProgramRevision] | None = None,
        execution_profile: str | None = None,
    ) -> tuple[ExecutionPlan, ...]:
        if task.cohort is not Cohort.FEEDBACK:
            raise StrategyViolation(
                "AgentProgram search is restricted to feedback tasks"
            )
        if not candidate_revision_ids:
            raise StrategyViolation("tournament requires at least one candidate")
        revision_order = (parent_revision_id, *candidate_revision_ids)
        if len(set(revision_order)) != len(revision_order):
            raise StrategyViolation("tournament revisions must be unique")
        generation = dict(generation_config or {})
        tournament_config_sha256 = content_sha256(
            {
                "tournament_id": tournament_id,
                "parent_revision_id": parent_revision_id,
                "candidate_revision_ids": list(candidate_revision_ids),
                "generation_config": generation,
                "execution_profile": execution_profile,
            }
        )
        plans: list[ExecutionPlan] = []
        for position, revision_id in enumerate(revision_order):
            arm = "search-parent" if position == 0 else "candidate"
            identity = canonical_json(
                (campaign_id, task.revision_id, tournament_id, revision_id)
            ).encode()
            metadata: dict[str, Any] = {
                "tournament_id": tournament_id,
                "parent_revision_id": None
                if position == 0
                else parent_revision_id,
                "dag_position": position,
                "generation_config": generation,
            }
            if revisions is not None:
                revision = revisions[revision_id]
                metadata.update(
                    {
                        "execution_profile": execution_profile,
                        "program_bundle_sha256": revision.bundle_sha256,
                        "program_prompt_sha256": revision.artifact_hash(
                            "PROGRAM-PROMPT.txt"
                        ),
                        "program_context_sha256": revision.artifact_hash(
                            "CONTEXT.json"
                        ),
                        "program_tool_policy_sha256": revision.artifact_hash(
                            "TOOL-POLICY.json"
                        ),
                        "program_capabilities_sha256": revision.artifact_hash(
                            "CAPABILITIES.json"
                        ),
                        "tournament_config_sha256": tournament_config_sha256,
                    }
                )
            plans.append(
                ExecutionPlan(
                    plan_id=f"plan-{hashlib.sha256(identity).hexdigest()[:24]}",
                    campaign_id=campaign_id,
                    strategy_id=self.strategy_id,
                    task=task,
                    candidate_revision_id=revision_id,
                    arm=arm,
                    model=model,
                    context_policy_id=context_policy_id,
                    tool_policy_id=tool_policy_id,
                    observer_policy_ids=observer_policy_ids,
                    native_evaluator_id=task.evaluator_id,
                    limits=limits,
                    holdout_scope="feedback-only",
                    metadata=metadata,
                )
            )
        return tuple(plans)

    def interpret(
        self,
        context: StrategyContext | Sequence[Receipt],
        receipts: Sequence[Receipt] | None = None,
    ) -> StrategyResult:
        campaign_id, normalized = interpretation_inputs(context, receipts)
        if self.execution_profile == "live" and isinstance(context, StrategyContext):
            plans = self.plan(context)
            plan_ids = {plan.plan_id for plan in plans}
            if any(receipt.plan_id not in plan_ids for receipt in normalized):
                raise StrategyViolation(
                    "live tournament receipt does not match a planned participant"
                )
            kinds_by_plan = {
                plan.plan_id: {
                    receipt.kind
                    for receipt in normalized
                    if receipt.plan_id == plan.plan_id
                }
                for plan in plans
            }
            required = {
                "workspace",
                "model",
                "native_evaluation",
                "execution_terminal",
            }
            complete = tuple(
                plan.plan_id
                for plan in plans
                if required.issubset(kinds_by_plan[plan.plan_id])
            )
            return StrategyResult(
                strategy_id=self.strategy_id,
                campaign_id=campaign_id,
                receipt_ids=tuple(receipt.receipt_id for receipt in normalized),
                observations={
                    "execution_scope": "live",
                    "participant_revision_ids": tuple(
                        plan.candidate_revision_id for plan in plans
                    ),
                    "complete_plan_ids": complete,
                    "plan_receipt_counts": {
                        plan_id: sum(
                            receipt.plan_id == plan_id for receipt in normalized
                        )
                        for plan_id in kinds_by_plan
                    },
                },
            )
        return StrategyResult(
            strategy_id=self.strategy_id,
            campaign_id=campaign_id,
            receipt_ids=tuple(receipt.receipt_id for receipt in normalized),
            observations={"tournament_receipt_count": len(normalized)},
        )

    def next_action(
        self,
        context: StrategyContext,
        claims: Sequence[Claim],
        *,
        decision: TournamentDecision | None = None,
    ) -> StrategyDecision:
        if self.execution_profile == "fixture":
            if claims:
                raise StrategyViolation("fixture tournament cannot consume gain claims")
            if decision is None:
                return advisory_decision(
                    strategy_id=self.strategy_id,
                    status=self.status,
                    action="await-tournament-decision",
                    reason="fixture executions require a hash-bound tournament decision",
                    claims=(),
                )
            if (
                decision.execution_scope != "fixture"
                or decision.tournament_id != context.inputs.get("tournament_id")
                or decision.parent_revision_id
                != context.inputs.get("parent_revision_id")
                or tuple(decision.participant_revision_ids[1:])
                != context.inputs.get("candidate_revision_ids")
            ):
                raise StrategyViolation("fixture tournament decision identity mismatch")
            action = (
                "advance-search-parent" if decision.advanced else "retain-search-parent"
            )
            return advisory_decision(
                strategy_id=self.strategy_id,
                status=self.status,
                action=action,
                reason=(
                    f"decision={decision.decision_sha256};"
                    f"winner={decision.winner_revision_id};"
                    "scope=fixture;native_gain_claimed=false"
                ),
                claims=(),
            )
        if self.execution_profile == "live":
            return self._live_next_action(context, claims)
        return advisory_decision(
            strategy_id=self.strategy_id,
            status=self.status,
            action="await-tournament-authority",
            reason="authoritative tournament selection is not yet wired",
            claims=claims,
        )

    def _live_next_action(
        self, context: StrategyContext, claims: Sequence[Claim]
    ) -> StrategyDecision:
        participants = (
            str(context.inputs.get("parent_revision_id")),
            *tuple(context.inputs.get("candidate_revision_ids", ())),
        )
        ordered_claims = tuple(sorted(claims, key=lambda claim: claim.claim_id))
        candidate_ids = {claim.candidate_id for claim in ordered_claims}
        claim_ids = tuple(claim.claim_id for claim in ordered_claims)
        if (
            not ordered_claims
            or len(set(claim_ids)) != len(claim_ids)
            or len(ordered_claims) != len(participants)
            or candidate_ids != set(participants)
            or any(claim.grade is ClaimGrade.E0 for claim in ordered_claims)
            or not self._claims_are_authority_bound(context, ordered_claims)
        ):
            return advisory_decision(
                strategy_id=self.strategy_id,
                status=self.status,
                action="await-tournament-authority",
                reason="live tournament requires complete participant Claims",
                claims=ordered_claims,
            )
        plans = self.plan(context)
        weights = {
            ClaimClassification.GAIN: 2,
            ClaimClassification.NEUTRAL: 0,
            ClaimClassification.REGRESSION: -2,
            ClaimClassification.INFRA_FAILURE: -1,
        }
        scores = {
            revision_id: sum(
                weights[claim.classification]
                for claim in ordered_claims
                if claim.candidate_id == revision_id
            )
            for revision_id in participants
        }
        highest = max(scores.values())
        tied = tuple(
            revision_id
            for revision_id in participants
            if scores[revision_id] == highest
        )
        parent_revision_id = participants[0]
        winner = (
            parent_revision_id
            if parent_revision_id in tied
            else min(tied)
        )
        decision_sha256 = content_sha256(
            {
                "tournament_id": context.inputs.get("tournament_id"),
                "execution_scope": "live",
                "parent_revision_id": parent_revision_id,
                "participant_revision_ids": list(participants),
                "program_bundle_sha256": [
                    [plan.candidate_revision_id, plan.metadata["program_bundle_sha256"]]
                    for plan in plans
                ],
                "claim_ids": list(claim_ids),
                "claim_sha256": [
                    [claim.claim_id, claim.content_sha256]
                    for claim in ordered_claims
                ],
                "scores": [[revision_id, scores[revision_id]] for revision_id in participants],
                "winner_revision_id": winner,
            }
        )
        return advisory_decision(
            strategy_id=self.strategy_id,
            status=self.status,
            action=(
                "advance-search-parent"
                if winner != parent_revision_id
                else "reject-candidates"
            ),
            reason=(
                f"decision={decision_sha256};winner={winner};"
                "scope=live;promotion_claimed=false"
            ),
            claims=ordered_claims,
        )

    def _claims_are_authority_bound(
        self, context: StrategyContext, claims: tuple[Claim, ...]
    ) -> bool:
        graph_value = context.inputs.get("claim_evidence_graph_root")
        store_value = context.inputs.get("claim_receipt_store_root")
        if (
            not isinstance(graph_value, str)
            or not graph_value.strip()
            or not isinstance(store_value, str)
            or not store_value.strip()
        ):
            return False
        graph_root = Path(graph_value)
        store_root = Path(store_value)
        if (
            not (graph_root / "evidence.jsonl").is_file()
            or not (graph_root / "claims.jsonl").is_file()
            or not (store_root / "receipts.jsonl").is_file()
        ):
            return False
        try:
            store = ReceiptStore(store_root)
            graph = EvidenceGraph.rebuild(graph_root, store)
            persisted = tuple(sorted(graph.latest_claims(), key=lambda row: row.claim_id))
            if persisted != claims or any(
                left.content_sha256 != right.content_sha256
                for left, right in zip(persisted, claims, strict=True)
            ):
                return False
            evidence_by_id = {
                row.evidence_id: row for row in graph.list_evidence()
            }
            receipts_by_id = {
                row.receipt_id: row for row in store.list_receipts()
            }
            plans_by_revision = {
                plan.candidate_revision_id: plan for plan in self.plan(context)
            }
            for claim in claims:
                plan = plans_by_revision.get(claim.candidate_id)
                if plan is None or not claim.evidence_ids:
                    return False
                selected = tuple(
                    evidence_by_id.get(evidence_id)
                    for evidence_id in claim.evidence_ids
                )
                if any(envelope is None for envelope in selected):
                    return False
                native_receipts: dict[str, Receipt] = {}
                for envelope in selected:
                    if envelope is None:  # narrowed above
                        return False
                    if (
                        envelope.grade is ClaimGrade.E0
                        or envelope.payload.get("plan_id") != plan.plan_id
                        or not envelope.receipt_ids
                    ):
                        return False
                    for receipt_id in envelope.receipt_ids:
                        receipt = receipts_by_id.get(receipt_id)
                        if receipt is None or receipt.plan_id != plan.plan_id:
                            return False
                        if receipt.kind == "native_evaluation":
                            native_receipts[receipt.receipt_id] = receipt
                if len(native_receipts) != 1:
                    return False
                native_receipt = next(iter(native_receipts.values()))
                model_receipt_id = native_receipt.payload.get("model_receipt_id")
                if not isinstance(model_receipt_id, str):
                    return False
                model_receipt = receipts_by_id.get(model_receipt_id)
                if (
                    model_receipt is None
                    or model_receipt.kind != "model"
                    or model_receipt.plan_id != plan.plan_id
                    or native_receipt.payload.get("model_artifact_sha256")
                    != model_receipt.artifact_sha256
                    or model_receipt.payload.get("program_bundle_sha256")
                    != plan.metadata.get("program_bundle_sha256")
                ):
                    return False
                revision_id = model_receipt.payload.get("revision_id")
                candidate_revision_id = model_receipt.payload.get(
                    "candidate_revision_id"
                )
                if (
                    revision_id is None
                    and candidate_revision_id is None
                ) or (
                    revision_id is not None
                    and revision_id != plan.candidate_revision_id
                ) or (
                    candidate_revision_id is not None
                    and candidate_revision_id != plan.candidate_revision_id
                ):
                    return False
        except (OSError, ValueError, ContractViolation, IntegrityError, ReceiptConflict):
            return False
        return True
