"""Candidate-DAG tournament planning for complete AgentProgram revisions."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from evolve.contracts import (
    Claim,
    Cohort,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
)

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

    def plan(self, context: StrategyContext) -> tuple[ExecutionPlan, ...]:
        required = {
            "parent_revision_id",
            "candidate_revision_ids",
            "tournament_id",
        }
        optional = {"generation_config"}
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
    ) -> tuple[ExecutionPlan, ...]:
        if task.cohort is not Cohort.FEEDBACK:
            raise StrategyViolation(
                "AgentProgram search is restricted to feedback tasks"
            )
        if not candidate_revision_ids:
            raise StrategyViolation("tournament requires at least one candidate")
        revisions = (parent_revision_id, *candidate_revision_ids)
        if len(set(revisions)) != len(revisions):
            raise StrategyViolation("tournament revisions must be unique")
        generation = dict(generation_config or {})
        plans: list[ExecutionPlan] = []
        for position, revision_id in enumerate(revisions):
            arm = "search-parent" if position == 0 else "candidate"
            identity = canonical_json(
                (campaign_id, task.revision_id, tournament_id, revision_id)
            ).encode()
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
                    metadata={
                        "tournament_id": tournament_id,
                        "parent_revision_id": None
                        if position == 0
                        else parent_revision_id,
                        "dag_position": position,
                        "generation_config": generation,
                    },
                )
            )
        return tuple(plans)

    def interpret(
        self,
        context: StrategyContext | Sequence[Receipt],
        receipts: Sequence[Receipt] | None = None,
    ) -> StrategyResult:
        campaign_id, normalized = interpretation_inputs(context, receipts)
        return StrategyResult(
            strategy_id=self.strategy_id,
            campaign_id=campaign_id,
            receipt_ids=tuple(receipt.receipt_id for receipt in normalized),
            observations={"tournament_receipt_count": len(normalized)},
        )

    def next_action(
        self, context: StrategyContext, claims: Sequence[Claim]
    ) -> StrategyDecision:
        return advisory_decision(
            strategy_id=self.strategy_id,
            status=self.status,
            action="await-tournament-authority",
            reason="authoritative tournament selection is not yet wired",
            claims=claims,
        )
