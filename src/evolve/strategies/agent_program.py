"""Candidate-DAG tournament planning for complete AgentProgram revisions."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from evolve.contracts import (
    Cohort,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
)

from .base import StrategyInterpretation, StrategyViolation


class AgentProgramSearchStrategy:
    strategy_id = "agent-program-search-v3"

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

    def interpret(self, receipts: Sequence[Receipt]) -> StrategyInterpretation:
        return StrategyInterpretation(
            strategy_id=self.strategy_id,
            receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
            observations={"tournament_receipt_count": len(receipts)},
        )
