"""Matched paired experiment strategy for external Skill candidates."""

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


class SkillPairedStrategy:
    strategy_id = "skill-paired-v3"

    def build_plans(
        self,
        *,
        campaign_id: str,
        task: TaskRevision,
        baseline_revision_id: str,
        taught_revision_id: str,
        model: ModelIdentity,
        context_policy_id: str,
        tool_policy_id: str,
        observer_policy_ids: tuple[str, ...],
        limits: ExecutionLimits,
        generation_config: Mapping[str, Any] | None = None,
        plan_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ExecutionPlan, ExecutionPlan]:
        if task.cohort is not Cohort.FEEDBACK:
            raise StrategyViolation(
                "Skill paired plans are restricted to feedback tasks"
            )
        generation = dict(generation_config or {})
        metadata = dict(plan_metadata or {})
        if "generation_config" in metadata:
            raise StrategyViolation("plan_metadata cannot replace generation_config")
        metadata["generation_config"] = generation

        def build(arm: str, candidate_revision_id: str) -> ExecutionPlan:
            return ExecutionPlan(
                plan_id=self._plan_id(
                    campaign_id, task.revision_id, candidate_revision_id, arm
                ),
                campaign_id=campaign_id,
                strategy_id=self.strategy_id,
                task=task,
                candidate_revision_id=candidate_revision_id,
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

        plans = (
            build("baseline", baseline_revision_id),
            build("taught", taught_revision_id),
        )
        self.validate_matched_pair(*plans)
        return plans

    @staticmethod
    def _plan_id(campaign: str, task: str, revision: str, arm: str) -> str:
        value = canonical_json((campaign, task, revision, arm)).encode()
        return f"plan-{hashlib.sha256(value).hexdigest()[:24]}"

    @staticmethod
    def validate_matched_pair(baseline: ExecutionPlan, taught: ExecutionPlan) -> None:
        if (baseline.arm, taught.arm) != ("baseline", "taught"):
            raise StrategyViolation("matched pair must be ordered baseline then taught")
        invariant_fields = (
            "campaign_id",
            "strategy_id",
            "task",
            "model",
            "context_policy_id",
            "tool_policy_id",
            "observer_policy_ids",
            "native_evaluator_id",
            "limits",
            "holdout_scope",
        )
        for field in invariant_fields:
            if getattr(baseline, field) != getattr(taught, field):
                raise StrategyViolation(f"matched invariant differs: {field}")
        if baseline.metadata != taught.metadata:
            if baseline.metadata.get("generation_config") != taught.metadata.get(
                "generation_config"
            ):
                raise StrategyViolation("matched invariant differs: generation_config")
            raise StrategyViolation("matched invariant differs: plan_metadata")
        if baseline.task.cohort is not Cohort.FEEDBACK:
            raise StrategyViolation("matched pair must use a feedback task")

    def interpret(self, receipts: Sequence[Receipt]) -> StrategyInterpretation:
        return StrategyInterpretation(
            strategy_id=self.strategy_id,
            receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
            observations={"arm_receipt_count": len(receipts)},
        )
