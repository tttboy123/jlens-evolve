"""Matched paired experiment strategy for external Skill candidates."""

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
    StrategyPhase,
    StrategyResult,
    StrategyStatus,
    StrategyViolation,
    advisory_decision,
    interpretation_inputs,
)


class SkillPairedStrategy:
    strategy_id = "skill-paired-v3"
    status = StrategyStatus.LIVE

    def plan(self, context: StrategyContext) -> tuple[ExecutionPlan, ...]:
        required = {"baseline_revision_id"}
        if context.phase is StrategyPhase.EXPERIMENT:
            required.add("taught_revision_id")
        optional = {"generation_config", "plan_metadata"}
        unknown = set(context.inputs) - required - optional
        missing = required - set(context.inputs)
        if missing or unknown:
            raise StrategyViolation(
                f"skill paired inputs invalid; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        generation = context.inputs.get("generation_config")
        metadata = context.inputs.get("plan_metadata")
        if generation is not None and not isinstance(generation, Mapping):
            raise StrategyViolation("generation_config must be a mapping")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise StrategyViolation("plan_metadata must be a mapping")
        return self._plans(
            campaign_id=context.campaign_id,
            task=context.task,
            baseline_revision_id=str(context.inputs["baseline_revision_id"]),
            taught_revision_id=(
                str(context.inputs["taught_revision_id"])
                if context.phase is StrategyPhase.EXPERIMENT
                else None
            ),
            model=context.model,
            context_policy_id=context.context_policy_id,
            tool_policy_id=context.tool_policy_id,
            observer_policy_ids=context.observer_policy_ids,
            limits=context.limits,
            generation_config=generation,
            plan_metadata=metadata,
        )

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
        plans = self.plan(
            StrategyContext(
                campaign_id=campaign_id,
                task=task,
                model=model,
                context_policy_id=context_policy_id,
                tool_policy_id=tool_policy_id,
                observer_policy_ids=observer_policy_ids,
                limits=limits,
                inputs={
                    "baseline_revision_id": baseline_revision_id,
                    "taught_revision_id": taught_revision_id,
                    "generation_config": generation_config or {},
                    "plan_metadata": plan_metadata or {},
                },
            )
        )
        if len(plans) != 2:
            raise StrategyViolation("paired planning did not produce two arms")
        return plans[0], plans[1]

    def _plans(
        self,
        *,
        campaign_id: str,
        task: TaskRevision,
        baseline_revision_id: str,
        taught_revision_id: str | None,
        model: ModelIdentity,
        context_policy_id: str,
        tool_policy_id: str,
        observer_policy_ids: tuple[str, ...],
        limits: ExecutionLimits,
        generation_config: Mapping[str, Any] | None = None,
        plan_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ExecutionPlan, ...]:
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

        baseline = build("baseline", baseline_revision_id)
        if taught_revision_id is None:
            return (baseline,)
        plans = (baseline, build("taught", taught_revision_id))
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
            observations={"arm_receipt_count": len(normalized)},
        )

    def next_action(
        self, context: StrategyContext, claims: Sequence[Claim]
    ) -> StrategyDecision:
        return advisory_decision(
            strategy_id=self.strategy_id,
            status=self.status,
            action="await-authoritative-claims",
            reason="promotion remains owned by the Claim and Governance authorities",
            claims=claims,
        )
