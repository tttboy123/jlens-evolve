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
    Cohort,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
    content_sha256,
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

    def __init__(self, *, execution_profile: str | None = None) -> None:
        if execution_profile not in {None, "fixture"}:
            raise StrategyViolation("unsupported AgentProgram execution profile")
        self.execution_profile = execution_profile
        self.status = (
            StrategyStatus.LIVE
            if execution_profile == "fixture"
            else StrategyStatus.NOT_YET_LIVE
        )

    def plan(self, context: StrategyContext) -> tuple[ExecutionPlan, ...]:
        required = {
            "parent_revision_id",
            "candidate_revision_ids",
            "tournament_id",
        }
        optional = {"generation_config"}
        if self.execution_profile == "fixture":
            required.update({"execution_profile", "revision_roots"})
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
        if self.execution_profile == "fixture":
            if context.inputs.get("execution_profile") != "fixture":
                raise StrategyViolation("fixture execution profile identity mismatch")
            roots = context.inputs["revision_roots"]
            if not isinstance(roots, Mapping):
                raise StrategyViolation("revision_roots must be a mapping")
            expected_revisions = (
                str(context.inputs["parent_revision_id"]),
                *candidate_ids,
            )
            if set(roots) != set(expected_revisions):
                raise StrategyViolation("fixture revision roots are incomplete")
            try:
                revisions = {
                    revision_id: AgentProgramRevision.load(
                        Path(str(roots[revision_id])).resolve()
                    )
                    for revision_id in expected_revisions
                }
            except AgentProgramViolation as error:
                raise StrategyViolation("fixture AgentProgram revision is invalid") from error
            if any(
                revision.revision_id != revision_id
                for revision_id, revision in revisions.items()
            ):
                raise StrategyViolation("fixture revision root identity mismatch")
            program_ids = {revision.program_id for revision in revisions.values()}
            if len(program_ids) != 1:
                raise StrategyViolation("fixture tournament mixes AgentPrograms")
            parent_revision_id = expected_revisions[0]
            if any(
                revisions[revision_id].parent_revision_id != parent_revision_id
                for revision_id in candidate_ids
            ):
                raise StrategyViolation("fixture candidate parent lineage mismatch")
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
                        "execution_profile": "fixture",
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
        return advisory_decision(
            strategy_id=self.strategy_id,
            status=self.status,
            action="await-tournament-authority",
            reason="authoritative tournament selection is not yet wired",
            claims=claims,
        )
