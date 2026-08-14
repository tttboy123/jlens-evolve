"""Read-only compatibility planning for immutable legacy evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LegacyImportStrategy:
    strategy_id = "legacy-import-v3"
    status = StrategyStatus.COMPATIBILITY

    def plan(self, context: StrategyContext) -> tuple[ExecutionPlan]:
        required = {
            "imported_revision_id",
            "legacy_artifact_sha256",
            "provenance_uri",
        }
        if set(context.inputs) != required:
            raise StrategyViolation(
                "legacy import inputs must be imported_revision_id, "
                "legacy_artifact_sha256, and provenance_uri"
            )
        return self._plans(
            campaign_id=context.campaign_id,
            task=context.task,
            imported_revision_id=str(context.inputs["imported_revision_id"]),
            legacy_artifact_sha256=str(
                context.inputs["legacy_artifact_sha256"]
            ),
            provenance_uri=str(context.inputs["provenance_uri"]),
            model=context.model,
            context_policy_id=context.context_policy_id,
            tool_policy_id=context.tool_policy_id,
            observer_policy_ids=context.observer_policy_ids,
            limits=context.limits,
        )

    def build_plans(
        self,
        *,
        campaign_id: str,
        task: TaskRevision,
        imported_revision_id: str,
        legacy_artifact_sha256: str,
        provenance_uri: str,
        model: ModelIdentity,
        context_policy_id: str,
        tool_policy_id: str,
        observer_policy_ids: tuple[str, ...],
        limits: ExecutionLimits,
    ) -> tuple[ExecutionPlan]:
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
                    "imported_revision_id": imported_revision_id,
                    "legacy_artifact_sha256": legacy_artifact_sha256,
                    "provenance_uri": provenance_uri,
                },
            )
        )

    def _plans(
        self,
        *,
        campaign_id: str,
        task: TaskRevision,
        imported_revision_id: str,
        legacy_artifact_sha256: str,
        provenance_uri: str,
        model: ModelIdentity,
        context_policy_id: str,
        tool_policy_id: str,
        observer_policy_ids: tuple[str, ...],
        limits: ExecutionLimits,
    ) -> tuple[ExecutionPlan]:
        if task.cohort is not Cohort.FEEDBACK:
            raise StrategyViolation(
                "Legacy import execution is restricted to feedback tasks"
            )
        if _SHA256.fullmatch(legacy_artifact_sha256) is None:
            raise StrategyViolation("legacy artifact must have a literal SHA-256")
        if not provenance_uri.strip():
            raise StrategyViolation("legacy import requires provenance")
        identity = canonical_json(
            (
                campaign_id,
                task.revision_id,
                imported_revision_id,
                legacy_artifact_sha256,
            )
        ).encode()
        plan = ExecutionPlan(
            plan_id=f"plan-{hashlib.sha256(identity).hexdigest()[:24]}",
            campaign_id=campaign_id,
            strategy_id=self.strategy_id,
            task=task,
            candidate_revision_id=imported_revision_id,
            arm="legacy-replay",
            model=model,
            context_policy_id=context_policy_id,
            tool_policy_id=tool_policy_id,
            observer_policy_ids=observer_policy_ids,
            native_evaluator_id=task.evaluator_id,
            limits=limits,
            holdout_scope="feedback-only",
            metadata={
                "legacy_artifact_sha256": legacy_artifact_sha256,
                "provenance_uri": provenance_uri,
                "compatibility_mode": "replay",
            },
        )
        return (plan,)

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
            observations={"replayed_receipt_count": len(normalized)},
        )

    def next_action(
        self, context: StrategyContext, claims: Sequence[Claim]
    ) -> StrategyDecision:
        return advisory_decision(
            strategy_id=self.strategy_id,
            status=self.status,
            action="use-legacy-replay",
            reason="legacy import remains a read-only compatibility path",
            claims=claims,
        )
