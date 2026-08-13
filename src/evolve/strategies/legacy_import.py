"""Read-only compatibility planning for immutable legacy evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

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

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LegacyImportStrategy:
    strategy_id = "legacy-import-v3"

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

    def interpret(self, receipts: Sequence[Receipt]) -> StrategyInterpretation:
        return StrategyInterpretation(
            strategy_id=self.strategy_id,
            receipt_ids=tuple(receipt.receipt_id for receipt in receipts),
            observations={"replayed_receipt_count": len(receipts)},
        )
