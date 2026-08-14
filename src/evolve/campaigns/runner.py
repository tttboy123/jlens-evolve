"""Strategy-neutral Campaign runner backed by Kernel and Runtime authorities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from evolve.contracts import (
    Authorization,
    Claim,
    ContractViolation,
    ExecutionPlan,
    MechanismPrediction,
    Receipt,
)
from evolve.kernel import CampaignController, CampaignSnapshot, CampaignStatus
from evolve.runtime import ExecutionResult, RuntimeEntry
from evolve.strategies import (
    EvolutionStrategy,
    StrategyContext,
    StrategyDecision,
    StrategyResult,
    StrategyStatus,
)


class CampaignRunStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    COMPATIBILITY = "compatibility"
    NOT_YET_LIVE = "not-yet-live"


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """Frozen strategy contexts and externally authorized Claims for one run."""

    campaign_id: str
    contexts: tuple[StrategyContext, ...]
    authorization: Authorization
    claims: tuple[Claim, ...] = ()
    mechanism_prediction: MechanismPrediction | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise ContractViolation("campaign_id must be non-empty text")
        if not self.contexts:
            raise ContractViolation("campaign spec requires at least one context")
        if self.authorization.campaign_id != self.campaign_id:
            raise ContractViolation("campaign spec authorization identity mismatch")
        if any(context.campaign_id != self.campaign_id for context in self.contexts):
            raise ContractViolation("strategy context campaign identity mismatch")


@dataclass(frozen=True, slots=True)
class CampaignRunResult:
    """Auditable projection; Claims are only those supplied by an authority."""

    campaign_id: str
    status: CampaignRunStatus
    plans: tuple[ExecutionPlan, ...]
    executions: tuple[ExecutionResult, ...]
    receipts: tuple[Receipt, ...]
    interpretations: tuple[StrategyResult, ...]
    decisions: tuple[StrategyDecision, ...]
    claims: tuple[Claim, ...]
    snapshot: CampaignSnapshot | None
    reason: str | None = None


class CampaignRunner:
    """Run live Strategy plans without becoming a fact or Claim authority."""

    def __init__(
        self,
        *,
        runtime: RuntimeEntry | None = None,
        controller: CampaignController | None = None,
    ) -> None:
        self._runtime = runtime
        self._controller = controller

    def run(
        self, spec: CampaignSpec, strategy: EvolutionStrategy
    ) -> CampaignRunResult:
        planned = tuple(
            (context, tuple(strategy.plan(context))) for context in spec.contexts
        )
        plans = tuple(plan for _, context_plans in planned for plan in context_plans)
        self._validate_plans(spec, strategy, plans)

        if strategy.status is StrategyStatus.NOT_YET_LIVE or (
            strategy.status is StrategyStatus.COMPATIBILITY
            and (self._runtime is None or self._controller is None)
        ):
            status = (
                CampaignRunStatus.COMPATIBILITY
                if strategy.status is StrategyStatus.COMPATIBILITY
                else CampaignRunStatus.NOT_YET_LIVE
            )
            return self._non_live_result(
                spec=spec,
                strategy=strategy,
                planned=planned,
                plans=plans,
                status=status,
                reason=(
                    "strategy is available only through a compatibility path"
                    if status is CampaignRunStatus.COMPATIBILITY
                    else "strategy execution is not yet wired to an authority"
                ),
            )

        if self._runtime is None or self._controller is None:
            return self._non_live_result(
                spec=spec,
                strategy=strategy,
                planned=planned,
                plans=plans,
                status=CampaignRunStatus.NOT_YET_LIVE,
                reason=(
                    "live execution requires both ExecutionRuntime and "
                    "CampaignController authorities"
                ),
            )

        self._validate_controller(spec)
        snapshot = self._controller.snapshot()
        terminal_replay = snapshot.status is CampaignStatus.COMPLETED
        if snapshot.status in {CampaignStatus.CREATED, CampaignStatus.PAUSED}:
            self._controller.start()
        elif snapshot.status not in {
            CampaignStatus.RUNNING,
            CampaignStatus.COMPLETED,
        }:
            raise ContractViolation(
                f"campaign runner cannot execute from {snapshot.status}"
            )

        executions: list[ExecutionResult] = []
        receipts: list[Receipt] = []
        receipt_ids_by_context: list[set[str]] = [set() for _ in planned]
        failure_reason: str | None = None
        for context_index, (_, context_plans) in enumerate(planned):
            for plan in context_plans:
                if not terminal_replay:
                    self._controller.submit(
                        plan,
                        reserved_model_calls=_reserved_model_calls(
                            self._runtime, plan
                        ),
                        remote=getattr(self._runtime, "remote", None),
                    )
                execution = (
                    self._runtime.execute(
                        plan,
                        spec.authorization,
                        mechanism_prediction=spec.mechanism_prediction,
                    )
                    if spec.mechanism_prediction is not None
                    else self._runtime.execute(plan, spec.authorization)
                )
                if not isinstance(execution, ExecutionResult):
                    raise ContractViolation(
                        "ExecutionRuntime returned an invalid result contract"
                    )
                executions.append(execution)
                receipts.extend(execution.receipts)
                receipt_ids_by_context[context_index].update(
                    receipt.receipt_id for receipt in execution.receipts
                )
                if not terminal_replay:
                    self._controller.record_result(
                        plan.plan_id,
                        actual_cost_cny=_execution_cost(execution),
                        actual_model_calls=_execution_model_calls(execution),
                        succeeded=execution.status == "completed",
                    )
                if execution.status != "completed":
                    failure_reason = (
                        f"plan {plan.plan_id} ended with {execution.status}"
                    )
                    break
            if failure_reason is not None:
                break

        if terminal_replay:
            run_status = (
                CampaignRunStatus.COMPATIBILITY
                if strategy.status is StrategyStatus.COMPATIBILITY
                else CampaignRunStatus.COMPLETED
            )
        elif failure_reason is None:
            self._controller.finalize(CampaignStatus.COMPLETED)
            run_status = (
                CampaignRunStatus.COMPATIBILITY
                if strategy.status is StrategyStatus.COMPATIBILITY
                else CampaignRunStatus.COMPLETED
            )
        else:
            self._controller.mark_partial(failure_reason)
            run_status = CampaignRunStatus.PARTIAL

        interpretations = tuple(
            strategy.interpret(
                context,
                tuple(
                    receipt
                    for receipt in receipts
                    if receipt.receipt_id in receipt_ids_by_context[index]
                ),
            )
            for index, (context, _) in enumerate(planned)
        )
        decisions = tuple(
            strategy.next_action(context, spec.claims)
            for context, _ in planned
        )
        return CampaignRunResult(
            campaign_id=spec.campaign_id,
            status=run_status,
            plans=plans,
            executions=tuple(executions),
            receipts=tuple(receipts),
            interpretations=interpretations,
            decisions=decisions,
            claims=spec.claims,
            snapshot=self._controller.snapshot(),
            reason=failure_reason,
        )

    def _non_live_result(
        self,
        *,
        spec: CampaignSpec,
        strategy: EvolutionStrategy,
        planned: tuple[tuple[StrategyContext, tuple[ExecutionPlan, ...]], ...],
        plans: tuple[ExecutionPlan, ...],
        status: CampaignRunStatus,
        reason: str,
    ) -> CampaignRunResult:
        return CampaignRunResult(
            campaign_id=spec.campaign_id,
            status=status,
            plans=plans,
            executions=(),
            receipts=(),
            interpretations=tuple(
                strategy.interpret(context, ()) for context, _ in planned
            ),
            decisions=tuple(
                strategy.next_action(context, spec.claims)
                for context, _ in planned
            ),
            claims=spec.claims,
            snapshot=None,
            reason=reason,
        )

    @staticmethod
    def _validate_plans(
        spec: CampaignSpec,
        strategy: EvolutionStrategy,
        plans: tuple[ExecutionPlan, ...],
    ) -> None:
        if not plans:
            raise ContractViolation("strategy produced no execution plans")
        if len({plan.plan_id for plan in plans}) != len(plans):
            raise ContractViolation("strategy produced duplicate plan identities")
        if any(plan.campaign_id != spec.campaign_id for plan in plans):
            raise ContractViolation("strategy plan campaign identity mismatch")
        if any(plan.strategy_id != strategy.strategy_id for plan in plans):
            raise ContractViolation("strategy plan strategy identity mismatch")

    def _validate_controller(self, spec: CampaignSpec) -> None:
        if self._controller is None:  # narrowed by caller
            raise ContractViolation("CampaignController authority is missing")
        snapshot = self._controller.snapshot()
        if snapshot.campaign_id != spec.campaign_id:
            raise ContractViolation("CampaignController campaign identity mismatch")
        if snapshot.authorization_id != spec.authorization.authorization_id:
            raise ContractViolation(
                "CampaignController authorization identity mismatch"
            )


def _execution_cost(execution: ExecutionResult) -> float:
    costs = tuple(
        receipt for receipt in execution.receipts if receipt.kind == "cost"
    )
    if len(costs) > 1:
        raise ContractViolation("ExecutionRuntime returned duplicate cost receipts")
    return float(costs[0].payload["cost_cny"]) if costs else 0.0


def _execution_model_calls(execution: ExecutionResult) -> int:
    return sum(receipt.kind == "model" for receipt in execution.receipts)


def _reserved_model_calls(runtime: RuntimeEntry, plan: ExecutionPlan) -> int:
    reservation = getattr(runtime, "reserved_model_calls", 1)
    value = reservation(plan) if callable(reservation) else reservation
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractViolation("runtime model-call reservation is invalid")
    return value


__all__ = [
    "CampaignRunner",
    "CampaignRunResult",
    "CampaignRunStatus",
    "CampaignSpec",
]
