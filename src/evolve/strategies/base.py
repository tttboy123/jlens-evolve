"""Public, context-driven Strategy seam for Campaign orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from evolve.contracts import (
    Claim,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
)


class StrategyViolation(ValueError):
    """A strategy request violates cohort or experiment invariants."""


class StrategyPhase(StrEnum):
    """Planning phase, allowing baseline evidence before candidate creation."""

    BASELINE_ONLY = "baseline-only"
    EXPERIMENT = "experiment"


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Frozen common inputs plus opaque, strategy-owned inputs."""

    campaign_id: str
    task: TaskRevision
    model: ModelIdentity
    context_policy_id: str
    tool_policy_id: str
    observer_policy_ids: tuple[str, ...]
    limits: ExecutionLimits
    phase: StrategyPhase = StrategyPhase.EXPERIMENT
    inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("campaign_id", "context_policy_id", "tool_policy_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise StrategyViolation(f"{name} must be non-empty text")
        if not self.observer_policy_ids:
            raise StrategyViolation("observer_policy_ids must not be empty")
        if not isinstance(self.inputs, Mapping):
            raise StrategyViolation("strategy inputs must be a mapping")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


class StrategyStatus(StrEnum):
    """Truthful operational maturity of a Strategy through CampaignRunner."""

    LIVE = "live"
    COMPATIBILITY = "compatibility"
    NOT_YET_LIVE = "not-yet-live"


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """Non-authoritative projection of receipts produced by a strategy.

    Governance remains the sole authority that can create promotion claims.
    """

    strategy_id: str
    campaign_id: str
    receipt_ids: tuple[str, ...]
    observations: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observations", MappingProxyType(dict(self.observations))
        )


# Kept as an import alias while callers migrate to the unified vocabulary.
StrategyInterpretation = StrategyResult


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Advisory next step; never a governance or promotion decision."""

    strategy_id: str
    status: StrategyStatus
    action: str
    reason: str
    claim_ids: tuple[str, ...] = ()


@runtime_checkable
class EvolutionStrategy(Protocol):
    """Pure plan factory and receipt interpreter; never an execution transport."""

    strategy_id: str
    status: StrategyStatus

    def plan(self, context: StrategyContext) -> tuple[ExecutionPlan, ...]: ...

    def interpret(
        self, context: StrategyContext, receipts: Sequence[Receipt]
    ) -> StrategyResult: ...

    def next_action(
        self, context: StrategyContext, claims: Sequence[Claim]
    ) -> StrategyDecision: ...


def interpretation_inputs(
    context: StrategyContext | Sequence[Receipt],
    receipts: Sequence[Receipt] | None,
) -> tuple[str, tuple[Receipt, ...]]:
    """Normalize the old ``interpret(receipts)`` compatibility call."""

    if isinstance(context, StrategyContext):
        normalized = tuple(receipts or ())
        if any(receipt.campaign_id != context.campaign_id for receipt in normalized):
            raise StrategyViolation("receipt campaign does not match strategy context")
        return context.campaign_id, normalized
    if receipts is not None:
        raise StrategyViolation("interpret received ambiguous receipt arguments")
    normalized = tuple(context)
    campaign_ids = {receipt.campaign_id for receipt in normalized}
    if len(campaign_ids) > 1:
        raise StrategyViolation("receipts span multiple campaigns")
    return next(iter(campaign_ids), "unspecified"), normalized


def advisory_decision(
    *,
    strategy_id: str,
    status: StrategyStatus,
    action: str,
    reason: str,
    claims: Sequence[Claim],
) -> StrategyDecision:
    return StrategyDecision(
        strategy_id=strategy_id,
        status=status,
        action=action,
        reason=reason,
        claim_ids=tuple(claim.claim_id for claim in claims),
    )
