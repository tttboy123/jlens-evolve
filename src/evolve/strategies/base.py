"""Public strategy seam for the evidence-centric execution runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from evolve.contracts import ExecutionPlan, Receipt


class StrategyViolation(ValueError):
    """A strategy request violates cohort or experiment invariants."""


@dataclass(frozen=True, slots=True)
class StrategyInterpretation:
    """Non-authoritative projection of receipts produced by a strategy.

    Governance remains the sole authority that can create promotion claims.
    """

    strategy_id: str
    receipt_ids: tuple[str, ...]
    observations: Mapping[str, Any]


@runtime_checkable
class EvolutionStrategy(Protocol):
    """Pure plan factory and receipt interpreter; never an execution transport."""

    strategy_id: str

    def build_plans(self, **kwargs: Any) -> tuple[ExecutionPlan, ...]: ...

    def interpret(self, receipts: Sequence[Receipt]) -> StrategyInterpretation: ...
