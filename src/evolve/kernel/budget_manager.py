"""Campaign-wide cost and model-call accounting."""

from __future__ import annotations

from dataclasses import dataclass

from evolve.contracts import ContractViolation


class BudgetExceeded(ContractViolation):
    """A reservation or final charge would exceed campaign authorization."""


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    max_cost_cny: float
    max_model_calls: int
    reserved_cost_cny: float
    reserved_model_calls: int
    spent_cost_cny: float
    spent_model_calls: int


class BudgetManager:
    """Reserve before dispatch and reconcile only after a recorded result."""

    def __init__(
        self,
        *,
        max_cost_cny: float,
        max_model_calls: int,
        reserved_cost_cny: float = 0,
        reserved_model_calls: int = 0,
        spent_cost_cny: float = 0,
        spent_model_calls: int = 0,
    ) -> None:
        self._max_cost_cny = max_cost_cny
        self._max_model_calls = max_model_calls
        self._reserved_cost_cny = reserved_cost_cny
        self._reserved_model_calls = reserved_model_calls
        self._spent_cost_cny = spent_cost_cny
        self._spent_model_calls = spent_model_calls

    def reserve(self, *, cost_cny: float, model_calls: int) -> None:
        if cost_cny < 0 or model_calls < 0:
            raise ContractViolation("budget reservations cannot be negative")
        prospective_cost = self._spent_cost_cny + self._reserved_cost_cny + cost_cny
        prospective_calls = (
            self._spent_model_calls + self._reserved_model_calls + model_calls
        )
        if prospective_cost > self._max_cost_cny:
            raise BudgetExceeded("campaign cost budget exhausted")
        if prospective_calls > self._max_model_calls:
            raise BudgetExceeded("campaign model call budget exhausted")
        self._reserved_cost_cny += cost_cny
        self._reserved_model_calls += model_calls

    def record(
        self,
        *,
        reserved_cost_cny: float,
        reserved_model_calls: int,
        actual_cost_cny: float,
        actual_model_calls: int,
    ) -> None:
        values = (
            reserved_cost_cny,
            reserved_model_calls,
            actual_cost_cny,
            actual_model_calls,
        )
        if any(value < 0 for value in values):
            raise ContractViolation("budget charges cannot be negative")
        if (
            reserved_cost_cny > self._reserved_cost_cny
            or reserved_model_calls > self._reserved_model_calls
        ):
            raise ContractViolation("result exceeds its outstanding reservation")
        if self._spent_cost_cny + actual_cost_cny > self._max_cost_cny:
            raise BudgetExceeded("actual campaign cost exceeds authorization")
        if self._spent_model_calls + actual_model_calls > self._max_model_calls:
            raise BudgetExceeded("actual campaign model calls exceed authorization")
        self._reserved_cost_cny -= reserved_cost_cny
        self._reserved_model_calls -= reserved_model_calls
        self._spent_cost_cny += actual_cost_cny
        self._spent_model_calls += actual_model_calls

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_cost_cny=self._max_cost_cny,
            max_model_calls=self._max_model_calls,
            reserved_cost_cny=self._reserved_cost_cny,
            reserved_model_calls=self._reserved_model_calls,
            spent_cost_cny=self._spent_cost_cny,
            spent_model_calls=self._spent_model_calls,
        )
