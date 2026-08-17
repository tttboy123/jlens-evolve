"""Task schema shared by deterministic operators and the synthetic generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskSchema:
    """Field naming contract for a record-cleaning task family."""

    identity_field: str = "account"
    value_field: str = "value"
    status_field: str = "state"
    accepted_status: str = "settled"
    currency_field: str | None = "currency"
    accepted_currency: str | None = "USD"
    round_decimals: int = 2


PAID_SCHEMA = TaskSchema(
    identity_field="user",
    value_field="amount",
    status_field="status",
    accepted_status="paid",
    currency_field=None,
    accepted_currency=None,
    round_decimals=2,
)

PAYOUT_SCHEMA = TaskSchema(
    identity_field="account",
    value_field="value",
    status_field="state",
    accepted_status="settled",
    currency_field="currency",
    accepted_currency="USD",
    round_decimals=2,
)

REFUND_SCHEMA = TaskSchema(
    identity_field="customer",
    value_field="refund_amount",
    status_field="decision",
    accepted_status="approved",
    currency_field=None,
    accepted_currency=None,
    round_decimals=2,
)
