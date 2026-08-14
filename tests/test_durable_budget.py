from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from evolve.kernel.budget_manager import (
    BudgetExceeded,
    DurableCostLedger,
    LedgerBusy,
    LedgerConflict,
    LedgerIntegrityError,
)


def test_durable_cost_ledger_survives_restart_and_deduplicates_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cost-ledger.jsonl"
    ledger = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=3
    )

    assert ledger.reserve("reservation-plan-1", cost_cny=2, model_calls=1) is True
    assert ledger.reserve("reservation-plan-1", cost_cny=2, model_calls=1) is False
    restarted = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=3
    )
    assert restarted.snapshot().reserved_cost_cny == 2
    assert restarted.snapshot().reserved_model_calls == 1

    assert (
        restarted.record(
            "reservation-plan-1",
            result_id="result-plan-1",
            actual_cost_cny=1.25,
            actual_model_calls=1,
        )
        is True
    )
    assert (
        restarted.record(
            "reservation-plan-1",
            result_id="result-plan-1",
            actual_cost_cny=1.25,
            actual_model_calls=1,
        )
        is False
    )
    final = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=3
    ).snapshot()
    assert (final.reserved_cost_cny, final.reserved_model_calls) == (0, 0)
    assert (final.spent_cost_cny, final.spent_model_calls) == (1.25, 1)

    with pytest.raises(LedgerConflict, match="reservation-plan-1"):
        restarted.reserve("reservation-plan-1", cost_cny=3, model_calls=1)


def test_durable_cost_ledger_fails_closed_on_writer_lock_and_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cost-ledger.jsonl"
    ledger = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=3
    )
    ledger.lease_path.write_text("held", encoding="utf-8")
    with pytest.raises(LedgerBusy, match="lease"):
        ledger.reserve("reservation-plan-1", cost_cny=2, model_calls=1)
    ledger.lease_path.unlink()
    ledger.reserve("reservation-plan-1", cost_cny=2, model_calls=1)

    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])
    payload["cost_cny"] = 9
    lines[-1] = json.dumps(payload)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerIntegrityError, match="hash mismatch"):
        DurableCostLedger(
            path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=3
        )


def test_concurrent_reservations_cannot_jointly_break_campaign_budget(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cost-ledger.jsonl"
    first = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=2
    )
    second = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=2
    )
    barrier = threading.Barrier(2)

    def reserve(ledger: DurableCostLedger, event_id: str) -> str:
        barrier.wait()
        try:
            return (
                "written"
                if ledger.reserve(event_id, cost_cny=6, model_calls=1)
                else "duplicate"
            )
        except (LedgerBusy, BudgetExceeded) as error:
            return type(error).__name__

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(
            pool.map(
                lambda args: reserve(*args),
                ((first, "reservation-a"), (second, "reservation-b")),
            )
        )

    assert outcomes.count("written") == 1
    snapshot = DurableCostLedger(
        path, campaign_id="campaign-1", max_cost_cny=10, max_model_calls=2
    ).snapshot()
    assert snapshot.spent_cost_cny + snapshot.reserved_cost_cny <= 10
    assert snapshot.spent_model_calls + snapshot.reserved_model_calls <= 2


def test_budget_exhaustion_prevents_dispatch_callable(tmp_path: Path) -> None:
    ledger = DurableCostLedger(
        tmp_path / "cost-ledger.jsonl",
        campaign_id="campaign-1",
        max_cost_cny=10,
        max_model_calls=1,
    )
    ledger.reserve("reservation-existing", cost_cny=10, model_calls=1)
    called = False

    def dispatch() -> str:
        nonlocal called
        called = True
        return "should-not-run"

    with pytest.raises(BudgetExceeded):
        ledger.dispatch_once(
            "reservation-blocked",
            cost_cny=0.01,
            model_calls=1,
            dispatch=dispatch,
        )
    assert called is False
