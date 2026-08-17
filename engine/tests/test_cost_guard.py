from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from skill_evolution_loop.cost_guard import (
    CostGuardError,
    CostPolicy,
    RunCostState,
    assert_restart_allowed,
    assess_cost_guard,
)


def _policy() -> CostPolicy:
    return CostPolicy.create(
        hourly_compute_cost_cny=7.24,
        maximum_run_cost_cny=30.0,
        maximum_runtime_seconds=4 * 3600,
        idle_timeout_seconds=15 * 60,
    )


def test_cost_guard_stops_before_projected_budget_overrun() -> None:
    started = datetime(2026, 8, 12, 8, tzinfo=UTC)
    state = RunCostState.create(
        started_at=started,
        last_progress_at=started + timedelta(hours=2),
        expected_remaining_seconds=3 * 3600,
    )

    decision = assess_cost_guard(
        policy=_policy(), state=state, now=started + timedelta(hours=2)
    )

    assert decision.stop_required is True
    assert decision.reasons == ("projected-cost-limit",)
    assert decision.projected_compute_cost_cny == pytest.approx(36.2)


def test_cost_guard_stops_on_runtime_or_idle_limit() -> None:
    started = datetime(2026, 8, 12, 8, tzinfo=UTC)
    state = RunCostState.create(
        started_at=started,
        last_progress_at=started + timedelta(hours=3, minutes=30),
        expected_remaining_seconds=0,
    )

    decision = assess_cost_guard(
        policy=_policy(), state=state, now=started + timedelta(hours=4)
    )

    assert decision.stop_required is True
    assert decision.reasons == ("maximum-runtime", "idle-timeout")


def test_restart_requires_stopped_instance_and_verified_evidence() -> None:
    assert_restart_allowed(
        instance_state="STOPPED",
        checkpoint_atomic=True,
        append_only_evidence_synced=True,
        recovery_verified=True,
    )

    with pytest.raises(CostGuardError, match="previous run is not recoverable"):
        assert_restart_allowed(
            instance_state="STOPPED",
            checkpoint_atomic=True,
            append_only_evidence_synced=False,
            recovery_verified=True,
        )

    with pytest.raises(CostGuardError, match="instance must be stopped"):
        assert_restart_allowed(
            instance_state="RUNNING",
            checkpoint_atomic=True,
            append_only_evidence_synced=True,
            recovery_verified=True,
        )
