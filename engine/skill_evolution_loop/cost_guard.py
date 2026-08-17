"""Provider-neutral cost and recovery gates for remote execution rounds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


class CostGuardError(ValueError):
    """Raised before an unsafe or unrecoverable cloud transition."""


def _require_aware(label: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CostGuardError(f"{label} must be timezone-aware")


@dataclass(frozen=True)
class CostPolicy:
    hourly_compute_cost_cny: float
    maximum_run_cost_cny: float
    maximum_runtime_seconds: int
    idle_timeout_seconds: int

    @classmethod
    def create(
        cls,
        *,
        hourly_compute_cost_cny: float,
        maximum_run_cost_cny: float,
        maximum_runtime_seconds: int,
        idle_timeout_seconds: int,
    ) -> CostPolicy:
        values = (hourly_compute_cost_cny, maximum_run_cost_cny)
        if any(type(value) not in {int, float} or value <= 0 for value in values):
            raise CostGuardError("cost limits must be positive")
        if (
            type(maximum_runtime_seconds) is not int
            or maximum_runtime_seconds < 1
            or type(idle_timeout_seconds) is not int
            or idle_timeout_seconds < 1
        ):
            raise CostGuardError("time limits must be positive integers")
        return cls(
            float(hourly_compute_cost_cny),
            float(maximum_run_cost_cny),
            maximum_runtime_seconds,
            idle_timeout_seconds,
        )


@dataclass(frozen=True)
class RunCostState:
    started_at: datetime
    last_progress_at: datetime
    expected_remaining_seconds: int

    @classmethod
    def create(
        cls,
        *,
        started_at: datetime,
        last_progress_at: datetime,
        expected_remaining_seconds: int,
    ) -> RunCostState:
        _require_aware("started_at", started_at)
        _require_aware("last_progress_at", last_progress_at)
        if last_progress_at < started_at:
            raise CostGuardError("last progress precedes run start")
        if (
            type(expected_remaining_seconds) is not int
            or expected_remaining_seconds < 0
        ):
            raise CostGuardError("expected remaining time must be non-negative")
        return cls(started_at, last_progress_at, expected_remaining_seconds)


@dataclass(frozen=True)
class CostDecision:
    stop_required: bool
    reasons: tuple[str, ...]
    elapsed_seconds: int
    idle_seconds: int
    accrued_compute_cost_cny: float
    projected_compute_cost_cny: float


def assess_cost_guard(
    *, policy: CostPolicy, state: RunCostState, now: datetime
) -> CostDecision:
    _require_aware("now", now)
    if now < state.last_progress_at:
        raise CostGuardError("assessment time precedes progress time")
    elapsed = int((now - state.started_at).total_seconds())
    idle = int((now - state.last_progress_at).total_seconds())
    accrued = elapsed / 3600 * policy.hourly_compute_cost_cny
    projected = (
        (elapsed + state.expected_remaining_seconds)
        / 3600
        * policy.hourly_compute_cost_cny
    )
    reasons: list[str] = []
    if elapsed >= policy.maximum_runtime_seconds:
        reasons.append("maximum-runtime")
    if idle >= policy.idle_timeout_seconds:
        reasons.append("idle-timeout")
    if projected > policy.maximum_run_cost_cny:
        reasons.append("projected-cost-limit")
    return CostDecision(
        stop_required=bool(reasons),
        reasons=tuple(reasons),
        elapsed_seconds=elapsed,
        idle_seconds=idle,
        accrued_compute_cost_cny=round(accrued, 6),
        projected_compute_cost_cny=round(projected, 6),
    )


def assert_restart_allowed(
    *,
    instance_state: str,
    checkpoint_atomic: bool,
    append_only_evidence_synced: bool,
    recovery_verified: bool,
) -> None:
    if instance_state.upper() != "STOPPED":
        raise CostGuardError("instance must be stopped before a reviewed restart")
    if not all((checkpoint_atomic, append_only_evidence_synced, recovery_verified)):
        raise CostGuardError("previous run is not recoverable")
