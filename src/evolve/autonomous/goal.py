"""Durable product goal state; no development-review states belong here."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from evolve.contracts import canonical_json

from .config import AutonomousEvolutionError


class GoalRunStatus(StrEnum):
    ACTIVE = "active"
    GOAL_REACHED = "goal_reached"
    NO_PROGRESS = "no_progress"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    MAX_CONSECUTIVE_INFRA_FAILURES = "max_consecutive_infra_failures"
    MAX_SAME_FAILURE_SIGNATURE = "max_same_failure_signature"
    DISK_LIMIT = "disk_limit"
    STOPPED_BY_USER = "stopped_by_user"
    BLOCKED_INTEGRITY = "blocked_integrity"


@dataclass(frozen=True, slots=True)
class GoalState:
    schema_version: int
    goal_id: str
    status: GoalRunStatus
    next_round_index: int
    rounds_completed: int
    no_progress_rounds: int
    consecutive_infra_failures: int
    native_gain_task_ids: tuple[str, ...]
    best_candidate_revision_id: str | None
    best_bundle_sha256: str | None
    same_failure_signature_sha256: str | None = None
    same_failure_signature_rounds: int = 0


class GoalStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def load_or_create(self, *, goal_id: str) -> GoalState:
        if not self.path.exists():
            state = GoalState(
                schema_version=1,
                goal_id=goal_id,
                status=GoalRunStatus.ACTIVE,
                next_round_index=0,
                rounds_completed=0,
                no_progress_rounds=0,
                consecutive_infra_failures=0,
                native_gain_task_ids=(),
                best_candidate_revision_id=None,
                best_bundle_sha256=None,
            )
            self.write(state)
            return state
        try:
            row = json.loads(self.path.read_text(encoding="utf-8"))
            raw_status = row["status"]
            if raw_status == "blocked_infrastructure":
                raw_status = "max_consecutive_infra_failures"
            state = GoalState(
                schema_version=int(row["schema_version"]),
                goal_id=str(row["goal_id"]),
                status=GoalRunStatus(raw_status),
                next_round_index=int(row["next_round_index"]),
                rounds_completed=int(row["rounds_completed"]),
                no_progress_rounds=int(row["no_progress_rounds"]),
                consecutive_infra_failures=int(row["consecutive_infra_failures"]),
                native_gain_task_ids=tuple(row["native_gain_task_ids"]),
                best_candidate_revision_id=row["best_candidate_revision_id"],
                best_bundle_sha256=row["best_bundle_sha256"],
                same_failure_signature_sha256=row.get(
                    "same_failure_signature_sha256"
                ),
                same_failure_signature_rounds=int(
                    row.get("same_failure_signature_rounds", 0)
                ),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            raise AutonomousEvolutionError("evolution state is invalid") from error
        if state.schema_version != 1 or state.goal_id != goal_id:
            raise AutonomousEvolutionError("evolution state goal identity mismatch")
        return state

    def write(self, state: GoalState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(asdict(state)) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
