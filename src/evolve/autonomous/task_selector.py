"""Deterministic, replayable selection from a feedback-only SWE-bench pool."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolve.contracts import canonical_json

from .config import AutonomousEvolutionError

_FORBIDDEN = ("holdout", "final-sealed", "final_sealed", "r076", "r078", "burned")


@dataclass(frozen=True, slots=True)
class TaskSelection:
    selection_id: str
    round_index: int
    selected_task_ids: tuple[str, ...]
    selected_projects: tuple[str, ...]
    selection_reason: tuple[str, ...]
    excluded: tuple[str, ...]
    tasks: tuple[Mapping[str, Any], ...]


class FeedbackTaskSelector:
    def __init__(
        self, task_pool: str | Path, *, source_pool: str | Path | None = None
    ) -> None:
        self.path = Path(task_pool).expanduser().resolve()
        self.source_pool = (
            Path(source_pool).expanduser().resolve()
            if source_pool is not None
            else None
        )
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise AutonomousEvolutionError("feedback task pool is unreadable") from error
        runtime: Mapping[str, Any] = {}
        if isinstance(rows, Mapping):
            runtime_row = rows.get("runtime", {})
            rows = rows.get("tasks")
            if not isinstance(runtime_row, Mapping):
                raise AutonomousEvolutionError("task pool runtime must be an object")
            runtime = dict(runtime_row)
        if not isinstance(rows, list) or not rows:
            raise AutonomousEvolutionError("feedback task pool must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise AutonomousEvolutionError("feedback task row must be an object")
            task = dict(row)
            rendered = canonical_json(task).casefold()
            task_id = task.get("instance_id")
            project = task.get("project")
            if (
                task.get("cohort") != "feedback"
                or any(term in rendered for term in _FORBIDDEN)
            ):
                raise AutonomousEvolutionError("task pool contains a forbidden cohort")
            if not isinstance(task_id, str) or not task_id or task_id in seen:
                raise AutonomousEvolutionError("task identities must be unique")
            if not isinstance(project, str) or not project:
                raise AutonomousEvolutionError("task project must be non-empty")
            source_uri = task.get("source_uri")
            if self.source_pool is not None:
                if not isinstance(source_uri, str) or not source_uri:
                    raise AutonomousEvolutionError(
                        "task source_uri is required by the configured source pool"
                    )
                source = Path(source_uri).expanduser().resolve()
                try:
                    source.relative_to(self.source_pool)
                except ValueError as error:
                    raise AutonomousEvolutionError(
                        "task source is outside the configured source pool"
                    ) from error
            task["task_fingerprint_sha256"] = hashlib.sha256(
                canonical_json(task).encode("utf-8")
            ).hexdigest()
            seen.add(task_id)
            normalized.append(task)
        self.tasks = tuple(normalized)
        self.runtime = dict(runtime)

    def select(
        self,
        *,
        round_index: int,
        count: int,
        prior_claims: Sequence[Mapping[str, Any]],
    ) -> TaskSelection:
        if round_index < 0 or count <= 0 or len(self.tasks) < count:
            raise AutonomousEvolutionError("task selection bounds are invalid")
        priorities = {
            str(claim.get("task_id")): str(claim.get("classification"))
            for claim in prior_claims
        }
        ranked = sorted(
            self.tasks,
            key=lambda task: (
                0
                if priorities.get(str(task["instance_id"]))
                in {"regression", "neutral"}
                else 1,
                float(task.get("estimated_cost", 0)),
                str(task["instance_id"]),
            ),
        )
        offset = (round_index * count) % len(ranked)
        rotated = ranked[offset:] + ranked[:offset]
        selected: list[Mapping[str, Any]] = []
        projects: set[str] = set()
        for task in rotated:
            project = str(task["project"])
            if len(selected) < count and (len(projects) < 2 or project not in projects):
                selected.append(task)
                projects.add(project)
        for task in rotated:
            if len(selected) == count:
                break
            if task not in selected:
                selected.append(task)
                projects.add(str(task["project"]))
        if len(selected) != count or len(projects) < 2:
            raise AutonomousEvolutionError(
                "selected tasks must cover at least two projects"
            )
        ids = tuple(str(task["instance_id"]) for task in selected)
        identity = {
            "round_index": round_index,
            "selected_task_ids": ids,
            "prior_claims": [dict(row) for row in prior_claims],
        }
        selection_id = "selection-" + hashlib.sha256(
            canonical_json(identity).encode("utf-8")
        ).hexdigest()[:24]
        return TaskSelection(
            selection_id=selection_id,
            round_index=round_index,
            selected_task_ids=ids,
            selected_projects=tuple(sorted(projects)),
            selection_reason=(
                "feedback-only",
                "project-diversity",
                "deterministic-round-rotation",
            ),
            excluded=tuple(
                str(task["instance_id"])
                for task in self.tasks
                if task not in selected
            ),
            tasks=tuple(selected),
        )
