"""Deterministic, replayable selection from a feedback-only SWE-bench pool."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from evolve.contracts import canonical_json

from .config import AutonomousEvolutionError

_FORBIDDEN = ("holdout", "final-sealed", "final_sealed", "r076", "r078", "burned")


@dataclass(frozen=True, slots=True)
class TaskSelection:
    selection_id: str
    round_index: int
    selection_context_sha256: str
    selection_context: Mapping[str, Any]
    selected_task_ids: tuple[str, ...]
    selected_projects: tuple[str, ...]
    selection_reason: tuple[str, ...]
    excluded: tuple[str, ...]
    tasks: tuple[Mapping[str, Any], ...]

    def selection_context_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable copy for TASK-SELECTION/INDEX projection."""

        payload = _thaw_json(self.selection_context)
        if not isinstance(payload, dict):  # Construction guarantees this invariant.
            raise AutonomousEvolutionError("selection context projection is invalid")
        return payload


@dataclass(frozen=True, slots=True)
class TaskSelectionContext:
    """Immutable, hash-bound inputs to the feedback-task selection policy.

    ``failure_signature_counts`` and ``task_selection_counts`` are keyed by task
    id. Historical claims retain their complete canonical JSON representation so
    replay identity cannot silently collapse to only the latest classification.
    """

    historical_claims: tuple[Mapping[str, Any], ...]
    current_best_revision_id: str | None
    current_best_supported_task_ids: tuple[str, ...]
    failure_signature_counts: Mapping[str, int]
    goal_gap: int
    task_selection_counts: Mapping[str, int]
    repeat_hard_cap: int
    excluded_task_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        frozen_claims: list[Mapping[str, Any]] = []
        for claim in self.historical_claims:
            if not isinstance(claim, Mapping):
                raise AutonomousEvolutionError(
                    "historical claims must contain only objects"
                )
            try:
                normalized = json.loads(canonical_json(dict(claim)))
            except (TypeError, ValueError) as error:
                raise AutonomousEvolutionError(
                    "historical claims must be canonical JSON objects"
                ) from error
            task_id = normalized.get("task_id")
            classification = normalized.get("classification")
            if not isinstance(task_id, str) or not task_id:
                raise AutonomousEvolutionError(
                    "historical claim task_id must be non-empty"
                )
            if not isinstance(classification, str) or not classification:
                raise AutonomousEvolutionError(
                    "historical claim classification must be non-empty"
                )
            frozen_claims.append(_freeze_mapping(normalized))
        object.__setattr__(self, "historical_claims", tuple(frozen_claims))

        revision = self.current_best_revision_id
        if revision is not None and (not isinstance(revision, str) or not revision):
            raise AutonomousEvolutionError(
                "current best revision id must be non-empty when present"
            )
        support = tuple(
            sorted(
                _validated_task_ids(
                    "current best supported task ids",
                    self.current_best_supported_task_ids,
                )
            )
        )
        object.__setattr__(self, "current_best_supported_task_ids", support)
        object.__setattr__(
            self,
            "failure_signature_counts",
            _validated_counts(
                "failure signature counts", self.failure_signature_counts
            ),
        )
        object.__setattr__(
            self,
            "task_selection_counts",
            _validated_counts("task selection counts", self.task_selection_counts),
        )
        object.__setattr__(
            self,
            "excluded_task_ids",
            _validated_task_ids("excluded task ids", self.excluded_task_ids),
        )
        if (
            isinstance(self.goal_gap, bool)
            or not isinstance(self.goal_gap, int)
            or self.goal_gap < 0
        ):
            raise AutonomousEvolutionError("goal gap must be a non-negative integer")
        if (
            isinstance(self.repeat_hard_cap, bool)
            or not isinstance(self.repeat_hard_cap, int)
            or self.repeat_hard_cap <= 0
        ):
            raise AutonomousEvolutionError("repeat hard cap must be a positive integer")

    def identity_payload(self) -> dict[str, Any]:
        payload = {
            "historical_claims": [
                _thaw_json(claim) for claim in self.historical_claims
            ],
            "current_best_revision_id": self.current_best_revision_id,
            "current_best_supported_task_ids": list(
                self.current_best_supported_task_ids
            ),
            "failure_signature_counts": dict(self.failure_signature_counts),
            "goal_gap": self.goal_gap,
            "task_selection_counts": dict(self.task_selection_counts),
            "repeat_hard_cap": self.repeat_hard_cap,
        }
        if self.excluded_task_ids:
            payload["excluded_task_ids"] = list(self.excluded_task_ids)
        return payload

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self.identity_payload()).encode("utf-8")
        ).hexdigest()


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
            estimated_cost = task.get("estimated_cost", 0)
            if (
                isinstance(estimated_cost, bool)
                or not isinstance(estimated_cost, (int, float))
                or not math.isfinite(float(estimated_cost))
                or estimated_cost < 0
            ):
                raise AutonomousEvolutionError(
                    "task estimated_cost must be finite and non-negative"
                )
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
        self.task_pool_sha256 = hashlib.sha256(
            canonical_json(normalized).encode("utf-8")
        ).hexdigest()
        self.runtime = dict(runtime)

    def select(
        self,
        *,
        round_index: int,
        count: int,
        prior_claims: Sequence[Mapping[str, Any]] | None = None,
        context: TaskSelectionContext | None = None,
    ) -> TaskSelection:
        if round_index < 0 or count <= 0 or len(self.tasks) < count:
            raise AutonomousEvolutionError("task selection bounds are invalid")
        if context is not None and prior_claims is not None:
            raise AutonomousEvolutionError(
                "selection accepts either context or prior_claims, not both"
            )
        claims = context.historical_claims if context is not None else prior_claims or ()
        priorities = {
            str(claim.get("task_id")): str(claim.get("classification"))
            for claim in claims
        }
        excluded = set(context.excluded_task_ids if context is not None else ())
        if context is None:
            ranked = sorted(
                [task for task in self.tasks if task["instance_id"] not in excluded],
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
            ranked = ranked[offset:] + ranked[:offset]
            context_payload: Mapping[str, Any] = {
                "schema_version": 1,
                "mode": "compatibility-prior-claims",
                "historical_claims": [dict(row) for row in claims],
            }
            reasons: tuple[str, ...] = (
                "feedback-only",
                *((("excluded-task-filter",) if excluded else ())),
                "project-diversity",
                "deterministic-round-rotation",
            )
        else:
            support = set(context.current_best_supported_task_ids)
            ranked = sorted(
                (
                    task
                    for task in self.tasks
                    if context.task_selection_counts.get(str(task["instance_id"]), 0)
                    < context.repeat_hard_cap
                    and str(task["instance_id"]) not in excluded
                ),
                key=lambda task: (
                    0
                    if context.goal_gap > 0
                    and str(task["instance_id"]) not in support
                    else 1,
                    0
                    if priorities.get(str(task["instance_id"]))
                    in {"regression", "neutral"}
                    else 1,
                    -context.failure_signature_counts.get(
                        str(task["instance_id"]), 0
                    ),
                    context.task_selection_counts.get(str(task["instance_id"]), 0),
                    float(task.get("estimated_cost", 0)),
                    str(task["instance_id"]),
                ),
            )
            context_payload = {
                "schema_version": 1,
                "mode": "stateful-v1",
                **context.identity_payload(),
            }
            reasons = (
                "feedback-only",
                *(
                    (f"excluded-task-count={len(excluded)}",)
                    if excluded
                    else ()
                ),
                f"goal-gap={context.goal_gap}",
                f"current-best={context.current_best_revision_id or 'none'}",
                "unsupported-current-best-first",
                "historical-regression-neutral-first",
                "repeated-failure-first",
                "least-selected-first",
                "cost-ascending",
                f"repeat-hard-cap={context.repeat_hard_cap}",
                "project-diversity",
                "deterministic-context-replay",
            )
        if len(ranked) < count:
            raise AutonomousEvolutionError(
                "repeat hard cap leaves too few selectable tasks"
            )
        selected: list[Mapping[str, Any]] = []
        projects: set[str] = set()
        if count > 1:
            first = ranked[0]
            selected.append(first)
            projects.add(str(first["project"]))
            for task in ranked[1:]:
                project = str(task["project"])
                if project not in projects:
                    selected.append(task)
                    projects.add(project)
                    break
        for task in ranked:
            if len(selected) == count:
                break
            if task not in selected:
                selected.append(task)
                projects.add(str(task["project"]))
        if len(selected) != count or (count > 1 and len(projects) < 2):
            raise AutonomousEvolutionError(
                "selected tasks must cover at least two projects"
            )
        ids = tuple(str(task["instance_id"]) for task in selected)
        canonical_context = json.loads(canonical_json(context_payload))
        selection_context_sha256 = hashlib.sha256(
            canonical_json(canonical_context).encode("utf-8")
        ).hexdigest()
        identity = {
            "round_index": round_index,
            "task_pool_sha256": self.task_pool_sha256,
            "selected_task_ids": ids,
            "ranked_selectable_task_ids": [
                str(task["instance_id"]) for task in ranked
            ],
            "selection_context_sha256": selection_context_sha256,
        }
        selection_id = "selection-" + hashlib.sha256(
            canonical_json(identity).encode("utf-8")
        ).hexdigest()[:24]
        return TaskSelection(
            selection_id=selection_id,
            round_index=round_index,
            selection_context_sha256=selection_context_sha256,
            selection_context=_freeze_mapping(canonical_context),
            selected_task_ids=ids,
            selected_projects=tuple(sorted(projects)),
            selection_reason=reasons,
            excluded=tuple(
                str(task["instance_id"])
                for task in self.tasks
                if task not in selected
            ),
            tasks=tuple(selected),
        )


def _validated_task_ids(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AutonomousEvolutionError(f"{name} must be a sequence")
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise AutonomousEvolutionError(f"{name} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise AutonomousEvolutionError(f"{name} must not contain duplicates")
    return normalized


def _validated_counts(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise AutonomousEvolutionError(f"{name} must be an object")
    normalized: dict[str, int] = {}
    for task_id, count in values.items():
        if not isinstance(task_id, str) or not task_id:
            raise AutonomousEvolutionError(f"{name} keys must be non-empty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise AutonomousEvolutionError(
                f"{name} values must be non-negative integers"
            )
        normalized[task_id] = count
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_json(item) for key, item in value.items()}
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
