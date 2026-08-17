"""Deterministic, score-blind trajectory observer for the real evolution bridge."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from real_evolution_bridge import ObservationFeatures


class TrajectoryObservationError(ValueError):
    """Raised when frozen trajectory evidence cannot be observed safely."""


_TEST_COMMAND = re.compile(
    r"(?:^|\s)(?:pytest|python\s+-m\s+pytest|cargo\s+test|go\s+test|"
    r"npm\s+(?:run\s+)?test|pnpm\s+(?:run\s+)?test|yarn\s+test|"
    r"mvn\s+test|gradle\s+test|make\s+test)(?:\s|$)",
    re.IGNORECASE,
)
_INSPECT_COMMAND = re.compile(
    r"(?:^|\s)(?:rg|grep|sed|find|ls|git\s+(?:status|diff|show|log)|"
    r"head|tail|cat)(?:\s|$)",
    re.IGNORECASE,
)


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _command(event: dict[str, Any]) -> str:
    commands = []
    for item in _walk(event):
        if not isinstance(item, dict):
            continue
        for key in ("command", "cmd"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                commands.append(value.strip())
    return "\n".join(commands)


def _is_edit(event: dict[str, Any]) -> bool:
    for item in _walk(event):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).lower()
        name = str(item.get("name", item.get("tool", ""))).lower()
        if kind in {"file_change", "file_edit", "apply_patch"}:
            return True
        if name in {"apply_patch", "write_file", "edit_file"}:
            return True
    return False


def _edit_paths(event: dict[str, Any]) -> tuple[str, ...]:
    paths = []
    for item in _walk(event):
        if not isinstance(item, dict):
            continue
        value = item.get("path")
        if isinstance(value, str) and value.strip():
            paths.append(value.strip().replace("\\", "/"))
    return tuple(dict.fromkeys(paths))


def _edit_scope_features(events: list[dict[str, Any]]) -> tuple[str, ...]:
    categories = set()
    edit_seen = False
    for event in events:
        if not _is_edit(event):
            continue
        edit_seen = True
        for raw_path in _edit_paths(event):
            path = PurePosixPath(raw_path)
            parts = {part.lower() for part in path.parts}
            name = path.name.lower()
            if name == "agents.md" or {".agents", ".codex"} & parts:
                categories.add("agent_program_artifact_edit")
            elif (
                {"test", "tests"} & parts
                or name.startswith("test_")
                or name.endswith(("_test.go", ".spec.ts", ".test.ts"))
            ):
                categories.add("test_artifact_edit")
            elif "docs" in parts or name.startswith("readme") or name.endswith(".md"):
                categories.add("documentation_edit")
            else:
                categories.add("task_source_edit")
    if edit_seen and not categories:
        categories.add("unknown_edit_scope")
    ordered = tuple(
        item
        for item in (
            "agent_program_artifact_edit",
            "task_source_edit",
            "test_artifact_edit",
            "documentation_edit",
            "unknown_edit_scope",
        )
        if item in categories
    )
    if len(categories) > 1:
        return (*ordered, "mixed_edit_scope")
    return ordered


def _load_events(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrajectoryObservationError("tool-event evidence is invalid") from error
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list) or any(
        not isinstance(item, dict) for item in events
    ):
        raise TrajectoryObservationError("tool-event evidence must contain objects")
    return events


_LANGUAGE_ALIASES = {"c++": "cpp"}


def _safe_condition(value: str) -> str:
    """Normalize a condition value to the pattern_miner identifier grammar.

    pattern_miner._IDENTIFIER allows [A-Za-z0-9._:/-] only; benchmark pools use
    language tags such as ``c++`` which would otherwise crash observation
    validation (incident: G1 confirmation, catchorg__Catch2-1616).
    """
    mapped = _LANGUAGE_ALIASES.get(str(value), str(value))
    cleaned = re.sub(r"[^A-Za-z0-9._:/-]", "_", mapped)
    return cleaned or "unknown"


class FrozenTrajectoryObserver:
    """Extract trace patterns without reading or changing native evaluator scores.

    This is the observable trajectory sidecar used for Codex API agents. It does not
    claim hidden-state/Jacobian access and never emits a correctness or admission bit.
    """

    observer_kind = "jlens_trajectory_sidecar"
    hidden_state_access = False
    admission_gate_allowed = False

    def __call__(self, context: Any) -> ObservationFeatures:
        events = _load_events(Path(context.tool_events_path))
        commands = [_command(event) for event in events]
        edit_indexes = [index for index, event in enumerate(events) if _is_edit(event)]
        inspect_indexes = [
            index
            for index, command in enumerate(commands)
            if _INSPECT_COMMAND.search(command)
        ]
        test_indexes = [
            index
            for index, command in enumerate(commands)
            if _TEST_COMMAND.search(command)
        ]
        features = []
        if edit_indexes:
            first_edit = edit_indexes[0]
            features.append(
                "inspect_before_edit"
                if any(index < first_edit for index in inspect_indexes)
                else "edit_before_inspect"
            )
            features.append(
                "test_before_edit"
                if any(index < first_edit for index in test_indexes)
                else "no_test_before_edit"
            )
            features.append(
                "test_after_edit"
                if any(index > first_edit for index in test_indexes)
                else "no_test_after_edit"
            )
            features.extend(_edit_scope_features(events))
        else:
            features.append("no_source_edit")
            features.append("tests_observed" if test_indexes else "no_tests_observed")
            features.append(
                "inspection_observed" if inspect_indexes else "no_inspection_observed"
            )
        if not events:
            features.append("empty_tool_trace")
        task = context.materialized_task
        conditions = tuple(
            sorted(
                {
                    f"benchmark:{_safe_condition(task.get('benchmark_id', 'unknown'))}",
                    f"language:{_safe_condition(task.get('language', 'unknown'))}",
                    f"repo:{_safe_condition(task.get('repo', 'unknown'))}",
                    f"stage:{_safe_condition(context.request.stage)}",
                }
            )
        )
        return ObservationFeatures(
            observed_features=tuple(dict.fromkeys(features)),
            conditions=conditions,
            expected_surfaces=(
                "prompt",
                "skills",
                "policy",
                "constrained_harness_code",
            ),
        )
