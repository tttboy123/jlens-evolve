from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from real_evolution_bridge import ObservationContext
from trace_observer import FrozenTrajectoryObserver


def _context(
    tmp_path: Path,
    events: list[dict],
    task: dict | None = None,
) -> ObservationContext:
    trajectory = tmp_path / "events.jsonl"
    trajectory.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    tool_events = tmp_path / "tool-events.json"
    tool_events.write_text(
        json.dumps({"schema_version": 1, "events": events}), encoding="utf-8"
    )
    evidence = tmp_path / "native.json"
    cost = tmp_path / "cost.json"
    safety = tmp_path / "safety.json"
    for path in (evidence, cost, safety):
        path.write_text("{}", encoding="utf-8")
    materialized_task = task or {
        "benchmark_id": "swe-bench-verified",
        "repo": "django/django",
        "language": "python",
    }
    return ObservationContext(
        request=SimpleNamespace(stage="scout"),
        materialized_task=materialized_task,
        trajectory_path=trajectory,
        tool_events_path=tool_events,
        native_evidence_path=evidence,
        cost_path=cost,
        safety_path=safety,
    )


def test_observer_normalizes_cpp_language_condition(tmp_path: Path):
    """Incident regression: language 'c++' must not crash pattern_miner validation."""
    context = _context(
        tmp_path,
        [],
        task={
            "benchmark_id": "multi-swe-bench-flash",
            "repo": "catchorg/Catch2",
            "language": "c++",
        },
    )
    features = FrozenTrajectoryObserver()(context)
    assert "language:cpp" in features.conditions
    assert "language:c++" not in features.conditions
    import re as _re

    identifier = _re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
    assert all(identifier.fullmatch(cond) for cond in features.conditions)


def test_trace_observer_detects_inspect_test_edit_and_retest_sequence(tmp_path: Path):
    context = _context(
        tmp_path,
        [
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "command": "rg -n bug src"},
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "pytest -q tests/test_bug.py",
                },
            },
            {
                "type": "item.completed",
                "item": {"type": "file_change", "path": "src/fix.py"},
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "pytest -q tests/test_bug.py",
                },
            },
        ],
    )

    features = FrozenTrajectoryObserver()(context)

    assert "inspect_before_edit" in features.observed_features
    assert "test_before_edit" in features.observed_features
    assert "test_after_edit" in features.observed_features
    assert features.conditions == (
        "benchmark:swe-bench-verified",
        "language:python",
        "repo:django/django",
        "stage:scout",
    )
    assert features.expected_surfaces == (
        "prompt",
        "skills",
        "policy",
        "constrained_harness_code",
    )


def test_trace_observer_records_missing_validation_as_observation_not_failure(
    tmp_path: Path,
):
    context = _context(
        tmp_path,
        [
            {
                "type": "item.completed",
                "item": {"type": "file_change", "path": "src/fix.py"},
            },
        ],
    )

    features = FrozenTrajectoryObserver()(context)

    assert "edit_before_inspect" in features.observed_features
    assert "no_test_before_edit" in features.observed_features
    assert "no_test_after_edit" in features.observed_features
    assert not any(
        "success" in item or "failure" in item for item in features.observed_features
    )


def test_trace_observer_separates_agent_artifact_and_task_source_edits(
    tmp_path: Path,
):
    context = _context(
        tmp_path,
        [
            {
                "type": "item.completed",
                "item": {
                    "type": "file_change",
                    "changes": [
                        {"path": "/workspace/AGENTS.md", "kind": "add"},
                        {"path": "/workspace/src/fix.py", "kind": "update"},
                    ],
                },
            }
        ],
    )

    features = FrozenTrajectoryObserver()(context)

    assert "agent_program_artifact_edit" in features.observed_features
    assert "task_source_edit" in features.observed_features
    assert "mixed_edit_scope" in features.observed_features
