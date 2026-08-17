"""Tests for the v2.5 local-lens skill compilation (T4 bridge)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from search_skill_bridge import (
    SearchSkillBridgeError,
    compile_local_lens_candidate,
)
from skill_registry import SkillRegistry


def _write_run_artifact(path: Path, *, layer_count: int = 32) -> Path:
    artifact = {
        "target_file": "django/db/models/fields/files.py",
        "patch": "--- a/django/db/models/fields/files.py\n+++ b/django/db/models/fields/files.py\n@@ -1,3 +1,4 @@\n+change\n",
        "tool_events": [
            {"kind": "read", "detail": {"path": "x"}, "layer_record_count": 0},
            {
                "kind": "generate",
                "detail": {"stage": "plan"},
                "layer_record_count": layer_count,
            },
        ],
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_compile_local_lens_candidate(tmp_path: Path) -> None:
    artifact = _write_run_artifact(tmp_path / "RUN-ARTIFACT.json")
    registry_root = tmp_path / "registry"
    candidate = compile_local_lens_candidate(
        run_artifact=artifact,
        registry_root=registry_root,
        skill_id="local-lens-django-storage",
        source_task_id="django__django-16493",
        reusable_instruction="Inspect deconstruct() and preserve callable storage semantics.",
    )
    assert candidate.status == "candidate"
    assert candidate.project_local_only is True
    assert candidate.auto_install is False
    assert candidate.active is False
    assert candidate.counterexamples
    assert candidate.known_failure_modes
    assert any("layer_records: 32" in line for line in candidate.content)

    registry = SkillRegistry(registry_root)
    latest = registry.latest(candidate.skill_id)
    assert latest is not None
    assert latest.revision_id == candidate.revision_id
    assert (registry_root / "skills" / candidate.skill_id / "SKILL.md").is_file()


def test_compile_refuses_without_layer_records(tmp_path: Path) -> None:
    artifact = _write_run_artifact(tmp_path / "RUN-ARTIFACT.json", layer_count=0)
    with pytest.raises(SearchSkillBridgeError, match="no layer_records"):
        compile_local_lens_candidate(
            run_artifact=artifact,
            registry_root=tmp_path / "registry",
            skill_id="s",
            source_task_id="t",
            reusable_instruction="instruction",
        )


def test_compile_requires_instruction(tmp_path: Path) -> None:
    artifact = _write_run_artifact(tmp_path / "RUN-ARTIFACT.json")
    with pytest.raises(SearchSkillBridgeError, match="reusable_instruction"):
        compile_local_lens_candidate(
            run_artifact=artifact,
            registry_root=tmp_path / "registry",
            skill_id="s",
            source_task_id="t",
            reusable_instruction="",
        )
