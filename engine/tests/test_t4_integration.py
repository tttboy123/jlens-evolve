"""Tests for t4_integration.py (v2.5 T4 landing path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from search_skill_bridge import SearchSkillBridgeError
from t4_integration import integrate


def _fake_artifact(tmp_path: Path, embedded: bool = True) -> Path:
    events = [
        {"kind": "read", "detail": {"path": "pkg/mod.py"}},
        {
            "kind": "generate",
            "detail": {"stage": "plan"},
            "layer_record_count": 32,
            "layer_records": (
                [
                    {
                        "step_label": "plan",
                        "layer": i,
                        "shape": [1, 8, 16],
                        "mean": float(i) * 0.01,
                        "l2_norm": float(i) + 1.0,
                    }
                    for i in range(32)
                ]
                if embedded
                else []
            ),
        },
        {"kind": "edit", "detail": {"path": "pkg/mod.py"}},
    ]
    artifact = tmp_path / "RUN-ARTIFACT.json"
    artifact.write_text(
        json.dumps(
            {
                "instance_id": "django__django-16493",
                "target_file": "pkg/mod.py",
                "patch": "diff --git a/pkg/mod.py b/pkg/mod.py",
                "tool_events": events,
                "layer_record_count": 32 if embedded else 0,
                "observational_boundary": "observational_not_causal",
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_integrate_compiles_candidate_and_records_ladder(tmp_path: Path) -> None:
    artifact = _fake_artifact(tmp_path)
    result = integrate(
        run_artifact=artifact,
        registry_root=tmp_path / "registry",
        skill_id="local-lens-django-16493",
        source_task_id="django__django-16493",
        instruction="Fix callable storage deconstruct bug",
    )
    assert result.candidate_skill_id == "local-lens-django-16493"
    assert result.pattern_card["observational_boundary"] == "observational_not_causal"
    assert result.pattern_card["layer_record_count"] == 32
    assert result.gate["paired_tasks"] == 0
    assert result.gate["passed"] is False  # no paired evidence yet
    assert result.ladder_status in {"candidate", "reviewed"}
    assert (tmp_path / "registry" / "ladder").exists()


def test_integrate_rejects_artifact_without_embedded_records(tmp_path: Path) -> None:
    artifact = _fake_artifact(tmp_path, embedded=False)
    with pytest.raises(SearchSkillBridgeError, match="no embedded layer_records"):
        integrate(
            run_artifact=artifact,
            registry_root=tmp_path / "registry",
            skill_id="x",
            source_task_id="django__django-16493",
            instruction="fix it",
        )


def test_integrate_with_gate_evidence(tmp_path: Path) -> None:
    artifact = _fake_artifact(tmp_path)
    paired = []
    for i in range(8):
        paired.append(
            {
                "task_uid": f"task-{i}",
                "role": "original",
                "native_score": 1.0,
                "safety_passed": True,
                "cost_units": 10,
                "matched_contract_sha256": "c" * 64,
                "native_evaluator_epoch": "epoch-1",
            }
        )
        paired.append(
            {
                "task_uid": f"task-{i}",
                "role": "candidate",
                "native_score": 1.0,
                "safety_passed": True,
                "cost_units": 10,
                "matched_contract_sha256": "c" * 64,
                "native_evaluator_epoch": "epoch-1",
            }
        )
    result = integrate(
        run_artifact=artifact,
        registry_root=tmp_path / "registry",
        skill_id="local-lens-django-16493",
        source_task_id="django__django-16493",
        instruction="Fix it",
        paired_evals=paired,
        expected_contract_sha256="c" * 64,
        expected_evaluator_epoch="epoch-1",
    )
    assert result.gate["paired_tasks"] == 8
    assert result.gate["passed"] is True
    assert result.ladder_status == "reviewed"
