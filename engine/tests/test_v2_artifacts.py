from __future__ import annotations

import json
from pathlib import Path

from artifact_verifier import verify_manifest
from v2_artifacts import V20_SOURCES, build_v2_manifest, snapshot_sources


def test_v20_source_snapshot_tracks_cloud_runner_contract():
    assert Path("swe_cloud_runner.py") in V20_SOURCES
    assert Path("tests/test_swe_cloud_runner.py") in V20_SOURCES


def test_source_snapshot_and_manifest_are_hash_verified(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "tests").mkdir()
    (project / "tests/test_runtime.py").write_text(
        "def test_value():\n    assert 1 == 1\n", encoding="utf-8"
    )
    artifacts = project / "artifacts/v2.0.0"
    for stage in ("v1.1.0-codex-target", "v2.0.0-meta-evolution"):
        stage_root = artifacts / stage
        stage_root.mkdir(parents=True)
        (stage_root / "DECISION.json").write_text(
            json.dumps({"decision": "accepted"}), encoding="utf-8"
        )
        (stage_root / "RESULT.zh-CN.md").write_text("通过\n", encoding="utf-8")
    (artifacts / "GOAL.zh-CN.md").write_text("goal\n", encoding="utf-8")

    source_manifest = snapshot_sources(
        project_root=project,
        stage_root=artifacts / "v1.1.0-codex-target",
        sources=(Path("runtime.py"), Path("tests/test_runtime.py")),
    )
    manifest = build_v2_manifest(artifacts)
    verification = verify_manifest(manifest)

    assert source_manifest["source_control"] == "sha256_snapshot_no_git"
    assert len(source_manifest["files"]) == 2
    assert verification["valid"] is True
    assert verification["stages_verified"] == 2
    assert verification["files_verified"] >= 7

    (artifacts / "v1.1.0-codex-target/RESULT.zh-CN.md").write_text(
        "tampered\n", encoding="utf-8"
    )
    assert verify_manifest(manifest)["valid"] is False
