import hashlib
import json
from pathlib import Path

from artifact_verifier import verify_manifest
from v21_artifacts import (
    V211_SOURCES,
    build_v21_manifest,
    snapshot_v21_sources,
    snapshot_v211_sources,
)


def test_v21_snapshot_and_manifest_are_read_only_verifiable(tmp_path: Path) -> None:
    project = tmp_path / "project"
    artifact_root = project / "artifacts/v2.1.0"
    stage = artifact_root / "v2.1.0-continuous-ab"
    stage.mkdir(parents=True)
    sources = (
        "benchmark_adapters.py",
        "benchmark_catalog.py",
        "continuous_ab.py",
        "continuous_ab_service.py",
        "continuous_ab_smoke.py",
        "freeze_benchmark_pool.py",
        "verify_continuous_ab.py",
        "v21_artifacts.py",
        "evolve_jlens_harbor.py",
        "pilot_admission.py",
        "benchmark_execution.py",
        "agent_arm_runner.py",
        "native_result_adapter.py",
        "release_candidate.py",
        "tests/test_benchmark_adapters.py",
        "tests/test_benchmark_catalog.py",
        "tests/test_continuous_ab.py",
        "tests/test_continuous_ab_service.py",
        "tests/test_continuous_ab_smoke.py",
        "tests/test_freeze_benchmark_pool.py",
        "tests/test_verify_continuous_ab.py",
        "tests/test_v21_artifacts.py",
        "tests/test_evolve_jlens_harbor.py",
        "tests/test_pilot_admission.py",
        "tests/test_benchmark_execution.py",
        "tests/test_agent_arm_runner.py",
        "tests/test_native_result_adapter.py",
    )
    for relative in sources:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
    (stage / "DECISION.json").write_text(
        json.dumps({"decision": "accepted"}), encoding="utf-8"
    )

    source_manifest = snapshot_v21_sources(
        project_root=project, artifact_root=artifact_root
    )
    manifest_path = build_v21_manifest(artifact_root)
    verification = verify_manifest(manifest_path)

    assert len(source_manifest["files"]) == len(sources)
    assert all(
        row["sha256"]
        == hashlib.sha256((project / row["source_path"]).read_bytes()).hexdigest()
        if not Path(row["source_path"]).is_absolute()
        else row["sha256"]
        == hashlib.sha256(Path(row["source_path"]).read_bytes()).hexdigest()
        for row in source_manifest["files"]
    )
    assert verification["valid"] is True
    assert not any(
        row["path"] == "MANIFEST.json"
        for row in json.loads(manifest_path.read_text())["stages"][0]["files"]
    )


def test_v211_snapshot_adds_second_stage_without_rewriting_v21_evidence(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    artifact_root = project / "artifacts/v2.1.0"
    old_stage = artifact_root / "v2.1.0-continuous-ab"
    new_stage = artifact_root / "v2.1.1-jlens-evolution"
    old_stage.mkdir(parents=True)
    new_stage.mkdir(parents=True)
    old_decision = old_stage / "DECISION.json"
    old_decision.write_text(json.dumps({"decision": "accepted"}), encoding="utf-8")
    (new_stage / "DECISION.json").write_text(
        json.dumps({"decision": "accepted"}), encoding="utf-8"
    )
    for relative in V211_SOURCES:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(relative), encoding="utf-8")
    old_hash = hashlib.sha256(old_decision.read_bytes()).hexdigest()

    snapshot = snapshot_v211_sources(project_root=project, artifact_root=artifact_root)
    manifest_path = build_v21_manifest(artifact_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = verify_manifest(manifest_path)

    assert len(snapshot["files"]) == len(V211_SOURCES)
    assert [stage["stage"] for stage in manifest["stages"]] == [
        "v2.1.0-continuous-ab",
        "v2.1.1-jlens-evolution",
    ]
    assert verification["valid"] is True
    assert hashlib.sha256(old_decision.read_bytes()).hexdigest() == old_hash


def _frozen_tmp_stage(tmp_path: Path):
    from v21_artifacts import V21_SOURCES

    project = tmp_path / "project"
    artifact_root = project / "artifacts/v2.1.0"
    stage = artifact_root / "v2.1.0-continuous-ab"
    stage.mkdir(parents=True)
    for relative in V21_SOURCES:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(relative), encoding="utf-8")
    (stage / "DECISION.json").write_text(
        json.dumps({"decision": "accepted"}), encoding="utf-8"
    )
    (stage / "FROZEN.json").write_text(
        json.dumps({"schema_version": 1, "frozen": True}), encoding="utf-8"
    )
    return project, artifact_root


def test_frozen_stage_refuses_resnapshot_without_override(tmp_path):
    from v21_artifacts import V21ArtifactError, snapshot_v21_sources

    project, artifact_root = _frozen_tmp_stage(tmp_path)
    try:
        snapshot_v21_sources(project_root=project, artifact_root=artifact_root)
        raise AssertionError("expected frozen-stage refusal")
    except V21ArtifactError as error:
        assert "is frozen" in str(error)
    assert not (artifact_root / "v2.1.0-continuous-ab/FREEZE-OVERRIDE.json").exists()


def test_frozen_stage_override_is_audited(tmp_path):
    from v21_artifacts import snapshot_v21_sources

    project, artifact_root = _frozen_tmp_stage(tmp_path)
    source_manifest = snapshot_v21_sources(
        project_root=project,
        artifact_root=artifact_root,
        frozen_override_reason="test-only audited override",
    )
    assert source_manifest["files"]
    override = json.loads(
        (artifact_root / "v2.1.0-continuous-ab/FREEZE-OVERRIDE.json").read_text()
    )
    assert override["frozen_override_reason"] == "test-only audited override"


def test_real_v210_stage_is_frozen_protected():
    from v21_artifacts import stage_is_frozen

    stage = (
        Path(__file__).resolve().parents[1] / "artifacts/v2.1.0/v2.1.0-continuous-ab"
    )
    assert stage_is_frozen(stage)
    assert (stage / "FROZEN.json").is_file()
