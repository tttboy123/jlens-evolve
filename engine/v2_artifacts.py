"""Create source snapshots and a SHA-256 manifest for the v2 evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/v2.0.0"
STAGES = ("v1.1.0-codex-target", "v2.0.0-meta-evolution")
V11_SOURCES = (
    Path("codex_target_runtime.py"),
    Path("codex_changeset.py"),
    Path("codex_evolution_runtime.py"),
    Path("tests/test_codex_target_runtime.py"),
    Path("tests/test_codex_changeset.py"),
    Path("tests/test_codex_evolution_runtime.py"),
)
V20_SOURCES = (
    Path("changeset_adapter.py"),
    Path("meta_evolution_runtime.py"),
    Path("live_codex_ab.py"),
    Path("multi_model_eval.py"),
    Path("swe_bench_adapter.py"),
    Path("swe_cloud_runner.py"),
    Path("v2_artifacts.py"),
    Path("evolve_service.py"),
    Path("release_candidate.py"),
    Path("codex_target_runtime.py"),
    Path("pytest.ini"),
    Path("tests/test_changeset_adapter.py"),
    Path("tests/test_meta_evolution_runtime.py"),
    Path("tests/test_live_codex_ab.py"),
    Path("tests/test_multi_model_eval.py"),
    Path("tests/test_swe_bench_adapter.py"),
    Path("tests/test_swe_cloud_runner.py"),
    Path("tests/test_evolve_service.py"),
    Path("tests/test_v2_artifacts.py"),
)


class V2ArtifactError(ValueError):
    """Raised when snapshot or manifest targets are unsafe or incomplete."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_source(project_root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise V2ArtifactError(f"unsafe source path: {relative}")
    source = (project_root / relative).resolve()
    if not source.is_relative_to(project_root.resolve()) or not source.is_file():
        raise V2ArtifactError(f"source file missing or outside project: {relative}")
    return source


def snapshot_sources(
    *, project_root: Path, stage_root: Path, sources: tuple[Path, ...]
) -> dict[str, Any]:
    project_root = project_root.resolve()
    stage_root = stage_root.resolve()
    snapshot_root = stage_root / "source-snapshot"
    rows = []
    for relative in sources:
        source = _safe_source(project_root, relative)
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_sha256 = _sha256_file(source)
        snapshot_sha256 = _sha256_file(target)
        if source_sha256 != snapshot_sha256:
            raise V2ArtifactError(f"snapshot hash mismatch: {relative}")
        rows.append(
            {
                "source_path": str(source),
                "snapshot_path": target.relative_to(stage_root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": snapshot_sha256,
            }
        )
    manifest = {
        "schema_version": 1,
        "source_control": "sha256_snapshot_no_git",
        "project_root": str(project_root),
        "files": rows,
    }
    (stage_root / "SOURCE-MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def snapshot_release_sources(
    *, project_root: Path = ROOT, artifact_root: Path = DEFAULT_ARTIFACT_ROOT
) -> dict[str, Any]:
    return {
        STAGES[0]: snapshot_sources(
            project_root=project_root,
            stage_root=artifact_root / STAGES[0],
            sources=V11_SOURCES,
        ),
        STAGES[1]: snapshot_sources(
            project_root=project_root,
            stage_root=artifact_root / STAGES[1],
            sources=V20_SOURCES,
        ),
    }


def build_v2_manifest(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    artifact_root = artifact_root.resolve()
    stages = []
    for stage_id in STAGES:
        stage_root = artifact_root / stage_id
        decision_path = stage_root / "DECISION.json"
        if not decision_path.is_file():
            raise V2ArtifactError(f"stage decision missing: {stage_id}")
        decision = json.loads(decision_path.read_text(encoding="utf-8")).get("decision")
        if decision != "accepted":
            raise V2ArtifactError(f"stage release decision is not accepted: {stage_id}")
        files = []
        for path in sorted(item for item in stage_root.rglob("*") if item.is_file()):
            files.append(
                {
                    "path": path.relative_to(artifact_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        stages.append({"stage": stage_id, "decision": decision, "files": files})
    manifest = {
        "schema_version": 1,
        "release_version": "2.0.0",
        "artifact_root": str(artifact_root),
        "source_control": "sha256_snapshot_no_git",
        "stages": stages,
    }
    manifest_path = artifact_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--snapshot-sources", action="store_true")
    args = parser.parse_args()
    if args.snapshot_sources:
        snapshot_release_sources(
            project_root=args.project_root, artifact_root=args.artifact_root
        )
    manifest = build_v2_manifest(args.artifact_root)
    print(
        json.dumps({"manifest": str(manifest), "status": "built"}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
