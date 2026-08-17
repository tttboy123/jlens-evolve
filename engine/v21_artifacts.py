"""Create source snapshots and a SHA-256 manifest for v2.1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from v2_artifacts import snapshot_sources

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts/v2.1.0"
STAGE_ID = "v2.1.0-continuous-ab"
V211_STAGE_ID = "v2.1.1-jlens-evolution"
V21_SOURCES = tuple(
    Path(item)
    for item in (
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
)
V211_SOURCES = tuple(
    Path(item)
    for item in (
        "pattern_miner.py",
        "mutation_proposer.py",
        "evolution_archive.py",
        "candidate_tournament.py",
        "evolution_controller.py",
        "evolution_runtime.py",
        "evolution_report.py",
        "evolution_fixture.py",
        "evolve_service.py",
        "agent_arm_runner.py",
        "benchmark_execution.py",
        "native_result_adapter.py",
        "verify_continuous_ab.py",
        "real_evolution_bridge.py",
        "real_mutation_proposer.py",
        "codex_mutation_caller.py",
        "trace_observer.py",
        "real_workspace_factory.py",
        "loopback_connect_proxy.py",
        "official_patch_evaluator.py",
        "real_evolution_run.py",
        "release_candidate.py",
        "v21_artifacts.py",
        "tests/test_pattern_miner.py",
        "tests/test_mutation_proposer.py",
        "tests/test_evolution_archive.py",
        "tests/test_candidate_tournament.py",
        "tests/test_evolution_controller.py",
        "tests/test_evolution_integration.py",
        "tests/test_agent_arm_runner.py",
        "tests/test_benchmark_execution.py",
        "tests/test_native_result_adapter.py",
        "tests/test_verify_continuous_ab.py",
        "tests/test_real_evolution_bridge.py",
        "tests/test_real_mutation_proposer.py",
        "tests/test_codex_mutation_caller.py",
        "tests/test_trace_observer.py",
        "tests/test_real_workspace_factory.py",
        "tests/test_loopback_connect_proxy.py",
        "tests/test_official_patch_evaluator.py",
        "tests/test_real_evolution_run.py",
        "tests/test_v21_artifacts.py",
    )
)


class V21ArtifactError(ValueError):
    """Raised when the v2.1 decision or artifact tree is incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


FROZEN_MARKER = "FROZEN.json"
FREEZE_OVERRIDE_MARKER = "FREEZE-OVERRIDE.json"


def stage_is_frozen(stage_root: Path) -> bool:
    return (stage_root / FROZEN_MARKER).is_file()


def _record_freeze_override(stage_root: Path, reason: str) -> None:
    record = {
        "schema_version": 1,
        "stage": stage_root.name,
        "frozen_override_reason": reason,
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = stage_root / FREEZE_OVERRIDE_MARKER
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _assert_snapshot_allowed(
    stage_root: Path, *, frozen_override_reason: str | None
) -> None:
    """Refuse to re-snapshot a frozen stage unless an audited override is given."""
    if not stage_is_frozen(stage_root):
        return
    if not frozen_override_reason:
        raise V21ArtifactError(
            f"stage {stage_root.name} is frozen ({FROZEN_MARKER} present); "
            f"refusing to re-snapshot (use --freeze-override <reason>)"
        )
    _record_freeze_override(stage_root, frozen_override_reason)


def snapshot_v21_sources(
    *,
    project_root: Path = ROOT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    frozen_override_reason: str | None = None,
) -> dict[str, Any]:
    stage_root = artifact_root / STAGE_ID
    _assert_snapshot_allowed(stage_root, frozen_override_reason=frozen_override_reason)
    return snapshot_sources(
        project_root=project_root,
        stage_root=stage_root,
        sources=V21_SOURCES,
    )


def snapshot_v211_sources(
    *,
    project_root: Path = ROOT,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    frozen_override_reason: str | None = None,
) -> dict[str, Any]:
    stage_root = artifact_root / V211_STAGE_ID
    _assert_snapshot_allowed(stage_root, frozen_override_reason=frozen_override_reason)
    return snapshot_sources(
        project_root=project_root,
        stage_root=stage_root,
        sources=V211_SOURCES,
    )


def build_v21_manifest(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    artifact_root = artifact_root.resolve()
    stages = []
    for stage_id in (STAGE_ID, V211_STAGE_ID):
        stage_root = artifact_root / stage_id
        decision_path = stage_root / "DECISION.json"
        if stage_id == V211_STAGE_ID and not decision_path.is_file():
            continue
        if not decision_path.is_file():
            raise V21ArtifactError(f"stage decision missing: {decision_path}")
        decision = json.loads(decision_path.read_text(encoding="utf-8")).get("decision")
        if decision != "accepted":
            raise V21ArtifactError(
                f"software stage decision is not accepted: {stage_id}={decision}"
            )
        files = [
            {
                "path": path.relative_to(artifact_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in sorted(item for item in stage_root.rglob("*") if item.is_file())
        ]
        stages.append({"stage": stage_id, "decision": decision, "files": files})
    manifest = {
        "schema_version": 1,
        "release_version": "2.1.x-local-evolution",
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
    parser.add_argument(
        "--freeze-override",
        type=str,
        default=None,
        help="audited reason to re-snapshot a frozen stage (writes FREEZE-OVERRIDE.json)",
    )
    args = parser.parse_args()
    if args.snapshot_sources:
        for snapshot in (
            lambda: snapshot_v21_sources(
                project_root=args.project_root,
                artifact_root=args.artifact_root,
                frozen_override_reason=args.freeze_override,
            ),
            lambda: snapshot_v211_sources(
                project_root=args.project_root,
                artifact_root=args.artifact_root,
                frozen_override_reason=args.freeze_override,
            ),
        ):
            try:
                snapshot()
            except V21ArtifactError as error:
                if "is frozen" in str(error):
                    print(f"SKIP frozen stage: {error}")
                else:
                    raise
    manifest = build_v21_manifest(args.artifact_root)
    print(json.dumps({"manifest": str(manifest), "status": "built"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
