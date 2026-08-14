"""Immutable product artifacts and manifest sealing for autonomous evolution."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolve.contracts import canonical_json
from evolve.reporting import AuditVerifier

from .config import AutonomousEvolutionError, ModelConfig
from .verification import VerifiedCampaignClaim


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def freeze_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).resolve()
    encoded = (canonical_json(payload) + "\n").encode()
    if target.exists():
        if target.read_bytes() != encoded:
            raise AutonomousEvolutionError(
                f"immutable autonomous artifact conflict: {target.name}"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, encoded) != len(encoded):
            raise AutonomousEvolutionError(
                f"partial autonomous artifact write: {target.name}"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model: ModelConfig) -> tuple[str, dict[str, str]]:
    files = {
        name: file_sha256(model.model_path / name)
        for name in model.model_identity_files
    }
    identity = hashlib.sha256(canonical_json(files).encode()).hexdigest()
    return identity, files


def export_best_harness(
    *,
    output_root: Path,
    round_root: Path,
    model_identity_sha256: str,
    candidate_id: str,
    candidate_revision_id: str,
    bundle_sha256: str,
    claims: Sequence[VerifiedCampaignClaim],
) -> Path:
    best_root = output_root.resolve() / "best"
    best_root.mkdir(parents=True, exist_ok=True)
    names = [
        "COMPILED-SKILL.json",
        "COMPILED-OPERATOR.json",
        "COMPILED-ROUTER.json",
        "COMPILED-REVISION.json",
    ]
    memory = round_root / "COMPILED-MEMORY-POLICY.json"
    if memory.is_file():
        names.append(memory.name)
    for name in names:
        source = round_root / name
        if not source.is_file():
            raise AutonomousEvolutionError(
                f"best Harness artifact is missing: {name}"
            )
        target = best_root / name
        if target.exists() and target.read_bytes() == source.read_bytes():
            continue
        temporary = best_root / f".{name}.new"
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    gains = tuple(
        claim.task_id for claim in claims if claim.classification == "gain"
    )
    regressions = tuple(
        claim.task_id for claim in claims if claim.classification == "regression"
    )
    harness = {
        "schema_version": 1,
        "model_identity_sha256": model_identity_sha256,
        "candidate_id": candidate_id,
        "candidate_revision_id": candidate_revision_id,
        "compiled_bundle_sha256": bundle_sha256,
        "skill_path": "COMPILED-SKILL.json",
        "operator_path": "COMPILED-OPERATOR.json",
        "router_path": "COMPILED-ROUTER.json",
        "memory_policy_path": (
            "COMPILED-MEMORY-POLICY.json" if memory.is_file() else None
        ),
        "supported_task_signatures": sorted({claim.task_id for claim in claims}),
        "native_gain_task_ids": sorted(gains),
        "regression_task_ids": sorted(regressions),
        "source_claim_ids": [claim.claim_id for claim in claims],
        "active": False,
    }
    return atomic_json(best_root / "BEST-HARNESS.json", harness)


def seal_manifest(root: str | Path) -> int:
    run_root = Path(root).resolve()
    manifest = run_root / "EVIDENCE-MANIFEST.json"
    entries = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path == manifest or path.name.endswith(".writer.lock"):
            continue
        entries.append(
            {"path": path.relative_to(run_root).as_posix(), "sha256": file_sha256(path)}
        )
    atomic_json(manifest, {"schema_version": 1, "entries": entries})
    return AuditVerifier().verify_manifest(manifest, root=run_root)
