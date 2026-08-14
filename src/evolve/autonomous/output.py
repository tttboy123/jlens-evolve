"""Immutable product artifacts and manifest sealing for autonomous evolution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolve.contracts import ContractViolation, canonical_json, content_sha256
from evolve.proposals import CompiledRevision
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
    compiled = CompiledRevision.load(round_root)
    if (
        compiled.change_set.candidate_id != candidate_id
        or compiled.change_set.revision_id != candidate_revision_id
        or compiled.bundle_sha256 != bundle_sha256
    ):
        raise AutonomousEvolutionError("best Harness compiled identity mismatch")
    names = [name for name, _ in compiled.artifact_sha256]
    names.append("COMPILED-REVISION.json")
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
    memory = round_root / "COMPILED-MEMORY-POLICY.json"
    if not memory.is_file():
        (best_root / "COMPILED-MEMORY-POLICY.json").unlink(missing_ok=True)
    loaded = CompiledRevision.load(best_root)
    if loaded.bundle_sha256 != bundle_sha256:
        raise AutonomousEvolutionError("exported best Harness failed reload")
    gains = tuple(
        claim.task_id for claim in claims if claim.classification == "gain"
    )
    regressions = tuple(
        claim.task_id for claim in claims if claim.classification == "regression"
    )
    harness = {
        "schema_version": 1,
        "harness_kind": "compiled-candidate",
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


def export_empty_harness(
    *, output_root: Path, model_identity_sha256: str
) -> Path:
    """Project the immutable no-intervention parent as the initial best Harness."""

    definition = {
        "kind": "empty-harness",
        "revision_id": "empty-harness-v1",
        "external_intervention": None,
    }
    harness = {
        "schema_version": 1,
        "harness_kind": "empty",
        "model_identity_sha256": model_identity_sha256,
        "candidate_id": None,
        "candidate_revision_id": "empty-harness-v1",
        "compiled_bundle_sha256": content_sha256(definition),
        "skill_path": None,
        "operator_path": None,
        "router_path": None,
        "memory_policy_path": None,
        "supported_task_signatures": [],
        "native_gain_task_ids": [],
        "regression_task_ids": [],
        "source_claim_ids": [],
        "active": False,
    }
    return atomic_json(output_root.resolve() / "best/BEST-HARNESS.json", harness)


def load_best_harness(
    path: str | Path, *, expected_model_identity_sha256: str
) -> CompiledRevision | None:
    """Hash-verify a BEST-HARNESS projection for the frozen model runtime."""

    return verify_best_harness(
        path,
        expected_model_identity_sha256=expected_model_identity_sha256,
    )


def verify_best_harness(
    path: str | Path,
    *,
    expected_model_identity_sha256: str,
    accepted_round: Mapping[str, Any] | None = None,
) -> CompiledRevision | None:
    """Verify BEST-HARNESS against runtime and optional accepted-round authority."""

    source = Path(path).resolve()
    try:
        harness = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractViolation("BEST-HARNESS is unreadable") from error
    if (
        not isinstance(harness, dict)
        or set(harness) != _BEST_HARNESS_FIELDS
        or harness.get("schema_version") != 1
    ):
        raise ContractViolation("BEST-HARNESS schema is invalid")
    if harness.get("model_identity_sha256") != expected_model_identity_sha256:
        raise ContractViolation("BEST-HARNESS model identity mismatch")
    if harness.get("active") is not False:
        raise ContractViolation("BEST-HARNESS must remain inactive")
    kind = harness.get("harness_kind", "compiled-candidate")
    if kind == "empty":
        expected = {
            "kind": "empty-harness",
            "revision_id": "empty-harness-v1",
            "external_intervention": None,
        }
        if (
            harness.get("candidate_id") is not None
            or harness.get("candidate_revision_id") != "empty-harness-v1"
            or harness.get("compiled_bundle_sha256") != content_sha256(expected)
            or any(
                harness.get(name) is not None
                for name in (
                    "skill_path",
                    "operator_path",
                    "router_path",
                    "memory_policy_path",
                )
                )
            ):
                raise ContractViolation("empty BEST-HARNESS identity is invalid")
        for name in (
            "supported_task_signatures",
            "native_gain_task_ids",
            "regression_task_ids",
            "source_claim_ids",
        ):
            if harness.get(name) != []:
                raise ContractViolation("empty BEST-HARNESS list projection is invalid")
        if accepted_round is not None:
            _verify_empty_round_projection(harness, accepted_round)
        return None
    if kind != "compiled-candidate":
        raise ContractViolation("BEST-HARNESS kind is unsupported")
    compiled = CompiledRevision.load(source.parent)
    if (
        harness.get("candidate_id") != compiled.change_set.candidate_id
        or harness.get("candidate_revision_id") != compiled.change_set.revision_id
        or harness.get("compiled_bundle_sha256") != compiled.bundle_sha256
    ):
        raise ContractViolation("BEST-HARNESS compiled identity mismatch")
    expected_paths = {
        "skill_path": "COMPILED-SKILL.json",
        "operator_path": "COMPILED-OPERATOR.json",
        "router_path": "COMPILED-ROUTER.json",
        "memory_policy_path": (
            "COMPILED-MEMORY-POLICY.json"
            if compiled.memory_policy is not None
            else None
        ),
    }
    if any(harness.get(name) != value for name, value in expected_paths.items()):
        raise ContractViolation("BEST-HARNESS artifact projection mismatch")
    for name in (
        "supported_task_signatures",
        "native_gain_task_ids",
        "regression_task_ids",
        "source_claim_ids",
    ):
        _string_list(harness.get(name), f"BEST-HARNESS {name}")
    if accepted_round is not None:
        _verify_accepted_round_projection(harness, accepted_round)
    return compiled


_BEST_HARNESS_FIELDS = {
    "schema_version",
    "harness_kind",
    "model_identity_sha256",
    "candidate_id",
    "candidate_revision_id",
    "compiled_bundle_sha256",
    "skill_path",
    "operator_path",
    "router_path",
    "memory_policy_path",
    "supported_task_signatures",
    "native_gain_task_ids",
    "regression_task_ids",
    "source_claim_ids",
    "active",
}


def _verify_empty_round_projection(
    harness: Mapping[str, Any], accepted_round: Mapping[str, Any]
) -> None:
    if (
        accepted_round.get("accepted_as_best") is not False
        or accepted_round.get("candidate_id") is not None
        or accepted_round.get("candidate_revision_id") != "empty-harness-v1"
        or accepted_round.get("compiled_bundle_sha256")
        != harness.get("compiled_bundle_sha256")
        or accepted_round.get("claims") not in (None, [])
    ):
        raise ContractViolation("BEST-HARNESS authoritative round projection mismatch")


def _verify_accepted_round_projection(
    harness: Mapping[str, Any], accepted_round: Mapping[str, Any]
) -> None:
    if (
        accepted_round.get("accepted_as_best") is not True
        or accepted_round.get("candidate_id") != harness.get("candidate_id")
        or accepted_round.get("candidate_revision_id")
        != harness.get("candidate_revision_id")
        or accepted_round.get("compiled_bundle_sha256")
        != harness.get("compiled_bundle_sha256")
    ):
        raise ContractViolation("BEST-HARNESS authoritative round projection mismatch")
    claims = accepted_round.get("claims")
    if not isinstance(claims, list):
        raise ContractViolation("BEST-HARNESS authoritative round projection mismatch")
    expected = _claim_projection(claims)
    actual = {
        "supported_task_signatures": harness.get("supported_task_signatures"),
        "native_gain_task_ids": harness.get("native_gain_task_ids"),
        "regression_task_ids": harness.get("regression_task_ids"),
        "source_claim_ids": harness.get("source_claim_ids"),
    }
    if actual != expected:
        raise ContractViolation("BEST-HARNESS authoritative round projection mismatch")


def _claim_projection(claims: Sequence[Any]) -> dict[str, list[str]]:
    rows: list[Mapping[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise ContractViolation("BEST-HARNESS authoritative claim is invalid")
        rows.append(claim)
    task_ids = [_required_text(row.get("task_id"), "task_id") for row in rows]
    claim_ids = [_required_text(row.get("claim_id"), "claim_id") for row in rows]
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or len(set(claim_ids)) != len(claim_ids)
    ):
        raise ContractViolation(
            "BEST-HARNESS authoritative claims must be non-empty and unique"
        )
    return {
        "supported_task_signatures": sorted(set(task_ids)),
        "native_gain_task_ids": sorted(
            task_id
            for task_id, row in zip(task_ids, rows, strict=True)
            if row.get("classification") == "gain"
        ),
        "regression_task_ids": sorted(
            task_id
            for task_id, row in zip(task_ids, rows, strict=True)
            if row.get("classification") == "regression"
        ),
        "source_claim_ids": claim_ids,
    }


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractViolation(f"BEST-HARNESS authoritative {field} is invalid")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ContractViolation(f"{field} is invalid")
    return value


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
