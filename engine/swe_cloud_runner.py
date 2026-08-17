"""Fail-closed primitives for ephemeral SWE-bench cloud batches."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PHASES = (
    "planned",
    "selected",
    "provisioned",
    "predictions_frozen",
    "evaluated",
    "evidence_verified",
    "cloud_released",
)


class RunnerError(ValueError):
    """Raised when a batch attempts to bypass a frozen runner contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_instances(
    instance_ids: Iterable[str],
    *,
    excluded: set[str] | frozenset[str],
    seed_material: str,
    count: int,
) -> list[dict[str, str]]:
    """Select a deterministic, unique task set without reading task content."""

    if count < 1:
        raise RunnerError("selection count must be positive")
    if not seed_material:
        raise RunnerError("seed material must be non-empty")
    eligible = sorted(
        {
            instance_id
            for instance_id in instance_ids
            if isinstance(instance_id, str)
            and instance_id
            and instance_id not in excluded
        }
    )
    if len(eligible) < count:
        raise RunnerError(
            f"not enough eligible instances: required={count} available={len(eligible)}"
        )

    def rank(instance_id: str) -> str:
        material = f"{seed_material}\0{instance_id}".encode()
        return hashlib.sha256(material).hexdigest()

    ordered = sorted(eligible, key=lambda instance_id: (rank(instance_id), instance_id))
    return [
        {"instance_id": instance_id, "rank_sha256": rank(instance_id)}
        for instance_id in ordered[:count]
    ]


def freeze_file(path: Path, manifest_path: Path, *, kind: str) -> dict[str, Any]:
    """Record a file digest that must match before a later stage may consume it."""

    path = path.resolve()
    manifest_path = manifest_path.resolve()
    if not path.is_file():
        raise RunnerError(f"freeze target missing: {path}")
    if not kind:
        raise RunnerError("freeze kind must be non-empty")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    relative_path = os.path.relpath(path, manifest_path.parent)
    payload = {
        "schema_version": "1.0",
        "kind": kind,
        "path": Path(relative_path).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "status": "frozen",
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_frozen_file(manifest_path: Path) -> str:
    """Fail if a frozen input was moved, resized, or changed."""

    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = (manifest_path.parent / payload["path"]).resolve()
    if not target.is_file():
        raise RunnerError(f"frozen file missing: {target}")
    actual = _sha256_file(target)
    if actual != payload.get("sha256") or target.stat().st_size != payload.get("bytes"):
        raise RunnerError("frozen file hash mismatch")
    return actual


def write_portable_checksums(
    root: Path, output_path: Path
) -> list[dict[str, str | int]]:
    """Write GNU-compatible SHA256SUMS containing only paths relative to root."""

    root = root.resolve()
    output_path = output_path.resolve()
    if not root.is_dir() or not output_path.is_relative_to(root):
        raise RunnerError("checksum output must be inside an existing evidence root")
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path != output_path
    )
    rows: list[dict[str, str | int]] = []
    lines = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise RunnerError("checksum path contains a newline")
        digest = _sha256_file(path)
        lines.append(f"{digest}  {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return rows


@dataclass(frozen=True)
class CloudResources:
    """Explicit resource inventory used by the terminal cleanup gate."""

    instance_id: str
    disk_ids: tuple[str, ...]
    security_group_ids: tuple[str, ...]
    ssh_key_id: str
    released_ids: frozenset[str]

    @property
    def required_ids(self) -> frozenset[str]:
        return frozenset(
            (
                self.instance_id,
                *self.disk_ids,
                *self.security_group_ids,
                self.ssh_key_id,
            )
        )


class RunnerLedger:
    """Durable monotonic batch state with terminal failure semantics."""

    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path.resolve()
        self.payload = payload

    @classmethod
    def create(cls, path: Path, *, batch_id: str) -> RunnerLedger:
        if not batch_id:
            raise RunnerError("batch id must be non-empty")
        ledger = cls(
            path,
            {
                "schema_version": "1.0",
                "batch_id": batch_id,
                "phase": "planned",
                "events": [{"phase": "planned"}],
            },
        )
        ledger._persist()
        return ledger

    @classmethod
    def load(cls, path: Path) -> RunnerLedger:
        path = path.resolve()
        if not path.is_file():
            raise RunnerError(f"ledger missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        phase = payload.get("phase")
        if phase not in {*PHASES, "failed"}:
            raise RunnerError(f"invalid ledger phase: {phase}")
        if not isinstance(payload.get("batch_id"), str) or not payload["batch_id"]:
            raise RunnerError("invalid ledger batch id")
        events = payload.get("events")
        if not isinstance(events, list) or not events:
            raise RunnerError("invalid ledger events")
        return cls(path, payload)

    @property
    def phase(self) -> str:
        return str(self.payload["phase"])

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    def advance(self, target_phase: str, **evidence: Any) -> None:
        if self.phase == "failed":
            raise RunnerError("runner is failed and cannot advance")
        try:
            current_index = PHASES.index(self.phase)
        except ValueError as error:
            raise RunnerError(f"unknown current phase: {self.phase}") from error
        expected = (
            PHASES[current_index + 1] if current_index + 1 < len(PHASES) else None
        )
        if target_phase != expected:
            raise RunnerError(
                f"invalid phase transition: {self.phase} -> {target_phase}; expected={expected}"
            )
        event = {"phase": target_phase, **evidence}
        self.payload["phase"] = target_phase
        self.payload["events"].append(event)
        self._persist()

    def run_checked(
        self, command: Sequence[str], *, evidence_path: Path
    ) -> subprocess.CompletedProcess[str]:
        if self.phase == "failed":
            raise RunnerError("runner is failed and cannot execute commands")
        if not command or not all(isinstance(item, str) and item for item in command):
            raise RunnerError("command must be a non-empty argv sequence")
        completed = subprocess.run(
            list(command), capture_output=True, text=True, check=False
        )
        evidence_path = evidence_path.resolve()
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(
                {
                    "command": list(command),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            self.payload["phase"] = "failed"
            self.payload["events"].append(
                {
                    "phase": "failed",
                    "reason": "command_failed",
                    "returncode": completed.returncode,
                    "evidence_path": str(evidence_path),
                }
            )
            self._persist()
        return completed

    def release_cloud(self, resources: CloudResources) -> None:
        missing = sorted(resources.required_ids - resources.released_ids)
        if missing:
            raise RunnerError(f"unreleased cloud resources: {', '.join(missing)}")
        self.advance(
            "cloud_released",
            released_resource_ids=sorted(resources.released_ids),
        )
