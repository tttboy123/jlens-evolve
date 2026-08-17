"""Unified, observational-only artifacts for trace and lens sidecars."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,191}")
_MODES = {"off", "trace", "logit_lens", "jlens"}
_STATUSES = {"disabled", "completed", "failed", "timeout"}


class ObservationContractError(ValueError):
    """Raised when an observer artifact violates the sidecar boundary."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceRef:
    path: str
    sha256: str
    role: str

    @classmethod
    def from_path(cls, path: Path, *, role: str) -> SourceRef:
        return cls(path=str(path.resolve()), sha256=_sha256_file(path), role=role)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRef:
        if not isinstance(data, dict) or set(data) != {"path", "sha256", "role"}:
            raise ObservationContractError("invalid source ref fields")
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            role=str(data["role"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "role": self.role}

    def validate(self) -> None:
        path = Path(self.path)
        if not path.is_absolute() or not path.is_file():
            raise ObservationContractError(f"source path is not a file: {self.path}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ObservationContractError("invalid source sha256")
        if _sha256_file(path) != self.sha256:
            raise ObservationContractError(f"source sha256 mismatch: {self.path}")
        if not self.role:
            raise ObservationContractError("source role must be non-empty")


@dataclass(frozen=True)
class ObservationArtifact:
    schema_version: int
    artifact_id: str
    observer_mode: str
    status: str
    runtime_outcome_fingerprint: str
    active_program_hash: str
    config_hashes: dict[str, str]
    source_refs: tuple[SourceRef, ...]
    features: dict[str, Any]
    causal_boundary: str
    used_for_admission: bool
    error: str | None

    _FIELDS = frozenset(
        {
            "schema_version",
            "artifact_id",
            "observer_mode",
            "status",
            "runtime_outcome_fingerprint",
            "active_program_hash",
            "config_hashes",
            "source_refs",
            "features",
            "causal_boundary",
            "used_for_admission",
            "error",
            "artifact_fingerprint",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationArtifact:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            unknown = sorted(set(data) - cls._FIELDS) if isinstance(data, dict) else []
            missing = sorted(cls._FIELDS - set(data)) if isinstance(data, dict) else []
            raise ObservationContractError(
                f"invalid ObservationArtifact fields; unknown={unknown}, missing={missing}"
            )
        if not isinstance(data["source_refs"], list):
            raise ObservationContractError("source_refs must be a list")
        if not isinstance(data["config_hashes"], dict):
            raise ObservationContractError("config_hashes must be a mapping")
        if not isinstance(data["features"], dict):
            raise ObservationContractError("features must be a mapping")
        artifact = cls(
            schema_version=data["schema_version"],
            artifact_id=data["artifact_id"],
            observer_mode=data["observer_mode"],
            status=data["status"],
            runtime_outcome_fingerprint=data["runtime_outcome_fingerprint"],
            active_program_hash=data["active_program_hash"],
            config_hashes=dict(data["config_hashes"]),
            source_refs=tuple(SourceRef.from_dict(row) for row in data["source_refs"]),
            features=dict(data["features"]),
            causal_boundary=data["causal_boundary"],
            used_for_admission=data["used_for_admission"],
            error=data["error"],
        )
        artifact.validate()
        if data["artifact_fingerprint"] != artifact.artifact_fingerprint:
            raise ObservationContractError("artifact fingerprint mismatch")
        return artifact

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "observer_mode": self.observer_mode,
            "status": self.status,
            "runtime_outcome_fingerprint": self.runtime_outcome_fingerprint,
            "active_program_hash": self.active_program_hash,
            "config_hashes": self.config_hashes,
            "source_refs": [source.to_dict() for source in self.source_refs],
            "features": self.features,
            "causal_boundary": self.causal_boundary,
            "used_for_admission": self.used_for_admission,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    @property
    def artifact_fingerprint(self) -> str:
        stable = {
            **self._content_dict(),
            "source_refs": [
                {"sha256": source.sha256, "role": source.role}
                for source in self.source_refs
            ],
        }
        return hashlib.sha256(_canonical_json(stable).encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ObservationContractError("unsupported observation schema")
        if _IDENTIFIER.fullmatch(self.artifact_id) is None:
            raise ObservationContractError("invalid artifact_id")
        if self.observer_mode not in _MODES:
            raise ObservationContractError(
                f"unknown observer mode: {self.observer_mode}"
            )
        if self.status not in _STATUSES:
            raise ObservationContractError(f"unknown observer status: {self.status}")
        if self.causal_boundary != "observational_not_causal":
            raise ObservationContractError("invalid causal boundary")
        if self.used_for_admission is not False:
            raise ObservationContractError(
                "observer artifact cannot be used for admission"
            )
        for name, value in (
            ("runtime_outcome_fingerprint", self.runtime_outcome_fingerprint),
            ("active_program_hash", self.active_program_hash),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ObservationContractError(f"invalid {name}")
        if not self.config_hashes or any(
            _SHA256.fullmatch(value) is None for value in self.config_hashes.values()
        ):
            raise ObservationContractError("invalid config_hashes")
        for source in self.source_refs:
            source.validate()
        if self.observer_mode == "off":
            if (
                self.status != "disabled"
                or self.features
                or self.source_refs
                or self.error
            ):
                raise ObservationContractError(
                    "off artifact must be empty and disabled"
                )
        elif self.status == "completed":
            if not self.features or not self.source_refs or self.error:
                raise ObservationContractError(
                    "completed artifact requires sources/features and no error"
                )
        elif not self.error:
            raise ObservationContractError("failed/timeout artifact requires error")


def _runtime_identity(runtime_run_dir: Path) -> tuple[dict[str, Any], str, str]:
    result_path = runtime_run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return (
        result,
        str(result["outcome_fingerprint"]),
        str(result["final"]["program_hash"]),
    )


def collect_observation(
    *,
    mode: str,
    runtime_run_dir: Path,
    lens_source: Path,
    config_hashes: dict[str, str],
) -> ObservationArtifact:
    """Collect one deterministic artifact after runtime completion."""
    if mode not in _MODES:
        raise ObservationContractError(f"unknown observer mode: {mode}")
    result, runtime_fingerprint, program_hash = _runtime_identity(runtime_run_dir)
    artifact_id = f"observation-{mode}-{runtime_fingerprint[:12]}"
    common = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "observer_mode": mode,
        "runtime_outcome_fingerprint": runtime_fingerprint,
        "active_program_hash": program_hash,
        "config_hashes": dict(config_hashes),
        "causal_boundary": "observational_not_causal",
        "used_for_admission": False,
        "error": None,
    }
    if mode == "off":
        artifact = ObservationArtifact(
            **common,
            status="disabled",
            source_refs=(),
            features={},
        )
        artifact.validate()
        return artifact

    if mode == "trace":
        events_path = runtime_run_dir / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        evaluations = [row for row in events if row["event_type"] == "evaluation"]
        features = {
            "public_evaluations": sum(
                row["partition"] == "public" for row in evaluations
            ),
            "sealed_evaluations": sum(
                row["partition"] == "sealed" for row in evaluations
            ),
            "accepted_candidates": int(result["search"]["accepted_candidates"]),
            "rejected_candidates": int(result["search"]["rejected_candidates"]),
            "runtime_decision": result["decision"],
            "public_gain": float(result["claims"]["agent_program_public_gain"]),
            "sealed_gain": float(result["claims"]["sealed_gain"]),
        }
        sources = (SourceRef.from_path(events_path, role="runtime_trace"),)
    else:
        strategy = json.loads(lens_source.read_text(encoding="utf-8"))
        evidence = strategy["evidence"]
        metric_key = (
            "logit_lens_score_eta_squared"
            if mode == "logit_lens"
            else "jlens_score_eta_squared"
        )
        features = {
            "score_eta_squared": float(evidence[metric_key]),
            "trace_edges": int(evidence["trace_edges"]),
            "unique_transitions": int(evidence["unique_transitions"]),
            "repeated_transition_fraction": float(
                evidence["repeated_transition_fraction"]
            ),
            "lineage_edges_are_independent": bool(
                evidence["lineage_edges_are_independent"]
            ),
            "jlens_incremental_supported": bool(
                evidence["jlens_incremental_supported"]
            ),
        }
        sources = (SourceRef.from_path(lens_source, role="historical_lens_analysis"),)
    artifact = ObservationArtifact(
        **common,
        status="completed",
        source_refs=sources,
        features=features,
    )
    artifact.validate()
    return artifact
