"""Read-only verifier for the Evolve x JLens staged artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "artifacts/v1.0.0/MANIFEST.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_target(root: Path, relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        return None
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify every listed file without changing manifest or artifact bytes."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    try:
        root = Path(manifest["artifact_root"]).resolve()
        stages = manifest["stages"]
    except (KeyError, TypeError) as error:
        return {
            "valid": False,
            "failures": [{"reason": "invalid_manifest_schema", "detail": str(error)}],
            "files_verified": 0,
            "stages_verified": 0,
            "verification_fingerprint": None,
        }
    files_verified = 0
    stages_verified = 0
    stage_summaries = []
    seen_stages: set[str] = set()
    for stage in stages:
        stage_id = stage.get("stage")
        if not isinstance(stage_id, str) or stage_id in seen_stages:
            failures.append({"stage": stage_id, "reason": "invalid_or_duplicate_stage"})
            continue
        seen_stages.add(stage_id)
        stage_failures_before = len(failures)
        decision = stage.get("decision")
        if decision != "accepted":
            failures.append(
                {"stage": stage_id, "reason": "stage_not_accepted", "actual": decision}
            )
        decision_payload: dict[str, Any] | None = None
        for entry in stage.get("files", []):
            relative = entry.get("path")
            target = _safe_target(root, relative) if isinstance(relative, str) else None
            if target is None:
                failures.append(
                    {"stage": stage_id, "path": relative, "reason": "unsafe_path"}
                )
                continue
            if not target.is_file():
                failures.append(
                    {"stage": stage_id, "path": relative, "reason": "missing_file"}
                )
                continue
            actual_bytes = target.stat().st_size
            if actual_bytes != entry.get("bytes"):
                failures.append(
                    {
                        "stage": stage_id,
                        "path": relative,
                        "reason": "bytes_mismatch",
                        "expected": entry.get("bytes"),
                        "actual": actual_bytes,
                    }
                )
                continue
            actual_sha256 = _sha256_file(target)
            if actual_sha256 != entry.get("sha256"):
                failures.append(
                    {
                        "stage": stage_id,
                        "path": relative,
                        "reason": "sha256_mismatch",
                        "expected": entry.get("sha256"),
                        "actual": actual_sha256,
                    }
                )
                continue
            files_verified += 1
            if relative.endswith("/DECISION.json"):
                decision_payload = json.loads(target.read_text(encoding="utf-8"))
        if decision_payload is None:
            failures.append({"stage": stage_id, "reason": "decision_file_missing"})
        elif decision_payload.get("decision") != decision:
            failures.append(
                {
                    "stage": stage_id,
                    "reason": "decision_mismatch",
                    "manifest": decision,
                    "file": decision_payload.get("decision"),
                }
            )
        stage_valid = len(failures) == stage_failures_before
        stages_verified += int(stage_valid)
        stage_summaries.append(
            {
                "stage": stage_id,
                "decision": decision,
                "files": len(stage.get("files", [])),
                "valid": stage_valid,
            }
        )
    stable = {
        "artifact_root": str(root),
        "files_verified": files_verified,
        "stage_summaries": stage_summaries,
        "failures": failures,
    }
    return {
        "valid": not failures,
        "manifest": str(manifest_path.resolve()),
        "artifact_root": str(root),
        "files_verified": files_verified,
        "stages_verified": stages_verified,
        "stage_summaries": stage_summaries,
        "failures": failures,
        "verification_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    report = verify_manifest(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
