from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evolve.cli import main
from evolve.contracts import canonical_json
from evolve.evidence import ReceiptStore
from evolve.reporting import AuditVerifier


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_config(tmp_path: Path, artifact: Path) -> Path:
    config = {
        "schema_version": 1,
        "campaign_id": "legacy-import-campaign-1",
        "imported_revision_id": "legacy-revision-7",
        "artifact_path": str(artifact),
        "artifact_sha256": _sha256(artifact.read_bytes()),
        "provenance_uri": "catalog://legacy/revision-7",
        "task": {
            "task_id": "sphinx-doc__sphinx-10435",
            "revision_id": "task-sphinx-10435-r1",
            "project": "sphinx",
            "cohort": "feedback",
            "source_sha256": _sha256(b"frozen-source"),
            "evaluator_id": "legacy-import-v1",
        },
        "model": {
            "provider": "legacy",
            "model": "frozen-artifact",
            "revision": "v1",
        },
    }
    path = tmp_path / "legacy-import.json"
    path.write_text(canonical_json(config) + "\n", encoding="utf-8")
    return path


def test_public_legacy_import_runs_through_campaign_authority(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy.json"
    artifact.write_bytes(b'{"legacy":"immutable"}\n')
    config = _write_config(tmp_path, artifact)
    output = tmp_path / "run"

    assert main(
        [
            "campaign",
            "import",
            "--strategy",
            "legacy",
            "--config",
            str(config),
            "--output",
            str(output),
        ]
    ) == 0

    report = json.loads((output / "CAMPAIGN-RESULT.json").read_text())
    assert report["status"] == "compatibility"
    assert report["claim_ids"] == []
    assert report["candidate_revision_ids"] == []
    assert report["capability_ids"] == []
    receipts = ReceiptStore(output / "receipt-store").list_receipts()
    assert len(receipts) == 1
    assert receipts[0].kind == "legacy_import"
    assert receipts[0].payload["compatibility_mode"] == "read-only"
    assert receipts[0].payload["provenance_uri"] == "catalog://legacy/revision-7"
    assert AuditVerifier().verify_manifest(
        output / "EVIDENCE-MANIFEST.json", root=output
    ) >= 3


def test_public_legacy_import_replay_is_idempotent(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy.json"
    artifact.write_bytes(b'{"legacy":"immutable"}\n')
    config = _write_config(tmp_path, artifact)
    output = tmp_path / "run"
    argv = [
        "campaign",
        "import",
        "--strategy",
        "legacy",
        "--config",
        str(config),
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    first = (output / "receipt-store" / "receipts.jsonl").read_bytes()
    assert main(argv) == 0

    assert (output / "receipt-store" / "receipts.jsonl").read_bytes() == first
    assert len(ReceiptStore(output / "receipt-store").list_receipts()) == 1


def test_public_legacy_import_rejects_artifact_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "legacy.json"
    artifact.write_bytes(b'{"legacy":"immutable"}\n')
    config = _write_config(tmp_path, artifact)
    artifact.write_bytes(b'{"legacy":"tampered"}\n')

    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        main(
            [
                "campaign",
                "import",
                "--strategy",
                "legacy",
                "--config",
                str(config),
                "--output",
                str(tmp_path / "run"),
            ]
        )
