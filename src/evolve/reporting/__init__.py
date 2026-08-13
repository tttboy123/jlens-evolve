"""Rebuildable campaign reporting and literal-SHA audit verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evolve.contracts import Claim, ContractViolation, Receipt, canonical_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class ReportPaths:
    json_path: Path
    markdown_path: Path
    manifest_path: Path


class CampaignReportProjector:
    """Create views strictly from receipts and claims, never hand-entered counts."""

    def project(
        self,
        *,
        campaign_id: str,
        receipts: Iterable[Receipt],
        claims: Iterable[Claim],
        final_commit_sha: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        receipt_rows = tuple(receipts)
        claim_rows = tuple(claims)
        if any(row.campaign_id != campaign_id for row in receipt_rows):
            raise ContractViolation("campaign report receipt identity mismatch")
        counts = Counter(str(row.classification) for row in claim_rows)
        cost = sum(
            float(row.payload.get("estimated_cost_cny", 0))
            for row in receipt_rows
            if row.kind == "cost"
        )
        return {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "final_commit_sha": final_commit_sha,
            "receipt_count": len(receipt_rows),
            "claim_count": len(claim_rows),
            "counts": dict(sorted(counts.items())),
            "actual_api_spend_cny": round(cost, 8),
            "receipt_sha256": [row.content_sha256 for row in receipt_rows],
            "claim_sha256": [row.content_sha256 for row in claim_rows],
            **dict(metadata or {}),
        }

    def write(self, report: dict[str, Any], root: Path) -> ReportPaths:
        root = root.resolve()
        json_path = root / "FINAL-REPORT.json"
        markdown_path = root / "FINAL-REPORT.md"
        manifest_path = root / "EVIDENCE-MANIFEST.json"
        _atomic_write(json_path, (canonical_json(report) + "\n").encode())
        markdown = (
            "# Campaign Report\n\n"
            f"- Campaign: `{report['campaign_id']}`\n"
            f"- Final commit: `{report['final_commit_sha']}`\n"
            f"- Receipts: {report['receipt_count']}\n"
            f"- Claims: {report['claim_count']}\n"
            f"- API spend (CNY): {report['actual_api_spend_cny']}\n"
            f"- Classifications: `{canonical_json(report['counts'])}`\n"
        )
        _atomic_write(markdown_path, markdown.encode())
        entries = [
            {"path": path.name, "sha256": _sha256(path)}
            for path in (json_path, markdown_path)
        ]
        _atomic_write(
            manifest_path,
            (canonical_json({"schema_version": 1, "entries": entries}) + "\n").encode(),
        )
        return ReportPaths(json_path, markdown_path, manifest_path)


class AuditVerifier:
    def verify_manifest(self, manifest_path: Path, *, root: Path) -> int:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractViolation("manifest is unreadable") from exc
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ContractViolation("manifest entries must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ContractViolation("manifest entry format is invalid")
            path = (root / str(entry["path"])).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise ContractViolation("manifest path escapes root") from exc
            if not path.is_file():
                raise ContractViolation(f"manifest artifact missing: {entry['path']}")
            if _sha256(path) != entry["sha256"]:
                raise ContractViolation(f"manifest hash mismatch: {entry['path']}")
        return len(entries)


__all__ = ["AuditVerifier", "CampaignReportProjector", "ReportPaths"]
