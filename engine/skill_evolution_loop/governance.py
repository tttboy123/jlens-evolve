"""Deterministic governance outputs for evolution rounds."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import ContractError, sha256_json
from .evolution_catalog import EvolutionCatalog


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_runtime_identity(round_dir: Path, files: Iterable[Path]) -> dict[str, Any]:
    """Freeze one immutable runtime identity receipt for a round."""

    root = round_dir.resolve()
    entries: dict[str, str] = {}
    for relative in files:
        path = (root / relative).resolve()
        if not path.is_file():
            raise ContractError(f"runtime identity file is missing: {relative}")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError("runtime identity file escapes round root") from exc
        entries[str(relative)] = _file_sha256(path)
    content = {
        "schema_version": 1,
        "round_dir": str(root),
        "files": entries,
        "digest": hashlib.sha256(
            json.dumps(entries, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    return {**content, "evidence_sha256": sha256_json(content)}


@dataclass(frozen=True)
class CostRow:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    kind: str
    round: str


def build_cost_ledger(rows: Iterable[CostRow]) -> dict[str, Any]:
    """Aggregate teacher/student call costs into one ledger."""

    normalized: list[dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    by_provider: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (
            type(row.prompt_tokens) is not int
            or type(row.completion_tokens) is not int
            or type(row.total_tokens) is not int
            or row.total_tokens != row.prompt_tokens + row.completion_tokens
        ):
            raise ContractError("cost ledger token fields are inconsistent")
        normalized.append(
            {
                "provider": row.provider,
                "model": row.model,
                "prompt_tokens": row.prompt_tokens,
                "completion_tokens": row.completion_tokens,
                "total_tokens": row.total_tokens,
                "kind": row.kind,
                "round": row.round,
            }
        )
        totals["prompt_tokens"] += row.prompt_tokens
        totals["completion_tokens"] += row.completion_tokens
        totals["total_tokens"] += row.total_tokens
        bucket = by_provider.setdefault(
            row.provider,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        for key in bucket:
            bucket[key] += getattr(row, key)
        bucket = by_model.setdefault(
            row.model,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        for key in bucket:
            bucket[key] += getattr(row, key)
    content = {
        "schema_version": 1,
        "rows": normalized,
        "totals": totals,
        "by_provider": by_provider,
        "by_model": by_model,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def build_evolution_overview(catalog_root: Path, output_path: Path) -> dict[str, Any]:
    """Write a human-readable evolution overview from the catalog."""

    catalog = EvolutionCatalog(catalog_root)
    records = catalog.search()
    skills = [r for r in records if r.record_type == "skills"]
    gains = [
        r
        for r in records
        if r.record_type == "experiments"
        and ("gain" in r.record_id or r.payload.get("gain"))
    ]
    regressions = [
        r
        for r in records
        if r.record_type == "experiments"
        and ("regression" in r.record_id or r.payload.get("teaching_regression"))
    ]
    failures = [r for r in records if r.record_type == "failure_clusters"]
    mechanisms = [r for r in records if r.record_type == "mechanisms"]
    infrastructure_gaps = [r for r in records if r.record_type == "infrastructure_gaps"]
    lines = [
        "# Evolution Overview",
        "",
        f"- Catalog records: {len(records)}",
        f"- Skills: {len(skills)} (all inactive)",
        f"- Native-gain experiments: {len(gains)}",
        f"- Regression experiments: {len(regressions)}",
        f"- Failure clusters: {len(failures)}",
        f"- Mechanisms: {len(mechanisms)}",
        f"- Infrastructure gaps: {len(infrastructure_gaps)}",
        "",
        "## Skills",
    ]
    for skill in skills:
        lines.append(f"- `{skill.record_id}` [{skill.status}] {skill.title}")
    lines += ["", "## Native gains"]
    for gain in gains:
        lines.append(f"- `{gain.record_id}` [{gain.status}] {gain.title}")
    lines += ["", "## Regressions"]
    for regression in regressions:
        lines.append(
            f"- `{regression.record_id}` [{regression.status}] {regression.title}"
        )
    lines += ["", "## Failure clusters"]
    for failure in failures:
        lines.append(f"- `{failure.record_id}` [{failure.status}] {failure.title}")
    lines += ["", "## Mechanisms"]
    for mechanism in mechanisms:
        lines.append(
            f"- `{mechanism.record_id}` [{mechanism.status}] {mechanism.title}"
        )
    lines += ["", "## Infrastructure gaps"]
    for gap in infrastructure_gaps:
        lines.append(f"- `{gap.record_id}` [{gap.status}] {gap.title}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    content = {
        "schema_version": 1,
        "catalog_records": len(records),
        "skill_count": len(skills),
        "gain_count": len(gains),
        "regression_count": len(regressions),
        "failure_cluster_count": len(failures),
        "mechanism_count": len(mechanisms),
        "infrastructure_gap_count": len(infrastructure_gaps),
        "overview_path": str(output_path),
    }
    return {**content, "evidence_sha256": sha256_json(content)}
