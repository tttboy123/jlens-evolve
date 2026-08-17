from __future__ import annotations

from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError, sha256_json
from skill_evolution_loop.governance import (
    CostRow,
    build_cost_ledger,
    build_evolution_overview,
    build_runtime_identity,
)


def test_runtime_identity_freeze_and_replay(tmp_path) -> None:
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")

    identity = build_runtime_identity(tmp_path, [Path("code.py")])

    content = {k: v for k, v in identity.items() if k != "evidence_sha256"}
    assert identity["evidence_sha256"] == sha256_json(content)
    assert len(identity["files"]["code.py"]) == 64
    assert all(c in "0123456789abcdef" for c in identity["files"]["code.py"])

    with pytest.raises(ContractError):
        build_runtime_identity(tmp_path, [Path("missing.py")])


def test_cost_ledger_aggregates_and_validates() -> None:
    rows = [
        CostRow("deepseek", "v4-flash", 100, 50, 150, "teacher", "r104"),
        CostRow("deepseek", "v4-flash", 200, 100, 300, "teacher", "r105"),
        CostRow("mlx", "qwen-4b", 10, 5, 15, "student", "r104"),
    ]

    ledger = build_cost_ledger(rows)
    content = {k: v for k, v in ledger.items() if k != "evidence_sha256"}
    assert ledger["evidence_sha256"] == sha256_json(content)
    assert ledger["totals"] == {
        "prompt_tokens": 310,
        "completion_tokens": 155,
        "total_tokens": 465,
    }
    assert ledger["by_model"]["v4-flash"]["total_tokens"] == 450

    bad = CostRow("deepseek", "v4-flash", 100, 100, 250, "teacher", "r104")
    with pytest.raises(ContractError):
        build_cost_ledger([bad])


def test_evolution_overview_writes_markdown(tmp_path) -> None:
    out = tmp_path / "EVOLUTION-OVERVIEW.md"
    summary = build_evolution_overview(
        Path(
            "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog"
        ),
        out,
    )

    assert out.is_file()
    assert "Catalog records" in out.read_text(encoding="utf-8")
    assert summary["catalog_records"] >= 60


def test_evolution_overview_surfaces_mechanisms_and_infrastructure_gaps(
    tmp_path,
) -> None:
    """The overview must surface every catalog record type.

    Round 5 (2026-08-13) added explicit ``## Mechanisms`` and
    ``## Infrastructure gaps`` sections because the previous overview
    silently dropped 23 ``mechanisms`` records (including
    ``r105-evidence-reanchored-20260813`` and
    ``r100-r102-baseline-asymmetry-resolved-20260813``) and 12
    ``infrastructure_gaps`` records. Catalog is the evidence source; a
    review artifact that hides 35 / 68 records is not auditable.
    """

    out = tmp_path / "EVOLUTION-OVERVIEW.md"
    summary = build_evolution_overview(
        Path(
            "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/evolution-catalog"
        ),
        out,
    )

    text = out.read_text(encoding="utf-8")
    assert "## Mechanisms" in text
    assert "## Infrastructure gaps" in text
    assert summary["mechanism_count"] >= 20
    assert summary["infrastructure_gap_count"] >= 10
    # The catalog's audit-fix records from this autonomous run must appear.
    assert "r105-evidence-reanchored-20260813" in text
    assert "r100-r102-baseline-asymmetry-resolved-20260813" in text
    # The evidence_sha256 is recomputed; assert the JSON body has the
    # new count fields.
    body = {k: v for k, v in summary.items() if k != "evidence_sha256"}
    assert (
        body["mechanism_count"]
        + body["infrastructure_gap_count"]
        + body["skill_count"]
        + body["gain_count"]
        + body["regression_count"]
        + body["failure_cluster_count"]
        <= body["catalog_records"]
    )
