from __future__ import annotations

from pathlib import Path

from continuous_ab_smoke import run_smoke


def test_smoke_uses_runtime_copy_and_makes_no_quality_claim(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    stage = root / "artifacts/v2.1.0/v2.1.0-continuous-ab"

    result = run_smoke(stage, tmp_path / "pass-1")

    assert result["status"] == "protocol_verified"
    assert result["quality_claim"] == "none"
    assert result["checks"]["frozen_pool_unchanged"] is True
    assert result["checks"]["matched_pair_completed"] is True
    assert result["checks"]["final_sealed_remains_unopened"] is True
    assert result["checks"]["prediction_tamper_rejected"] is True
    assert result["checks"]["planned_round_crash_reconciled"] is True
    assert result["checks"]["early_promotion_rejected"] is True
    assert result["checks"]["registry_tamper_rejected"] is True
    assert result["facts"]["completed_rounds"] == 1
