from pathlib import Path

from release_verification import run_release_verification

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "artifacts/v1.0.0/v1.0.0-release/configs/release.json"


def test_release_verification_reconstructs_all_stage_gates(tmp_path: Path):
    result = run_release_verification(
        config_path=CONFIG, output_dir=tmp_path / "verification"
    )

    assert result["decision"] == "accepted"
    assert all(result["contract_checks"].values())
    assert len(result["stage_evidence"]) == 9
    assert all(evidence["accepted"] for evidence in result["stage_evidence"].values())
    assert result["manifest"]["valid"]
    assert result["cli"]["run"]["decision"] == "accepted"
    assert result["cli"]["inspect"]["decision"] == "accepted"
    assert result["cli"]["verify"]["valid"]
    assert set(result["cli"]["rollback_plans"]) == {
        "agent-code",
        "agent-program",
        "evaluator",
        "skill",
    }
    assert all(
        plan["applied"] is False for plan in result["cli"]["rollback_plans"].values()
    )
    assert result["claims"]["production_deployed"] is False
