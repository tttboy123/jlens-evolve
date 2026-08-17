import json
from pathlib import Path

from hardening_runtime import run_hardening_experiment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "artifacts/v1.0.0/v0.8.0-hardening/configs/experiment.json"


def _runner(*, integration_config: Path, output_dir: Path, timeout_seconds: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "decision": "accepted",
        "experiment_fingerprint": "c" * 64,
        "contract_checks": {
            "observer_never_admission": True,
            "only_admission_publishes_active": True,
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def test_hardening_matrix_preserves_recovery_and_authority_contracts(tmp_path: Path):
    result = run_hardening_experiment(
        config_path=CONFIG,
        output_dir=tmp_path / "hardening",
        integration_runner=_runner,
    )

    assert result["decision"] == "accepted"
    assert all(result["contract_checks"].values())
    assert result["scenarios"]["after_prepare"]["final_status"] == "completed"
    assert result["scenarios"]["after_integration"]["busy_before_expiry"]
    assert result["scenarios"]["after_integration"]["attempt"] == 1
    assert result["scenarios"]["corrupt_result"]["attempt"] == 2
    assert result["scenarios"]["migration"]["preserved_rows"]
    assert result["scenarios"]["timeout"]["final_status"] == "failed"
    assert result["claims"] == {
        "distributed_exactly_once": False,
        "global_skill_installs": 0,
        "model_calls": 0,
        "model_weights_frozen": True,
        "network_calls": 0,
        "production_ready": False,
    }
