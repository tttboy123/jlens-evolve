from __future__ import annotations

import json
from pathlib import Path

import pytest

from integration_runtime import IntegrationContractError, run_integration

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "artifacts/v1.0.0/v0.7.0-integration/configs/experiment.json"


def test_vertical_slice_integrates_components_without_authority_leak(tmp_path: Path):
    result = run_integration(config_path=CONFIG, output_dir=tmp_path / "integration")

    assert result["decision"] == "accepted"
    assert result["component_decisions"] == {
        "agent_program": "accepted",
        "agent_code": "accepted",
        "evaluator_shadow": "accepted",
        "observer_failure": "accepted",
        "observer_formal": "accepted",
        "psi": "accepted",
    }
    assert result["observer"]["incremental"] == "not_supported"
    assert result["observer"]["failure_isolated"] is True
    assert result["psi"]["candidate_status"] == "transfer_verified"
    assert result["agent_code"]["rollback_to_parent"] is True
    assert result["evaluator_shadow"]["active_changed"] is False
    assert result["evaluator_shadow"]["auto_promoted"] is False
    assert set(result["authorities_present"]) == {
        "admit",
        "execute",
        "observe",
        "persist",
        "propose",
    }
    active_envelopes = [
        row for row in result["envelopes"] if row["active_ref"] is not None
    ]
    assert len(active_envelopes) == 1
    assert active_envelopes[0]["authority"] == "admit"
    assert result["claims"]["global_skill_installs"] == 0


def test_duplicate_operation_is_idempotent_without_new_log_rows(tmp_path: Path):
    output = tmp_path / "integration"
    first = run_integration(config_path=CONFIG, output_dir=output)
    first_log = (output / "operation-log.jsonl").read_text(encoding="utf-8")
    second = run_integration(config_path=CONFIG, output_dir=output)
    second_log = (output / "operation-log.jsonl").read_text(encoding="utf-8")

    assert first["experiment_fingerprint"] == second["experiment_fingerprint"]
    assert second["idempotent_replay"] is True
    assert first_log == second_log


def test_same_operation_with_changed_contract_is_rejected(tmp_path: Path):
    output = tmp_path / "integration"
    run_integration(config_path=CONFIG, output_dir=output)
    changed = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed["observer_replays"] = 2
    conflicting = tmp_path / "conflict.json"
    conflicting.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(IntegrationContractError, match="operation contract conflict"):
        run_integration(config_path=conflicting, output_dir=output)
