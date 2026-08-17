from __future__ import annotations

import json
from pathlib import Path

from psi_runtime import run_psi_experiment

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v1.0.0/v0.4.0-psi-skill-library"


def test_two_target_matched_ab_is_noninferior_and_reports_strict_gain(tmp_path: Path):
    result = run_psi_experiment(
        config_path=STAGE / "configs/experiment.json",
        candidate_path=STAGE / "configs/skill-candidates.json",
        output_dir=tmp_path / "psi",
    )

    assert result["decision"] == "accepted"
    assert result["psi_capability"] == "passed"
    assert result["contract_checks"]["only_skill_refs_changed"] is True
    assert result["contract_checks"]["sealed_after_public_persistence"] is True
    assert result["contract_checks"]["project_local_registry_only"] is True
    assert set(result["targets"]) == {
        "payout-record-cleaning-v1",
        "refund-record-cleaning-v1",
    }
    for target in result["targets"].values():
        assert target["sealed_noninferior_seeds"] == 3
        assert target["sealed_mean_delta"] >= 0
        assert target["public_mean_delta"] >= 0
    assert result["strict_transfer_benefit"] is True
    assert result["source_replay"] == {
        "public_passed": 13,
        "public_total": 13,
        "sealed_passed": 6,
        "sealed_total": 6,
    }
    assert result["candidate_status"] == "transfer_verified"
    assert result["legacy_candidate_status"] == "rejected"


def test_sealed_event_occurs_after_all_public_results_are_persisted(tmp_path: Path):
    output = tmp_path / "psi"
    run_psi_experiment(
        config_path=STAGE / "configs/experiment.json",
        candidate_path=STAGE / "configs/skill-candidates.json",
        output_dir=output,
    )
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sealed_open = next(
        i for i, row in enumerate(events) if row["event_type"] == "sealed_opened"
    )

    assert all(row["partition"] == "public" for row in events[:sealed_open])
    assert all(row["partition"] == "sealed" for row in events[sealed_open + 1 :])
    marker = events[sealed_open]
    assert marker["public_results_persisted"] is True
    assert (output / "public-results.json").is_file()


def test_registry_keeps_historical_negative_and_does_not_install_globally(
    tmp_path: Path,
):
    output = tmp_path / "psi"
    result = run_psi_experiment(
        config_path=STAGE / "configs/experiment.json",
        candidate_path=STAGE / "configs/skill-candidates.json",
        output_dir=output,
    )

    revisions = [
        json.loads(line)
        for line in (output / "registry/registry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    legacy = [
        row
        for row in revisions
        if row["skill_id"] == "legacy-record-cleaning-bundle-v1"
    ]
    assert legacy[-1]["status"] == "rejected"
    assert "negative" in legacy[-1]["status_reason"]
    assert result["claims"]["global_skill_installs"] == 0
    assert all(row["project_local_only"] is True for row in revisions)
    assert all(row["active"] is False for row in revisions)
