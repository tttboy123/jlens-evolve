from __future__ import annotations

import json
from pathlib import Path

import pytest

from observer_runtime import run_observer_matrix

ROOT = Path(__file__).resolve().parents[1]
OBSERVER_CONFIG = (
    ROOT / "artifacts/v1.0.0/v0.3.0-jlens-observer/configs/experiment.json"
)
AGENT_CONFIGS = ROOT / "artifacts/v1.0.0/v0.2.0-agent-program/configs"


def test_matched_matrix_is_runtime_equivalent_and_reports_negative_jlens_gain(
    tmp_path,
):
    result = run_observer_matrix(
        observer_config_path=OBSERVER_CONFIG,
        agent_config_dir=AGENT_CONFIGS,
        output_dir=tmp_path / "matrix",
        replays_per_mode=2,
    )

    assert result["decision"] == "accepted"
    assert result["mechanism_checks"]["one_runtime_outcome_fingerprint"] is True
    assert result["mechanism_checks"]["one_active_program_hash"] is True
    assert result["mechanism_checks"]["all_runtime_results_equal"] is True
    assert result["mechanism_checks"]["observer_never_used_for_admission"] is True
    assert result["conditions"]["off"]["statuses"] == ["disabled", "disabled"]
    for mode in ("trace", "logit_lens", "jlens"):
        assert result["conditions"][mode]["statuses"] == [
            "completed",
            "completed",
        ]
        assert result["conditions"][mode]["unique_artifact_fingerprints"] == 1
    assert result["jlens_incremental"]["advantage"] == pytest.approx(
        0.9953156431246746 - 0.9970722769529216
    )
    assert result["jlens_incremental"]["required_margin"] == 0.01
    assert result["jlens_incremental"]["conclusion"] == "not_supported"


def test_observer_is_submitted_after_runtime_persistence(tmp_path):
    output = tmp_path / "matrix"
    run_observer_matrix(
        observer_config_path=OBSERVER_CONFIG,
        agent_config_dir=AGENT_CONFIGS,
        output_dir=output,
        replays_per_mode=1,
    )
    events = [
        json.loads(line)
        for line in (output / "matrix-events.jsonl").read_text().splitlines()
    ]

    for mode in ("trace", "logit_lens", "jlens"):
        mode_events = [event for event in events if event["observer_mode"] == mode]
        names = [event["event_type"] for event in mode_events]
        assert names.index("runtime_complete") < names.index("observer_submitted")
        assert names.index("observer_submitted") < names.index("observer_complete")
        runtime_event = next(
            event for event in mode_events if event["event_type"] == "runtime_complete"
        )
        assert Path(runtime_event["result_path"]).is_file()


def test_injected_collector_failure_does_not_change_runtime_or_active_program(tmp_path):
    result = run_observer_matrix(
        observer_config_path=OBSERVER_CONFIG,
        agent_config_dir=AGENT_CONFIGS,
        output_dir=tmp_path / "failure",
        replays_per_mode=1,
        inject_failure_mode="jlens",
    )

    assert result["conditions"]["jlens"]["statuses"] == ["failed"]
    assert result["failure_injection"] == {
        "mode": "jlens",
        "isolated": True,
    }
    assert result["mechanism_checks"]["one_runtime_outcome_fingerprint"] is True
    assert result["mechanism_checks"]["one_active_program_hash"] is True
    assert result["conditions"]["jlens"]["runtime_decisions"] == ["accepted"]
    artifact_path = next((tmp_path / "failure/observations").glob("jlens-*.json"))
    artifact = json.loads(artifact_path.read_text())
    assert artifact["status"] == "failed"
    assert artifact["used_for_admission"] is False
