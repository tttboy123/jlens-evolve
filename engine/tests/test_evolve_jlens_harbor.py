from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuous_ab import ABContractError
from evolve_jlens_harbor import (
    validate_prediction_receipt,
    write_prediction_receipt,
)


def test_harbor_receipt_proves_trajectory_frozen_before_verifier(tmp_path: Path):
    trajectory = tmp_path / "trajectory.json"
    receipt_path = tmp_path / "frozen-prediction.json"
    result_path = tmp_path / "result.json"
    trajectory.write_text('{"steps": []}\n', encoding="utf-8")
    write_prediction_receipt(
        receipt_path=receipt_path,
        trajectory_path=trajectory,
        round_id="round-000001",
        arm="baseline",
        agent_program_sha256="a" * 64,
        baseline_contract_sha256="b" * 64,
        harness_revision="c" * 40,
    )
    result_path.write_text(
        json.dumps(
            {
                "agent_execution": {"finished_at": "2026-08-03T10:00:00+00:00"},
                "verifier": {"started_at": "2026-08-03T10:00:01+00:00"},
            }
        ),
        encoding="utf-8",
    )

    receipt = validate_prediction_receipt(
        receipt_path=receipt_path,
        trajectory_path=trajectory,
        result_path=result_path,
        expected_round_id="round-000001",
        expected_arm="baseline",
        expected_agent_program_sha256="a" * 64,
        expected_baseline_contract_sha256="b" * 64,
        expected_harness_revision="c" * 40,
    )

    assert receipt["prediction_frozen_before_evaluator"] is True


def test_harbor_receipt_rejects_changed_trajectory_or_reversed_phase_order(
    tmp_path: Path,
):
    trajectory = tmp_path / "trajectory.json"
    receipt_path = tmp_path / "frozen-prediction.json"
    result_path = tmp_path / "result.json"
    trajectory.write_text('{"steps": []}\n', encoding="utf-8")
    write_prediction_receipt(
        receipt_path=receipt_path,
        trajectory_path=trajectory,
        round_id="round-000001",
        arm="evolved",
        agent_program_sha256="a" * 64,
        baseline_contract_sha256="b" * 64,
        harness_revision="c" * 40,
    )
    result_path.write_text(
        json.dumps(
            {
                "agent_execution": {"finished_at": "2026-08-03T10:00:00+00:00"},
                "verifier": {"started_at": "2026-08-03T09:59:59+00:00"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ABContractError, match="phase order"):
        validate_prediction_receipt(
            receipt_path=receipt_path,
            trajectory_path=trajectory,
            result_path=result_path,
            expected_round_id="round-000001",
            expected_arm="evolved",
            expected_agent_program_sha256="a" * 64,
            expected_baseline_contract_sha256="b" * 64,
            expected_harness_revision="c" * 40,
        )

    trajectory.write_text('{"steps": ["tampered"]}\n', encoding="utf-8")
    with pytest.raises(ABContractError, match="trajectory.*tampered"):
        validate_prediction_receipt(
            receipt_path=receipt_path,
            trajectory_path=trajectory,
            result_path=result_path,
            expected_round_id="round-000001",
            expected_arm="evolved",
            expected_agent_program_sha256="a" * 64,
            expected_baseline_contract_sha256="b" * 64,
            expected_harness_revision="c" * 40,
        )
