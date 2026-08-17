from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from continuous_ab import ABContractError
from evolve_jlens_harbor import write_prediction_receipt
from native_result_adapter import (
    PatchAdmissionContract,
    normalize_harbor_result,
    normalize_patch_result,
)


def _sha256_json(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_patch_receipt(
    root: Path,
    *,
    benchmark_id: str,
    arm: str = "baseline",
    instance_id: str = "repo__repo-1",
) -> tuple[Path, Path]:
    prediction = root / f"{arm}.patch"
    prediction.write_text(
        "diff --git a/source.py b/source.py\n"
        "--- a/source.py\n"
        "+++ b/source.py\n"
        "@@ -1 +1 @@\n-old\n+new\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "1.0",
        "round_id": "round-000001",
        "arm": arm,
        "task_uid": "d" * 64,
        "benchmark_id": benchmark_id,
        "instance_id": instance_id,
        "agent_program_sha256": "a" * 64,
        "baseline_contract_sha256": "b" * 64,
        "matched_contract_sha256": "c" * 64,
        "evaluator_epoch": "native-v1",
        "returncode": 0,
        "timed_out": False,
        "elapsed_seconds": 12.5,
        "execution_success": True,
        "workspace_head_unchanged": True,
        "usage": {
            "input_tokens": 100,
            "cached_input_tokens": 25,
            "output_tokens": 20,
            "total_tokens": 120,
        },
        "token_budget": 4096,
        "token_budget_exceeded": False,
        "prediction": {
            "path": str(prediction.resolve()),
            "bytes": prediction.stat().st_size,
            "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
            "frozen": True,
        },
    }
    receipt["integrity_sha256"] = _sha256_json(receipt)
    receipt_path = root / f"{arm}-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return prediction, receipt_path


def _contract(benchmark_id: str, *, instance_id: str = "repo__repo-1"):
    return PatchAdmissionContract(
        round_id="round-000001",
        arm="baseline",
        task_uid="d" * 64,
        benchmark_id=benchmark_id,
        instance_id=instance_id,
        agent_program_sha256="a" * 64,
        baseline_contract_sha256="b" * 64,
        evaluator_epoch="native-v1",
    )


@pytest.mark.parametrize(
    "benchmark_id", ("swe-bench-verified", "swe-bench-multilingual")
)
def test_swe_report_is_normalized_without_dropping_unresolved_or_regressions(
    tmp_path: Path, benchmark_id: str
):
    prediction, receipt = _write_patch_receipt(tmp_path, benchmark_id=benchmark_id)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "repo__repo-1": {
                    "resolved": False,
                    "patch_successfully_applied": True,
                    "tests_status": {
                        "PASS_TO_PASS": {"success": [], "failure": ["a", "b"]},
                        "PASS_TO_FAIL": {"success": [], "failure": ["c"]},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    admission = normalize_patch_result(
        contract=_contract(benchmark_id),
        prediction_path=prediction,
        agent_receipt_path=receipt,
        evaluator_report_path=report,
        evidence_path=tmp_path / "normalized.json",
    )

    assert admission.result.resolved is False
    assert admission.result.regression_failures == 3
    assert admission.result.safe is True
    assert admission.result.input_tokens == 100
    assert (
        admission.evidence["native_report"]["sha256"]
        == hashlib.sha256(report.read_bytes()).hexdigest()
    )

    prediction.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ABContractError, match="prediction.*tampered"):
        normalize_patch_result(
            contract=_contract(benchmark_id),
            prediction_path=prediction,
            agent_receipt_path=receipt,
            evaluator_report_path=report,
            evidence_path=tmp_path / "rejected.json",
        )


def test_swe_schema2_error_report_is_preserved_as_infrastructure_invalid(
    tmp_path: Path,
):
    benchmark_id = "swe-bench-multilingual"
    prediction, receipt = _write_patch_receipt(tmp_path, benchmark_id=benchmark_id)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "submitted_ids": ["repo__repo-1"],
                "completed_ids": [],
                "resolved_ids": [],
                "unresolved_ids": [],
                "error_ids": ["repo__repo-1"],
                "empty_patch_ids": [],
            }
        ),
        encoding="utf-8",
    )

    admission = normalize_patch_result(
        contract=_contract(benchmark_id),
        prediction_path=prediction,
        agent_receipt_path=receipt,
        evaluator_report_path=report,
        evidence_path=tmp_path / "normalized.json",
    )

    assert admission.result.resolved is False
    assert admission.result.safe is False
    assert admission.evidence["outcome"]["native_valid"] is False
    assert admission.evidence["outcome"]["native_error"] == "evaluator_error"


@pytest.mark.parametrize("instance_id", ["example/repo:pr-7", "example__repo-7"])
def test_multi_swe_report_uses_native_validity_and_counts_new_failures(
    tmp_path: Path, instance_id: str
):
    prediction, receipt = _write_patch_receipt(
        tmp_path,
        benchmark_id="multi-swe-bench-flash",
        instance_id=instance_id,
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "org": "example",
                "repo": "repo",
                "number": 7,
                "valid": False,
                "error_msg": "regression",
                "test_patch_result": {
                    "passed_tests": ["stable", "other"],
                    "failed_tests": ["target"],
                },
                "fix_patch_result": {
                    "passed_tests": ["target", "other"],
                    "failed_tests": ["stable"],
                },
            }
        ),
        encoding="utf-8",
    )

    admission = normalize_patch_result(
        contract=_contract("multi-swe-bench-flash", instance_id=instance_id),
        prediction_path=prediction,
        agent_receipt_path=receipt,
        evaluator_report_path=report,
        evidence_path=tmp_path / "normalized.json",
    )

    assert admission.result.resolved is False
    assert admission.result.regression_failures == 1
    assert admission.result.benchmark_family == "multi_swe_bench"


def test_harbor_result_requires_pre_verifier_receipt_and_uses_native_reward(
    tmp_path: Path,
):
    trajectory = tmp_path / "trajectory.json"
    receipt = tmp_path / "frozen-prediction.json"
    result = tmp_path / "result.json"
    trajectory.write_text('{"steps": []}\n', encoding="utf-8")
    write_prediction_receipt(
        receipt_path=receipt,
        trajectory_path=trajectory,
        round_id="round-000001",
        arm="evolved",
        agent_program_sha256="a" * 64,
        baseline_contract_sha256="b" * 64,
        harness_revision="c" * 40,
    )
    result.write_text(
        json.dumps(
            {
                "task_name": "terminal-task",
                "agent_execution": {
                    "started_at": "2026-08-03T10:00:00+00:00",
                    "finished_at": "2026-08-03T10:00:12+00:00",
                },
                "verifier": {"started_at": "2026-08-03T10:00:13+00:00"},
                "agent_result": {
                    "n_input_tokens": 200,
                    "n_output_tokens": 40,
                },
                "verifier_result": {"rewards": {"reward": 1}},
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )

    admission = normalize_harbor_result(
        result_path=result,
        receipt_path=receipt,
        trajectory_path=trajectory,
        evidence_path=tmp_path / "normalized.json",
        expected_round_id="round-000001",
        expected_arm="evolved",
        expected_instance_id="terminal-task",
        expected_agent_program_sha256="a" * 64,
        expected_baseline_contract_sha256="b" * 64,
        expected_harness_revision="c" * 40,
        evaluator_epoch="native-v1",
    )

    assert admission.result.resolved is True
    assert admission.result.safe is True
    assert admission.result.total_tokens == 240
    assert admission.result.elapsed_seconds == 12
    assert admission.result.benchmark_family == "terminal_bench"
