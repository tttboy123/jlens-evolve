"""Normalize pinned native benchmark reports into auditable matched-arm results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from continuous_ab import ABContractError, ArmResult
from evolve_jlens_harbor import validate_prediction_receipt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ABContractError(f"{label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ABContractError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ABContractError(f"{label} must be a JSON object")
    return payload


def _file_evidence(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_evidence(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload["integrity_sha256"] = _sha256_json(payload)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


@dataclass(frozen=True)
class PatchAdmissionContract:
    round_id: str
    arm: str
    task_uid: str
    benchmark_id: str
    instance_id: str
    agent_program_sha256: str
    baseline_contract_sha256: str
    evaluator_epoch: str

    def __post_init__(self) -> None:
        if (
            self.arm not in {"baseline", "evolved", "original", "parent", "candidate"}
            and re.fullmatch(r"candidate-[0-9]+", self.arm) is None
        ):
            raise ABContractError("patch admission arm is invalid")
        if self.benchmark_id not in {
            "swe-bench-verified",
            "swe-bench-multilingual",
            "multi-swe-bench-flash",
        }:
            raise ABContractError("patch admission benchmark is unsupported")
        if not all(
            (
                self.round_id,
                self.task_uid,
                self.instance_id,
                self.evaluator_epoch,
            )
        ):
            raise ABContractError("patch admission identity is incomplete")


@dataclass(frozen=True)
class NormalizedAdmission:
    result: ArmResult
    evidence_path: Path
    evidence: dict[str, Any]


def _validate_patch_receipt(
    *,
    contract: PatchAdmissionContract,
    receipt_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    receipt = _read_object(receipt_path, label="Agent receipt")
    integrity = receipt.pop("integrity_sha256", None)
    if integrity != _sha256_json(receipt):
        raise ABContractError("Agent receipt was tampered")
    receipt["integrity_sha256"] = integrity
    expected = {
        "round_id": contract.round_id,
        "arm": contract.arm,
        "task_uid": contract.task_uid,
        "benchmark_id": contract.benchmark_id,
        "instance_id": contract.instance_id,
        "agent_program_sha256": contract.agent_program_sha256,
        "baseline_contract_sha256": contract.baseline_contract_sha256,
        "evaluator_epoch": contract.evaluator_epoch,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ABContractError("Agent receipt contract mismatch")
    prediction_path = prediction_path.resolve()
    frozen = receipt.get("prediction")
    if not isinstance(frozen, dict):
        raise ABContractError("Agent receipt has no frozen prediction")
    try:
        recorded_path = Path(str(frozen.get("path"))).resolve()
    except (OSError, ValueError) as error:
        raise ABContractError("Agent prediction path is invalid") from error
    if (
        recorded_path != prediction_path
        or not prediction_path.is_file()
        or frozen.get("frozen") is not True
        or frozen.get("bytes") != prediction_path.stat().st_size
        or frozen.get("sha256") != _sha256_file(prediction_path)
    ):
        raise ABContractError("frozen prediction was tampered")
    return receipt


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ABContractError(f"{field} must be a non-negative integer")
    return value


def _patch_costs(receipt: dict[str, Any]) -> tuple[int, int, float, bool]:
    usage = receipt.get("usage")
    if not isinstance(usage, dict):
        raise ABContractError("Agent receipt usage is missing")
    input_tokens = _nonnegative_int(usage.get("input_tokens"), field="input_tokens")
    output_tokens = _nonnegative_int(usage.get("output_tokens"), field="output_tokens")
    elapsed = receipt.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise ABContractError("elapsed_seconds must be finite and non-negative")
    safe = (
        receipt.get("returncode") == 0
        and receipt.get("execution_success") is True
        and receipt.get("timed_out") is False
        and receipt.get("token_budget_exceeded") is False
        and receipt.get("workspace_head_unchanged") is True
    )
    return input_tokens, output_tokens, float(elapsed), safe


def _failure_count(value: Any, *, field: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ABContractError(f"{field} must be a list of test names")
    return len(set(value))


def _normalize_swe_report(
    report: dict[str, Any], instance_id: str
) -> tuple[bool, int, dict[str, Any]]:
    if report.get("schema_version") == 2:
        submitted = _string_set(report.get("submitted_ids"), field="submitted_ids")
        completed = _string_set(report.get("completed_ids"), field="completed_ids")
        resolved_ids = _string_set(report.get("resolved_ids"), field="resolved_ids")
        unresolved_ids = _string_set(
            report.get("unresolved_ids"), field="unresolved_ids"
        )
        error_ids = _string_set(report.get("error_ids"), field="error_ids")
        empty_ids = _string_set(report.get("empty_patch_ids"), field="empty_patch_ids")
        if instance_id not in submitted:
            raise ABContractError(
                "SWE-bench summary does not contain the frozen instance"
            )
        outcomes = {
            "resolved": instance_id in resolved_ids,
            "unresolved": instance_id in unresolved_ids,
            "evaluator_error": instance_id in error_ids,
            "empty_patch": instance_id in empty_ids,
        }
        if sum(outcomes.values()) != 1:
            raise ABContractError("SWE-bench summary outcome is ambiguous")
        if (
            outcomes["resolved"] or outcomes["unresolved"]
        ) and instance_id not in completed:
            raise ABContractError("SWE-bench completed outcome is inconsistent")
        native_error = next(
            (name for name in ("evaluator_error", "empty_patch") if outcomes[name]),
            None,
        )
        native_valid = native_error is None
        resolved = outcomes["resolved"]
        return (
            resolved,
            0,
            {
                "resolved": resolved,
                "patch_successfully_applied": native_valid,
                "regression_test_names": [],
                "native_valid": native_valid,
                "native_error": native_error,
                "report_schema_version": 2,
            },
        )
    if instance_id not in report or not isinstance(report[instance_id], dict):
        raise ABContractError("SWE-bench report does not contain the frozen instance")
    row = report[instance_id]
    resolved = row.get("resolved")
    applied = row.get("patch_successfully_applied")
    if not isinstance(resolved, bool) or not isinstance(applied, bool):
        raise ABContractError("SWE-bench report outcome fields are invalid")
    statuses = row.get("tests_status")
    if not isinstance(statuses, dict):
        raise ABContractError("SWE-bench tests_status is missing")
    regression_names: set[str] = set()
    for category in ("PASS_TO_PASS", "PASS_TO_FAIL"):
        bucket = statuses.get(category, {})
        if not isinstance(bucket, dict):
            raise ABContractError("SWE-bench test status bucket is invalid")
        failures = bucket.get("failure", [])
        _failure_count(failures, field=f"{category}.failure")
        regression_names.update(failures)
    summary = {
        "resolved": resolved,
        "patch_successfully_applied": applied,
        "regression_test_names": sorted(regression_names),
        "native_valid": True,
        "native_error": None,
    }
    return resolved, len(regression_names), summary


def _string_set(value: Any, *, field: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ABContractError(f"{field} must be a list of test names")
    return set(value)


def _normalize_multi_swe_report(
    report: dict[str, Any], instance_id: str
) -> tuple[bool, int, dict[str, Any]]:
    org = report.get("org")
    repo = report.get("repo")
    number = report.get("number")
    if (
        not isinstance(org, str)
        or not isinstance(repo, str)
        or not isinstance(number, int)
    ):
        raise ABContractError("Multi-SWE-bench report identity is invalid")
    accepted_ids = {f"{org}/{repo}:pr-{number}", f"{org}__{repo}-{number}"}
    if instance_id not in accepted_ids:
        raise ABContractError("Multi-SWE-bench report instance mismatch")
    valid = report.get("valid")
    if not isinstance(valid, bool):
        raise ABContractError("Multi-SWE-bench native validity is missing")
    before = report.get("test_patch_result")
    after = report.get("fix_patch_result")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ABContractError("Multi-SWE-bench test results are missing")
    previously_passing = _string_set(
        before.get("passed_tests"), field="test_patch_result.passed_tests"
    )
    now_failing = _string_set(
        after.get("failed_tests"), field="fix_patch_result.failed_tests"
    )
    regressions = sorted(previously_passing & now_failing)
    summary = {
        "resolved": valid,
        "native_valid": valid,
        "native_error": report.get("error_msg"),
        "regression_test_names": regressions,
    }
    return valid, len(regressions), summary


def normalize_patch_result(
    *,
    contract: PatchAdmissionContract,
    prediction_path: Path,
    agent_receipt_path: Path,
    evaluator_report_path: Path,
    evidence_path: Path,
) -> NormalizedAdmission:
    """Admit one frozen patch and its pinned official evaluator report."""

    receipt = _validate_patch_receipt(
        contract=contract,
        receipt_path=agent_receipt_path,
        prediction_path=prediction_path,
    )
    report = _read_object(evaluator_report_path, label="native evaluator report")
    input_tokens, output_tokens, elapsed, safe = _patch_costs(receipt)
    if contract.benchmark_id in {
        "swe-bench-verified",
        "swe-bench-multilingual",
    }:
        resolved, regressions, outcome = _normalize_swe_report(
            report, contract.instance_id
        )
        safe = safe and outcome["native_valid"]
        family = "swe_bench"
    else:
        resolved, regressions, outcome = _normalize_multi_swe_report(
            report, contract.instance_id
        )
        family = "multi_swe_bench"
    result = ArmResult(
        arm=("baseline" if contract.arm in {"baseline", "original"} else "evolved"),
        resolved=resolved,
        regression_failures=regressions,
        safe=safe,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        benchmark_family=family,
        evaluator_epoch=contract.evaluator_epoch,
    )
    evidence = _write_evidence(
        evidence_path,
        {
            "schema_version": "1.0",
            "kind": "native_patch_result_admission",
            "contract": asdict(contract),
            "result": asdict(result),
            "outcome": outcome,
            "prediction": _file_evidence(prediction_path),
            "agent_receipt": _file_evidence(agent_receipt_path),
            "native_report": _file_evidence(evaluator_report_path),
        },
    )
    return NormalizedAdmission(result, evidence_path.resolve(), evidence)


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ABContractError(f"Harbor {field} timestamp is missing")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ABContractError(f"Harbor {field} timestamp is invalid") from error


def _harbor_tokens(result: dict[str, Any]) -> tuple[int, int]:
    contexts: list[dict[str, Any]] = []
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        contexts.append(agent_result)
    elif isinstance(result.get("step_results"), list):
        contexts.extend(
            step["agent_result"]
            for step in result["step_results"]
            if isinstance(step, dict) and isinstance(step.get("agent_result"), dict)
        )
    if not contexts:
        raise ABContractError("Harbor Agent token usage is missing")
    inputs = sum(
        _nonnegative_int(context.get("n_input_tokens"), field="n_input_tokens")
        for context in contexts
    )
    outputs = sum(
        _nonnegative_int(context.get("n_output_tokens"), field="n_output_tokens")
        for context in contexts
    )
    return inputs, outputs


def _harbor_reward(result: dict[str, Any]) -> float:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    if not isinstance(rewards, dict) or not rewards:
        raise ABContractError("Harbor verifier reward is missing")
    if "reward" in rewards:
        value = rewards["reward"]
    elif len(rewards) == 1:
        value = next(iter(rewards.values()))
    else:
        raise ABContractError("Harbor primary reward is ambiguous")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 <= float(value) <= 1
    ):
        raise ABContractError("Harbor reward must be finite and within [0, 1]")
    return float(value)


def normalize_harbor_result(
    *,
    result_path: Path,
    receipt_path: Path,
    trajectory_path: Path,
    evidence_path: Path,
    expected_round_id: str,
    expected_arm: str,
    expected_instance_id: str,
    expected_agent_program_sha256: str,
    expected_baseline_contract_sha256: str,
    expected_harness_revision: str,
    evaluator_epoch: str,
) -> NormalizedAdmission:
    """Admit a Terminal-Bench Harbor result after pre-verifier trajectory proof."""

    if expected_arm not in {"baseline", "evolved"}:
        raise ABContractError("Harbor admission arm is invalid")
    validate_prediction_receipt(
        receipt_path=receipt_path,
        trajectory_path=trajectory_path,
        result_path=result_path,
        expected_round_id=expected_round_id,
        expected_arm=expected_arm,
        expected_agent_program_sha256=expected_agent_program_sha256,
        expected_baseline_contract_sha256=expected_baseline_contract_sha256,
        expected_harness_revision=expected_harness_revision,
    )
    native = _read_object(result_path, label="Harbor native result")
    if native.get("task_name") != expected_instance_id:
        raise ABContractError("Harbor native result instance mismatch")
    reward = _harbor_reward(native)
    input_tokens, output_tokens = _harbor_tokens(native)
    execution = native.get("agent_execution")
    if not isinstance(execution, dict):
        raise ABContractError("Harbor agent timing is missing")
    started = _parse_datetime(execution.get("started_at"), field="agent started")
    finished = _parse_datetime(execution.get("finished_at"), field="agent finished")
    elapsed = (finished - started).total_seconds()
    if elapsed < 0:
        raise ABContractError("Harbor agent duration is negative")
    result = ArmResult(
        arm=expected_arm,
        resolved=reward == 1.0,
        regression_failures=0,
        safe=native.get("exception_info") is None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
        benchmark_family="terminal_bench",
        evaluator_epoch=evaluator_epoch,
    )
    evidence = _write_evidence(
        evidence_path,
        {
            "schema_version": "1.0",
            "kind": "native_harbor_result_admission",
            "identity": {
                "round_id": expected_round_id,
                "arm": expected_arm,
                "instance_id": expected_instance_id,
                "agent_program_sha256": expected_agent_program_sha256,
                "baseline_contract_sha256": expected_baseline_contract_sha256,
                "harness_revision": expected_harness_revision,
                "evaluator_epoch": evaluator_epoch,
            },
            "result": asdict(result),
            "outcome": {"native_reward": reward, "resolved": reward == 1.0},
            "prediction": _file_evidence(trajectory_path),
            "prediction_receipt": _file_evidence(receipt_path),
            "native_report": _file_evidence(result_path),
        },
    )
    return NormalizedAdmission(result, evidence_path.resolve(), evidence)
