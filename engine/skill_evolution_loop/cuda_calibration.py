"""Freeze and evaluate the three-task MLX-versus-CUDA migration gate."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .contracts import canonical_json, sha256_json
from .model_transport import ModelTransport, PromptGenerationRequest


class CalibrationError(ValueError):
    pass


def load_calibration_evidence(path: Path, label: str) -> dict[str, Any]:
    return _load_evidence(path, label)


def replay_cuda_calibration(
    *,
    manifest: dict[str, Any],
    evidence_root: Path,
    transport: ModelTransport,
    max_tokens: int,
) -> dict[str, Any]:
    """Replay exact rendered prompts remotely with append-only cell receipts."""

    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise CalibrationError("calibration manifest cells are invalid")
    if type(max_tokens) is not int or max_tokens < 1:
        raise CalibrationError("calibration max_tokens must be positive")
    completed: list[dict[str, Any]] = []
    generation_seconds = 0.0
    for reference in sorted(cells, key=lambda row: row["cell_id"]):
        cell_id = reference.get("cell_id")
        if not isinstance(cell_id, str) or "/" not in cell_id:
            raise CalibrationError("calibration cell ID is invalid")
        task_id, condition_id = cell_id.split("/", 1)
        cell_root = evidence_root / "cells" / task_id / condition_id
        receipt_path = cell_root / "RESPONSE.json"
        if receipt_path.exists():
            receipt = _load_evidence(receipt_path, "CUDA replay receipt")
            if receipt.get("cell_id") != cell_id or receipt.get(
                "manifest_evidence_sha256"
            ) != manifest.get("evidence_sha256"):
                raise CalibrationError("CUDA replay receipt identity mismatch")
            completed.append(receipt)
            continue
        prompt_path = Path(str(reference.get("prompt_path")))
        try:
            prompt = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CalibrationError("calibration prompt is unreadable") from exc
        if _sha(prompt_path) != reference.get("prompt_sha256"):
            raise CalibrationError("calibration prompt SHA mismatch")
        request = PromptGenerationRequest.create(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=0,
        )
        started = time.monotonic()
        response = transport.generate_prompt(request)
        elapsed = time.monotonic() - started
        generation_seconds += elapsed
        cell_root.mkdir(parents=True, exist_ok=True)
        output_path = cell_root / "generation-output-000.txt"
        if output_path.exists():
            raise CalibrationError("orphan CUDA calibration output exists")
        output_path.write_text(response.text, encoding="utf-8")
        content = {
            "schema_version": 1,
            "cell_id": cell_id,
            "manifest_evidence_sha256": manifest.get("evidence_sha256"),
            "prompt_sha256": reference["prompt_sha256"],
            "request_sha256": response.request_sha256,
            "response_sha256": response.response_sha256,
            "output_sha256": _sha(output_path),
            "model": response.model,
            "finish_reason": response.finish_reason,
            "usage": response.usage,
            "generation_seconds": round(elapsed, 6),
            "transport_identity": transport.identity(),
            "generation_parameters": {
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 0,
            },
            "network_calls_performed": True,
        }
        receipt = {**content, "evidence_sha256": sha256_json(content)}
        _freeze(receipt_path, receipt)
        completed.append(receipt)
    projection_content = {
        "schema_version": 1,
        "manifest_evidence_sha256": manifest.get("evidence_sha256"),
        "planned_cells": len(cells),
        "completed_cells": len(completed),
        "usage_totals": _sum_usage(completed),
        "effective_token_ratio": _effective_token_ratio(completed),
        "generation_seconds": round(
            sum(float(row.get("generation_seconds", 0.0)) for row in completed), 6
        ),
        "cell_evidence_fingerprint": sha256_json(
            sorted((row["cell_id"], row["evidence_sha256"]) for row in completed)
        ),
        "status": "complete" if len(completed) == len(cells) else "partial",
        "network_calls_performed": bool(completed),
    }
    projection = {
        **projection_content,
        "evidence_sha256": sha256_json(projection_content),
    }
    progress_path = evidence_root / "PROGRESS.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(canonical_json(projection) + "\n", encoding="utf-8")
    return projection


def _sum_usage(receipts: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for receipt in receipts:
        usage = receipt.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if type(value) is int and value >= 0:
                totals[key] = totals.get(key, 0) + value
    return dict(sorted(totals.items()))


def _effective_token_ratio(receipts: list[dict[str, Any]]) -> float | None:
    usage = _sum_usage(receipts)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", completion)
    if total <= 0:
        return None
    return round(completion / total, 6)


def build_calibration_manifest(
    *, cells_root: Path, task_ids: tuple[str, ...], output_path: Path
) -> dict[str, Any]:
    if len(task_ids) != 3 or len(set(task_ids)) != 3:
        raise CalibrationError("calibration requires exactly three unique tasks")
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_folder = cells_root / task_id
        condition_folders = sorted(
            path
            for path in task_folder.iterdir()
            if path.is_dir() and path.name.endswith(("-baseline", "-taught"))
        )
        if len(condition_folders) != 2:
            raise CalibrationError("calibration task requires one baseline/taught pair")
        for condition in condition_folders:
            attempt_path = condition / "ATTEMPT.json"
            try:
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                trace = attempt["generation_trace"][0]
                prompt_path = condition / trace["prompt_path"]
                output = condition / trace["path"]
            except (OSError, KeyError, IndexError, json.JSONDecodeError) as exc:
                raise CalibrationError(
                    "calibration attempt evidence is invalid"
                ) from exc
            rows.append(
                {
                    "cell_id": f"{task_id}/{condition.name}",
                    "task_id": task_id,
                    "condition_id": condition.name,
                    "prompt_path": prompt_path.as_posix(),
                    "prompt_sha256": _sha(prompt_path),
                    "mlx_output_path": output.as_posix(),
                    "mlx_output_sha256": _sha(output),
                    "mlx_structural_valid": bool(
                        attempt.get("attempt", {}).get("structural_valid")
                    ),
                    "mlx_patch_sha256": attempt.get("attempt", {}).get("patch_sha256"),
                }
            )
    rows.sort(key=lambda row: row["cell_id"])
    content = {
        "schema_version": 1,
        "gate_kind": "three-task-mlx-vs-cuda",
        "task_count": 3,
        "cell_count": 6,
        "task_ids": task_ids,
        "cells": rows,
        "acceptance": {
            "minimum_structural_agreement": 5,
            "maximum_safety_regressions": 0,
            "native_evaluator_separate": True,
        },
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path, report)
    return report


def evaluate_calibration(
    *, manifest: dict[str, Any], cuda_results: dict[str, Any]
) -> dict[str, Any]:
    references = {row["cell_id"]: row for row in manifest.get("cells", [])}
    results = {row["cell_id"]: row for row in cuda_results.get("cells", [])}
    if not references or references.keys() != results.keys():
        raise CalibrationError("CUDA results do not match calibration cells")
    agreement = sum(
        bool(reference["mlx_structural_valid"])
        == bool(results[cell]["structural_valid"])
        for cell, reference in references.items()
    )
    regressions = sum(bool(row.get("safety_regression")) for row in results.values())
    gate_passed = agreement >= 5 and regressions == 0
    content = {
        "schema_version": 1,
        "gate_kind": "three-task-mlx-vs-cuda",
        "cell_count": len(references),
        "structural_agreement_count": agreement,
        "safety_regression_count": regressions,
        "gate_passed": gate_passed,
        "native_evaluator_separate": True,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def collect_cuda_calibration(
    *,
    manifest: dict[str, Any],
    experiment_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Verify formal-runner cells and freeze the CUDA half of calibration."""

    manifest_cells = manifest.get("cells")
    if not isinstance(manifest_cells, list) or not manifest_cells:
        raise CalibrationError("calibration manifest cells are invalid")
    results: list[dict[str, Any]] = []
    for reference in manifest_cells:
        if not isinstance(reference, dict):
            raise CalibrationError("calibration manifest cell is invalid")
        task_id = reference.get("task_id")
        condition_id = reference.get("condition_id")
        if not isinstance(task_id, str) or not isinstance(condition_id, str):
            raise CalibrationError("calibration manifest cell identity is invalid")
        cell = experiment_root / "cells" / task_id / condition_id
        attempt_path = cell / "ATTEMPT.json"
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CalibrationError("CUDA calibration attempt is unreadable") from exc
        content = {
            key: value for key, value in attempt.items() if key != "evidence_sha256"
        }
        if attempt.get("evidence_sha256") != sha256_json(content):
            raise CalibrationError("CUDA calibration attempt evidence SHA mismatch")
        if (
            attempt.get("task", {}).get("task_id") != task_id
            or attempt.get("condition", {}).get("condition_id") != condition_id
        ):
            raise CalibrationError("CUDA calibration attempt identity mismatch")
        traces = attempt.get("generation_trace")
        if not isinstance(traces, list) or not traces:
            raise CalibrationError("CUDA calibration generation trace is missing")
        first = traces[0]
        try:
            prompt = cell / first["prompt_path"]
            output = cell / first["path"]
        except (KeyError, TypeError) as exc:
            raise CalibrationError(
                "CUDA calibration generation trace is invalid"
            ) from exc
        prompt_sha = _sha(prompt)
        output_sha = _sha(output)
        if prompt_sha != first.get("prompt_sha256"):
            raise CalibrationError("CUDA calibration prompt receipt mismatch")
        if prompt_sha != reference.get("prompt_sha256"):
            raise CalibrationError("CUDA calibration prompt drift")
        if output_sha != first.get("sha256"):
            raise CalibrationError("CUDA calibration output receipt mismatch")
        outcome = attempt.get("attempt")
        if not isinstance(outcome, dict):
            raise CalibrationError("CUDA calibration outcome is invalid")
        structural = outcome.get("structural_valid")
        if type(structural) is not bool:
            raise CalibrationError("CUDA calibration structural outcome is invalid")
        results.append(
            {
                "cell_id": reference["cell_id"],
                "task_id": task_id,
                "condition_id": condition_id,
                "prompt_sha256": prompt_sha,
                "output_sha256": output_sha,
                "attempt_evidence_sha256": attempt["evidence_sha256"],
                "structural_valid": structural,
                "failure_reason": outcome.get("failure_reason"),
                "patch_sha256": outcome.get("patch_sha256"),
                "safety_regression": (
                    bool(reference.get("mlx_structural_valid")) and not structural
                ),
            }
        )
    results.sort(key=lambda row: row["cell_id"])
    result_content = {
        "schema_version": 1,
        "gate_kind": "three-task-mlx-vs-cuda-results",
        "manifest_evidence_sha256": manifest.get("evidence_sha256"),
        "cell_count": len(results),
        "cells": results,
        "network_calls_performed": True,
    }
    report = {**result_content, "evidence_sha256": sha256_json(result_content)}
    _freeze(output_path, report)
    return report


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(path: Path, value: dict[str, Any]) -> None:
    content = canonical_json(value) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise CalibrationError("frozen calibration manifest does not match replay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_evidence(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise CalibrationError(f"{label} is invalid")
    content = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if value.get("evidence_sha256") != sha256_json(content):
        raise CalibrationError(f"{label} evidence SHA mismatch")
    return value
