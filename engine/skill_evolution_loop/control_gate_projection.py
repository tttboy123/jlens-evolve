"""Append-only projections for normalized convergence and independent safety."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .convergence_gate import normalized_convergence_metrics
from .independent_safety import build_independent_safety_report
from .statistical_capability_gate import evaluate_statistical_capability_gate


def _load(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.resolve().read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value, raw


def _verify_embedded_sha(value: dict[str, Any], *, label: str) -> None:
    evidence_sha = value.get("evidence_sha256")
    content = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if evidence_sha != sha256_json(content):
        raise ContractError(f"{label} evidence sha256 mismatch")


def _freeze(path: Path, report: dict[str, Any], *, label: str) -> None:
    output = path.resolve()
    if output.exists():
        existing, _raw = _load(output, label=f"existing {label}")
        if existing != report:
            raise ContractError(f"existing {label} does not match replay")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")


def freeze_convergence_projection(
    *, source_path: Path, output_path: Path
) -> dict[str, Any]:
    source, raw = _load(source_path, label="convergence source")
    _verify_embedded_sha(source, label="convergence source")
    required = {
        "schema_version",
        "contract",
        "generation_id",
        "original_sha256",
        "parent_sha256",
        "parent_vs_original",
        "per_candidate",
        "network_calls_performed",
        "evidence_sha256",
    }
    if set(source) != required or source.get("schema_version") != 1:
        raise ContractError("convergence source fields are invalid")
    if (
        source.get("contract") != "paired-convergence-input-v1"
        or source.get("network_calls_performed") is not False
        or not isinstance(source.get("per_candidate"), dict)
    ):
        raise ContractError("convergence source boundary is invalid")
    try:
        metrics = normalized_convergence_metrics(
            per_candidate=source["per_candidate"],
            parent_vs_original=source["parent_vs_original"],
        )
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    content = {
        "schema_version": 1,
        "contract": "parent-relative-convergence-projection-v1",
        "generation_id": source["generation_id"],
        "original_sha256": source["original_sha256"],
        "parent_sha256": source["parent_sha256"],
        "source_evidence_sha256": source["evidence_sha256"],
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "metrics": metrics,
        "network_calls_performed": False,
        "core_engine_mutated": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path, report, label="convergence projection")
    return report


def freeze_independent_safety_projection(
    *,
    subject_sha256: str,
    receipt_paths: tuple[Path, ...],
    output_path: Path,
) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for path in sorted((item.resolve() for item in receipt_paths), key=str):
        receipt, raw = _load(path, label="independent safety receipt")
        _verify_embedded_sha(receipt, label="independent safety receipt")
        if (
            receipt.get("schema_version") != 1
            or receipt.get("contract") != "independent-safety-probe-v1"
            or receipt.get("subject_sha256") != subject_sha256
            or receipt.get("network_calls_performed") is not False
        ):
            raise ContractError("independent safety receipt boundary is invalid")
        probes.append(
            {
                key: value
                for key, value in receipt.items()
                if key
                in {
                    "category",
                    "probe_id",
                    "passed",
                    "evaluator_valid",
                    "evaluator",
                    "error",
                }
            }
            | {"evidence_sha256": receipt["evidence_sha256"]}
        )
        sources.append(
            {
                "path": str(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "evidence_sha256": receipt["evidence_sha256"],
            }
        )
    base = build_independent_safety_report(
        subject_sha256=subject_sha256, probes=tuple(probes)
    )
    content = {
        **{key: value for key, value in base.items() if key != "evidence_sha256"},
        "source_receipts": sources,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path, report, label="independent safety projection")
    return report


def freeze_runtime_identity_projection(
    *, source_path: Path, output_path: Path
) -> dict[str, Any]:
    """Freeze the immutable identity of one student execution."""

    source, raw = _load(source_path, label="runtime identity source")
    _verify_embedded_sha(source, label="runtime identity source")
    required = {
        "schema_version",
        "contract",
        "run_id",
        "model",
        "tokenizer",
        "runtime",
        "generation",
        "transport",
        "skill",
        "framework",
        "network_calls_performed",
        "evidence_sha256",
    }
    if (
        set(source) != required
        or source.get("schema_version") != 1
        or source.get("contract") != "runtime-identity-input-v1"
        or source.get("network_calls_performed") is not False
        or not isinstance(source.get("run_id"), str)
        or not source["run_id"].strip()
    ):
        raise ContractError("runtime identity source boundary is invalid")

    def exact_object(name: str, fields: set[str]) -> dict[str, Any]:
        value = source.get(name)
        if not isinstance(value, dict) or set(value) != fields:
            raise ContractError(f"runtime identity {name} fields are invalid")
        return value

    def sha(value: Any, *, label: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"runtime identity {label} is not a sha256")
        return value

    model = exact_object(
        "model", {"family", "revision", "weights_sha256", "quantization"}
    )
    tokenizer = exact_object("tokenizer", {"tokenizer_sha256", "chat_template_sha256"})
    runtime = exact_object("runtime", {"engine", "version", "container_digest"})
    generation = exact_object("generation", {"parameters", "seed"})
    transport = exact_object("transport", {"contract", "source_sha256"})
    skill = exact_object("skill", {"sha256", "status"})
    framework = exact_object("framework", {"sha256"})
    for label, value in (
        ("model family", model["family"]),
        ("model revision", model["revision"]),
        ("model quantization", model["quantization"]),
        ("runtime engine", runtime["engine"]),
        ("runtime version", runtime["version"]),
        ("transport contract", transport["contract"]),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"runtime identity {label} is empty")
    sha(model["weights_sha256"], label="model weights")
    sha(tokenizer["tokenizer_sha256"], label="tokenizer")
    sha(tokenizer["chat_template_sha256"], label="chat template")
    sha(transport["source_sha256"], label="transport source")
    sha(skill["sha256"], label="Skill")
    sha(framework["sha256"], label="framework")
    digest = runtime["container_digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ContractError("runtime identity container digest is not immutable")
    sha(digest.removeprefix("sha256:"), label="container digest")
    if (
        not isinstance(generation["parameters"], dict)
        or not generation["parameters"]
        or not isinstance(generation["seed"], int)
        or isinstance(generation["seed"], bool)
        or skill["status"] != "inactive"
    ):
        raise ContractError("runtime identity generation or Skill boundary is invalid")

    content = {
        "schema_version": 1,
        "contract": "replayable-runtime-identity-v1",
        "run_id": source["run_id"],
        "complete": True,
        "model": model,
        "tokenizer": tokenizer,
        "runtime": runtime,
        "generation": generation,
        "transport": transport,
        "skill": skill,
        "framework": framework,
        "source_evidence_sha256": source["evidence_sha256"],
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "network_calls_performed": False,
        "core_engine_mutated": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path, report, label="runtime identity projection")
    return report


def freeze_accelerator_preflight_projection(
    *, source_path: Path, output_path: Path
) -> dict[str, Any]:
    """Admit paid model work only after a replayable accelerator smoke."""

    source, raw = _load(source_path, label="accelerator preflight source")
    _verify_embedded_sha(source, label="accelerator preflight source")
    required = {
        "schema_version",
        "contract",
        "run_id",
        "installer",
        "accelerator",
        "container_runtime",
        "reboot_completed",
        "network_calls_performed",
        "evidence_sha256",
    }
    if (
        set(source) != required
        or source.get("schema_version") != 1
        or source.get("contract") != "accelerator-runtime-preflight-input-v1"
        or not isinstance(source.get("run_id"), str)
        or not source["run_id"].strip()
        or source.get("network_calls_performed") is not True
    ):
        raise ContractError("accelerator preflight source boundary is invalid")

    def exact_object(name: str, fields: set[str]) -> dict[str, Any]:
        value = source.get(name)
        if not isinstance(value, dict) or set(value) != fields:
            raise ContractError(f"accelerator preflight {name} fields are invalid")
        return value

    def sha(value: Any, *, label: str) -> str:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ContractError(f"accelerator preflight {label} is not a sha256")
        return value

    installer = exact_object("installer", {"kind", "source_sha256"})
    accelerator = exact_object(
        "accelerator",
        {"vendor", "model", "device_uuid", "driver_version", "management_cli"},
    )
    container_runtime = exact_object(
        "container_runtime",
        {"engine", "version", "gpu_smoke_passed", "gpu_smoke_sha256"},
    )
    for label, value in (
        ("installer kind", installer["kind"]),
        ("accelerator vendor", accelerator["vendor"]),
        ("accelerator model", accelerator["model"]),
        ("accelerator device uuid", accelerator["device_uuid"]),
        ("accelerator driver version", accelerator["driver_version"]),
        ("accelerator management cli", accelerator["management_cli"]),
        ("container engine", container_runtime["engine"]),
        ("container version", container_runtime["version"]),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ContractError(f"accelerator preflight {label} is empty")
    sha(installer["source_sha256"], label="installer source")
    sha(container_runtime["gpu_smoke_sha256"], label="GPU smoke")
    if (
        source["reboot_completed"] is not True
        or container_runtime["gpu_smoke_passed"] is not True
    ):
        raise ContractError("accelerator preflight runtime is not ready")

    content = {
        "schema_version": 1,
        "contract": "accelerator-runtime-preflight-v1",
        "run_id": source["run_id"],
        "ready": True,
        "model_download_admitted": True,
        "installer": installer,
        "accelerator": accelerator,
        "container_runtime": container_runtime,
        "reboot_completed": True,
        "source_evidence_sha256": source["evidence_sha256"],
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_network_calls_performed": True,
        "projection_network_calls_performed": False,
        "core_engine_mutated": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path, report, label="accelerator preflight projection")
    return report


def freeze_statistical_capability_projection(
    *,
    feedback_path: Path,
    holdout_path: Path,
    independent_safety_path: Path,
    runtime_identity_path: Path,
    cost_receipt_path: Path,
    catalog_audit_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind all strict capability inputs to one replayable gap report."""

    paths = {
        "feedback": feedback_path,
        "holdout": holdout_path,
        "independent_safety": independent_safety_path,
        "runtime_identity": runtime_identity_path,
        "cost_receipt": cost_receipt_path,
        "catalog_audit": catalog_audit_path,
    }
    values: dict[str, dict[str, Any]] = {}
    source_files: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        value, raw = _load(path, label=f"{name} capability evidence")
        values[name] = value
        source_files[name] = {
            "path": str(path.resolve()),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
        }
    base = evaluate_statistical_capability_gate(**values)
    content = {
        **{key: value for key, value in base.items() if key != "evidence_sha256"},
        "source_files": source_files,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path, report, label="statistical capability projection")
    return report
