from __future__ import annotations

from pathlib import Path

import pytest

from skill_evolution_loop.__main__ import _parser
from skill_evolution_loop.contracts import ContractError, canonical_json, sha256_json
from skill_evolution_loop.control_gate_projection import (
    freeze_accelerator_preflight_projection,
    freeze_convergence_projection,
    freeze_independent_safety_projection,
    freeze_runtime_identity_projection,
    freeze_statistical_capability_projection,
)


def _metric(delta: float, *, fingerprint: str = "a" * 64) -> dict[str, object]:
    return {
        "paired_tasks": 20,
        "paired_task_fingerprint": fingerprint,
        "native_score_delta_mean": delta,
        "cost_delta_mean": 0.0,
        "safety_regression": False,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def test_convergence_projection_freezes_parent_relative_evidence(
    tmp_path: Path,
) -> None:
    source_content = {
        "schema_version": 1,
        "contract": "paired-convergence-input-v1",
        "generation_id": "generation-7",
        "original_sha256": "1" * 64,
        "parent_sha256": "2" * 64,
        "parent_vs_original": _metric(0.4),
        "per_candidate": {
            "3" * 64: {
                "vs_original": _metric(0.405),
                "vs_parent": _metric(0.005),
            }
        },
        "network_calls_performed": False,
    }
    source = {
        **source_content,
        "evidence_sha256": sha256_json(source_content),
    }
    source_path = tmp_path / "input.json"
    output_path = tmp_path / "projection.json"
    _write_json(source_path, source)

    report = freeze_convergence_projection(
        source_path=source_path, output_path=output_path
    )
    replay = freeze_convergence_projection(
        source_path=source_path, output_path=output_path
    )

    assert replay == report
    assert report["metrics"]["normalized_mean_abs_delta"] == 0.005
    assert report["metrics"]["sample_gate"]["complete"] is True
    assert report["source_evidence_sha256"] == source["evidence_sha256"]
    assert report["network_calls_performed"] is False


def test_convergence_projection_rejects_pair_identity_drift(tmp_path: Path) -> None:
    source_content = {
        "schema_version": 1,
        "contract": "paired-convergence-input-v1",
        "generation_id": "generation-7",
        "original_sha256": "1" * 64,
        "parent_sha256": "2" * 64,
        "parent_vs_original": _metric(0.4, fingerprint="a" * 64),
        "per_candidate": {
            "3" * 64: {
                "vs_original": _metric(0.405, fingerprint="b" * 64),
                "vs_parent": _metric(0.005, fingerprint="a" * 64),
            }
        },
        "network_calls_performed": False,
    }
    source = {**source_content, "evidence_sha256": sha256_json(source_content)}
    source_path = tmp_path / "input.json"
    _write_json(source_path, source)

    with pytest.raises(ContractError, match="paired task fingerprint"):
        freeze_convergence_projection(
            source_path=source_path, output_path=tmp_path / "projection.json"
        )


def _probe_receipt(subject: str, category: str) -> dict[str, object]:
    content = {
        "schema_version": 1,
        "contract": "independent-safety-probe-v1",
        "subject_sha256": subject,
        "category": category,
        "probe_id": f"probe-{category}",
        "passed": True,
        "evaluator_valid": True,
        "evaluator": f"sandbox-{category}-v1",
        "error": None,
        "network_calls_performed": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def test_safety_projection_loads_four_self_verifying_receipts(tmp_path: Path) -> None:
    subject = "f" * 64
    paths = []
    for category in (
        "dangerous-command",
        "http-5xx",
        "private-data-exposure",
        "unauthorized-side-effect",
    ):
        path = tmp_path / f"{category}.json"
        _write_json(path, _probe_receipt(subject, category))
        paths.append(path)

    output = tmp_path / "SAFETY.json"
    report = freeze_independent_safety_projection(
        subject_sha256=subject,
        receipt_paths=tuple(paths),
        output_path=output,
    )
    replay = freeze_independent_safety_projection(
        subject_sha256=subject,
        receipt_paths=tuple(reversed(paths)),
        output_path=output,
    )

    assert replay == report
    assert report["suite_passed"] is True
    assert report["probe_count"] == 4
    assert len(report["source_receipts"]) == 4
    assert report["native_admission_reused"] is False


def test_safety_projection_rejects_tampered_receipt(tmp_path: Path) -> None:
    subject = "f" * 64
    paths = []
    for category in (
        "dangerous-command",
        "http-5xx",
        "private-data-exposure",
        "unauthorized-side-effect",
    ):
        path = tmp_path / f"{category}.json"
        receipt = _probe_receipt(subject, category)
        if category == "http-5xx":
            receipt["passed"] = False
        _write_json(path, receipt)
        paths.append(path)

    with pytest.raises(ContractError, match="evidence sha256 mismatch"):
        freeze_independent_safety_projection(
            subject_sha256=subject,
            receipt_paths=tuple(paths),
            output_path=tmp_path / "SAFETY.json",
        )


def test_control_gate_cli_exposes_generic_projection_commands() -> None:
    convergence = _parser().parse_args(
        [
            "convergence-project",
            "--source",
            "input.json",
            "--out",
            "projection.json",
        ]
    )
    safety = _parser().parse_args(
        [
            "independent-safety-project",
            "--subject-sha256",
            "f" * 64,
            "--probe",
            "one.json",
            "--probe",
            "two.json",
            "--out",
            "safety.json",
        ]
    )
    statistical = _parser().parse_args(
        [
            "statistical-capability-project",
            "--feedback",
            "feedback.json",
            "--holdout",
            "holdout.json",
            "--independent-safety",
            "safety.json",
            "--runtime-identity",
            "identity.json",
            "--cost-receipt",
            "cost.json",
            "--catalog-audit",
            "catalog.json",
            "--out",
            "capability.json",
        ]
    )
    identity = _parser().parse_args(
        [
            "runtime-identity-project",
            "--source",
            "runtime-input.json",
            "--out",
            "runtime-identity.json",
        ]
    )
    accelerator = _parser().parse_args(
        [
            "accelerator-preflight-project",
            "--source",
            "accelerator-input.json",
            "--out",
            "accelerator-preflight.json",
        ]
    )

    assert convergence.source == Path("input.json")
    assert safety.probes == [Path("one.json"), Path("two.json")]
    assert statistical.feedback == Path("feedback.json")
    assert statistical.catalog_audit == Path("catalog.json")
    assert identity.source == Path("runtime-input.json")
    assert accelerator.source == Path("accelerator-input.json")


def _runtime_identity_input() -> dict[str, object]:
    content = {
        "schema_version": 1,
        "contract": "runtime-identity-input-v1",
        "run_id": "feedback-r083",
        "model": {
            "family": "student-model",
            "revision": "immutable-revision",
            "weights_sha256": "1" * 64,
            "quantization": "awq-int4",
        },
        "tokenizer": {
            "tokenizer_sha256": "2" * 64,
            "chat_template_sha256": "3" * 64,
        },
        "runtime": {
            "engine": "cuda-serving-runtime",
            "version": "1.2.3",
            "container_digest": "sha256:" + "4" * 64,
        },
        "generation": {
            "parameters": {
                "temperature": 0.0,
                "max_tokens": 1536,
            },
            "seed": 7,
        },
        "transport": {
            "contract": "openai-compatible-v1",
            "source_sha256": "5" * 64,
        },
        "skill": {"sha256": "6" * 64, "status": "inactive"},
        "framework": {"sha256": "7" * 64},
        "network_calls_performed": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def test_runtime_identity_projection_freezes_complete_replayable_identity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "runtime-input.json"
    output_path = tmp_path / "RUNTIME-IDENTITY.json"
    _write_json(source_path, _runtime_identity_input())

    report = freeze_runtime_identity_projection(
        source_path=source_path, output_path=output_path
    )
    replay = freeze_runtime_identity_projection(
        source_path=source_path, output_path=output_path
    )

    assert replay == report
    assert report["contract"] == "replayable-runtime-identity-v1"
    assert report["complete"] is True
    assert report["skill"]["status"] == "inactive"
    assert report["source_file_sha256"]
    assert report["network_calls_performed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("runtime", "container_digest"), "latest"),
        (("generation", "seed"), None),
        (("skill", "status"), "active"),
        (("transport", "source_sha256"), "short"),
    ],
)
def test_runtime_identity_projection_rejects_incomplete_or_mutable_identity(
    tmp_path: Path, path: tuple[str, str], value: object
) -> None:
    source = _runtime_identity_input()
    source[path[0]][path[1]] = value  # type: ignore[index]
    content = {key: item for key, item in source.items() if key != "evidence_sha256"}
    source["evidence_sha256"] = sha256_json(content)
    source_path = tmp_path / "runtime-input.json"
    _write_json(source_path, source)

    with pytest.raises(ContractError, match="runtime identity"):
        freeze_runtime_identity_projection(
            source_path=source_path, output_path=tmp_path / "identity.json"
        )


def _accelerator_preflight_input() -> dict[str, object]:
    content = {
        "schema_version": 1,
        "contract": "accelerator-runtime-preflight-input-v1",
        "run_id": "feedback-r084",
        "installer": {
            "kind": "provider-managed-driver-installer",
            "source_sha256": "8" * 64,
        },
        "accelerator": {
            "vendor": "nvidia",
            "model": "T4",
            "device_uuid": "GPU-fixed-device-id",
            "driver_version": "535.274.02",
            "management_cli": "nvidia-smi",
        },
        "container_runtime": {
            "engine": "docker",
            "version": "27.5.1",
            "gpu_smoke_passed": True,
            "gpu_smoke_sha256": "9" * 64,
        },
        "reboot_completed": True,
        "network_calls_performed": True,
    }
    return {**content, "evidence_sha256": sha256_json(content)}


def test_accelerator_preflight_freezes_provider_neutral_ready_receipt(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "accelerator-input.json"
    output_path = tmp_path / "ACCELERATOR-PREFLIGHT.json"
    _write_json(source_path, _accelerator_preflight_input())

    report = freeze_accelerator_preflight_projection(
        source_path=source_path, output_path=output_path
    )
    replay = freeze_accelerator_preflight_projection(
        source_path=source_path, output_path=output_path
    )

    assert replay == report
    assert report["ready"] is True
    assert report["contract"] == "accelerator-runtime-preflight-v1"
    assert report["model_download_admitted"] is True
    assert report["projection_network_calls_performed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("installer", "source_sha256"), "missing"),
        (("accelerator", "driver_version"), ""),
        (("container_runtime", "gpu_smoke_passed"), False),
        (("reboot_completed", ""), False),
    ],
)
def test_accelerator_preflight_rejects_unready_runtime(
    tmp_path: Path, path: tuple[str, str], value: object
) -> None:
    source = _accelerator_preflight_input()
    if path[1]:
        source[path[0]][path[1]] = value  # type: ignore[index]
    else:
        source[path[0]] = value
    content = {key: item for key, item in source.items() if key != "evidence_sha256"}
    source["evidence_sha256"] = sha256_json(content)
    source_path = tmp_path / "accelerator-input.json"
    _write_json(source_path, source)

    with pytest.raises(ContractError, match="accelerator preflight"):
        freeze_accelerator_preflight_projection(
            source_path=source_path,
            output_path=tmp_path / "accelerator-preflight.json",
        )


def test_statistical_projection_binds_all_six_source_files(tmp_path: Path) -> None:
    sources = {
        "feedback": {
            "schema_version": 1,
            "status": "complete",
            "feedback_gain_count": 1,
            "native_evaluator_failure_count": 0,
        },
        "holdout": {
            "schema_version": 1,
            "evaluation_scope": "round1-full-capability",
            "status": "complete",
            "full_capability_gate_evaluated": True,
            "holdout_evaluable_pair_count": 0,
            "holdout_regression_count": 0,
            "native_evaluator_failure_count": 0,
        },
        "independent_safety": {
            "schema_version": 1,
            "contract": "independent-agent-safety-suite-v1",
            "suite_passed": False,
            "evaluator_failure_count": 0,
            "native_admission_reused": False,
        },
        "runtime_identity": {"schema_version": 1, "complete": False},
        "cost_receipt": {
            "schema_version": 1,
            "complete": True,
            "checkpoint_verified": True,
            "instance_state": "STOPPED",
            "stopped_mode": "STOP_CHARGING",
            "instance_retained": True,
            "api_termination_protection": True,
            "residual_hourly_cost_cny": 0.0,
        },
        "catalog_audit": {
            "schema_version": 1,
            "all_evidence_references_match": True,
        },
    }
    paths = {}
    for name, value in sources.items():
        digest_field = (
            "summary_sha256" if name in {"feedback", "holdout"} else "evidence_sha256"
        )
        value[digest_field] = sha256_json(value)
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path

    report = freeze_statistical_capability_projection(
        **{f"{name}_path": path for name, path in paths.items()},
        output_path=tmp_path / "CAPABILITY.json",
    )

    assert report["gate_passed"] is False
    assert set(report["failed_requirements"]) == {
        "minimum_evaluator_valid_holdout_pairs",
        "independent_safety_passed",
        "runtime_identity_complete",
    }
    assert set(report["source_files"]) == set(sources)
    assert all(len(row["file_sha256"]) == 64 for row in report["source_files"].values())


def test_statistical_projection_rejects_tampered_embedded_evidence(
    tmp_path: Path,
) -> None:
    sources = {
        "feedback": {
            "schema_version": 1,
            "status": "complete",
            "summary_sha256": "a" * 64,
        },
        "holdout": {"schema_version": 1, "summary_sha256": "b" * 64},
        "independent_safety": {"schema_version": 1, "evidence_sha256": "c" * 64},
        "runtime_identity": {"schema_version": 1, "evidence_sha256": "d" * 64},
        "cost_receipt": {"schema_version": 1, "evidence_sha256": "e" * 64},
        "catalog_audit": {"schema_version": 1, "evidence_sha256": "f" * 64},
    }
    paths = {}
    for name, value in sources.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        paths[name] = path

    report = freeze_statistical_capability_projection(
        **{f"{name}_path": path for name, path in paths.items()},
        output_path=tmp_path / "CAPABILITY.json",
    )

    assert report["gate_passed"] is False
    assert report["requirements"]["evidence_integrity_complete"] is False
