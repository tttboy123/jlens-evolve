"""Provider-neutral admission policy for asynchronous accelerator bootstrap."""

from __future__ import annotations

from typing import Any

from .contracts import ContractError, sha256_json


def decide_accelerator_bootstrap(
    *,
    elapsed_seconds: int,
    timeout_seconds: int,
    management_cli_available: bool,
    container_runtime_available: bool,
    gpu_smoke_passed: bool,
) -> dict[str, Any]:
    """Choose WAIT, VERIFY, ADMIT, or STOP-WORKER without provider coupling."""

    if (
        type(elapsed_seconds) is not int
        or elapsed_seconds < 0
        or type(timeout_seconds) is not int
        or timeout_seconds < 1
    ):
        raise ContractError("accelerator bootstrap timing is invalid")
    if any(
        type(value) is not bool
        for value in (
            management_cli_available,
            container_runtime_available,
            gpu_smoke_passed,
        )
    ):
        raise ContractError("accelerator bootstrap observation is invalid")
    if gpu_smoke_passed and not (
        management_cli_available and container_runtime_available
    ):
        raise ContractError("GPU smoke cannot pass without runtime prerequisites")

    ready_for_smoke = management_cli_available and container_runtime_available
    if gpu_smoke_passed:
        decision = "admit"
    elif ready_for_smoke:
        decision = "verify-gpu-smoke"
    elif elapsed_seconds >= timeout_seconds:
        decision = "stop-worker"
    else:
        decision = "wait"
    content = {
        "schema_version": 1,
        "contract": "accelerator-bootstrap-admission-decision-v1",
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout_seconds,
        "remaining_seconds": max(0, timeout_seconds - elapsed_seconds),
        "management_cli_available": management_cli_available,
        "container_runtime_available": container_runtime_available,
        "gpu_smoke_passed": gpu_smoke_passed,
        "decision": decision,
        "paid_model_work_admitted": decision == "admit",
        "network_calls_performed": False,
        "provider_special_case": False,
    }
    return {**content, "evidence_sha256": sha256_json(content)}
