from __future__ import annotations

from skill_evolution_loop.accelerator_bootstrap import decide_accelerator_bootstrap


def test_bootstrap_waits_inside_grace_period_when_runtime_is_missing() -> None:
    result = decide_accelerator_bootstrap(
        elapsed_seconds=190,
        timeout_seconds=600,
        management_cli_available=False,
        container_runtime_available=False,
        gpu_smoke_passed=False,
    )

    assert result["decision"] == "wait"
    assert result["paid_model_work_admitted"] is False
    assert result["remaining_seconds"] == 410


def test_bootstrap_verifies_after_driver_and_container_appear() -> None:
    result = decide_accelerator_bootstrap(
        elapsed_seconds=240,
        timeout_seconds=600,
        management_cli_available=True,
        container_runtime_available=True,
        gpu_smoke_passed=False,
    )

    assert result["decision"] == "verify-gpu-smoke"
    assert result["paid_model_work_admitted"] is False


def test_bootstrap_admits_only_after_gpu_smoke() -> None:
    result = decide_accelerator_bootstrap(
        elapsed_seconds=270,
        timeout_seconds=600,
        management_cli_available=True,
        container_runtime_available=True,
        gpu_smoke_passed=True,
    )

    assert result["decision"] == "admit"
    assert result["paid_model_work_admitted"] is True


def test_bootstrap_stops_worker_after_timeout_without_terminating_it() -> None:
    result = decide_accelerator_bootstrap(
        elapsed_seconds=600,
        timeout_seconds=600,
        management_cli_available=False,
        container_runtime_available=False,
        gpu_smoke_passed=False,
    )

    assert result["decision"] == "stop-worker"
    assert result["paid_model_work_admitted"] is False
    assert result["remaining_seconds"] == 0
