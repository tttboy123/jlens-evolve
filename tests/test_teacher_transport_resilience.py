from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evolve.autonomous.config import AutonomousEvolutionConfig
from evolve.contracts import ContractViolation
from evolve.teachers import (
    DeepSeekCompatibleTeacherTransport,
    OpenAICompatibleTeacherTransport,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _raw() -> dict[str, object]:
    return {
        "model": "deepseek-chat",
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


def _http_error(code: int) -> Exception:
    return __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
        "https://teacher.invalid", code, "error", {}, None
    )


def test_transport_retries_transient_http_then_succeeds() -> None:
    calls: list[int] = []
    raw = _raw()

    def opener(request: Any, timeout: float) -> _Response:
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(502)
        return _Response(raw)

    transport = OpenAICompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="teacher-model",
        api_key="secret",
        max_retries=3,
        retry_base_delay=0.0,
        opener=opener,
    )
    assert transport({"request_id": "r", "max_output_tokens": 10}) == raw
    assert len(calls) == 3


def test_transport_retries_urlerror_then_succeeds() -> None:
    calls: list[int] = []
    raw = _raw()

    def opener(request: Any, timeout: float) -> _Response:
        calls.append(1)
        if len(calls) == 1:
            raise __import__("urllib.error", fromlist=["URLError"]).URLError(
                "connection reset"
            )
        return _Response(raw)

    transport = OpenAICompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="teacher-model",
        api_key="secret",
        max_retries=3,
        retry_base_delay=0.0,
        opener=opener,
    )
    assert transport({"request_id": "r", "max_output_tokens": 10}) == raw
    assert len(calls) == 2


def test_transport_does_not_retry_non_retryable_http() -> None:
    calls: list[int] = []

    def opener(request: Any, timeout: float) -> _Response:
        calls.append(1)
        raise _http_error(404)

    transport = OpenAICompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="teacher-model",
        api_key="secret",
        max_retries=5,
        retry_base_delay=0.0,
        opener=opener,
    )
    with pytest.raises(ContractViolation) as excinfo:
        transport({"request_id": "r", "max_output_tokens": 10})
    assert len(calls) == 1
    assert isinstance(excinfo.value.__cause__, __import__("urllib.error", fromlist=["HTTPError"]).HTTPError)
    assert excinfo.value.__cause__.code == 404


def test_transport_exhausts_retries_and_preserves_cause() -> None:
    calls: list[int] = []

    def opener(request: Any, timeout: float) -> _Response:
        calls.append(1)
        raise _http_error(503)

    transport = OpenAICompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="teacher-model",
        api_key="secret",
        max_retries=2,
        retry_base_delay=0.0,
        opener=opener,
    )
    with pytest.raises(ContractViolation) as excinfo:
        transport({"request_id": "r", "max_output_tokens": 10})
    assert len(calls) == 3  # initial + 2 retries
    assert excinfo.value.__cause__ is not None
    assert excinfo.value.__cause__.code == 503


def test_transport_body_respects_request_temperature() -> None:
    captured: list[Any] = []

    def opener(request: Any, timeout: float) -> _Response:
        captured.append(json.loads(request.data))
        return _Response(_raw())

    transport = OpenAICompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="teacher-model",
        api_key="secret",
        opener=opener,
    )
    transport({"request_id": "r", "max_output_tokens": 10, "temperature": 0.7})
    assert captured[0]["temperature"] == 0.7
    transport({"request_id": "r2", "max_output_tokens": 10})
    assert captured[1]["temperature"] == 0.0


def test_deepseek_body_keeps_thinking_disabled_with_temperature() -> None:
    captured: list[Any] = []

    def opener(request: Any, timeout: float) -> _Response:
        captured.append(json.loads(request.data))
        return _Response(_raw())

    transport = DeepSeekCompatibleTeacherTransport(
        endpoint="https://teacher.invalid/chat/completions",
        model="deepseek-chat",
        api_key="secret",
        opener=opener,
    )
    transport({"request_id": "r", "max_output_tokens": 10, "temperature": 1.0})
    assert captured[0]["temperature"] == 1.0
    assert captured[0]["thinking"] == {"type": "disabled"}


def _write_config(root: Path, teacher: dict[str, Any]) -> Path:
    model_dir = root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (root / "task_pool.json").write_text("[]", encoding="utf-8")
    (root / "source_pool").mkdir()
    (root / "harness").mkdir()
    (root / "evaluator.py").write_text("", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "goal": {
            "goal_id": "g",
            "description": "d",
            "target_native_gains": 1,
            "max_rounds": 3,
            "no_progress_patience": 2,
        },
        "model": {
            "provider": "local-mlx",
            "model_path": str(model_dir),
            "model_identity_files": ["config.json"],
        },
        "swe_bench": {
            "task_pool": str(root / "task_pool.json"),
            "source_pool": str(root / "source_pool"),
            "official_harness": str(root / "harness"),
            "official_evaluator": str(root / "evaluator.py"),
            "cohort": "feedback",
        },
        "teacher": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "endpoint": "https://api.deepseek.com/chat/completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "budget_cny": 20.0,
            "max_output_tokens": 512,
            **teacher,
        },
        "execution": {
            "tasks_per_campaign": 3,
            "qwen_prescreen_count": 0,
            "native_finalist_count": 1,
            "seed": 0,
        },
    }
    path = root / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_teacher_resilience_defaults(tmp_path: Path) -> None:
    config = AutonomousEvolutionConfig.load(_write_config(tmp_path, {}))
    assert config.teacher.timeout_seconds == 60.0
    assert config.teacher.max_retries == 3
    assert config.teacher.retry_base_delay == 1.0
    assert config.teacher.temperature == 0.0


def test_config_teacher_resilience_overrides(tmp_path: Path) -> None:
    config = AutonomousEvolutionConfig.load(
        _write_config(
            tmp_path,
            {
                "timeout_seconds": 120,
                "max_retries": 4,
                "retry_base_delay": 0.5,
                "temperature": 0.7,
            },
        )
    )
    assert config.teacher.timeout_seconds == 120.0
    assert config.teacher.max_retries == 4
    assert config.teacher.retry_base_delay == 0.5
    assert config.teacher.temperature == 0.7


def test_config_temperature_schedule(tmp_path: Path) -> None:
    config = AutonomousEvolutionConfig.load(
        _write_config(
            tmp_path,
            {"temperature_schedule": [0.0, 0.7, 1.1]},
        )
    )
    assert config.teacher.temperature_schedule == (0.0, 0.7, 1.1)


def test_config_temperature_schedule_defaults(tmp_path: Path) -> None:
    config = AutonomousEvolutionConfig.load(_write_config(tmp_path, {}))
    assert config.teacher.temperature_schedule == (0.0,)


def test_config_rejects_bad_temperature_schedule(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        AutonomousEvolutionConfig.load(
            _write_config(tmp_path, {"temperature_schedule": [0.0, 3.5]})
        )


def test_round_temperature_rotates_through_schedule() -> None:
    from evolve.autonomous.runner import _round_temperature

    schedule = (0.0, 0.7, 1.1)
    assert _round_temperature(schedule, 0) == 0.0
    assert _round_temperature(schedule, 1) == 0.7
    assert _round_temperature(schedule, 2) == 1.1
    assert _round_temperature(schedule, 3) == 0.0
    assert _round_temperature(schedule, 7) == 0.7
    assert _round_temperature((), 4) == 0.0
