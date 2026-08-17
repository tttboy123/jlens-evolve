"""Tests for the v2.5 multi-model teaching API capability (T3)."""

from __future__ import annotations

from typing import Any

import pytest

from teacher_api import (
    TeacherCallError,
    TeacherClient,
    TeacherConfig,
    TeacherProvider,
    TeacherResponse,
    TeacherSample,
    combine_weighted,
    weighted_messages,
)


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload


def _fake_transport(
    payload: dict[str, Any], status_code: int = 200, capture: list | None = None
):
    def transport(url, headers=None, json=None, timeout=None):
        if capture is not None:
            capture.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(payload, status_code)

    return transport


def _client(provider: TeacherProvider, transport) -> TeacherClient:
    config = TeacherConfig(
        provider=provider,
        api_base=f"https://example.{provider.value}.invalid/v1",
        model=f"model-{provider.value}",
        api_key_env="EVOLVE_TEST_TEACHER_KEY",
    )
    return TeacherClient(config, api_key="test-key", transport=transport)


def test_openai_compatible_parse_and_no_key_guard():
    capture: list = []
    client = _client(
        TeacherProvider.DEEPSEEK,
        _fake_transport(
            {
                "choices": [{"message": {"content": "teaching text"}}],
                "model": "deepseek-chat",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            capture=capture,
        ),
    )
    sample = TeacherSample(messages=[{"role": "user", "content": "hi"}], weight=2.0)
    resp = client.complete(sample)
    assert isinstance(resp, TeacherResponse)
    assert resp.text == "teaching text"
    assert resp.weight == 2.0
    assert resp.model == "deepseek-chat"
    assert resp.usage["prompt_tokens"] == 10
    assert capture[0]["url"].endswith("/chat/completions")
    assert capture[0]["headers"]["Authorization"] == "Bearer test-key"
    assert capture[0]["json"]["max_tokens"] == 2000

    # No key => capability guard, no call made.
    unconfigured = TeacherClient(
        TeacherConfig(
            provider=TeacherProvider.OPENAI,
            api_base="https://example.invalid/v1",
            model="gpt-4o",
            api_key_env="EVOLVE_TEST_MISSING_KEY",
        )
    )
    with pytest.raises(TeacherCallError, match="not configured"):
        unconfigured.complete(sample)


def test_openai_adapter():
    client = _client(
        TeacherProvider.OPENAI,
        _fake_transport(
            {"choices": [{"message": {"content": "gpt text"}}], "model": "gpt-4o"},
        ),
    )
    resp = client.complete(TeacherSample(messages=[{"role": "user", "content": "q"}]))
    assert resp.text == "gpt text"
    assert resp.provider is TeacherProvider.OPENAI


def test_claude_native_adapter():
    capture: list = []
    client = _client(
        TeacherProvider.CLAUDE,
        _fake_transport(
            {
                "content": [{"type": "text", "text": "claude text"}],
                "model": "claude-sonnet-4-5",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
            capture=capture,
        ),
    )
    resp = client.complete(TeacherSample(messages=[{"role": "user", "content": "q"}]))
    assert resp.text == "claude text"
    assert resp.usage["input_tokens"] == 7
    assert capture[0]["url"].endswith("/messages")
    assert capture[0]["headers"]["x-api-key"] == "test-key"


def test_http_error_surfaces():
    client = _client(
        TeacherProvider.DEEPSEEK,
        _fake_transport({"error": "boom"}, status_code=429),
    )
    with pytest.raises(TeacherCallError, match="HTTP 429"):
        client.complete(TeacherSample(messages=[{"role": "user", "content": "q"}]))


def test_output_token_limit_is_forwarded_and_bounded():
    capture: list = []
    client = _client(
        TeacherProvider.DEEPSEEK,
        _fake_transport({"choices": [{"message": {"content": "ok"}}]}, capture=capture),
    )

    client.complete(
        TeacherSample(
            messages=[{"role": "user", "content": "q"}], max_output_tokens=321
        )
    )
    assert capture[0]["json"]["max_tokens"] == 321
    with pytest.raises(TeacherCallError, match="max_output_tokens"):
        client.complete(
            TeacherSample(
                messages=[{"role": "user", "content": "q"}], max_output_tokens=0
            )
        )


def test_deepseek_v4_flash_supports_large_json_thinking_calls():
    capture: list = []
    client = TeacherClient(
        TeacherConfig(
            provider=TeacherProvider.DEEPSEEK,
            api_base="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key_env="EVOLVE_TEST_TEACHER_KEY",
        ),
        api_key="test-key",
        transport=_fake_transport(
            {
                "choices": [{"message": {"content": "{}"}}],
                "model": "deepseek-v4-flash",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            capture=capture,
        ),
    )

    client.complete(
        TeacherSample(
            messages=[{"role": "user", "content": "Return JSON."}],
            max_output_tokens=384_000,
            response_format={"type": "json_object"},
            thinking=True,
            reasoning_effort="high",
        )
    )

    payload = capture[0]["json"]
    assert payload["max_tokens"] == 384_000
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"


def test_non_v4_teacher_keeps_legacy_output_ceiling():
    client = _client(
        TeacherProvider.OPENAI,
        _fake_transport({"choices": [{"message": {"content": "ok"}}]}),
    )

    with pytest.raises(TeacherCallError, match="max_output_tokens"):
        client.complete(
            TeacherSample(
                messages=[{"role": "user", "content": "q"}],
                max_output_tokens=4_097,
            )
        )


def test_deepseek_standard_environment_key_is_an_alias(monkeypatch):
    monkeypatch.delenv("EVOLVE_TEACHER_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "alias-key")

    client = TeacherClient.from_env(TeacherProvider.DEEPSEEK)

    assert client.api_key == "alias-key"
    assert client.config.model == "deepseek-v4-flash"


def test_weighted_messages_and_combine():
    parts = [
        weighted_messages([{"role": "system", "content": "sys"}]),
        weighted_messages([{"role": "user", "content": "task"}]),
    ]
    assert [m["role"] for m in parts[0]] == ["system"]

    responses = [
        TeacherResponse(
            provider=TeacherProvider.DEEPSEEK, model="ds", text="a", weight=1.0
        ),
        TeacherResponse(
            provider=TeacherProvider.OPENAI, model="gpt", text="b", weight=3.0
        ),
    ]
    out = combine_weighted(responses)
    assert "[deepseek:ds:w=1]" in out
    assert "[openai:gpt:w=3]" in out
    assert "a" in out and "b" in out


def test_no_real_network_with_fake_transport():
    # The fake transport proves the client path never hits the network in tests.
    client = _client(
        TeacherProvider.OPENAI,
        _fake_transport(
            {"choices": [{"message": {"content": "x"}}], "model": "gpt-4o"}
        ),
    )
    resp = client.complete(TeacherSample(messages=[{"role": "user", "content": "q"}]))
    assert resp.text == "x"
