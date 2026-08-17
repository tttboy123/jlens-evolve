from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _relay_module():
    path = Path(__file__).parents[1] / "deploy/tencent/tat-openai-relay.py"
    spec = importlib.util.spec_from_file_location("tat_openai_relay", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tat_relay_compresses_large_request_and_response(monkeypatch) -> None:
    module = _relay_module()
    relay = module.TatRelay(
        instance_id="ins-test",
        region="ap-test",
        remote_base_url="http://127.0.0.1:8000",
        timeout_seconds=10,
        poll_seconds=0,
    )
    response = json.dumps(
        {"choices": [{"text": "x" * 20_000, "finish_reason": "stop"}]}
    ).encode()
    task_output = module.encode_tat_payload(response)
    invocations = iter(
        [
            {"InvocationId": "inv-test"},
            {
                "InvocationTaskSet": [
                    {
                        "TaskStatus": "SUCCESS",
                        "TaskResult": {"Output": task_output},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(relay, "_tccli", lambda *args: next(invocations))
    body = json.dumps({"model": "m", "prompt": "repeat " * 5_000}).encode()

    assert relay.forward("/v1/completions", body) == response
    assert len(relay.last_tat_content) < module.MAX_TAT_CONTENT


def test_tat_relay_rejects_uncompressed_command_over_limit(monkeypatch) -> None:
    module = _relay_module()
    relay = module.TatRelay(
        instance_id="ins-test",
        region="ap-test",
        remote_base_url="http://127.0.0.1:8000",
        timeout_seconds=10,
    )
    monkeypatch.setattr(module, "MAX_TAT_CONTENT", 8)

    try:
        relay.forward("/v1/completions", b'{"prompt":"x"}')
    except module.RelayError as exc:
        assert "TAT command limit" in str(exc)
    else:
        raise AssertionError("expected command limit rejection")


def test_tat_relay_wraps_invalid_remote_envelope(monkeypatch) -> None:
    module = _relay_module()
    relay = module.TatRelay(
        instance_id="ins-test",
        region="ap-test",
        remote_base_url="http://127.0.0.1:8000",
        timeout_seconds=10,
        poll_seconds=0,
    )
    invocations = iter(
        [
            {"InvocationId": "inv-test"},
            {
                "InvocationTaskSet": [
                    {
                        "TaskStatus": "SUCCESS",
                        "TaskResult": {"Output": "Y3VybDogKDIyKSByZW1vdGUgZXJyb3I="},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(relay, "_tccli", lambda *args: next(invocations))

    try:
        relay.forward("/v1/completions", b'{"prompt":"x"}')
    except module.RelayError as exc:
        assert str(exc) == "TAT response envelope is invalid"
    else:
        raise AssertionError("expected invalid envelope rejection")


def test_tat_relay_maps_cloud_gateway_calls(monkeypatch) -> None:
    module = _relay_module()
    relay = module.TatRelay(
        instance_id="ins-test",
        region="ap-test",
        remote_base_url="http://127.0.0.1:8000",
        timeout_seconds=10,
    )
    calls = []

    class FakeMcp:
        def call(self, *, tool, arguments):
            calls.append((tool, arguments))
            if tool == "tencent_api_mutate":
                return {"InvocationId": "inv-test"}
            return {"InvocationTaskSet": []}

    relay._mcp = FakeMcp()

    assert relay._run_command("encoded") == {"InvocationId": "inv-test"}
    assert relay._describe_invocation_tasks("inv-test") == {"InvocationTaskSet": []}
    assert calls[0][0] == "tencent_api_mutate"
    assert calls[0][1]["force"] is True
    assert calls[0][1]["body"]["InstanceIds"] == ["ins-test"]
    assert calls[1][0] == "tencent_api_read"
    assert calls[1][1]["body"]["HideOutput"] is False
