from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import LoopRevision
from skill_evolution_loop.mlx_student import MlxStructuredGenerator
from skill_evolution_loop.model_transport import (
    ChatGenerationRequest,
    ChatGenerationResponse,
    FileCachedModelTransport,
    OpenAICompatibleTransport,
    PromptGenerationRequest,
    TransportError,
)
from skill_evolution_loop.student_adapter import StudentTask


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_openai_transport_sends_generic_chat_request_and_freezes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(
            {
                "model": "same-base-cuda",
                "choices": [
                    {"message": {"content": "result"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        )

    monkeypatch.setenv("TEST_MODEL_TOKEN", "secret")
    transport = OpenAICompatibleTransport(
        base_url="http://127.0.0.1:8000",
        model="same-base-cuda",
        api_key_env="TEST_MODEL_TOKEN",
        timeout_seconds=7,
        urlopen=fake_urlopen,
    )
    request = ChatGenerationRequest.create(
        messages=(("system", "system"), ("user", "task")),
        max_tokens=64,
        temperature=0.0,
        seed=17,
    )

    response = transport.generate(request)

    sent = captured["request"]
    assert json.loads(sent.data)["seed"] == 17  # type: ignore[attr-defined]
    assert sent.headers["Authorization"] == "Bearer secret"  # type: ignore[attr-defined]
    assert response.text == "result"
    assert response.request_sha256 == request.fingerprint
    identity = transport.identity()
    assert identity["transport_implementation_sha256"] == (
        __import__("hashlib")
        .sha256(
            Path(
                __import__(
                    "skill_evolution_loop.model_transport", fromlist=["x"]
                ).__file__
            ).read_bytes()
        )
        .hexdigest()
    )
    assert "secret" not in json.dumps(identity)


def test_openai_transport_rejects_remote_plaintext_endpoint() -> None:
    with pytest.raises(TransportError, match="HTTPS"):
        OpenAICompatibleTransport(
            base_url="http://203.0.113.1:8000",
            model="model",
        )


def test_openai_transport_supports_rendered_prompt_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        captured["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        return _Response(
            {
                "model": "same-base-cuda",
                "choices": [{"text": "completion", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            }
        )

    transport = OpenAICompatibleTransport(
        base_url="http://localhost:8000",
        model="same-base-cuda",
        urlopen=fake_urlopen,
    )

    response = transport.generate_prompt(
        PromptGenerationRequest.create(prompt="rendered", max_tokens=32, seed=9)
    )

    assert captured["body"]["prompt"] == "rendered"  # type: ignore[index]
    assert response.text == "completion"


def test_file_cached_transport_reuses_prompt_generation_across_instances(
    tmp_path: Path,
) -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls = 0

        def generate_prompt(self, request):
            self.calls += 1
            return ChatGenerationResponse(
                text="cached completion",
                model="qwen-cuda",
                finish_reason="stop",
                usage={"prompt_tokens": 9, "completion_tokens": 3},
                request_sha256=request.fingerprint,
                response_sha256="a" * 64,
            )

        def generate(self, request):
            raise AssertionError("chat path not used")

        def identity(self):
            return {
                "kind": "fixture",
                "model": "qwen-cuda",
                "transport_implementation_sha256": "b" * 64,
            }

    delegate = Delegate()
    request = PromptGenerationRequest.create(
        prompt="stable rendered prompt", max_tokens=256, seed=0
    )

    first = FileCachedModelTransport(delegate=delegate, cache_root=tmp_path / "cache")
    second = FileCachedModelTransport(delegate=delegate, cache_root=tmp_path / "cache")

    assert first.generate_prompt(request).text == "cached completion"
    assert second.generate_prompt(request).text == "cached completion"
    assert delegate.calls == 1
    assert first.cache_stats() == {"hits": 0, "misses": 1, "writes": 1}
    assert second.cache_stats() == {"hits": 1, "misses": 0, "writes": 0}
    assert second.aggregate_metrics() == {
        "cache_entries": 1,
        "remote_calls": 1,
        "prompt_tokens": 9,
        "completion_tokens": 3,
        "total_tokens": 12,
        "current_process_cache_hits": 1,
        "current_process_cache_misses": 0,
        "current_process_cache_writes": 0,
    }
    cache_files = list((tmp_path / "cache/prompt").glob("*.json"))
    assert len(cache_files) == 1
    frozen = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert frozen["request_sha256"] == request.fingerprint
    assert frozen["transport_identity_sha256"]


def test_cached_transport_aggregate_metrics_fail_closed_on_invalid_usage(
    tmp_path: Path,
) -> None:
    class Delegate:
        def identity(self):
            return {
                "kind": "fixture",
                "transport_implementation_sha256": "b" * 64,
            }

    cache = tmp_path / "cache/prompt"
    cache.mkdir(parents=True)
    (cache / "invalid.json").write_text(
        json.dumps({"response": {"usage": {"prompt_tokens": "9"}}}),
        encoding="utf-8",
    )
    transport = FileCachedModelTransport(
        delegate=Delegate(), cache_root=tmp_path / "cache"
    )

    with pytest.raises(TransportError, match="aggregate metrics"):
        transport.aggregate_metrics()


def test_structured_generator_can_offload_inference_without_loading_mlx(
    tmp_path,
) -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return "\n".join(item["content"] for item in messages)

    class Transport:
        def __init__(self) -> None:
            self.requests = []

        def generate_prompt(self, request):
            self.requests.append(request)
            return type(
                "Response",
                (),
                {
                    "text": '{"file":"a.py","search":"x = 1","replace":"x = 2","diagnostic":"ok"}'
                },
            )()

        def identity(self):
            return {"kind": "fixture", "transport_implementation_sha256": "c" * 64}

    target = tmp_path / "a.py"
    target.write_text("x = 1\n")
    task = StudentTask(
        task_id="task-1",
        checkout=tmp_path,
        instruction="change x",
        allowed_targets=("a.py",),
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="skill-1",
        revision_id="revision-1",
        parent_revision_id=None,
        source_round=1,
        protocol="structured-edit",
        skill_text="Use exact spans.",
        prompt_template="Return JSON.",
        eval_note="fixture",
    )
    transport = Transport()
    generator = MlxStructuredGenerator(
        model_path=str(tmp_path),
        model_transport=transport,
        tokenizer_loader=lambda _path: Tokenizer(),
    )

    assert "x = 2" in generator(task, revision)
    assert transport.requests[0].prompt == generator.generation_prompt_trace()[0]
    assert generator.generation_config()["execution_mode"] == "remote-transport"
