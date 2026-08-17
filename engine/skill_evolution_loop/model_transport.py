"""Generic model transport boundary for local and remote execution planes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

from .contracts import canonical_json, sha256_json


class TransportError(RuntimeError):
    """Fail-closed transport or protocol error."""


@dataclass(frozen=True)
class ChatGenerationRequest:
    messages: tuple[tuple[str, str], ...]
    max_tokens: int
    temperature: float
    seed: int
    stop: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        messages: tuple[tuple[str, str], ...],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        stop: tuple[str, ...] = (),
    ) -> ChatGenerationRequest:
        if not messages or any(
            role not in {"system", "user", "assistant"} or not content
            for role, content in messages
        ):
            raise TransportError("messages are invalid")
        if max_tokens < 1 or temperature < 0 or type(seed) is not int:
            raise TransportError("generation parameters are invalid")
        return cls(messages, max_tokens, temperature, seed, stop)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": role, "content": content} for role, content in self.messages
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "stop": list(self.stop),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class PromptGenerationRequest:
    prompt: str
    max_tokens: int
    temperature: float
    seed: int
    stop: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        stop: tuple[str, ...] = (),
    ) -> PromptGenerationRequest:
        if not prompt or max_tokens < 1 or temperature < 0 or type(seed) is not int:
            raise TransportError("prompt generation parameters are invalid")
        return cls(prompt, max_tokens, temperature, seed, stop)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "stop": list(self.stop),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class ChatGenerationResponse:
    text: str
    model: str
    finish_reason: str
    usage: dict[str, int]
    request_sha256: str
    response_sha256: str


class ModelTransport(Protocol):
    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResponse: ...

    def generate_prompt(
        self, request: PromptGenerationRequest
    ) -> ChatGenerationResponse: ...

    def identity(self) -> dict[str, Any]: ...


class FileCachedModelTransport:
    """Append-only, identity-bound cache for deterministic model requests."""

    def __init__(self, *, delegate: ModelTransport, cache_root: Path) -> None:
        self.delegate = delegate
        self.cache_root = cache_root.resolve()
        self._hits = 0
        self._misses = 0
        self._writes = 0

    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResponse:
        return self._cached("chat", request, self.delegate.generate)

    def generate_prompt(
        self, request: PromptGenerationRequest
    ) -> ChatGenerationResponse:
        return self._cached("prompt", request, self.delegate.generate_prompt)

    def _cached(
        self,
        kind: str,
        request: ChatGenerationRequest | PromptGenerationRequest,
        generate: Callable[[Any], ChatGenerationResponse],
    ) -> ChatGenerationResponse:
        identity_sha = sha256_json(self.delegate.identity())
        cache_key = sha256_json(
            {
                "schema_version": 1,
                "kind": kind,
                "request_sha256": request.fingerprint,
                "transport_identity_sha256": identity_sha,
            }
        )
        target_dir = self.cache_root / kind
        target = target_dir / f"{cache_key}.json"
        if target.is_file():
            try:
                frozen = json.loads(target.read_text(encoding="utf-8"))
                response = frozen["response"]
                if (
                    frozen["cache_key"] != cache_key
                    or frozen["request_sha256"] != request.fingerprint
                    or frozen["transport_identity_sha256"] != identity_sha
                    or response["request_sha256"] != request.fingerprint
                ):
                    raise TransportError("cached model response identity mismatch")
                loaded = ChatGenerationResponse(**response)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
            ) as exc:
                raise TransportError("cached model response is invalid") from exc
            self._hits += 1
            return loaded
        self._misses += 1
        response = generate(request)
        content = {
            "schema_version": 1,
            "cache_key": cache_key,
            "kind": kind,
            "request_sha256": request.fingerprint,
            "transport_identity_sha256": identity_sha,
            "response": {
                "text": response.text,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
                "request_sha256": response.request_sha256,
                "response_sha256": response.response_sha256,
            },
        }
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_dir,
            prefix=f".{cache_key}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(content) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            try:
                os.link(temporary, target)
                self._writes += 1
            except FileExistsError:
                existing = target.read_text(encoding="utf-8")
                if existing != canonical_json(content) + "\n":
                    raise TransportError("model cache collision")
        finally:
            temporary.unlink(missing_ok=True)
        return response

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "file-cached-model-transport",
            "delegate": self.delegate.identity(),
            "cache_schema": "identity-request-v1",
        }

    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "writes": self._writes}

    def aggregate_metrics(self) -> dict[str, int]:
        """Rebuild remote-call and token totals from the persistent cache."""

        totals = {
            "cache_entries": 0,
            "remote_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        try:
            cache_files = sorted(self.cache_root.glob("*/*.json"))
            for path in cache_files:
                frozen = json.loads(path.read_text(encoding="utf-8"))
                if frozen.get("schema_version") != 1:
                    raise TransportError(
                        "cached transport aggregate metrics are invalid"
                    )
                usage = frozen["response"]["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get(
                    "total_tokens", prompt_tokens + completion_tokens
                )
                if any(
                    type(value) is not int or value < 0
                    for value in (prompt_tokens, completion_tokens, total_tokens)
                ):
                    raise TransportError(
                        "cached transport aggregate metrics are invalid"
                    )
                totals["cache_entries"] += 1
                totals["remote_calls"] += 1
                totals["prompt_tokens"] += prompt_tokens
                totals["completion_tokens"] += completion_tokens
                totals["total_tokens"] += total_tokens
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            raise TransportError(
                "cached transport aggregate metrics are invalid"
            ) from exc
        return {
            **totals,
            "current_process_cache_hits": self._hits,
            "current_process_cache_misses": self._misses,
            "current_process_cache_writes": self._writes,
        }


UrlOpen = Callable[..., Any]


class OpenAICompatibleTransport:
    """Small dependency-free adapter; cloud/provider details stay outside core."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 120.0,
        urlopen: UrlOpen = urllib.request.urlopen,
    ) -> None:
        parsed = urlparse(base_url)
        local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme not in ({"http", "https"} if local else {"https"}):
            raise TransportError("remote model endpoint must use HTTPS")
        if not parsed.hostname or not model.strip() or timeout_seconds <= 0:
            raise TransportError("transport configuration is invalid")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self._urlopen = urlopen

    def generate(self, request: ChatGenerationRequest) -> ChatGenerationResponse:
        return self._request(
            path="chat/completions",
            body={"model": self.model, **request.to_dict()},
            request_sha256=request.fingerprint,
            content_key="message",
        )

    def generate_prompt(
        self, request: PromptGenerationRequest
    ) -> ChatGenerationResponse:
        return self._request(
            path="completions",
            body={"model": self.model, **request.to_dict()},
            request_sha256=request.fingerprint,
            content_key="text",
        )

    def _request(
        self,
        *,
        path: str,
        body: dict[str, Any],
        request_sha256: str,
        content_key: str,
    ) -> ChatGenerationResponse:
        if not body.get("stop"):
            body.pop("stop")
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            token = os.environ.get(self.api_key_env)
            if not token:
                raise TransportError(f"missing API credential env: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {token}"
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/{path}",
            data=canonical_json(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with self._urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise TransportError("model endpoint request failed") from exc
        try:
            choice = payload["choices"][0]
            text = (
                choice["message"]["content"]
                if content_key == "message"
                else choice[content_key]
            )
            finish_reason = choice["finish_reason"]
            model = payload.get("model", self.model)
            usage = payload.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError("model endpoint response is invalid") from exc
        if not isinstance(text, str) or not isinstance(usage, dict):
            raise TransportError("model endpoint response is invalid")
        response_content = {
            "text": text,
            "model": str(model),
            "finish_reason": str(finish_reason),
            "usage": usage,
            "request_sha256": request_sha256,
        }
        return ChatGenerationResponse(
            **response_content,
            response_sha256=sha256_json(response_content),
        )

    def identity(self) -> dict[str, Any]:
        implementation = Path(__file__).read_bytes()
        return {
            "schema_version": 1,
            "kind": "openai-compatible-chat",
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "transport_implementation_sha256": hashlib.sha256(
                implementation
            ).hexdigest(),
        }
