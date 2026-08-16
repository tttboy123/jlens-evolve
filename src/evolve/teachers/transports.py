"""Network and frozen-replay transports for Teacher proposals."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from evolve.contracts import ContractViolation, canonical_json


@runtime_checkable
class TeacherTransport(Protocol):
    """A single Teacher inference boundary.

    The protocol intentionally matches a plain callable so existing injected
    test functions and v1 callers remain valid implementations.
    """

    def __call__(self, request: dict[str, object]) -> dict[str, object]: ...


_CANDIDATE_SCHEMA = {
    "protocol": "execution protocol identifier",
    "prompt_template": "complete prompt template for the Student model",
    "skill_text": "complete evolved Skill, not a diff",
    "operator": "complete executable Operator object",
    "router": "complete task-to-Operator Router object",
    "memory_policy": "optional Memory Policy object or null",
    "preconditions": "conditions required before this candidate may execute",
    "expected_external_effect": "observable task or environment outcome",
    "expected_internal_effect": "observable harness decision or state outcome",
    "falsification": "an explicit condition that rejects the candidate",
    "eval_note": "native evaluation procedure",
}

_FIELD_CONTRACTS: dict[str, object] = {
    **_CANDIDATE_SCHEMA,
    "operator": {
        "id": "non-empty operator id",
        "kind": "zero-arg",
        "arguments": [],
        "instruction": "non-empty executable instruction",
    },
    "router": {
        "routes": {"<selected task instance_id>": "<operator id>"}
    },
    "memory_policy": "null or a non-empty JSON object",
    "preconditions": ["non-empty execution condition"],
    "expected_external_effect": "non-empty string or JSON object",
    "expected_internal_effect": "non-empty string or JSON object",
    "falsification": "non-empty rejection condition string or JSON object",
}


_RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def _is_retryable(error: Exception) -> bool:
    """Transient network/gateway failures are retried; contract bugs are not."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _RETRYABLE_HTTP_CODES
    if isinstance(error, urllib.error.URLError):
        return True
    if isinstance(error, OSError):
        return True
    if isinstance(error, json.JSONDecodeError):
        # A gateway can answer 200 with an HTML error page; retry once is cheap.
        return True
    return False


class OpenAICompatibleTeacherTransport:
    """Send one deterministic JSON-mode request to a compatible endpoint."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        base_url: str | None = None,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_seconds: float = 60,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if endpoint is not None and base_url is not None and endpoint != base_url:
            raise ContractViolation("Teacher endpoint identity is ambiguous")
        selected_endpoint = endpoint or base_url
        if not isinstance(selected_endpoint, str) or not selected_endpoint.strip():
            raise ContractViolation("Teacher endpoint must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ContractViolation("Teacher model must be non-empty")
        if api_key is None and api_key_env is None:
            raise ContractViolation("Teacher API key or environment name is required")
        if timeout_seconds <= 0:
            raise ContractViolation("Teacher timeout must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ContractViolation("Teacher max_retries must be a non-negative integer")
        if isinstance(retry_base_delay, bool) or not isinstance(retry_base_delay, (int, float)) or retry_base_delay < 0:
            raise ContractViolation("Teacher retry_base_delay must be non-negative")
        self.endpoint = selected_endpoint
        self.model = model
        self._api_key = api_key
        self._api_key_env = api_key_env
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_base_delay = float(retry_base_delay)
        self._opener = opener

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        if not isinstance(request, dict):
            raise ContractViolation("Teacher transport request must be an object")
        key = self._api_key
        if key is None and self._api_key_env is not None:
            key = os.environ.get(self._api_key_env)
        if not key:
            raise ContractViolation("Teacher API key is missing")
        body = self._body(request)
        http_request = urllib.request.Request(
            self.endpoint,
            data=canonical_json(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(
                    http_request, timeout=self.timeout_seconds
                ) as response:
                    raw = json.loads(response.read())
                break
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                urllib.error.URLError,
            ) as error:
                last_error = error
                if not _is_retryable(error) or attempt >= self.max_retries:
                    raise ContractViolation(
                        "Teacher transport request failed"
                    ) from error
                delay = self.retry_base_delay * (2**attempt) + random.uniform(
                    0.0, 0.5
                )
                time.sleep(delay)
        else:  # pragma: no cover - defensive; the loop raises on final failure
            assert last_error is not None
            raise ContractViolation("Teacher transport request failed") from last_error
        if not isinstance(raw, dict):
            raise ContractViolation("Teacher transport response must be an object")
        return raw

    def _body(self, request: Mapping[str, object]) -> dict[str, object]:
        max_tokens = request.get("max_output_tokens", 4096)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ContractViolation("Teacher max_output_tokens must be an integer")
        temperature = request.get("temperature", 0.0)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not (0.0 <= float(temperature) <= 2.0)
        ):
            raise ContractViolation("Teacher temperature must be a number in [0, 2]")
        system = canonical_json(
            {
                "purpose": "evolve an external Skill/Harness for a frozen model",
                "output_contract": {
                    "return_direct_object": True,
                    "top_level_keys": list(_CANDIDATE_SCHEMA),
                    "forbidden_wrapper_keys": [
                        "candidate",
                        "candidate_schema",
                        "schema",
                    ],
                    "no_additional_top_level_keys": True,
                },
                "field_contracts": _FIELD_CONTRACTS,
                "constraints": [
                    "return exactly one direct Candidate JSON object",
                    "do not wrap the object in candidate, candidate_schema, or schema",
                    "include every top_level_keys field exactly once",
                    "candidate remains inactive",
                    "do not change weights, evaluator, cohort, or activation state",
                ],
            }
        )
        return {
            "model": self.model,
            "temperature": float(temperature),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": canonical_json(request)},
            ],
            "max_tokens": max_tokens,
        }


class DeepSeekCompatibleTeacherTransport(OpenAICompatibleTeacherTransport):
    """OpenAI-compatible transport with deterministic DeepSeek controls."""

    def _body(self, request: Mapping[str, object]) -> dict[str, object]:
        body = super()._body(request)
        body["thinking"] = {"type": "disabled"}
        return body


class FrozenReplayTeacherTransport:
    """Replay an exact raw Teacher response for one exact frozen request."""

    def __init__(
        self,
        *,
        request_path: str | Path,
        response_path: str | Path,
    ) -> None:
        self.request_path = Path(request_path).resolve()
        self.response_path = Path(response_path).resolve()
        try:
            self._request_bytes = self.request_path.read_bytes()
            self._response_bytes = self.response_path.read_bytes()
            request = json.loads(self._request_bytes)
            response = json.loads(self._response_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ContractViolation("frozen Teacher replay is unreadable") from error
        if not isinstance(request, dict) or not isinstance(response, dict):
            raise ContractViolation("frozen Teacher replay must contain objects")
        self._request = request
        self._response = response

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        try:
            if (
                self.request_path.read_bytes() != self._request_bytes
                or self.response_path.read_bytes() != self._response_bytes
            ):
                raise ContractViolation("frozen Teacher replay identity changed")
        except OSError as error:
            raise ContractViolation("frozen Teacher replay is missing") from error
        if request != self._request:
            raise ContractViolation("frozen Teacher request identity mismatch")
        replay = json.loads(self._response_bytes)
        assert isinstance(replay, dict)
        return replay


class FrozenReplayDirectoryTeacherTransport:
    """Select one exact request/response pair by the product request id."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ContractViolation("frozen Teacher replay directory is missing")

    def __call__(self, request: dict[str, object]) -> dict[str, object]:
        request_id = request.get("request_id")
        if (
            not isinstance(request_id, str)
            or not request_id
            or Path(request_id).name != request_id
        ):
            raise ContractViolation("frozen Teacher request_id is unsafe")
        pair = self.root / request_id
        if not pair.is_dir():
            raise ContractViolation("frozen Teacher replay pair is missing")
        return FrozenReplayTeacherTransport(
            request_path=pair / "TEACHER-REQUEST.json",
            response_path=pair / "TEACHER-RESPONSE.json",
        )(request)


def build_teacher_transport(
    *,
    provider: str,
    model: str,
    endpoint: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    timeout_seconds: float = 60,
    max_retries: int = 3,
    retry_base_delay: float = 1.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> TeacherTransport:
    """Construct the provider adapter used by the autonomous runner."""

    if not isinstance(provider, str) or not provider.strip():
        raise ContractViolation("Teacher provider must be non-empty")
    normalized = provider.strip().casefold()
    if normalized in {"frozen", "frozen-replay"}:
        return FrozenReplayDirectoryTeacherTransport(endpoint)
    transport_type: type[OpenAICompatibleTeacherTransport]
    if normalized in {"deepseek", "deepseek-compatible"}:
        transport_type = DeepSeekCompatibleTeacherTransport
    elif normalized in {"openai", "openai-compatible"}:
        transport_type = OpenAICompatibleTeacherTransport
    else:
        raise ContractViolation(f"unsupported Teacher provider: {provider}")
    return transport_type(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
        opener=opener,
    )
