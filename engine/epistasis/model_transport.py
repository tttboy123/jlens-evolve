"""Model-agnostic transport so any vendor (local MLX, DeepSeek, OpenAI, ...)
can drive the same experiment suite.

``StubTransport`` returns a fixed repair source (used by tests and CI);
``OpenAICompatTransport`` talks to any OpenAI-compatible ``/chat/completions``
endpoint, which covers mlx_lm.server, vLLM, DeepSeek, and hosted providers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ModelTransport(Protocol):
    model_id: str

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str: ...


@dataclass(frozen=True, slots=True)
class StubTransport:
    """Deterministic transport returning a configurable repair source."""

    repair_source: str
    model_id: str = "stub-deterministic"

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        del prompt, temperature
        return f"```python\n{self.repair_source.strip()}\n```"


@dataclass(frozen=True, slots=True)
class OpenAICompatTransport:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_seconds: float = 180.0

    @property
    def model_id(self) -> str:
        return self.model

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"model endpoint returned HTTP {exc.code}: {detail[:400]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surface transport failures
            raise RuntimeError(f"model transport failed: {exc}") from exc
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected model response shape: {exc}") from exc


def load_transport(config_path: str | Path | None) -> ModelTransport:
    """Load a transport from a JSON config or fall back to the stub."""
    if config_path is None:
        return StubTransport(repair_source="")
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    provider = str(data.get("provider", "openai"))
    if provider == "stub":
        return StubTransport(
            repair_source=str(data.get("repair_source", "")),
            model_id=str(data.get("model", "stub-deterministic")),
        )
    if provider != "openai":
        raise ValueError(f"unsupported transport provider: {provider}")
    api_key_env = str(data.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"transport requires env var {api_key_env} (set it before running)"
        )
    return OpenAICompatTransport(
        base_url=str(data["base_url"]),
        model=str(data["model"]),
        api_key=api_key,
        temperature=float(data.get("temperature", 0.7)),
        max_tokens=int(data.get("max_tokens", 1024)),
        timeout_seconds=float(data.get("timeout_seconds", 180.0)),
    )
