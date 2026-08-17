"""Multi-model teaching API capability (v2.5 / T3).

Exposes a unified, configuration-driven interface for "teacher" model calls:
DeepSeek / OpenAI (GPT-4o) / Claude, with weighted inputs. This module is
capability-only: it never calls an external API unless an explicitly configured
client or environment keys are supplied. Tests inject a fake transport.

Boundaries:
- Teaching data feeds inactive candidate Skills / PatternCards only.
- Model weights stay frozen (no training / SFT / LoRA / RL).
- External teacher calls require explicit user authorization and a separate
  budget; they are NOT part of the DeepSeek 2000-call evaluation budget.
- Evidence is append-only; negative evidence is preserved.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests


class TeacherProvider(str, Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    CLAUDE = "claude"


class TeacherCallError(ValueError):
    """Raised when a teacher call is not configured or the provider fails."""


@dataclass(frozen=True)
class TeacherSample:
    """One teaching input with a weight used for downstream synthesis."""

    messages: list[dict[str, str]]
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: int = 2000
    response_format: dict[str, str] | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class TeacherResponse:
    provider: TeacherProvider
    model: str
    text: str
    weight: float = 1.0
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherConfig:
    provider: TeacherProvider
    api_base: str
    model: str
    api_key_env: str
    timeout_s: float = 60.0
    api_key_env_aliases: tuple[str, ...] = ()


_DEFAULT_CONFIGS: dict[TeacherProvider, TeacherConfig] = {
    TeacherProvider.DEEPSEEK: TeacherConfig(
        provider=TeacherProvider.DEEPSEEK,
        api_base=os.environ.get(
            "EVOLVE_TEACHER_DEEPSEEK_API_BASE", "https://api.deepseek.com"
        ),
        model=os.environ.get(
            "EVOLVE_TEACHER_DEEPSEEK_MODEL", "deepseek-v4-flash"
        ),
        api_key_env="EVOLVE_TEACHER_DEEPSEEK_API_KEY",
        api_key_env_aliases=("DEEPSEEK_API_KEY",),
        timeout_s=600.0,
    ),
    TeacherProvider.OPENAI: TeacherConfig(
        provider=TeacherProvider.OPENAI,
        api_base=os.environ.get(
            "EVOLVE_TEACHER_OPENAI_API_BASE", "https://api.openai.com/v1"
        ),
        model=os.environ.get("EVOLVE_TEACHER_OPENAI_MODEL", "gpt-4o"),
        api_key_env="OPENAI_API_KEY",
    ),
    TeacherProvider.CLAUDE: TeacherConfig(
        provider=TeacherProvider.CLAUDE,
        api_base=os.environ.get(
            "EVOLVE_TEACHER_ANTHROPIC_API_BASE", "https://api.anthropic.com/v1"
        ),
        model=os.environ.get("EVOLVE_TEACHER_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        api_key_env="ANTHROPIC_API_KEY",
    ),
}

Transport = Callable[..., requests.Response]


class TeacherClient:
    """Configured teacher client.

    ``transport`` is injectable for tests; the default uses ``requests.post``
    and is only reached when the caller has supplied an API key explicitly.
    """

    def __init__(
        self,
        config: TeacherConfig,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        configured_key = os.environ.get(config.api_key_env)
        if configured_key is None:
            configured_key = next(
                (
                    value
                    for name in config.api_key_env_aliases
                    if (value := os.environ.get(name))
                ),
                None,
            )
        self.api_key = api_key if api_key is not None else configured_key
        self._transport = transport or requests.post

    @classmethod
    def from_env(cls, provider: TeacherProvider | str) -> TeacherClient:
        p = TeacherProvider(provider)
        return cls(_DEFAULT_CONFIGS[p])

    def complete(self, sample: TeacherSample) -> TeacherResponse:
        maximum_output_tokens = (
            384_000
            if self.config.provider is TeacherProvider.DEEPSEEK
            and self.config.model.startswith("deepseek-v4-")
            else 4_096
        )
        if (
            type(sample.max_output_tokens) is not int
            or not 1 <= sample.max_output_tokens <= maximum_output_tokens
        ):
            raise TeacherCallError(
                "teacher max_output_tokens must be between 1 and "
                f"{maximum_output_tokens}"
            )
        if sample.response_format is not None and sample.response_format != {
            "type": "json_object"
        }:
            raise TeacherCallError("unsupported teacher response_format")
        if sample.reasoning_effort not in {None, "low", "high", "max"}:
            raise TeacherCallError("unsupported teacher reasoning_effort")
        if (
            (sample.thinking is not None or sample.reasoning_effort is not None)
            and self.config.provider is not TeacherProvider.DEEPSEEK
        ):
            raise TeacherCallError("thinking controls require DeepSeek")
        if not self.api_key:
            accepted = ", ".join(
                (self.config.api_key_env, *self.config.api_key_env_aliases)
            )
            raise TeacherCallError(
                f"{self.config.provider.value}: not configured "
                f"(missing env: {accepted}); no external call was made"
            )
        if self.config.provider is TeacherProvider.CLAUDE:
            return self._complete_claude(sample)
        return self._complete_openai_compatible(sample)

    def _complete_openai_compatible(self, sample: TeacherSample) -> TeacherResponse:
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": sample.messages,
            "max_tokens": sample.max_output_tokens,
        }
        if sample.thinking is not True:
            payload["temperature"] = 0.2
        if sample.response_format is not None:
            payload["response_format"] = sample.response_format
        if sample.thinking is not None:
            payload["thinking"] = {
                "type": "enabled" if sample.thinking else "disabled"
            }
        if sample.reasoning_effort is not None:
            payload["reasoning_effort"] = sample.reasoning_effort
        resp = self._transport(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.config.timeout_s,
        )
        try:
            data = resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise TeacherCallError(
                f"non-JSON response from {self.config.provider.value}"
            ) from exc
        if resp.status_code >= 400:
            raise TeacherCallError(
                f"{self.config.provider.value} HTTP {resp.status_code}: {str(data)[:300]}"
            )
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return TeacherResponse(
            provider=self.config.provider,
            model=data.get("model", self.config.model),
            text=text,
            weight=sample.weight,
            usage=usage,
            metadata=sample.metadata,
        )

    def _complete_claude(self, sample: TeacherSample) -> TeacherResponse:
        url = f"{self.config.api_base.rstrip('/')}/messages"
        payload = {
            "model": self.config.model,
            "messages": sample.messages,
            "max_tokens": sample.max_output_tokens,
            "temperature": 0.2,
        }
        resp = self._transport(
            url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_s,
        )
        try:
            data = resp.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise TeacherCallError("non-JSON response from claude") from exc
        if resp.status_code >= 400:
            raise TeacherCallError(f"claude HTTP {resp.status_code}: {str(data)[:300]}")
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        usage = data.get("usage", {})
        return TeacherResponse(
            provider=self.config.provider,
            model=data.get("model", self.config.model),
            text=text,
            weight=sample.weight,
            usage=usage,
            metadata=sample.metadata,
        )


def weighted_messages(*parts: list[dict[str, str]]) -> list[dict[str, str]]:
    """Concatenate message segments into one sample's messages."""
    merged: list[dict[str, str]] = []
    for part in parts:
        merged.extend(part)
    return merged


def combine_weighted(responses: list[TeacherResponse]) -> str:
    """Simple weighted synthesis: weighted concatenation, labeled by provider.

    This is intentionally deterministic and append-only; richer synthesis
    (e.g., majority voting) can be layered on top later.
    """
    blocks: list[str] = []
    for r in sorted(responses, key=lambda x: (x.provider.value, x.model)):
        prefix = f"[{r.provider.value}:{r.model}:w={r.weight:g}]"
        blocks.append(f"{prefix}\n{r.text}")
    return "\n\n".join(blocks)
