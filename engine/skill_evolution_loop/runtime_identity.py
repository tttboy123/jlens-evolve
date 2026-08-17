"""Immutable runtime identity shared by every model execution plane."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import canonical_json, sha256_json


@dataclass(frozen=True)
class RuntimeIdentity:
    model_repository: str
    model_revision: str
    model_weights_sha256: tuple[str, ...]
    quantization: str
    tokenizer_sha256: str
    chat_template_sha256: str
    container_digest: str
    runtime: str
    generation_parameters: dict[str, Any]
    transport_sha256: str
    skill_sha256: str
    framework_sha256: str

    @classmethod
    def create(cls, **fields: Any) -> RuntimeIdentity:
        identity = cls(**fields)
        identity.validate()
        return identity

    def validate(self) -> None:
        for label, value in (
            ("model_repository", self.model_repository),
            ("model_revision", self.model_revision),
            ("quantization", self.quantization),
            ("container_digest", self.container_digest),
            ("runtime", self.runtime),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty")
        shas = (
            *self.model_weights_sha256,
            self.tokenizer_sha256,
            self.chat_template_sha256,
            self.transport_sha256,
            self.skill_sha256,
            self.framework_sha256,
        )
        if not self.model_weights_sha256 or any(not _sha(value) for value in shas):
            raise ValueError("runtime identity SHA is invalid")
        if not isinstance(self.generation_parameters, dict):
            raise ValueError("generation parameters must be an object")

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "model_weights_sha256": self.model_weights_sha256,
            "quantization": self.quantization,
            "tokenizer_sha256": self.tokenizer_sha256,
            "chat_template_sha256": self.chat_template_sha256,
            "container_digest": self.container_digest,
            "runtime": self.runtime,
            "generation_parameters": self.generation_parameters,
            "transport_sha256": self.transport_sha256,
            "skill_sha256": self.skill_sha256,
            "framework_sha256": self.framework_sha256,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.content_dict(), "identity_sha256": self.fingerprint}


def freeze_runtime_identity(path: str | Path, identity: RuntimeIdentity) -> Path:
    destination = Path(path)
    content = canonical_json(identity.to_dict()) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError("runtime identity is immutable")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
