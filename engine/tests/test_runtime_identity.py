from __future__ import annotations

import json

import pytest

from skill_evolution_loop.runtime_identity import (
    RuntimeIdentity,
    freeze_runtime_identity,
)


def test_runtime_identity_freezes_every_required_dimension(tmp_path) -> None:
    identity = RuntimeIdentity.create(
        model_repository="Qwen/Qwen3.5-4B",
        model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        model_weights_sha256=("a" * 64,),
        quantization="bf16",
        tokenizer_sha256="b" * 64,
        chat_template_sha256="c" * 64,
        container_digest="sha256:" + "d" * 64,
        runtime="vllm-main",
        generation_parameters={"temperature": 0.0, "seed": 0},
        transport_sha256="e" * 64,
        skill_sha256="f" * 64,
        framework_sha256="1" * 64,
    )

    path = freeze_runtime_identity(tmp_path / "IDENTITY.json", identity)
    value = json.loads(path.read_text())

    assert value["identity_sha256"] == identity.fingerprint
    assert freeze_runtime_identity(path, identity) == path


def test_runtime_identity_refuses_overwrite(tmp_path) -> None:
    fields = dict(
        model_repository="model",
        model_revision="revision",
        model_weights_sha256=("a" * 64,),
        quantization="bf16",
        tokenizer_sha256="b" * 64,
        chat_template_sha256="c" * 64,
        container_digest="sha256:" + "d" * 64,
        runtime="runtime",
        generation_parameters={"seed": 0},
        transport_sha256="e" * 64,
        skill_sha256="f" * 64,
        framework_sha256="1" * 64,
    )
    path = tmp_path / "IDENTITY.json"
    freeze_runtime_identity(path, RuntimeIdentity.create(**fields))

    with pytest.raises(ValueError, match="immutable"):
        freeze_runtime_identity(
            path,
            RuntimeIdentity.create(**{**fields, "quantization": "awq-4bit"}),
        )
