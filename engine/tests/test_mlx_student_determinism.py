"""Determinism tests for the local MLX generation entry."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from skill_evolution_loop.contracts import LoopRevision
from skill_evolution_loop.mlx_student import MlxStructuredGenerator
from skill_evolution_loop.student_adapter import StudentTask


def _git_checkout(path: Path) -> Path:
    import subprocess

    path.mkdir(parents=True)
    (path / "src").mkdir()
    (path / "src" / "example.py").write_text("def answer():\n    return 1\n")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    return path


def _revision() -> LoopRevision:
    return LoopRevision.create(
        skill_id="local-edit-skill",
        revision_id="rev-det",
        parent_revision_id=None,
        source_round=0,
        protocol="structured-search-replace-v1",
        skill_text="Return one bounded edit.",
        prompt_template="Edit as JSON.",
        eval_note="offline fixture",
    )


def _task(checkout: Path) -> StudentTask:
    return StudentTask.create(
        task_id="det-task",
        checkout=checkout,
        instruction="Make answer return two.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )


def test_local_generation_reseeds_mlx_prng(tmp_path: Path, monkeypatch) -> None:
    checkout = _git_checkout(tmp_path / "repo")
    calls: list[int] = []

    fake_mlx = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.random = types.SimpleNamespace(seed=lambda value: calls.append(value))
    fake_mlx.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)

    prompts: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=128,
        seed=7,
        temperature=0.0,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda _model, _tokenizer, *, prompt, max_tokens: (
            prompts.append(prompt) or "{}"
        ),
    )
    generator(_task(checkout), _revision())

    assert calls, "local generation must reseed the MLX PRNG"
    assert calls[0] == 7
    config = generator.generation_config()
    assert config["seed"] == 7
    assert config["temperature"] == 0.0
    assert config["execution_mode"] == "local-mlx"


def test_remote_generation_pins_seed_zero(tmp_path: Path) -> None:
    checkout = _git_checkout(tmp_path / "repo")
    seen: list[dict] = []

    class FakeTransport:
        def generate_prompt(self, request):
            seen.append(
                {
                    "temperature": request.temperature,
                    "seed": request.seed,
                    "max_tokens": request.max_tokens,
                }
            )
            return types.SimpleNamespace(text="{}")

        def identity(self):
            return "fake-transport"

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            return "\n".join(row["content"] for row in messages)

    generator = MlxStructuredGenerator(
        model_path="fixture-model",
        max_tokens=128,
        seed=9,
        temperature=0.0,
        model_transport=FakeTransport(),
        tokenizer_loader=lambda _path: FakeTokenizer(),
    )
    generator(_task(checkout), _revision())

    assert seen, "remote transport must be called"
    assert seen[0]["temperature"] == 0.0
    assert seen[0]["seed"] == 0
