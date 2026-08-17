from __future__ import annotations

import json
import subprocess
from pathlib import Path

from skill_evolution_loop import LoopRevision, StudentTask
from skill_evolution_loop.symbol_rewrite import (
    MlxSymbolRewriteGenerator,
    SymbolRewriteAdapter,
    build_symbol_conditions,
)


def _checkout(path: Path) -> Path:
    path.mkdir()
    (path / "src").mkdir()
    (path / "src/example.py").write_text(
        "def helper():\n    return 0\n\n\ndef answer(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=path,
        check=True,
    )
    return path


def _task(checkout: Path) -> StudentTask:
    return StudentTask.create(
        task_id="symbol-fixture",
        checkout=checkout,
        instruction="The answer function must return the input value plus two.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )


def _revision(*, taught: bool) -> LoopRevision:
    return LoopRevision.create(
        skill_id="symbol-fixture-skill",
        revision_id="symbol-taught" if taught else "symbol-baseline",
        parent_revision_id=None,
        source_round=1 if taught else 0,
        protocol="python-symbol-rewrite-v1",
        skill_text=(
            "Transformation: preserve the complete function and change the returned "
            "offset from one to two."
            if taught
            else "No additional domain teaching is provided."
        ),
        prompt_template=(
            "Return exactly one JSON object with file, symbol, replacement, and "
            "diagnostic."
        ),
        eval_note="fixture",
    )


def test_symbol_rewrite_adapter_replaces_one_ast_located_definition(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    raw = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "answer",
            "replacement": "def answer(value):\n    return value + 2\n",
            "diagnostic": "change the answer offset",
        }
    )
    adapter = SymbolRewriteAdapter(generator=lambda _task, _revision: raw)

    attempt = adapter.run(_task(checkout), _revision(taught=True))

    assert attempt.structural_valid is True
    assert attempt.failure_reason is None
    assert attempt.target_file == "src/example.py"
    assert "def helper" not in attempt.patch
    assert "+    return value + 2" in attempt.patch
    adapter.apply(attempt)
    assert "return value + 2" in (checkout / "src/example.py").read_text()


def test_symbol_rewrite_adapter_rejects_a_replacement_for_another_symbol(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    raw = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "answer",
            "replacement": "def invented(value):\n    return value + 2\n",
            "diagnostic": "wrong symbol",
        }
    )
    adapter = SymbolRewriteAdapter(generator=lambda _task, _revision: raw)

    attempt = adapter.run(_task(checkout), _revision(taught=True))

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "apply-fail"
    assert "same symbol" in attempt.detail


def test_symbol_rewrite_adapter_resolves_one_unique_unqualified_method(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    (checkout / "src/example.py").write_text(
        "class Writer:\n    def visit_literal(self, value):\n        return value\n",
        encoding="utf-8",
    )
    raw = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "visit_literal",
            "replacement": (
                "    def visit_literal(self, value):\n        return value.strip()\n"
            ),
            "diagnostic": "trim the literal",
        }
    )
    adapter = SymbolRewriteAdapter(generator=lambda _task, _revision: raw)

    attempt = adapter.run(_task(checkout), _revision(taught=True))

    assert attempt.structural_valid is True
    assert "+        return value.strip()" in attempt.patch


def test_symbol_rewrite_generator_keeps_user_prompt_fixed_between_arms(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered_messages: list[list[dict[str, str]]] = []
    loads: list[str] = []
    output = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "answer",
            "replacement": "def answer(value):\n    return value + 2\n",
            "diagnostic": "fixture",
        }
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            rendered_messages.append(messages)
            assert tokenize is False
            return "\n".join(row["content"] for row in messages)

    def load(path: str):
        loads.append(path)
        return object(), Tokenizer()

    generator = MlxSymbolRewriteGenerator(
        model_path="fixture-model",
        max_tokens=512,
        loader=load,
        text_generator=lambda *_args, **_kwargs: output,
    )

    generator(_task(checkout), _revision(taught=False))
    generator(_task(checkout), _revision(taught=True))

    assert loads == ["fixture-model"]
    assert rendered_messages[0][1] == rendered_messages[1][1]
    assert rendered_messages[0][0] != rendered_messages[1][0]
    assert "def answer(value)" in rendered_messages[0][1]["content"]
    assert "def helper" not in rendered_messages[0][1]["content"]
    assert generator.generation_trace_kinds() == ("symbol-rewrite-attempt-0",)
    assert generator.generation_trace_results() == ({"status": "generated"},)
    assert generator.generation_config()["action_space"] == ("python-symbol-rewrite-v1")
    assert generator.generation_config()["prompt_rendering"] == (
        "chat-template-tokenize-false-v1"
    )


def test_symbol_conditions_vary_only_the_teaching_content() -> None:
    conditions = build_symbol_conditions(
        taught_skill="Transformation: use the aligned defaults vector in both loops.",
        parent_revision_id="r5",
        source_round=5,
        generation_config={"generator": "fixture"},
    )

    assert [condition.condition_id for condition in conditions] == [
        "symbol-baseline",
        "symbol-taught",
    ]
    baseline, taught = conditions
    assert baseline.mechanism == taught.mechanism == "symbol"
    assert baseline.revision.protocol == taught.revision.protocol
    assert baseline.revision.prompt_template == taught.revision.prompt_template
    assert baseline.revision.skill_text != taught.revision.skill_text
    assert baseline.generation_config == taught.generation_config


def test_symbol_context_selector_ignores_noisy_issue_body_terms(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (checkout / "src").mkdir()
    (checkout / "src/example.py").write_text(
        "def signature(subject):\n"
        "    # https python has environment details\n"
        "    return subject\n\n\n"
        "def signature_from_str(arg):\n"
        "    defaults = arg.defaults\n"
        "    positional = arg.positional\n"
        "    return defaults, positional\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    captured: list[str] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            captured.append(messages[1]["content"])
            return "prompt"

    generator = MlxSymbolRewriteGenerator(
        model_path="fixture-model",
        max_tokens=64,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: "{}",
    )
    task = StudentTask.create(
        task_id="noisy-issue",
        checkout=checkout,
        instruction=(
            "Default positional argument vanished\n"
            "Environment: python has https links and unrelated details"
        ),
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )

    generator(task, _revision(taught=False))

    assert "def signature_from_str" in captured[0]
    assert "def signature(subject)" not in captured[0]
