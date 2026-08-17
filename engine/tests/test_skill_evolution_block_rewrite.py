from __future__ import annotations

import json
import subprocess
from pathlib import Path

from skill_evolution_loop import LoopRevision, StudentTask
from skill_evolution_loop.block_rewrite import (
    LineBlockRewriteAdapter,
    MlxLineBlockRewriteGenerator,
    build_block_conditions,
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
    return path


def _task(checkout: Path) -> StudentTask:
    return StudentTask.create(
        task_id="block-fixture",
        checkout=checkout,
        instruction="The answer function must return the input value plus two.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )


def _revision(*, taught: bool) -> LoopRevision:
    return LoopRevision.create(
        skill_id="block-fixture-skill",
        revision_id="block-taught" if taught else "block-baseline",
        parent_revision_id=None,
        source_round=1 if taught else 0,
        protocol="python-line-block-rewrite-v1",
        skill_text=(
            "Transformation: change the returned offset from one to two."
            if taught
            else "No additional domain teaching is provided."
        ),
        prompt_template=(
            "Return exactly one JSON object with file, symbol, start_line, end_line, "
            "replacement, and diagnostic."
        ),
        eval_note="fixture",
    )


def test_line_block_adapter_rewrites_a_range_inside_one_symbol(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    raw = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "answer",
            "start_line": 6,
            "end_line": 6,
            "replacement": "    return value + 2\n",
            "diagnostic": "change the answer offset",
        }
    )
    adapter = LineBlockRewriteAdapter(generator=lambda _task, _revision: raw)

    attempt = adapter.run(_task(checkout), _revision(taught=True))

    assert attempt.structural_valid is True
    assert "+    return value + 2" in attempt.patch


def test_line_block_adapter_rejects_a_range_outside_the_named_symbol(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    raw = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "answer",
            "start_line": 2,
            "end_line": 2,
            "replacement": "    return 10\n",
            "diagnostic": "edits helper instead",
        }
    )
    adapter = LineBlockRewriteAdapter(generator=lambda _task, _revision: raw)

    attempt = adapter.run(_task(checkout), _revision(taught=True))

    assert attempt.structural_valid is False
    assert attempt.failure_reason == "apply-fail"
    assert "inside the named symbol" in attempt.detail


def test_line_block_generator_numbers_one_fixed_symbol_for_both_arms(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    rendered_messages: list[list[dict[str, str]]] = []
    output = json.dumps(
        {
            "file": "src/example.py",
            "symbol": "answer",
            "start_line": 6,
            "end_line": 6,
            "replacement": "    return value + 2\n",
            "diagnostic": "fixture",
        }
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            rendered_messages.append(messages)
            return "\n".join(row["content"] for row in messages)

    generator = MlxLineBlockRewriteGenerator(
        model_path="fixture-model",
        max_tokens=256,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: output,
    )

    generator(_task(checkout), _revision(taught=False))
    generator(_task(checkout), _revision(taught=True))

    assert rendered_messages[0][1] == rendered_messages[1][1]
    assert rendered_messages[0][0] != rendered_messages[1][0]
    assert "000005|def answer(value):" in rendered_messages[0][1]["content"]
    assert "000006|    return value + 1" in rendered_messages[0][1]["content"]
    assert "def helper" not in rendered_messages[0][1]["content"]
    assert generator.generation_config()["action_space"] == (
        "python-line-block-rewrite-v1"
    )


def test_block_conditions_keep_the_action_contract_fixed() -> None:
    conditions = build_block_conditions(
        taught_skill="Transformation: use one aligned defaults vector.",
        parent_revision_id="r5",
        source_round=5,
        generation_config={"generator": "fixture"},
    )

    baseline, taught = conditions
    assert [row.condition_id for row in conditions] == [
        "block-baseline",
        "block-taught",
    ]
    assert baseline.mechanism == taught.mechanism == "block"
    assert baseline.revision.protocol == taught.revision.protocol
    assert baseline.revision.prompt_template == taught.revision.prompt_template
    assert baseline.revision.skill_text != taught.revision.skill_text
