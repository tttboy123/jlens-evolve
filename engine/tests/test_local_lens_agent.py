"""Tests for the v2.5 local JLens agent (T4): step loop, tools, evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_lens_agent import LensAgentError, LocalLensAgent


def _fake_generator(script: list[dict]):
    idx = {"n": 0}

    def gen(messages, max_tokens: int):
        item = script[min(idx["n"], len(script) - 1)]
        idx["n"] += 1
        return json.dumps(item)

    return gen


def _fake_capture(model, tokenizer, text):
    return [
        {
            "layer": i,
            "is_linear": i % 4 != 0,
            "shape": [1, 5, 2560],
            "mean": float(i),
            "l2_norm": float(i + 1),
        }
        for i in range(32)
    ]


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (repo / "test_calc.py").write_text(
        "from calc import add\nassert add(1, 2) == 3\n", encoding="utf-8"
    )
    import subprocess

    subprocess.run(["git", "init", "-q", str(repo)], check=False)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=False)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=False,
    )
    return repo


def test_step_loop_parses_actions_and_writes_evidence(tmp_path: Path, tiny_repo: Path):
    script = [
        {"tool": "list"},
        {"tool": "read", "path": "calc.py"},
        {
            "tool": "write",
            "path": "calc.py",
            "content": "def add(a, b):\n    return a + b + 1\n",
        },
        {"tool": "finish", "message": "fixed"},
    ]
    out = tmp_path / "out"
    agent = LocalLensAgent(
        repo=tiny_repo,
        issue="make add return a+b+1",
        out_dir=out,
        max_steps=4,
        generator=_fake_generator(script),
        capture=_fake_capture,
    )
    result = agent.run()

    assert result["steps_completed"] == 4
    assert (out / "meta.json").is_file()
    assert (out / "STEPS.jsonl").is_file()
    assert (out / "final.diff").is_file()
    lines = [json.loads(l) for l in (out / "STEPS.jsonl").read_text().splitlines()]
    assert len(lines) == 4
    for line in lines:
        assert len(line["layer_records"]) == 32
        assert line["step_sha256"]
    assert (
        "add(1, 2) == 3" in (tiny_repo / "test_calc.py").read_text()
    )  # test unchanged
    assert "b + 1" in (tiny_repo / "calc.py").read_text()
    assert "+1" in (out / "final.diff").read_text()


def test_tool_allowlist_blocks_invalid_command(tiny_repo: Path, tmp_path: Path):
    script = [{"tool": "run", "cmd": "rm -rf /"}]
    agent = LocalLensAgent(
        repo=tiny_repo,
        issue="x",
        out_dir=tmp_path / "out",
        max_steps=1,
        generator=_fake_generator(script),
        capture=_fake_capture,
    )
    agent.run()
    lines = [
        json.loads(l) for l in (tmp_path / "out/STEPS.jsonl").read_text().splitlines()
    ]
    assert lines[0]["tool_ok"] is False
    assert "ERROR" in lines[0]["tool_result"]


def test_path_escape_rejected(tiny_repo: Path, tmp_path: Path):
    script = [{"tool": "read", "path": "../outside.txt"}]
    agent = LocalLensAgent(
        repo=tiny_repo,
        issue="x",
        out_dir=tmp_path / "out",
        max_steps=1,
        generator=_fake_generator(script),
        capture=_fake_capture,
    )
    agent.run()
    lines = [
        json.loads(l) for l in (tmp_path / "out/STEPS.jsonl").read_text().splitlines()
    ]
    assert lines[0]["tool_ok"] is False
    assert "escapes repo" in lines[0]["tool_result"]


def test_no_json_action_raises(tiny_repo: Path, tmp_path: Path):
    def bad_gen(messages, max_tokens):
        return "no action here"

    with pytest.raises(LensAgentError, match="no JSON action"):
        LocalLensAgent(
            repo=tiny_repo,
            issue="x",
            out_dir=tmp_path / "out",
            max_steps=1,
            generator=bad_gen,
            capture=_fake_capture,
        ).run()
