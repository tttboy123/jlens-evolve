"""Majority-vote tests for StudentAdapter.run_majority."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from skill_evolution_loop.contracts import LoopRevision
from skill_evolution_loop.student_adapter import StudentAdapter, StudentTask


def _git_checkout(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "src").mkdir()
    (path / "src" / "example.py").write_text("def answer():\n    return 1\n")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    return path


def _revision() -> LoopRevision:
    return LoopRevision.create(
        skill_id="local-edit-skill",
        revision_id="rev-majority",
        parent_revision_id=None,
        source_round=0,
        protocol="structured-search-replace-v1",
        skill_text="Return one bounded edit.",
        prompt_template="Edit as JSON.",
        eval_note="offline fixture",
    )


def _task(checkout: Path) -> StudentTask:
    return StudentTask.create(
        task_id="majority-task",
        checkout=checkout,
        instruction="Make answer return two.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )


def _raw(replace: str) -> str:
    return json.dumps(
        {
            "file": "src/example.py",
            "search": "    return 1\n",
            "replace": replace,
            "diagnostic": "fix",
        }
    )


class FakeGenerator:
    def __init__(self) -> None:
        self.seed = 0

    def __call__(self, task: StudentTask, revision: LoopRevision) -> str:
        remainder = self.seed % 3
        if remainder == 0:
            return _raw("    return 2\n")
        if remainder == 1:
            return _raw("    return 2\n")
        return "not-a-json-edit"


def test_majority_selects_common_valid_patch(tmp_path: Path) -> None:
    checkout = _git_checkout(tmp_path / "repo")
    task = _task(checkout)
    revision = _revision()
    adapter = StudentAdapter(generator=FakeGenerator())

    result = adapter.run_majority(task, revision, samples=3, seed_base=0)

    assert result.samples == 3
    assert result.attempt.structural_valid is True
    assert result.attempt.patch_sha256 is not None
    assert result.votes[max(result.votes, key=result.votes.get)] == 2
    assert result.selected_seed == 0
    assert "return 2" in result.attempt.patch


def test_majority_returns_failure_when_all_samples_invalid(tmp_path: Path) -> None:
    checkout = _git_checkout(tmp_path / "repo")
    task = _task(checkout)
    revision = _revision()

    class AlwaysBad(FakeGenerator):
        def __call__(self, task, revision):
            return "not-a-json-edit"

    adapter = StudentAdapter(generator=AlwaysBad())
    result = adapter.run_majority(task, revision, samples=3, seed_base=0)

    assert result.attempt.structural_valid is False
    assert result.votes == {}
    assert result.selected_seed is None
