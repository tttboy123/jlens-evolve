from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from skill_evolution_loop import round1_taskset
from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.round1_taskset import select_round1_targets


def _checkout(path: Path) -> Path:
    path.mkdir()
    (path / "src").mkdir()
    (path / "src/answer.rs").write_text(
        "pub fn calculate_limit(value: i32) -> i32 { value + 1 }\n",
        encoding="utf-8",
    )
    (path / "src/other.rs").write_text("pub fn unrelated() {}\n", encoding="utf-8")
    (path / "tests").mkdir()
    (path / "tests/answer.rs").write_text("calculate_limit(1);\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


def test_round1_target_selector_is_issue_ranked_and_excludes_tests(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")

    targets = select_round1_targets(
        checkout,
        "calculate_limit should add two when given a value",
        "rust",
    )

    assert targets[0] == "src/answer.rs"
    assert all("tests/" not in target for target in targets)


def test_round1_target_selector_freezes_bounded_top_k(tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "repo")
    for index in range(12):
        (checkout / "src" / f"candidate_{index}.rs").write_text(
            f"pub fn calculate_limit_{index}() {{}}\n", encoding="utf-8"
        )

    targets = select_round1_targets(
        checkout,
        "calculate_limit should add two",
        "rust",
        max_targets=8,
    )

    assert len(targets) == 8
    assert targets[0] == "src/answer.rs"


def test_round1_target_selector_rejects_unknown_language(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="unsupported"):
        select_round1_targets(_checkout(tmp_path / "repo"), "fixture", "polyglot")


def test_round1_target_selector_rejects_unbounded_candidate_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContractError, match="between 1 and 64"):
        select_round1_targets(
            _checkout(tmp_path / "repo"),
            "fixture",
            "rust",
            max_targets=65,
        )


def test_round1_target_selector_allocates_scan_budget_by_path_relevance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path / "repo")
    fillers = checkout / "aaa"
    fillers.mkdir()
    for index in range(8):
        (fillers / f"filler_{index}.rs").write_text(
            "let unrelated = 1;\n" * 20,
            encoding="utf-8",
        )
    target = checkout / "zzz" / "rare_boundary.rs"
    target.parent.mkdir()
    target.write_text("pub fn fix() {}\n", encoding="utf-8")
    monkeypatch.setattr(round1_taskset, "_MAX_CONTENT_SCAN_BYTES", 500)

    targets = select_round1_targets(
        checkout,
        "rare boundary is broken",
        "rust",
        max_targets=1,
    )

    assert targets == ("zzz/rare_boundary.rs",)


def test_round1_target_selector_excludes_language_specific_test_names(
    tmp_path: Path,
) -> None:
    checkout = _checkout(tmp_path / "repo")
    (checkout / "src/answer.go").write_text(
        "func calculate_limit() {}\n", encoding="utf-8"
    )
    (checkout / "src/answer_test.go").write_text(
        "func calculate_limit() {}\n", encoding="utf-8"
    )
    (checkout / "src/Answer.java").write_text("class Answer {}\n", encoding="utf-8")
    (checkout / "src/AnswerTest.java").write_text(
        "class AnswerTest {}\n", encoding="utf-8"
    )

    go_targets = select_round1_targets(checkout, "calculate_limit", "go")
    java_targets = select_round1_targets(checkout, "Answer", "java")

    assert "src/answer_test.go" not in go_targets
    assert "src/AnswerTest.java" not in java_targets
