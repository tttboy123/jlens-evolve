from __future__ import annotations

from benchmark_catalog import (
    PINNED_SOURCES,
    build_execution_command,
    normalize_multi_swe_task,
    normalize_swe_task,
    normalize_terminal_tasks,
)


def test_all_real_sources_and_harnesses_are_commit_pinned():
    assert set(PINNED_SOURCES) == {
        "swe-bench-verified",
        "swe-bench-multilingual",
        "multi-swe-bench-flash",
        "terminal-bench-2",
    }
    for source in PINNED_SOURCES.values():
        assert len(source.dataset_revision) == 40
        assert len(source.harness_revision) == 40
        assert source.dataset_revision not in {"main", "latest"}
        assert source.harness_revision not in {"main", "latest"}


def test_swe_normalizer_freezes_identity_without_embedding_gold_patch():
    source = PINNED_SOURCES["swe-bench-verified"]
    row = {
        "repo": "django/django",
        "instance_id": "django__django-12345",
        "base_commit": "a1b2c3d4",
        "problem_statement": "Fix the behavior.",
        "patch": "gold must not enter the task contract",
        "test_patch": "hidden tests",
    }

    task = normalize_swe_task(source, row, row_index=7)
    contract = task.to_dict()

    assert task.instance_id == "django__django-12345"
    assert source.dataset_revision in task.instruction_ref
    assert source.harness_revision in task.grader_ref
    assert "gold must not enter" not in str(contract)
    assert "https://github.com/django/django/issues/12345" in task.overlap_keys


def test_multi_swe_and_terminal_normalizers_produce_executable_contracts():
    multi_source = PINNED_SOURCES["multi-swe-bench-flash"]
    multi = normalize_multi_swe_task(
        multi_source,
        {
            "org": "rust-lang",
            "repo": "cargo",
            "number": 42,
            "instance_id": "rust-lang__cargo-42",
            "language": "rust",
            "base": {"sha": "deadbeef"},
            "body": "fix it",
            "fix_patch": "gold",
            "test_patch": "tests",
        },
        row_index=0,
    )
    terminal_source = PINNED_SOURCES["terminal-bench-2"]
    tasks = normalize_terminal_tasks(
        terminal_source,
        [
            "alpha/instruction.md",
            "alpha/task.toml",
            "alpha/environment/Dockerfile",
            "alpha/tests/test.sh",
            "beta/instruction.md",
            "beta/task.toml",
            "beta/environment/Dockerfile",
            "beta/tests/test.sh",
        ],
    )

    assert multi.language == "rust"
    assert multi.environment_ref.startswith("multi-swe://")
    assert len(tasks) == 2
    assert all(task.grader_ref.endswith("/tests") for task in tasks)

    swe_command = build_execution_command(
        PINNED_SOURCES["swe-bench-verified"],
        predictions_path="/tmp/predictions.jsonl",
        run_id="round-001-baseline",
        instance_ids=("django__django-12345",),
    )
    terminal_command = build_execution_command(
        terminal_source,
        predictions_path="/tmp/ignored.jsonl",
        run_id="round-001-baseline",
        instance_ids=("alpha",),
        model="gpt-5.6-sol",
        reasoning="low",
        arm="baseline",
        agent_program_sha256="a" * 64,
        baseline_contract_sha256="b" * 64,
    )

    assert swe_command[1:3] == ("-m", "swebench.harness.run_evaluation")
    assert (
        PINNED_SOURCES["swe-bench-verified"].dataset_revision
        in swe_command[swe_command.index("--dataset_name") + 1]
    )
    assert terminal_command[:3] == ("harbor", "run", "--path")
    assert terminal_source.dataset_revision in terminal_command[3]
    assert "--task-name" in terminal_command
    assert "evolve_jlens_harbor:FrozenCodexAgent" in terminal_command
    assert terminal_command[terminal_command.index("--model") + 1] == "gpt-5.6-sol"
    assert "arm=baseline" in terminal_command
    assert "reasoning_effort=low" in terminal_command
