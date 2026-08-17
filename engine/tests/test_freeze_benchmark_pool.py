from __future__ import annotations

from freeze_benchmark_pool import assemble_registry


def test_snapshot_assembly_builds_four_native_executable_adapters():
    swe_row = {
        "repo": "django/django",
        "instance_id": "django__django-1",
        "base_commit": "abc12345",
        "problem_statement": "fix",
    }
    multi_row = {
        "org": "rust-lang",
        "repo": "cargo",
        "number": 2,
        "instance_id": "rust-lang__cargo-2",
        "language": "rust",
        "base": {"sha": "def67890"},
        "body": "fix",
    }
    terminal_paths = [
        "terminal-task/instruction.md",
        "terminal-task/task.toml",
        "terminal-task/environment/Dockerfile",
        "terminal-task/tests/test.sh",
    ]

    registry = assemble_registry(
        verified_rows=[swe_row],
        multilingual_rows=[{**swe_row, "instance_id": "django__django-3"}],
        multi_swe_rows=[multi_row],
        terminal_paths=terminal_paths,
    )

    assert registry.adapter_ids == (
        "multi-swe-bench-flash",
        "swe-bench-multilingual",
        "swe-bench-verified",
        "terminal-bench-2",
    )
    assert len(registry.tasks()) == 4
