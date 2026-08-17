from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_adapters import BenchmarkRegistry, StaticBenchmarkAdapter, TaskPool
from benchmark_catalog import (
    PINNED_SOURCES,
    normalize_swe_task,
    normalize_terminal_tasks,
)
from benchmark_execution import ExecutionContractError, materialize_claimed_task
from continuous_ab import BaselineContract
from continuous_ab_service import ContinuousABService


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _baseline() -> BaselineContract:
    return BaselineContract(
        experiment_id="materializer-test",
        agent_program_sha256="a" * 64,
        model="codex",
        reasoning="low",
        token_budget=4096,
        timeout_seconds=1800,
        tools=("shell", "apply_patch"),
        retries=0,
        evaluator_epoch="native-v1",
    )


def _runtime_for_task(tmp_path: Path, task) -> ContinuousABService:
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id=task.benchmark_id,
            revision=task.benchmark_revision,
            executable=True,
            tasks=(task,),
        )
    )
    pool_path = tmp_path / "TASK_POOL.json"
    TaskPool.build(
        registry=registry,
        seed_material="materializer",
        target_count=1,
        promotion_count=0,
        final_sealed_count=0,
    ).save(pool_path)
    return ContinuousABService.initialize(
        tmp_path / "runtime",
        frozen_pool_path=pool_path,
        baseline=_baseline(),
        initial_active_agent_sha256="a" * 64,
    )


def test_claimed_swe_task_materializes_instruction_without_gold(tmp_path: Path):
    row = {
        "repo": "django/django",
        "instance_id": "django__django-12345",
        "base_commit": "deadbeef",
        "problem_statement": "Fix the public behavior.",
        "patch": "SECRET GOLD PATCH",
        "test_patch": "SECRET TEST PATCH",
    }
    source = PINNED_SOURCES["swe-bench-verified"]
    task = normalize_swe_task(source, row, row_index=0)
    service = _runtime_for_task(tmp_path, task)
    planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
    pool_root = tmp_path / "pool"
    input_path = pool_root / "harness-inputs/swe-bench-verified.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    materialized = materialize_claimed_task(
        runtime_root=tmp_path / "runtime",
        round_id=planned["round_id"],
        pool_root=pool_root,
    )

    rendered = json.dumps(materialized, ensure_ascii=False)
    assert materialized["instruction"] == "Fix the public behavior."
    assert materialized["source_content_sha256"] == _canonical_sha(row)
    assert "SECRET GOLD" not in rendered
    assert service.round_state(planned["round_id"])["task_input"]["frozen"] is True


def test_materializer_rejects_source_changed_after_pool_freeze(tmp_path: Path):
    row = {
        "repo": "django/django",
        "instance_id": "django__django-12345",
        "base_commit": "deadbeef",
        "problem_statement": "Original.",
    }
    source = PINNED_SOURCES["swe-bench-verified"]
    task = normalize_swe_task(source, row, row_index=0)
    service = _runtime_for_task(tmp_path, task)
    planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
    pool_root = tmp_path / "pool"
    input_path = pool_root / "harness-inputs/swe-bench-verified.jsonl"
    input_path.parent.mkdir(parents=True)
    input_path.write_text(
        json.dumps({**row, "problem_statement": "Changed."}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ExecutionContractError, match="content hash"):
        materialize_claimed_task(
            runtime_root=tmp_path / "runtime",
            round_id=planned["round_id"],
            pool_root=pool_root,
        )


def test_terminal_instruction_requires_pinned_checkout_head(tmp_path: Path):
    dataset = tmp_path / "terminal-dataset"
    task_dir = dataset / "alpha"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Repair the service.\n")
    (task_dir / "task.toml").write_text("version = '1'\n")
    (task_dir / "environment/Dockerfile").write_text("FROM scratch\n")
    (task_dir / "tests/test.sh").write_text("exit 0\n")
    subprocess.run(["git", "init", "-q"], cwd=dataset, check=True)
    subprocess.run(["git", "add", "."], cwd=dataset, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=dataset,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dataset,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = replace(PINNED_SOURCES["terminal-bench-2"], dataset_revision=revision)
    task = normalize_terminal_tasks(
        source,
        [
            "alpha/instruction.md",
            "alpha/task.toml",
            "alpha/environment/Dockerfile",
            "alpha/tests/test.sh",
        ],
    )[0]
    service = _runtime_for_task(tmp_path, task)
    planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)

    materialized = materialize_claimed_task(
        runtime_root=tmp_path / "runtime",
        round_id=planned["round_id"],
        pool_root=tmp_path / "unused-pool",
        terminal_dataset_root=dataset,
    )
    assert materialized["instruction"] == "Repair the service."

    subprocess.run(
        ["git", "commit", "--allow-empty", "-qm", "drift"],
        cwd=dataset,
        check=False,
    )
    second = tmp_path / "second"
    service = _runtime_for_task(second, task)
    planned = service.plan_round(partition="search", evolved_agent_sha256="b" * 64)
    with pytest.raises(ExecutionContractError, match="checkout revision"):
        materialize_claimed_task(
            runtime_root=second / "runtime",
            round_id=planned["round_id"],
            pool_root=tmp_path / "unused-pool",
            terminal_dataset_root=dataset,
        )
