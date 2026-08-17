from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_adapters import (
    BenchmarkContractError,
    BenchmarkRegistry,
    BenchmarkTask,
    StaticBenchmarkAdapter,
    TaskPool,
)


def _task(
    benchmark_id: str,
    index: int,
    *,
    overlap: str | None = None,
    revision: str = "rev-2026-08-03-sha256",
) -> BenchmarkTask:
    digest = f"{index:064x}"[-64:]
    return BenchmarkTask(
        benchmark_id=benchmark_id,
        benchmark_revision=revision,
        instance_id=f"task-{index}",
        task_family="repo_issue",
        language="python",
        repo=f"example/repo-{index % 41}",
        base_commit=f"commit-{index}",
        environment_ref=f"docker://example/task-{index}@sha256:{digest}",
        grader_ref=f"grader://tests/{index}@{digest}",
        instruction_ref=f"manifest://{benchmark_id}/task-{index}",
        source_url=f"https://example.test/{benchmark_id}/{index}",
        license_id="test-only",
        overlap_keys=(overlap or f"issue://{benchmark_id}/{index}",),
        content_sha256=digest,
    )


def test_registry_requires_pinned_unique_executable_adapters():
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id="swe-verified",
            revision="frozen-sha-1",
            executable=True,
            tasks=(_task("swe-verified", 1),),
        )
    )

    with pytest.raises(BenchmarkContractError, match="pinned"):
        registry.register(
            StaticBenchmarkAdapter(
                adapter_id="moving",
                revision="latest",
                executable=True,
                tasks=(_task("moving", 2, revision="latest"),),
            )
        )
    with pytest.raises(BenchmarkContractError, match="already registered"):
        registry.register(
            StaticBenchmarkAdapter(
                adapter_id="swe-verified",
                revision="frozen-sha-2",
                executable=True,
                tasks=(_task("swe-verified", 3),),
            )
        )


def test_pool_deduplicates_cross_dataset_overlap_and_freezes_300_tasks():
    registry = BenchmarkRegistry()
    adapter_ids = (
        "swe-verified",
        "swe-multilingual",
        "multi-swe-flash",
        "terminal-bench",
    )
    next_index = 0
    for adapter_id in adapter_ids:
        tasks = []
        for _ in range(80):
            tasks.append(_task(adapter_id, next_index))
            next_index += 1
        registry.register(
            StaticBenchmarkAdapter(
                adapter_id=adapter_id,
                revision=f"{adapter_id}-frozen-v1",
                executable=True,
                tasks=tuple(tasks),
            )
        )
    duplicate = _task(
        "terminal-bench",
        999,
        overlap="issue://swe-verified/0",
    )
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id="duplicate-probe",
            revision="duplicate-probe-frozen-v1",
            executable=True,
            tasks=(duplicate,),
        )
    )

    pool = TaskPool.build(
        registry=registry,
        seed_material="continuous-ab-v1",
        target_count=300,
        promotion_count=80,
        final_sealed_count=60,
    )

    assert len(pool.records) == 300
    assert len(pool.adapters) >= 4
    assert len(pool.duplicates) == 1
    assert set(adapter_ids).issubset({row.benchmark_id for row in pool.records})
    assert sum(row.assigned_partition == "final_sealed" for row in pool.records) == 60
    assert sum(row.assigned_partition == "promotion" for row in pool.records) == 80
    assert all(row.state == "unopened" for row in pool.records)
    assert all(
        row.task_contract["task_uid"] == row.task_uid
        and row.task_contract["environment_ref"]
        and row.task_contract["grader_ref"]
        for row in pool.records
    )


def test_task_pool_lifecycle_is_monotonic_and_persistent(tmp_path: Path):
    registry = BenchmarkRegistry()
    registry.register(
        StaticBenchmarkAdapter(
            adapter_id="fixture",
            revision="fixture-frozen-v1",
            executable=True,
            tasks=tuple(_task("fixture", index) for index in range(70)),
        )
    )
    pool = TaskPool.build(
        registry=registry,
        seed_material="lifecycle-v1",
        target_count=70,
        promotion_count=5,
        final_sealed_count=60,
    )
    path = tmp_path / "pool.json"
    pool.save(path)
    restored = TaskPool.load(path)
    search_task = next(
        row for row in restored.records if row.assigned_partition == "search"
    )

    restored.claim(search_task.task_uid, "search")
    restored.retire(search_task.task_uid)

    with pytest.raises(BenchmarkContractError, match="retired"):
        restored.claim(search_task.task_uid, "search")
    final_task = next(
        row for row in restored.records if row.assigned_partition == "final_sealed"
    )
    with pytest.raises(BenchmarkContractError, match="assigned partition"):
        restored.claim(final_task.task_uid, "search")


def test_pool_honors_predeclared_adapter_and_partition_quotas():
    registry = BenchmarkRegistry()
    adapter_ids = ("verified", "multilingual", "multi-swe", "terminal")
    next_index = 0
    for adapter_id in adapter_ids:
        tasks = tuple(
            _task(adapter_id, index) for index in range(next_index, next_index + 80)
        )
        next_index += 80
        registry.register(
            StaticBenchmarkAdapter(
                adapter_id=adapter_id,
                revision=f"{adapter_id}-frozen-v1",
                executable=True,
                tasks=tasks,
            )
        )
    quotas = {
        adapter_id: {"search": 40, "promotion": 20, "final_sealed": 15}
        for adapter_id in adapter_ids
    }

    pool = TaskPool.build(
        registry=registry,
        seed_material="stratified-v1",
        target_count=300,
        promotion_count=80,
        final_sealed_count=60,
        partition_quotas=quotas,
        retired_instance_ids=frozenset({"task-0"}),
    )

    for adapter_id in adapter_ids:
        for partition, expected in quotas[adapter_id].items():
            assert (
                sum(
                    row.benchmark_id == adapter_id
                    and row.assigned_partition == partition
                    for row in pool.records
                )
                == expected
            )
    search_stream = [
        row.benchmark_id for row in pool.records if row.assigned_partition == "search"
    ]
    assert search_stream[:8] == sorted(adapter_ids) * 2
    assert all(row.instance_id != "task-0" for row in pool.records)
    assert pool.retired_exclusions == [
        {
            "benchmark_id": "verified",
            "instance_id": "task-0",
            "reason": "prior_use_retired",
        }
    ]
