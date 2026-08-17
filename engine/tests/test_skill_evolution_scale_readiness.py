from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.scale_readiness import freeze_round1_scale_readiness


def _pool(path: Path) -> Path:
    rows = []
    for number, (benchmark, language) in enumerate(
        [
            ("swe-bench-verified", "python"),
            ("swe-bench-multilingual", "polyglot"),
            ("multi-swe-bench-flash", "rust"),
            ("terminal-bench-2", "shell"),
        ],
        1,
    ):
        rows.append(
            {
                "assigned_partition": "search",
                "state": "unopened",
                "task_uid": f"task-{number}",
                "instance_id": f"instance-{number}",
                "task_contract": {
                    "benchmark_id": benchmark,
                    "language": language,
                },
            }
        )
    path.write_text(json.dumps({"records": rows}), encoding="utf-8")
    return path


def test_scale_readiness_blocks_without_opening_sealed_partitions(
    tmp_path: Path,
) -> None:
    result = freeze_round1_scale_readiness(
        task_pool_path=_pool(tmp_path / "pool.json"),
        output_path=tmp_path / "readiness.json",
        target_tasks=2,
    )

    assert result["status"] == "blocked"
    assert result["fully_compatible_tasks"] == 1
    assert result["additional_renderer_compatible_tasks_required"] == 1
    assert result["promotion_partition_opened"] is False
    assert result["final_sealed_partition_opened"] is False
    assert result["selected_task_uids"] == []


def test_scale_readiness_selects_only_supported_search_tasks(tmp_path: Path) -> None:
    result = freeze_round1_scale_readiness(
        task_pool_path=_pool(tmp_path / "pool.json"),
        output_path=tmp_path / "readiness.json",
        target_tasks=2,
        renderer_languages=frozenset({"python", "rust"}),
    )

    assert result["status"] == "ready"
    assert result["selected_task_uids"] == ["task-1", "task-3"]
    assert result["native_gap_counts_by_benchmark"] == {"terminal-bench-2": 1}


def test_scale_readiness_is_append_only(tmp_path: Path) -> None:
    pool = _pool(tmp_path / "pool.json")
    output = tmp_path / "readiness.json"
    freeze_round1_scale_readiness(
        task_pool_path=pool,
        output_path=output,
        target_tasks=1,
    )

    with pytest.raises(ContractError, match="does not match replay"):
        freeze_round1_scale_readiness(
            task_pool_path=pool,
            output_path=output,
            target_tasks=2,
        )


def test_scale_readiness_excludes_instances_opened_after_pool_freeze(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path / "pool.json")
    opened = tmp_path / "opened.json"
    opened.write_text(
        json.dumps({"tasks": [{"instance_id": "missing"}]}), encoding="utf-8"
    )
    raw_pool = json.loads(pool.read_text(encoding="utf-8"))
    raw_pool["records"][0]["instance_id"] = "already-opened"
    pool.write_text(json.dumps(raw_pool), encoding="utf-8")
    opened.write_text(
        json.dumps({"tasks": [{"instance_id": "already-opened"}]}),
        encoding="utf-8",
    )

    result = freeze_round1_scale_readiness(
        task_pool_path=pool,
        output_path=tmp_path / "readiness.json",
        target_tasks=1,
        opened_taskset_paths=(opened,),
    )

    assert result["status"] == "blocked"
    assert result["runtime_opened_exclusion_count"] == 1
    assert result["runtime_opened_instance_ids"] == ["already-opened"]
    assert result["fully_compatible_tasks"] == 0
