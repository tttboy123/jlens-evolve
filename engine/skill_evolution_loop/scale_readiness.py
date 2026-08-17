"""Evidence-backed readiness gate for the 60-task Round 1 expansion."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json

_PATCH_BENCHMARKS = frozenset(
    {
        "swe-bench-verified",
        "swe-bench-multilingual",
        "multi-swe-bench-flash",
    }
)


def freeze_round1_scale_readiness(
    *,
    task_pool_path: Path,
    output_path: Path,
    target_tasks: int = 60,
    partition: str = "search",
    renderer_languages: frozenset[str] = frozenset({"python"}),
    opened_taskset_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Freeze whether the current renderer/native stack can run 60 search tasks."""

    if type(target_tasks) is not int or target_tasks < 1:
        raise ContractError("Round 1 target_tasks must be a positive integer")
    if not partition or not renderer_languages:
        raise ContractError("Round 1 scale policy is empty")
    source = task_pool_path.resolve()
    try:
        raw = source.read_bytes()
        task_pool = json.loads(raw)
        records = task_pool["records"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        raise ContractError("Round 1 task pool is unreadable") from exc
    if not isinstance(records, list):
        raise ContractError("Round 1 task pool records are invalid")

    opened_instance_ids: set[str] = set()
    opened_sources: list[dict[str, str]] = []
    for opened_path in sorted(path.resolve() for path in opened_taskset_paths):
        try:
            opened_raw = opened_path.read_bytes()
            opened = json.loads(opened_raw)
            tasks = opened["tasks"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            raise ContractError("opened Round 1 taskset is unreadable") from exc
        if not isinstance(tasks, list):
            raise ContractError("opened Round 1 taskset tasks are invalid")
        for task in tasks:
            instance_id = task.get("instance_id") if isinstance(task, dict) else None
            if not isinstance(instance_id, str) or not instance_id:
                raise ContractError("opened Round 1 instance identity is invalid")
            opened_instance_ids.add(instance_id)
        opened_sources.append(
            {
                "path": str(opened_path),
                "sha256": hashlib.sha256(opened_raw).hexdigest(),
            }
        )

    rows: list[dict[str, Any]] = []
    excluded_opened: list[dict[str, Any]] = []
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("assigned_partition") != partition
        ):
            continue
        contract = record.get("task_contract")
        if not isinstance(contract, dict):
            raise ContractError("Round 1 task contract is invalid")
        benchmark = contract.get("benchmark_id")
        language = contract.get("language")
        task_uid = record.get("task_uid")
        if not all(isinstance(value, str) for value in (benchmark, language, task_uid)):
            raise ContractError("Round 1 task identity is invalid")
        if record.get("state") != "unopened":
            continue
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str):
            raise ContractError("Round 1 instance identity is invalid")
        if instance_id in opened_instance_ids:
            excluded_opened.append(record)
            continue
        rows.append(record)

    native_compatible = [
        row for row in rows if row["task_contract"]["benchmark_id"] in _PATCH_BENCHMARKS
    ]
    fully_compatible = [
        row
        for row in native_compatible
        if row["task_contract"]["language"] in renderer_languages
    ]
    renderer_gap = [row for row in native_compatible if row not in fully_compatible]
    native_gap = [row for row in rows if row not in native_compatible]
    compatible_count = len(fully_compatible)
    missing = max(0, target_tasks - compatible_count)
    status = "ready" if missing == 0 else "blocked"
    selected = fully_compatible[:target_tasks] if status == "ready" else []

    def counts(source_rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(
            sorted(Counter(row["task_contract"][field] for row in source_rows).items())
        )

    content = {
        "schema_version": 1,
        "status": status,
        "partition": partition,
        "target_tasks": target_tasks,
        "unopened_partition_tasks": len(rows),
        "runtime_opened_exclusion_count": len(excluded_opened),
        "runtime_opened_instance_ids": sorted(
            row["instance_id"] for row in excluded_opened
        ),
        "native_patch_compatible_tasks": len(native_compatible),
        "fully_compatible_tasks": compatible_count,
        "additional_renderer_compatible_tasks_required": missing,
        "renderer_languages": sorted(renderer_languages),
        "supported_native_benchmarks": sorted(_PATCH_BENCHMARKS),
        "partition_counts_by_benchmark": counts(rows, "benchmark_id"),
        "partition_counts_by_language": counts(rows, "language"),
        "renderer_gap_counts_by_language": counts(renderer_gap, "language"),
        "native_gap_counts_by_benchmark": counts(native_gap, "benchmark_id"),
        "selected_task_uids": [row["task_uid"] for row in selected],
        "source_task_pool": str(source),
        "source_task_pool_sha256": hashlib.sha256(raw).hexdigest(),
        "source_opened_tasksets": opened_sources,
        "promotion_partition_opened": False,
        "final_sealed_partition_opened": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                "Round 1 scale readiness evidence is unreadable"
            ) from exc
        if existing != report:
            raise ContractError("frozen Round 1 scale readiness does not match replay")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
