"""Freeze the full unopened search candidate universe before qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .round1_selection import _task_id

_PATCH_BENCHMARKS = frozenset(
    {"swe-bench-verified", "swe-bench-multilingual", "multi-swe-bench-flash"}
)
_EXPLICIT_LANGUAGES = frozenset(
    {
        "c",
        "c++",
        "go",
        "java",
        "javascript",
        "php",
        "python",
        "ruby",
        "rust",
        "typescript",
    }
)


def freeze_round1_candidates(
    *,
    task_pool_path: Path,
    opened_taskset_paths: tuple[Path, ...],
    output_path: Path,
    partition: str = "search",
    candidate_limit: int | None = None,
) -> dict[str, Any]:
    """Freeze native-compatible search tasks without consulting answer fields."""

    if partition not in {"search", "promotion"}:
        raise ContractError("Round 1 candidate partition is invalid")
    if candidate_limit is not None and (
        type(candidate_limit) is not int or candidate_limit < 1
    ):
        raise ContractError("Round 1 candidate limit is invalid")
    source = task_pool_path.resolve()
    try:
        raw = source.read_bytes()
        records = json.loads(raw)["records"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        raise ContractError("Round 1 candidate task pool is unreadable") from exc
    if not isinstance(records, list):
        raise ContractError("Round 1 candidate task pool records are invalid")
    opened: set[str] = set()
    opened_sources: list[dict[str, str]] = []
    for path in sorted(row.resolve() for row in opened_taskset_paths):
        try:
            opened_raw = path.read_bytes()
            tasks = json.loads(opened_raw)["tasks"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            raise ContractError("Round 1 opened TaskSet is unreadable") from exc
        if not isinstance(tasks, list):
            raise ContractError("Round 1 opened TaskSet tasks are invalid")
        for task in tasks:
            instance_id = task.get("instance_id") if isinstance(task, dict) else None
            if not isinstance(instance_id, str):
                raise ContractError("Round 1 opened task identity is invalid")
            opened.add(instance_id)
        opened_sources.append(
            {"path": str(path), "sha256": hashlib.sha256(opened_raw).hexdigest()}
        )

    tasks: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("Round 1 candidate record is invalid")
        contract = record.get("task_contract")
        instance_id = record.get("instance_id")
        if (
            record.get("assigned_partition") != partition
            or record.get("state") != "unopened"
            or not isinstance(contract, dict)
            or contract.get("benchmark_id") not in _PATCH_BENCHMARKS
            or instance_id in opened
        ):
            continue
        language = contract.get("language")
        if language not in _EXPLICIT_LANGUAGES and language != "polyglot":
            continue
        values = (
            record.get("task_uid"),
            instance_id,
            contract.get("benchmark_id"),
            contract.get("base_commit"),
            contract.get("repo"),
            language,
            record.get("identity_fingerprint"),
            contract.get("content_sha256"),
        )
        if not all(isinstance(value, str) and value for value in values):
            raise ContractError("Round 1 candidate identity is invalid")
        tasks.append(
            {
                "task_id": _task_id(instance_id),
                "task_uid": record["task_uid"],
                "instance_id": instance_id,
                "benchmark_id": contract["benchmark_id"],
                "benchmark_revision": contract["benchmark_revision"],
                "base_commit": contract["base_commit"],
                "repo": contract["repo"],
                "declared_language": language,
                "identity_fingerprint": record["identity_fingerprint"],
                "content_sha256": contract["content_sha256"],
            }
        )
    tasks.sort(key=lambda row: row["task_uid"])
    if candidate_limit is not None:
        tasks = sorted(
            tasks,
            key=lambda row: hashlib.sha256(row["task_uid"].encode()).hexdigest(),
        )[:candidate_limit]
        tasks.sort(key=lambda row: row["task_uid"])
    if not tasks or len({row["task_uid"] for row in tasks}) != len(tasks):
        raise ContractError("Round 1 candidate universe is empty or ambiguous")
    content = {
        "schema_version": 1,
        "status": "frozen",
        "selection_id": (
            "round1-search-candidate-universe-v1"
            if partition == "search" and candidate_limit is None
            else f"round1-{partition}-candidate-{len(tasks)}-v2"
        ),
        "task_count": len(tasks),
        "tasks": tasks,
        "source_task_pool": str(source),
        "source_task_pool_sha256": hashlib.sha256(raw).hexdigest(),
        "source_opened_tasksets": opened_sources,
        "runtime_opened_instance_ids": sorted(opened),
        "promotion_partition_opened": partition == "promotion",
        "final_sealed_partition_opened": False,
        "answer_fields_read": False,
        "network_calls_performed": False,
    }
    if partition != "search" or candidate_limit is not None:
        content["source_partition"] = partition
        content["candidate_limit"] = candidate_limit
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 candidate evidence is unreadable") from exc
        if existing != report:
            raise ContractError("frozen Round 1 candidates do not match replay")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
