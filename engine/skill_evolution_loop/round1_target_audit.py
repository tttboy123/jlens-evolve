"""Freeze evaluator-only target coverage evidence for Round 1."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .eval_manifest import EvaluationTaskSet
from .target_audit import GoldPatchReference, audit_target_coverage

_DATASETS = {
    "swe-bench-verified": (
        Path("harness-inputs/swe-bench-verified.jsonl"),
        "patch",
    ),
    "swe-bench-multilingual": (
        Path("harness-inputs/swe-bench-multilingual.jsonl"),
        "patch",
    ),
    "multi-swe-bench-flash": (
        Path("inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"),
        "fix_patch",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_taskset(path: Path) -> tuple[EvaluationTaskSet, str]:
    resolved = path.resolve()
    try:
        raw = resolved.read_bytes()
        payload = json.loads(raw)
        taskset = EvaluationTaskSet.from_dict(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ContractError) as exc:
        raise ContractError("Round 1 target audit TaskSet is unreadable") from exc
    taskset.validate()
    return taskset, hashlib.sha256(raw).hexdigest()


def _load_references(
    pool_root: Path,
) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    indexed: dict[tuple[str, str], str] = {}
    dataset_hashes: dict[str, str] = {}
    for benchmark_id, (relative, answer_field) in _DATASETS.items():
        path = (pool_root / relative).resolve()
        if not path.is_file():
            raise ContractError("Round 1 target audit dataset is missing")
        dataset_hashes[benchmark_id] = _sha256(path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ContractError("Round 1 target audit dataset is unreadable") from exc
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            patch = row.get(answer_field)
            key = (benchmark_id, instance_id)
            if (
                not isinstance(instance_id, str)
                or not isinstance(patch, str)
                or not patch.strip()
                or key in indexed
            ):
                raise ContractError("Round 1 target audit reference is invalid")
            indexed[key] = patch
    return indexed, dataset_hashes


def _freeze(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 target audit evidence is unreadable") from exc
        if existing != payload:
            raise ContractError("frozen Round 1 target audit does not match replay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def audit_round1_targets(
    *,
    taskset_path: Path,
    pool_root: Path,
    output_path: Path,
    mechanism_max_files: int = 1,
    routes_path: Path | None = None,
) -> dict[str, Any]:
    """Audit selected targets without exposing answer content to the Student.

    The benchmark answer field is read only in this evaluator boundary. The
    frozen report contains reference hashes and implementation paths, never
    patch hunks or replacement content.
    """

    taskset, taskset_file_sha256 = _load_taskset(taskset_path)
    indexed, dataset_hashes = _load_references(pool_root.resolve())
    capacities: dict[str, int] = {}
    routes_file_sha256: str | None = None
    if routes_path is not None:
        route_path = routes_path.resolve()
        try:
            route_raw = route_path.read_bytes()
            route_wrapper = json.loads(route_raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 mechanism routes are unreadable") from exc
        route_content = {
            key: value
            for key, value in route_wrapper.items()
            if key != "evidence_sha256"
        }
        if route_wrapper.get("evidence_sha256") != sha256_json(route_content):
            raise ContractError("Round 1 mechanism routes were tampered")
        routes = route_wrapper.get("routes")
        if not isinstance(routes, dict) or set(routes) != {
            task.task_id for task in taskset.tasks
        }:
            raise ContractError("Round 1 mechanism routes do not match TaskSet")
        for task_id, mechanism in routes.items():
            if mechanism not in {"operator", "span"}:
                raise ContractError("Round 1 mechanism route is invalid")
            capacities[task_id] = 1 if mechanism == "operator" else 2
        routes_file_sha256 = hashlib.sha256(route_raw).hexdigest()

    with tempfile.TemporaryDirectory(prefix="jlens-round1-target-audit-") as temp:
        reference_root = Path(temp)
        references: list[GoldPatchReference] = []
        for task in taskset.tasks:
            key = (task.benchmark_id, task.instance_id)
            patch = indexed.get(key)
            if patch is None:
                raise ContractError(
                    f"Round 1 target audit reference is missing: {task.task_id}"
                )
            patch_path = reference_root / f"{task.task_id}.patch"
            patch_path.write_text(patch, encoding="utf-8")
            references.append(
                GoldPatchReference(task_id=task.task_id, patch_path=patch_path)
            )
        audit = audit_target_coverage(
            taskset,
            references,
            mechanism_max_files=mechanism_max_files,
            mechanism_max_files_by_task=capacities,
        )

    content = {
        "schema_version": 1,
        "evaluator_only": True,
        "student_visible": False,
        "answer_fields_read_by_evaluator": ["fix_patch", "patch"],
        "answer_content_serialized": False,
        "taskset_file_sha256": taskset_file_sha256,
        "dataset_file_sha256": dict(sorted(dataset_hashes.items())),
        "mechanism_routes_file_sha256": routes_file_sha256,
        "audit": audit.to_dict(),
    }
    payload = {**content, "evidence_sha256": sha256_json(content)}
    _freeze(output_path.resolve(), payload)
    return payload
