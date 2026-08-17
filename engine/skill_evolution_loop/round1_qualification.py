"""Evaluator-only qualification of the full Round 1 candidate universe."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .round1_selection import _verified
from .round1_taskset import (
    _EXCLUDED_PARTS,
    _LANGUAGE_EXTENSIONS,
    _checkout,
    _is_test_path,
    _multi_instruction,
    _remove_checkout,
    select_round1_targets,
)
from .target_audit import reference_implementation_targets

_DATASETS = {
    "swe-bench-verified": (
        Path("harness-inputs/swe-bench-verified.jsonl"),
        "problem_statement",
        "patch",
    ),
    "swe-bench-multilingual": (
        Path("harness-inputs/swe-bench-multilingual.jsonl"),
        "problem_statement",
        "patch",
    ),
    "multi-swe-bench-flash": (
        Path("inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"),
        "multi",
        "fix_patch",
    ),
}


def infer_repo_language(checkout: Path) -> str | None:
    """Infer one supported language from source bytes, never from a patch."""

    totals: Counter[str] = Counter()
    for path in sorted(checkout.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(checkout)
        if _is_test_path(relative) or {
            part.lower() for part in relative.parts
        }.intersection(_EXCLUDED_PARTS):
            continue
        suffix = path.suffix.lower()
        try:
            size = min(path.stat().st_size, 300_000)
        except OSError:
            continue
        for language, suffixes in _LANGUAGE_EXTENSIONS.items():
            if suffix in suffixes:
                totals[language] += size
    if not totals:
        return None
    # A .h-only tie is ambiguous; prefer C++ only when C++-specific files exist.
    cpp_specific = sum(
        min(path.stat().st_size, 300_000)
        for path in checkout.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".cc", ".cpp", ".cxx", ".hh", ".hpp"}
    )
    if cpp_specific:
        totals["c++"] += cpp_specific
    return sorted(totals, key=lambda language: (-totals[language], language))[0]


def _dataset_index(
    pool_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for benchmark, (relative, _instruction_field, _patch_field) in _DATASETS.items():
        path = (pool_root / relative).resolve()
        if not path.is_file():
            raise ContractError("Round 1 qualification dataset is missing")
        raw = path.read_bytes()
        hashes[benchmark] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            key = (benchmark, instance_id)
            if not isinstance(instance_id, str) or key in indexed:
                raise ContractError("Round 1 qualification dataset is ambiguous")
            indexed[key] = row
    return indexed, hashes


def _instruction(benchmark: str, row: dict[str, Any]) -> str:
    _relative, field, _patch_field = _DATASETS[benchmark]
    value = _multi_instruction(row) if field == "multi" else str(row.get(field, ""))
    if not value.strip():
        raise ContractError("Round 1 qualification instruction is empty")
    return value


def _reference_patch(benchmark: str, row: dict[str, Any]) -> str:
    _relative, _field, patch_field = _DATASETS[benchmark]
    value = row.get(patch_field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError("Round 1 qualification reference patch is empty")
    return value


def _freeze(path: Path, payload: dict[str, Any], label: str) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"{label} is unreadable") from exc
        if existing != payload:
            raise ContractError(f"frozen {label} does not match replay")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _qualification_cell_evidence_fingerprint(rows: list[dict[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "task_id": row["task_id"],
                "evidence_sha256": row["evidence_sha256"],
            }
            for row in sorted(rows, key=lambda item: item["task_id"])
        ]
    )


def qualify_round1_candidates(
    *,
    candidate_path: Path,
    source_summary_path: Path,
    pool_root: Path,
    evidence_root: Path,
    workspace_root: Path,
    max_candidates: int = 32,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """Resume candidate qualification while keeping answer content evaluator-only."""

    # Qualification is evaluator-only and never serialized into the Student
    # prompt. A wider admission scan is therefore safe and prevents a top-32
    # retrieval miss from being confused with an action-capacity failure.
    if type(max_candidates) is not int or not 1 <= max_candidates <= 64:
        raise ContractError("Round 1 qualification candidate limit is invalid")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ContractError("Round 1 qualification max_tasks is invalid")
    candidate_raw, candidates = _verified(
        candidate_path, label="Round 1 candidate universe"
    )
    if candidates.get("status") != "frozen" or candidates.get("task_count", 0) < 1:
        raise ContractError("Round 1 candidate universe is not frozen")
    source_raw, source_summary = _verified_source_summary(source_summary_path)
    if source_summary.get("status") != "complete":
        raise ContractError("Round 1 candidate sources are incomplete")
    cache = Path(source_summary["cache_root"]).resolve()
    datasets, dataset_hashes = _dataset_index(pool_root.resolve())
    evidence = evidence_root.resolve()
    workspace = workspace_root.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    executed = 0
    for candidate in candidates["tasks"]:
        task_id = candidate["task_id"]
        cell = evidence / "cells" / task_id / "QUALIFICATION.json"
        if cell.is_file():
            wrapper = json.loads(cell.read_text(encoding="utf-8"))
            content = {
                key: value for key, value in wrapper.items() if key != "evidence_sha256"
            }
            if wrapper.get("evidence_sha256") != sha256_json(content):
                raise ContractError("Round 1 qualification cell was tampered")
            rows.append(wrapper)
            continue
        if max_tasks is not None and executed >= max_tasks:
            continue
        key = (candidate["benchmark_id"], candidate["instance_id"])
        dataset = datasets.get(key)
        if dataset is None:
            raise ContractError("Round 1 qualification dataset row is missing")
        source = cache / candidate["repo"]
        if not (source / ".git").is_dir():
            raise ContractError("Round 1 qualification source is missing")
        with tempfile.TemporaryDirectory(
            prefix=f"{task_id}-", dir=workspace
        ) as temporary:
            checkout = _checkout(
                source, candidate["base_commit"], Path(temporary) / "checkout"
            )
            try:
                language = candidate["declared_language"]
                if language == "polyglot":
                    language = infer_repo_language(checkout)
                targets = (
                    select_round1_targets(
                        checkout,
                        _instruction(candidate["benchmark_id"], dataset),
                        language,
                        max_targets=max_candidates,
                    )
                    if language in _LANGUAGE_EXTENSIONS
                    else ()
                )
            finally:
                _remove_checkout(source, checkout)
        patch = _reference_patch(candidate["benchmark_id"], dataset)
        implementation = reference_implementation_targets(patch)
        if not implementation:
            raise ContractError("Round 1 qualification has no implementation target")
        missing = tuple(sorted(set(implementation) - set(targets)))
        ready = bool(
            language in _LANGUAGE_EXTENSIONS
            and not missing
            and len(implementation) == 1
        )
        content = {
            "schema_version": 1,
            "task_id": task_id,
            "task_uid": candidate["task_uid"],
            "instance_id": candidate["instance_id"],
            "benchmark_id": candidate["benchmark_id"],
            "repo": candidate["repo"],
            "base_commit": candidate["base_commit"],
            "declared_language": candidate["declared_language"],
            "resolved_language": language,
            "retrieval_limit": max_candidates,
            "retrieved_targets": list(targets),
            "reference_patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            "implementation_targets": list(implementation),
            "missing_targets": list(missing),
            "within_single_file_capacity": len(implementation) == 1,
            "ready": ready,
            "evaluator_only": True,
            "student_visible": False,
            "answer_content_serialized": False,
            "network_calls_performed": True,
        }
        wrapper = {**content, "evidence_sha256": sha256_json(content)}
        _freeze(cell, wrapper, "Round 1 qualification cell")
        rows.append(wrapper)
        executed += 1
    planned = candidates["task_count"]
    status = "complete" if len(rows) == planned else "partial"
    ready_rows = [row for row in rows if row["ready"]]
    cell_evidence_fingerprint = _qualification_cell_evidence_fingerprint(rows)
    content = {
        "schema_version": 1,
        "status": status,
        "planned_tasks": planned,
        "completed_tasks": len(rows),
        "ready_tasks": len(ready_rows),
        "ready_task_uids": sorted(row["task_uid"] for row in ready_rows),
        "cell_evidence_fingerprint": cell_evidence_fingerprint,
        "resolved_language_counts": dict(
            sorted(Counter(str(row["resolved_language"]) for row in rows).items())
        ),
        "failure_counts": {
            "unsupported_language": sum(
                row["resolved_language"] not in _LANGUAGE_EXTENSIONS for row in rows
            ),
            "retrieval_miss": sum(bool(row["missing_targets"]) for row in rows),
            "multi_file": sum(not row["within_single_file_capacity"] for row in rows),
        },
        "candidate_file_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "candidate_evidence_sha256": candidates["evidence_sha256"],
        "source_summary_file_sha256": hashlib.sha256(source_raw).hexdigest(),
        "source_summary_sha256": source_summary["summary_sha256"],
        "dataset_file_sha256": dict(sorted(dataset_hashes.items())),
        "retrieval_limit": max_candidates,
        "evaluator_only": True,
        "student_visible": False,
        "answer_content_serialized": False,
        "promotion_partition_opened": candidates.get(
            "promotion_partition_opened", False
        ),
        "final_sealed_partition_opened": candidates.get(
            "final_sealed_partition_opened", False
        ),
        "network_calls_performed": bool(rows),
    }
    summary = {**content, "summary_sha256": sha256_json(content)}
    name = "SUMMARY.json" if status == "complete" else "PROGRESS.json"
    if status == "complete":
        _freeze(evidence / name, summary, "Round 1 qualification summary")
    else:
        (evidence / name).write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def _verified_source_summary(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.resolve().read_bytes()
        summary = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("Round 1 candidate source summary is unreadable") from exc
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != sha256_json(content):
        raise ContractError("Round 1 candidate source summary was tampered")
    return raw, summary
