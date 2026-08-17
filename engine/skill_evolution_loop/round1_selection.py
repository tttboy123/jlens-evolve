"""Freeze the leakage-free 60-task Round 1 identity and cohort split."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json


def _verified(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.resolve().read_bytes()
        report = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is unreadable") from exc
    if not isinstance(report, dict):
        raise ContractError(f"{label} must be an object")
    integrity = report.get("evidence_sha256")
    content = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if integrity != sha256_json(content):
        raise ContractError(f"{label} evidence was tampered")
    return raw, report


def _task_id(instance_id: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9._-]+", "-", instance_id).strip("-.")
    if not rendered:
        raise ContractError("Round 1 instance id is unsafe")
    return f"round1-{rendered}"[:255]


def freeze_round1_selection(
    *,
    readiness_path: Path,
    task_pool_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Bind a ready scale preflight to an exact balanced 30+30 task selection."""

    readiness_raw, readiness = _verified(
        readiness_path, label="Round 1 scale readiness"
    )
    if (
        readiness.get("status") != "ready"
        or readiness.get("target_tasks") != 60
        or len(readiness.get("selected_task_uids", [])) != 60
        or readiness.get("promotion_partition_opened") is not False
        or readiness.get("final_sealed_partition_opened") is not False
    ):
        raise ContractError("Round 1 scale readiness has not unlocked selection")
    pool = task_pool_path.resolve()
    try:
        pool_raw = pool.read_bytes()
        task_pool = json.loads(pool_raw)
        records = task_pool["records"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
        raise ContractError("Round 1 task pool is unreadable") from exc
    if hashlib.sha256(pool_raw).hexdigest() != readiness.get("source_task_pool_sha256"):
        raise ContractError("Round 1 task pool does not match readiness")
    selected_uids = readiness["selected_task_uids"]
    indexed = {row.get("task_uid"): row for row in records if isinstance(row, dict)}
    if len(indexed) != len(records) or any(uid not in indexed for uid in selected_uids):
        raise ContractError("Round 1 selected task identity is ambiguous")
    opened = set(readiness.get("runtime_opened_instance_ids", []))
    selected = [indexed[uid] for uid in selected_uids]
    for row in selected:
        contract = row.get("task_contract")
        if (
            row.get("assigned_partition") != "search"
            or row.get("state") != "unopened"
            or row.get("instance_id") in opened
            or not isinstance(contract, dict)
        ):
            raise ContractError("Round 1 selected task crossed a frozen boundary")

    ranked = sorted(
        selected,
        key=lambda row: hashlib.sha256(row["task_uid"].encode()).hexdigest(),
    )
    feedback_uids = {row["task_uid"] for row in ranked[:30]}
    rows = []
    for row in selected:
        contract = row["task_contract"]
        language = contract["language"]
        rows.append(
            {
                "task_id": _task_id(row["instance_id"]),
                "task_uid": row["task_uid"],
                "instance_id": row["instance_id"],
                "benchmark_id": row["benchmark_id"],
                "benchmark_revision": contract["benchmark_revision"],
                "base_commit": contract["base_commit"],
                "repo": contract["repo"],
                "language": language,
                "cohort": (
                    "feedback" if row["task_uid"] in feedback_uids else "holdout"
                ),
                "mechanism": "operator" if language == "python" else "span",
                "identity_fingerprint": row["identity_fingerprint"],
                "content_sha256": contract["content_sha256"],
            }
        )
    if len({row["task_id"] for row in rows}) != 60:
        raise ContractError("Round 1 task ids are not unique")

    content = {
        "schema_version": 1,
        "status": "frozen",
        "selection_id": "round1-search-only-60-v1",
        "task_count": len(rows),
        "cohort_counts": dict(sorted(Counter(row["cohort"] for row in rows).items())),
        "benchmark_counts": dict(
            sorted(Counter(row["benchmark_id"] for row in rows).items())
        ),
        "language_counts": dict(
            sorted(Counter(row["language"] for row in rows).items())
        ),
        "mechanism_counts": dict(
            sorted(Counter(row["mechanism"] for row in rows).items())
        ),
        "tasks": rows,
        "source_readiness": str(readiness_path.resolve()),
        "source_readiness_file_sha256": hashlib.sha256(readiness_raw).hexdigest(),
        "source_readiness_evidence_sha256": readiness["evidence_sha256"],
        "source_task_pool": str(pool),
        "source_task_pool_sha256": hashlib.sha256(pool_raw).hexdigest(),
        "runtime_opened_instance_ids": sorted(opened),
        "promotion_partition_opened": False,
        "final_sealed_partition_opened": False,
        "gold_fields_included": False,
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 selection evidence is unreadable") from exc
        if existing != report:
            raise ContractError("frozen Round 1 selection does not match replay")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report
