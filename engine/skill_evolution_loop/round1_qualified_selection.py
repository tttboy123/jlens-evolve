"""Freeze 60 tasks admitted by evaluator-only mechanism qualification."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import ContractError, canonical_json, sha256_json
from .round1_selection import _verified


def freeze_round1_qualified_selection(
    *,
    candidate_path: Path,
    qualification_summary_path: Path,
    qualification_cells_root: Path,
    bundle_qualification_path: Path,
    output_path: Path,
    target_tasks: int = 60,
    feedback_tasks: int | None = None,
) -> dict[str, Any]:
    """Select only tasks expressible by the frozen operator/span mechanisms."""

    if type(target_tasks) is not int or target_tasks < 3:
        raise ContractError("Round 1 qualified target count is invalid")
    if feedback_tasks is None:
        feedback_tasks = target_tasks // 2
    if (
        type(feedback_tasks) is not int
        or feedback_tasks < 0
        or feedback_tasks > target_tasks
    ):
        raise ContractError("Round 1 qualified feedback count is invalid")
    candidate_raw, candidates = _verified(
        candidate_path, label="Round 1 candidate universe"
    )
    summary_raw, summary = _verified_summary(qualification_summary_path)
    if (
        summary.get("status") != "complete"
        or summary.get("planned_tasks") != candidates.get("task_count")
        or summary.get("completed_tasks") != candidates.get("task_count")
        or summary.get("evaluator_only") is not True
        or summary.get("student_visible") is not False
        or summary.get("answer_content_serialized") is not False
    ):
        raise ContractError("Round 1 candidate qualification is incomplete")
    bundle_raw, bundle = _verified(
        bundle_qualification_path, label="Round 1 span bundle qualification"
    )
    if (
        bundle.get("status") != "qualified"
        or bundle.get("max_bundle_files") != 2
        or bundle.get("atomic_apply_required") is not True
    ):
        raise ContractError("Round 1 span bundle mechanism is not qualified")

    indexed = {row["task_uid"]: row for row in candidates["tasks"]}
    admitted: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for task_uid, candidate in indexed.items():
        cell_path = (
            qualification_cells_root.resolve()
            / candidate["task_id"]
            / "QUALIFICATION.json"
        )
        try:
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 qualification cell is unreadable") from exc
        content = {
            key: value for key, value in cell.items() if key != "evidence_sha256"
        }
        if (
            cell.get("evidence_sha256") != sha256_json(content)
            or cell.get("task_uid") != task_uid
        ):
            raise ContractError("Round 1 qualification cell was tampered")
        language = cell.get("resolved_language")
        mechanism = "operator" if language == "python" else "span"
        file_count = len(cell.get("implementation_targets", []))
        capacity = 1 if mechanism == "operator" else 2
        if (
            isinstance(language, str)
            and language in bundle.get("supported_languages", []) + ["python"]
            and not cell.get("missing_targets")
            and 1 <= file_count <= capacity
        ):
            admitted.append((candidate, cell, mechanism))
    if len(admitted) < target_tasks:
        raise ContractError("Round 1 qualified task count is below target")
    admitted.sort(
        key=lambda row: hashlib.sha256(row[0]["task_uid"].encode()).hexdigest()
    )
    selected = admitted[:target_tasks]
    feedback_uids = {row[0]["task_uid"] for row in selected[:feedback_tasks]}
    rows: list[dict[str, Any]] = []
    for candidate, cell, mechanism in selected:
        rows.append(
            {
                "task_id": candidate["task_id"],
                "task_uid": candidate["task_uid"],
                "instance_id": candidate["instance_id"],
                "benchmark_id": candidate["benchmark_id"],
                "benchmark_revision": candidate["benchmark_revision"],
                "base_commit": candidate["base_commit"],
                "repo": candidate["repo"],
                "language": cell["resolved_language"],
                "cohort": (
                    "feedback" if candidate["task_uid"] in feedback_uids else "holdout"
                ),
                "mechanism": mechanism,
                "retrieval_limit": cell["retrieval_limit"],
                "retrieved_targets": cell["retrieved_targets"],
                "identity_fingerprint": candidate["identity_fingerprint"],
                "content_sha256": candidate["content_sha256"],
                "qualification_cell_evidence_sha256": cell["evidence_sha256"],
            }
        )
    content = {
        "schema_version": 1,
        "status": "frozen",
        "selection_id": (
            "round1-search-qualified-60-v2"
            if target_tasks == 60 and feedback_tasks == 30
            else f"round1-search-qualified-{target_tasks}-feedback-{feedback_tasks}-v3"
        ),
        "task_count": len(rows),
        "qualified_candidate_count": len(admitted),
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
        "source_candidate_file_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "source_candidate_evidence_sha256": candidates["evidence_sha256"],
        "source_qualification_summary_file_sha256": hashlib.sha256(
            summary_raw
        ).hexdigest(),
        "source_qualification_summary_sha256": summary["summary_sha256"],
        "source_bundle_qualification_file_sha256": hashlib.sha256(
            bundle_raw
        ).hexdigest(),
        "source_bundle_qualification_evidence_sha256": bundle["evidence_sha256"],
        "qualification_used_for_admission_only": True,
        "gold_fields_included": False,
        "reference_paths_included": False,
        "promotion_partition_opened": candidates.get(
            "promotion_partition_opened", False
        ),
        "final_sealed_partition_opened": candidates.get(
            "final_sealed_partition_opened", False
        ),
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError("Round 1 qualified selection is unreadable") from exc
        if existing != report:
            raise ContractError(
                "frozen Round 1 qualified selection does not match replay"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def _verified_summary(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.resolve().read_bytes()
        summary = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("Round 1 qualification summary is unreadable") from exc
    content = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if summary.get("summary_sha256") != sha256_json(content):
        raise ContractError("Round 1 qualification summary was tampered")
    return raw, summary
