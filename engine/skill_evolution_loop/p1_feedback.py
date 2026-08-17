"""Prepare a replayable, holdout-free P1 parent request from frozen evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .composition import load_frozen_attempt
from .contracts import (
    ContractError,
    FailureEvidence,
    FeedbackPackage,
    LoopRevision,
    ParentModelRequest,
    canonical_json,
    sha256_json,
)


def freeze_p1_parent_request(
    *,
    composition_path: Path,
    semantic_review_path: Path,
    output_path: Path,
    condition_id: str = "structured-taught",
) -> dict[str, Any]:
    """Build the next-round parent input using feedback-arm hashes and diagnostics."""
    composition = _load_json(composition_path, "composition")
    composition_sha = composition.get("composition_sha256")
    composition_content = {
        key: value for key, value in composition.items() if key != "composition_sha256"
    }
    if composition_sha != sha256_json(composition_content):
        raise ContractError("P1 composition sha256 mismatch")
    if composition.get("status") != "complete":
        raise ContractError("P1 composition must be complete")

    semantic = _load_json(semantic_review_path, "semantic review")
    if semantic.get("taskset_fingerprint") != composition.get("taskset_fingerprint"):
        raise ContractError("P1 semantic review taskset mismatch")
    if semantic.get("composition_sha256") != composition_sha:
        raise ContractError("P1 semantic review composition mismatch")
    semantic_rows = semantic.get("rows")
    if not isinstance(semantic_rows, list):
        raise ContractError("P1 semantic review rows are invalid")
    diagnostics = {
        str(row.get("task_id")): str(row.get("evidence", ""))
        for row in semantic_rows
        if isinstance(row, dict) and row.get("cohort") == "feedback"
    }
    if not diagnostics or any(not value.strip() for value in diagnostics.values()):
        raise ContractError("P1 feedback diagnostics are incomplete")

    cells = composition.get("cells")
    if not isinstance(cells, list):
        raise ContractError("P1 composition cells are invalid")
    selected = [
        row
        for row in cells
        if isinstance(row, dict)
        and row.get("condition_id") == condition_id
        and row.get("task_id") in diagnostics
    ]
    if {str(row.get("task_id")) for row in selected} != set(diagnostics):
        raise ContractError("P1 feedback condition evidence is incomplete")

    evidence: list[FailureEvidence] = []
    revision: LoopRevision | None = None
    for cell in sorted(selected, key=lambda row: str(row["task_id"])):
        attempt_path = Path(str(cell["attempt_path"]))
        attempt = load_frozen_attempt(attempt_path)
        if attempt.get("evidence_sha256") != cell.get("evidence_sha256"):
            raise ContractError("P1 composed cell evidence sha256 mismatch")
        if attempt.get("task", {}).get("cohort") != "feedback":
            raise ContractError("holdout evidence is prohibited from parent request")
        current = LoopRevision.from_dict(attempt["condition"]["revision"])
        if revision is None:
            revision = current
        elif revision.fingerprint != current.fingerprint:
            raise ContractError("P1 feedback cells use different Skill revisions")
        outcome = attempt.get("attempt", {})
        reason = outcome.get("failure_reason") or "native-unresolved"
        if not isinstance(reason, str):
            raise ContractError("P1 feedback failure reason is invalid")
        evidence.append(
            FailureEvidence.create(
                task_id=str(cell["task_id"]),
                reason_code=reason,
                diagnostic_summary=diagnostics[str(cell["task_id"])],
                raw_output_sha256=str(outcome["raw_output_sha256"]),
                extracted_edit_sha256=outcome.get("patch_sha256"),
                apply_error=outcome.get("detail") if reason == "apply-fail" else None,
            )
        )
    if revision is None:
        raise ContractError("P1 feedback produced no current revision")
    feedback = FeedbackPackage.create(
        current_round=0,
        arm_evidence=evidence,
        previous_eval_note=(
            "P1 Round 0: structural improvement without native capability gain; "
            "hunk mechanism stopped after 0/12 structural validity."
        ),
        no_progress=False,
        rejected_fingerprints=[revision.fingerprint],
    )
    request = ParentModelRequest.create(
        feedback=feedback,
        current_revision=revision,
    )
    content = {
        "schema_version": 1,
        "source_composition_sha256": composition_sha,
        "condition_id": condition_id,
        "feedback_task_count": len(evidence),
        "holdout_task_ids_included": False,
        "parent_request_sha256": request.sha256,
        "parent_request": request.to_dict(),
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    output = output_path.resolve()
    if output.exists():
        existing = _load_json(output, "parent request evidence")
        if existing != report:
            raise ContractError("frozen parent request does not match evidence")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    return report


def freeze_p1_round_parent_request(
    *,
    experiment_root: Path,
    semantic_review_path: Path,
    output_path: Path,
    condition_id: str = "structured-taught",
    rejected_fingerprints: list[str] | None = None,
) -> dict[str, Any]:
    """Build a later-round request from one frozen paired experiment.

    Only feedback-arm diagnostics and attempt hashes cross the parent boundary.
    Holdout rows are used by the independent evaluator but are deliberately not
    copied into this request.
    """
    root = experiment_root.resolve()
    summary = _load_json(root / "SUMMARY.json", "experiment summary")
    summary_sha = summary.get("summary_sha256")
    summary_content = {
        key: value for key, value in summary.items() if key != "summary_sha256"
    }
    if summary_sha != sha256_json(summary_content):
        raise ContractError("P1 experiment summary sha256 mismatch")
    if summary.get("status") != "complete":
        raise ContractError("P1 experiment must be complete")
    if summary.get("network_calls_performed") is not False:
        raise ContractError("P1 experiment must be local and replayable")

    semantic = _load_json(semantic_review_path, "semantic review")
    if semantic.get("taskset_fingerprint") != summary.get("taskset_fingerprint"):
        raise ContractError("P1 semantic review taskset mismatch")
    if semantic.get("experiment_summary_sha256") != summary_sha:
        raise ContractError("P1 semantic review experiment mismatch")
    if semantic.get("network_calls_performed") is not False:
        raise ContractError("P1 semantic review must be offline")
    semantic_rows = semantic.get("rows")
    if not isinstance(semantic_rows, list):
        raise ContractError("P1 semantic review rows are invalid")
    diagnostics = {
        str(row.get("task_id")): str(row.get("evidence", ""))
        for row in semantic_rows
        if isinstance(row, dict) and row.get("cohort") == "feedback"
    }
    if not diagnostics or any(not value.strip() for value in diagnostics.values()):
        raise ContractError("P1 feedback diagnostics are incomplete")

    evidence: list[FailureEvidence] = []
    revision: LoopRevision | None = None
    for task_id in sorted(diagnostics):
        attempt_path = root / "cells" / task_id / condition_id / "ATTEMPT.json"
        attempt = load_frozen_attempt(attempt_path)
        if attempt.get("taskset_fingerprint") != summary.get("taskset_fingerprint"):
            raise ContractError("P1 feedback attempt taskset mismatch")
        task = attempt.get("task", {})
        if task.get("cohort") != "feedback" or task.get("task_id") != task_id:
            raise ContractError("holdout evidence is prohibited from parent request")
        if attempt.get("condition", {}).get("condition_id") != condition_id:
            raise ContractError("P1 feedback attempt condition mismatch")
        current = LoopRevision.from_dict(attempt["condition"]["revision"])
        if revision is None:
            revision = current
        elif revision.fingerprint != current.fingerprint:
            raise ContractError("P1 feedback cells use different Skill revisions")
        outcome = attempt.get("attempt", {})
        reason = outcome.get("failure_reason") or "native-unresolved"
        if not isinstance(reason, str):
            raise ContractError("P1 feedback failure reason is invalid")
        evidence.append(
            FailureEvidence.create(
                task_id=task_id,
                reason_code=reason,
                diagnostic_summary=diagnostics[task_id],
                raw_output_sha256=str(outcome["raw_output_sha256"]),
                extracted_edit_sha256=outcome.get("patch_sha256"),
                apply_error=outcome.get("detail") if reason == "apply-fail" else None,
            )
        )
    if revision is None:
        raise ContractError("P1 feedback produced no current revision")
    if revision.source_round < 1:
        raise ContractError("later-round feedback requires a parent revision")

    metrics = summary.get("condition_metrics", {})
    baseline = metrics.get("structured-baseline", {})
    taught = metrics.get(condition_id, {})
    native_gain = semantic.get("feedback_native_gain")
    regressions = semantic.get("holdout_native_regressions")
    if type(native_gain) is not int or type(regressions) is not int:
        raise ContractError("P1 semantic native metrics are invalid")
    previous_note = (
        f"P1 Round {revision.source_round}: feedback native gain={native_gain}; "
        f"holdout native regressions={regressions}; structured baseline="
        f"{baseline.get('structural_valid')}/{baseline.get('completed')}; taught="
        f"{taught.get('structural_valid')}/{taught.get('completed')}. "
        "Replace the prior Skill entirely when its mechanism conflicts with the "
        "frozen structured-search-replace protocol."
    )
    rejected = list(
        dict.fromkeys([*(rejected_fingerprints or []), revision.fingerprint])
    )
    feedback = FeedbackPackage.create(
        current_round=revision.source_round,
        arm_evidence=evidence,
        previous_eval_note=previous_note,
        no_progress=native_gain == 0,
        rejected_fingerprints=rejected,
    )
    request = ParentModelRequest.create(feedback=feedback, current_revision=revision)
    content = {
        "schema_version": 1,
        "source_experiment_summary_sha256": summary_sha,
        "condition_id": condition_id,
        "feedback_task_count": len(evidence),
        "holdout_task_ids_included": False,
        "parent_request_sha256": request.sha256,
        "parent_request": request.to_dict(),
        "network_calls_performed": False,
    }
    report = {**content, "evidence_sha256": sha256_json(content)}
    _freeze_report(output_path, report, "parent request evidence")
    return report


def _freeze_report(path: Path, report: dict[str, Any], label: str) -> None:
    output = path.resolve()
    if output.exists():
        if _load_json(output, label) != report:
            raise ContractError(f"frozen {label} does not match evidence")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"P1 {label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ContractError(f"P1 {label} must be an object")
    return value
