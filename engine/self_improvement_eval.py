"""Auditable RSI and PSI evaluation over persisted evolution records.

Definitions used by this project:

* RSI (recursive self-improvement) requires evidence that the improvement
  operator itself changed and became more productive, in addition to ordinary
  candidate-level gains.
* PSI (persistent self-improvement) requires gains to survive process resume and
  verified lessons to transfer across task identities.

The distinction prevents a single successful mutation from being mislabeled as
recursive or persistent self-improvement.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def evaluate_rsi(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda row: int(row.get("iteration", 0)))
    improving = [
        row
        for row in ordered
        if row.get("accepted")
        and float(row.get("child_score", 0.0)) > float(row.get("parent_score", 0.0))
    ]
    accepted_regressions = [
        row
        for row in ordered
        if row.get("accepted")
        and (
            row.get("regressed_cases")
            or float(row.get("child_score", 0.0)) < float(row.get("parent_score", 0.0))
        )
    ]

    longest = current = 0
    previous_child: float | None = None
    for row in improving:
        parent = float(row.get("parent_score", 0.0))
        child = float(row.get("child_score", 0.0))
        if previous_child is not None and abs(parent - previous_child) <= 1e-12:
            current += 1
        else:
            current = 1
        previous_child = child
        longest = max(longest, current)

    revisions = [row for row in ordered if row.get("operator_revision")]
    productive_revisions = [
        row
        for row in revisions
        if float(row.get("post_revision_yield", 0.0))
        > float(row.get("pre_revision_yield", 0.0))
    ]
    operator_improved = bool(productive_revisions)
    candidate_improved = bool(improving)
    rsi_pass = (
        candidate_improved
        and longest >= 2
        and operator_improved
        and not accepted_regressions
    )
    return {
        "definition": "recursive self-improvement",
        "candidate_improved": candidate_improved,
        "strict_improvement_depth": longest,
        "strict_improvement_events": len(improving),
        "accepted_regressions": len(accepted_regressions),
        "operator_revisions": len(revisions),
        "productive_operator_revisions": len(productive_revisions),
        "operator_improved": operator_improved,
        "rsi_pass": rsi_pass,
    }


def evaluate_psi(manifests: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(manifests)
    resumed = [row for row in rows if row.get("resumed_from")]
    resume_checks = [
        float(row.get("final_holdout_score", 0.0))
        >= float(row.get("pre_resume_best_score", 0.0))
        for row in resumed
        if row.get("pre_resume_best_score") is not None
        and row.get("final_holdout_score") is not None
    ]
    same_search_resume_pass = bool(resume_checks) and all(resume_checks)

    transfer_rows = []
    for row in rows:
        task_id = str(row.get("task_id", ""))
        sources = row.get("retrieved_lesson_sources", []) or []
        if any(str(source.get("task_id", "")) != task_id for source in sources):
            transfer_rows.append(row)
    transfer_gains = [
        float(row.get("final_holdout_score", 0.0))
        - float(row.get("initial_holdout_score", 0.0))
        for row in transfer_rows
        if row.get("initial_holdout_score") is not None
        and row.get("final_holdout_score") is not None
    ]
    cross_task_transfer_observed = bool(transfer_rows)
    cross_task_holdout_gain = max(transfer_gains) if transfer_gains else None
    cross_task_transfer_pass = bool(transfer_gains) and cross_task_holdout_gain >= 0.0
    return {
        "definition": "persistent self-improvement",
        "resume_trials": len(resume_checks),
        "same_search_resume_pass": same_search_resume_pass,
        "cross_task_transfer_trials": len(transfer_rows),
        "cross_task_transfer_observed": cross_task_transfer_observed,
        "cross_task_holdout_gain": cross_task_holdout_gain,
        "cross_task_transfer_pass": cross_task_transfer_pass,
        "psi_pass": same_search_resume_pass and cross_task_transfer_pass,
    }


def evaluate_psi_ab(
    manifests: Iterable[dict[str, Any]], *, experiment_id: str
) -> dict[str, Any]:
    """Evaluate a matched control/transfer experiment on one target task.

    Non-inferiority is the PSI persistence gate.  A strict treatment benefit is
    reported separately so a tie is never described as a positive effect.
    """
    rows = [
        row
        for row in manifests
        if str(row.get("psi_experiment_id", "")) == experiment_id
    ]
    arms: dict[str, list[dict[str, Any]]] = {"control": [], "transfer": []}
    for row in rows:
        arm = str(row.get("psi_arm", ""))
        if arm in arms:
            arms[arm].append(row)

    def latest(arm: str) -> dict[str, Any] | None:
        candidates = arms[arm]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda row: (
                str(row.get("completed_at", "")),
                str(row.get("manifest_id", "")),
            ),
        )

    control = latest("control")
    transfer = latest("transfer")
    contract_keys = (
        "task_id",
        "task_family",
        "config_hash",
        "evaluator_hash",
        "initial_hash",
        "search_protocol_hash",
        "model_id",
        "iterations_requested",
    )
    contract_matched = bool(control and transfer) and all(
        control.get(key) == transfer.get(key) for key in contract_keys
    )
    modes_valid = bool(control and transfer) and (
        control.get("experience_mode") == "off"
        and transfer.get("experience_mode") == "cross-task"
    )
    target_task = str(transfer.get("task_id", "")) if transfer else ""
    foreign_sources = [
        source
        for source in (transfer or {}).get("retrieved_lesson_sources", []) or []
        if str(source.get("task_id", "")) not in {"", target_task}
    ]
    cross_task_provenance = bool(foreign_sources)

    control_initial = (
        float(control.get("initial_holdout_score", 0.0)) if control else None
    )
    control_final = float(control.get("final_holdout_score", 0.0)) if control else None
    transfer_initial = (
        float(transfer.get("initial_holdout_score", 0.0)) if transfer else None
    )
    transfer_final = (
        float(transfer.get("final_holdout_score", 0.0)) if transfer else None
    )
    control_public = float(control.get("best_public_score", 0.0)) if control else None
    transfer_public = (
        float(transfer.get("best_public_score", 0.0)) if transfer else None
    )
    control_gain = (
        control_final - control_initial
        if control_final is not None and control_initial is not None
        else None
    )
    transfer_gain = (
        transfer_final - transfer_initial
        if transfer_final is not None and transfer_initial is not None
        else None
    )
    holdout_delta = (
        transfer_final - control_final
        if transfer_final is not None and control_final is not None
        else None
    )
    gain_delta = (
        transfer_gain - control_gain
        if transfer_gain is not None and control_gain is not None
        else None
    )
    public_delta = (
        transfer_public - control_public
        if transfer_public is not None and control_public is not None
        else None
    )
    noninferior = bool(
        holdout_delta is not None
        and gain_delta is not None
        and transfer_gain is not None
        and holdout_delta >= -1e-12
        and gain_delta >= -1e-12
        and transfer_gain >= -1e-12
    )
    strict_transfer_benefit = bool(
        holdout_delta is not None
        and gain_delta is not None
        and (holdout_delta > 1e-12 or gain_delta > 1e-12)
    )
    psi_ab_pass = bool(
        contract_matched and modes_valid and cross_task_provenance and noninferior
    )
    return {
        "definition": "matched cross-task persistent self-improvement A/B",
        "experiment_id": experiment_id,
        "control_run_id": control.get("run_id") if control else None,
        "transfer_run_id": transfer.get("run_id") if transfer else None,
        "task_id": control.get("task_id") if control else None,
        "model_id": control.get("model_id") if control else None,
        "search_protocol_hash": (
            control.get("search_protocol_hash") if control else None
        ),
        "contract_keys": list(contract_keys),
        "contract_matched": contract_matched,
        "experience_modes_valid": modes_valid,
        "cross_task_provenance": cross_task_provenance,
        "foreign_lesson_sources": foreign_sources,
        "control_public_score": control_public,
        "transfer_public_score": transfer_public,
        "public_score_delta_vs_control": public_delta,
        "control_initial_holdout": control_initial,
        "control_final_holdout": control_final,
        "control_holdout_gain": control_gain,
        "transfer_initial_holdout": transfer_initial,
        "transfer_final_holdout": transfer_final,
        "transfer_holdout_gain": transfer_gain,
        "holdout_delta_vs_control": holdout_delta,
        "gain_delta_vs_control": gain_delta,
        "noninferior_to_control": noninferior,
        "strict_transfer_benefit": strict_transfer_benefit,
        "psi_ab_pass": psi_ab_pass,
    }
