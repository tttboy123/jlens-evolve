"""Receipt-derived, Teacher-safe context for the next autonomous round."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evolve.evidence import ReceiptStore

from .config import AutonomousEvolutionError
from .verification import CampaignOutcomeVerifier, VerifiedCampaignClaim

_TEACHER_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "instance_id",
        "task_id",
        "project",
        "repo",
        "benchmark_id",
        "cohort",
        "base_revision",
        "benchmark_base_commit",
        "source_revision",
        "instruction",
        "instruction_sha256",
        "allowed_targets",
        "catalog_fingerprint",
        "fingerprint",
        "task_fingerprint_sha256",
        "estimated_cost",
        "mechanism_route",
        "route",
    }
)


def teacher_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Project only explicitly approved task facts into a remote request.

    An allowlist is deliberate here: a future task-pool field must not silently
    cross the Teacher trust boundary merely because its name was not anticipated
    by a denylist.
    """

    return {key: task[key] for key in sorted(_TEACHER_TASK_FIELDS & task.keys())}


def rebuild_campaign_feedback(
    *, round_root: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct pre-schema feedback without mutating its sealed round."""

    prescreen = _load_json_object(round_root / "PRESCREEN-RESULT.json")
    campaign_result = _load_json_object(round_root / "CAMPAIGN-RESULT.json")
    campaign_status = campaign_result.get("campaign_status")
    if campaign_status == "screened_out":
        claims: tuple[VerifiedCampaignClaim, ...] = ()
    elif campaign_status == "completed":
        selection = _load_json_object(round_root / "TASK-SELECTION.json")
        selected_task_ids = selection.get("selected_task_ids")
        if not isinstance(selected_task_ids, list) or not all(
            isinstance(task_id, str) and task_id for task_id in selected_task_ids
        ):
            raise AutonomousEvolutionError("legacy round task selection is invalid")
        claims = CampaignOutcomeVerifier().verify(
            round_root=round_root,
            result=campaign_result,
            selected_task_ids=selected_task_ids,
            candidate_id=_required_text(payload.get("candidate_id"), "candidate id"),
            candidate_revision_id=_required_text(
                payload.get("candidate_revision_id"), "candidate revision"
            ),
            candidate_bundle_sha256=_required_sha256(
                payload.get("compiled_bundle_sha256"), "compiled bundle"
            ),
        )
    else:
        raise AutonomousEvolutionError(
            "legacy round campaign status cannot produce feedback"
        )
    return project_campaign_feedback(
        round_root=round_root,
        campaign_result=campaign_result,
        claims=claims,
        prescreen=prescreen,
        candidate_id=_required_text(payload.get("candidate_id"), "candidate id"),
    )


def project_campaign_feedback(
    *,
    round_root: Path,
    campaign_result: Mapping[str, Any],
    claims: Sequence[VerifiedCampaignClaim],
    prescreen: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    """Project paired model/native facts from hash-verified immutable receipts."""

    campaign_status = _optional_text(
        campaign_result.get("campaign_status"), "campaign status"
    )
    campaign_id = _optional_text(campaign_result.get("campaign_id"), "campaign id")
    if not claims:
        feedback: dict[str, Any] = {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "campaign_status": campaign_status,
            "task_pairs": [],
        }
        if campaign_status == "screened_out":
            feedback["prescreen"] = _prescreen_feedback(
                round_root=round_root,
                prescreen=prescreen,
                candidate_id=candidate_id,
            )
        return feedback
    if campaign_id is None:
        raise AutonomousEvolutionError("completed campaign feedback has no campaign id")

    receipts = ReceiptStore(round_root / "receipt-store").list_receipts()
    if any(receipt.campaign_id != campaign_id for receipt in receipts):
        raise AutonomousEvolutionError("campaign receipt identity mismatch")
    receipts_by_plan: dict[str, list[Any]] = {}
    for receipt in receipts:
        receipts_by_plan.setdefault(receipt.plan_id, []).append(receipt)

    arms_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    for plan_id, plan_receipts in receipts_by_plan.items():
        model = _only_receipt(plan_receipts, "model", plan_id)
        task_revision_id = _required_text(
            model.payload.get("task_revision_id"), "model task revision"
        )
        arm = _required_text(model.payload.get("arm"), "model arm")
        if arm not in {"baseline", "taught"}:
            raise AutonomousEvolutionError("campaign model receipt has invalid arm")
        external = _only_receipt(plan_receipts, "external_trace", plan_id)
        native = _only_receipt(plan_receipts, "native_evaluation", plan_id)
        terminal = _only_receipt(plan_receipts, "execution_terminal", plan_id)
        patch_sha256 = _optional_sha256(
            model.payload.get("patch_sha256"), "model patch"
        )
        if (
            external.payload.get("model_receipt_id") != model.receipt_id
            or native.payload.get("model_receipt_id") != model.receipt_id
            or external.payload.get("model_artifact_sha256") != model.artifact_sha256
            or native.payload.get("model_artifact_sha256") != model.artifact_sha256
            or external.payload.get("task_revision_id") != task_revision_id
            or native.payload.get("task_revision_id") != task_revision_id
            or external.payload.get("arm") != arm
            or native.payload.get("arm") != arm
            or external.payload.get("prediction_sha256") != patch_sha256
            or external.payload.get("patch_sha256") != patch_sha256
            or native.payload.get("prediction_sha256") != patch_sha256
        ):
            raise AutonomousEvolutionError(
                "campaign feedback receipt lineage is inconsistent"
            )
        for field in (
            "candidate_consumed",
            "candidate_revision_id",
            "candidate_bundle_sha256",
            "structural_valid",
            "failure_reason",
        ):
            if external.payload.get(field) != model.payload.get(field):
                raise AutonomousEvolutionError(
                    "campaign external trace disagrees with model receipt"
                )
        failure_reason, failure_reason_raw_sha256 = _failure_category(
            model.payload.get("failure_reason"), "model failure reason"
        )
        native_failure = native.payload.get("native_error")
        if native_failure is None:
            native_failure = native.payload.get("evaluator_error")
        native_error, native_error_raw_sha256 = _failure_category(
            native_failure, "native error"
        )
        key = (task_revision_id, arm)
        if key in arms_by_task:
            raise AutonomousEvolutionError("campaign feedback contains duplicate arm")
        arms_by_task[key] = {
            "plan_id": plan_id,
            "execution_status": _required_text(
                terminal.payload.get("status"), "execution status"
            ),
            "execution_terminal_receipt_id": terminal.receipt_id,
            "execution_terminal_receipt_sha256": terminal.content_sha256,
            "model_receipt_id": model.receipt_id,
            "model_receipt_sha256": model.content_sha256,
            "model_artifact_sha256": model.artifact_sha256,
            "external_trace_receipt_id": external.receipt_id,
            "external_trace_receipt_sha256": external.content_sha256,
            "external_trace_artifact_sha256": external.artifact_sha256,
            "candidate_consumed": _optional_bool(
                model.payload.get("candidate_consumed"), "candidate consumed"
            ),
            "candidate_revision_id": _optional_text(
                model.payload.get("candidate_revision_id"), "candidate revision"
            ),
            "candidate_bundle_sha256": _optional_sha256(
                model.payload.get("candidate_bundle_sha256"), "candidate bundle"
            ),
            "patch_sha256": patch_sha256,
            "structural_valid": _optional_bool(
                model.payload.get("structural_valid"), "structural validity"
            ),
            "failure_reason": failure_reason,
            "failure_reason_raw_sha256": failure_reason_raw_sha256,
            "native_receipt_id": native.receipt_id,
            "native_receipt_sha256": native.content_sha256,
            "native_artifact_sha256": native.artifact_sha256,
            "resolved": _optional_bool(native.payload.get("resolved"), "native result"),
            "native_error": native_error,
            "native_error_raw_sha256": native_error_raw_sha256,
            "native_report_sha256": _optional_sha256(
                native.payload.get("native_report_sha256"), "native report"
            ),
            "official_receipt_sha256": _optional_sha256(
                native.payload.get("official_receipt_sha256"), "official receipt"
            ),
        }

    verified_by_id = {claim.claim_id: claim for claim in claims}
    raw_claims = campaign_result.get("claims")
    if not isinstance(raw_claims, list):
        raise AutonomousEvolutionError("campaign feedback claim binding is missing")
    task_pairs: list[dict[str, Any]] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, Mapping):
            raise AutonomousEvolutionError("campaign feedback claim binding is invalid")
        claim_id = _required_text(raw_claim.get("claim_id"), "claim id")
        verified = verified_by_id.get(claim_id)
        if verified is None:
            raise AutonomousEvolutionError(
                "campaign feedback claim was not independently verified"
            )
        task_revision_id = _required_text(
            raw_claim.get("task_revision_id"), "claim task revision"
        )
        try:
            baseline = arms_by_task.pop((task_revision_id, "baseline"))
            taught = arms_by_task.pop((task_revision_id, "taught"))
        except KeyError as error:
            raise AutonomousEvolutionError(
                "campaign feedback has an incomplete paired execution"
            ) from error
        paired_receipt_ids = {
            baseline["model_receipt_id"],
            baseline["external_trace_receipt_id"],
            baseline["native_receipt_id"],
            taught["model_receipt_id"],
            taught["external_trace_receipt_id"],
            taught["native_receipt_id"],
        }
        if (
            len(verified.counterfactual_receipt_ids) != 6
            or set(verified.counterfactual_receipt_ids) != paired_receipt_ids
        ):
            raise AutonomousEvolutionError(
                "campaign counterfactual receipts do not bind the projected pair"
            )
        task_pairs.append(
            {
                "task_id": verified.task_id,
                "task_revision_id": task_revision_id,
                "baseline": baseline,
                "taught": taught,
                "claim": {
                    "claim_id": verified.claim_id,
                    "classification": verified.classification,
                    "grade": verified.grade,
                    "counterfactual_pair_sha256": (
                        verified.counterfactual_pair_sha256
                    ),
                    "counterfactual_receipt_ids": list(
                        verified.counterfactual_receipt_ids
                    ),
                },
            }
        )
    if arms_by_task or len(task_pairs) != len(claims):
        raise AutonomousEvolutionError(
            "campaign feedback does not cover every verified task pair"
        )
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "campaign_status": campaign_status,
        "task_pairs": task_pairs,
    }


def campaign_failure_facts(
    campaign_feedback: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Remove per-run lineage IDs while retaining model/native failure identity."""

    facts: list[dict[str, Any]] = []
    task_pairs = campaign_feedback.get("task_pairs", ())
    if not isinstance(task_pairs, Sequence):
        return facts
    for pair in task_pairs:
        if not isinstance(pair, Mapping):
            continue
        fact: dict[str, Any] = {
            "task_id": pair.get("task_id"),
            "task_revision_id": pair.get("task_revision_id"),
        }
        for arm in ("baseline", "taught"):
            arm_result = pair.get(arm)
            if not isinstance(arm_result, Mapping):
                continue
            fact[arm] = {
                "execution_status": arm_result.get("execution_status"),
                "patch_sha256": arm_result.get("patch_sha256"),
                "structural_valid": arm_result.get("structural_valid"),
                "failure_reason": arm_result.get("failure_reason"),
                "resolved": arm_result.get("resolved"),
                "native_error": arm_result.get("native_error"),
                "native_report_sha256": arm_result.get("native_report_sha256"),
                "official_receipt_sha256": arm_result.get(
                    "official_receipt_sha256"
                ),
            }
        facts.append(fact)
    prescreen = campaign_feedback.get("prescreen")
    if isinstance(prescreen, Mapping):
        facts.append(
            {
                "prescreen": {
                    field: prescreen.get(field)
                    for field in (
                        "candidate_revision_id",
                        "candidate_bundle_sha256",
                        "task_revision_id",
                        "patch_sha256",
                        "structural_valid",
                        "patch_applicable",
                        "failure_reason",
                    )
                }
            }
        )
    return facts


def _prescreen_feedback(
    *, round_root: Path, prescreen: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    status = _required_text(prescreen.get("status"), "prescreen status")
    model_receipt_ids = prescreen.get("model_receipt_ids")
    if (
        status != "completed"
        or not isinstance(model_receipt_ids, list)
        or len(model_receipt_ids) != 1
    ):
        raise AutonomousEvolutionError(
            "screened-out candidate lacks one completed prescreen model receipt"
        )
    receipts = ReceiptStore(round_root / "prescreen/receipt-store").list_receipts()
    model = next(
        (
            receipt
            for receipt in receipts
            if receipt.receipt_id == model_receipt_ids[0] and receipt.kind == "model"
        ),
        None,
    )
    if model is None:
        raise AutonomousEvolutionError("screened-out candidate model receipt is missing")
    plan_receipts = [receipt for receipt in receipts if receipt.plan_id == model.plan_id]
    external = _only_receipt(plan_receipts, "external_trace", model.plan_id)
    terminal = _only_receipt(plan_receipts, "execution_terminal", model.plan_id)
    candidate_revision_id = _required_text(
        prescreen.get("candidate_revision_id"), "prescreen candidate revision"
    )
    candidate_bundle_sha256 = _required_sha256(
        prescreen.get("candidate_bundle_sha256"), "prescreen candidate bundle"
    )
    patch_sha256 = _optional_sha256(
        model.payload.get("patch_sha256"), "prescreen model patch"
    )
    structural_valid = _optional_bool(
        model.payload.get("structural_valid"), "prescreen structural validity"
    )
    if (
        model.payload.get("candidate_consumed") is not True
        or model.payload.get("candidate_revision_id") != candidate_revision_id
        or model.payload.get("candidate_bundle_sha256") != candidate_bundle_sha256
        or structural_valid
        != _optional_bool(prescreen.get("structural_valid"), "prescreen result")
        or external.payload.get("model_receipt_id") != model.receipt_id
        or external.payload.get("model_artifact_sha256") != model.artifact_sha256
        or external.payload.get("prediction_sha256") != patch_sha256
        or external.payload.get("patch_sha256") != patch_sha256
        or terminal.payload.get("status") != "completed"
    ):
        raise AutonomousEvolutionError(
            "screened-out candidate receipt lineage is inconsistent"
        )
    failure_reason, failure_reason_raw_sha256 = _failure_category(
        model.payload.get("failure_reason"), "prescreen failure reason"
    )
    return {
        "candidate_id": candidate_id,
        "candidate_revision_id": candidate_revision_id,
        "candidate_bundle_sha256": candidate_bundle_sha256,
        "task_revision_id": _required_text(
            model.payload.get("task_revision_id"), "prescreen task revision"
        ),
        "plan_id": model.plan_id,
        "model_receipt_id": model.receipt_id,
        "model_receipt_sha256": model.content_sha256,
        "model_artifact_sha256": model.artifact_sha256,
        "external_trace_receipt_id": external.receipt_id,
        "external_trace_receipt_sha256": external.content_sha256,
        "execution_terminal_receipt_id": terminal.receipt_id,
        "execution_terminal_receipt_sha256": terminal.content_sha256,
        "patch_sha256": patch_sha256,
        "structural_valid": structural_valid,
        "patch_applicable": _optional_bool(
            prescreen.get("patch_applicable"), "prescreen patch applicability"
        ),
        "failure_reason": failure_reason,
        "failure_reason_raw_sha256": failure_reason_raw_sha256,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AutonomousEvolutionError(
            f"frozen autonomous artifact is unreadable: {path.name}"
        ) from error
    if not isinstance(payload, dict):
        raise AutonomousEvolutionError(
            f"frozen autonomous artifact is not an object: {path.name}"
        )
    return payload


def _only_receipt(receipts: Sequence[Any], kind: str, plan_id: str) -> Any:
    matched = [receipt for receipt in receipts if receipt.kind == kind]
    if len(matched) != 1:
        raise AutonomousEvolutionError(
            f"campaign feedback plan {plan_id} requires one {kind} receipt"
        )
    return matched[0]


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise AutonomousEvolutionError(f"campaign feedback {field} is invalid")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field)


def _optional_bool(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise AutonomousEvolutionError(f"campaign feedback {field} is invalid")
    return value


def _optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AutonomousEvolutionError(f"campaign feedback {field} is not SHA-256")
    return value


def _required_sha256(value: Any, field: str) -> str:
    digest = _optional_sha256(value, field)
    if digest is None:
        raise AutonomousEvolutionError(f"campaign feedback {field} is missing")
    return digest


_FAILURE_CATEGORIES = frozenset(
    {
        "empty_patch",
        "expression-used-for-statement",
        "malformed-hunk",
        "selector-no-match",
        "unresolved",
    }
)


def _failure_category(value: Any, field: str) -> tuple[str | None, str | None]:
    raw = _optional_text(value, field)
    if raw is None:
        return None, None
    normalized = raw.strip().casefold()
    if normalized in _FAILURE_CATEGORIES:
        return normalized, None
    return "other", hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "campaign_failure_facts",
    "project_campaign_feedback",
    "rebuild_campaign_feedback",
    "teacher_task",
]
