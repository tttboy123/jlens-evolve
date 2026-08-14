"""Build strict counterfactual lineage from independently stored facts."""

from __future__ import annotations

from typing import Any, Mapping

from evolve.alignment.native_pair import MATCHED_IDENTITY_FIELDS
from evolve.contracts import (
    ContractViolation,
    CounterfactualArmEvidence,
    EvidenceEnvelope,
    MatchedCounterfactualPair,
    Receipt,
    content_sha256,
)


def build_matched_counterfactual_pair(
    *,
    candidate_id: str,
    candidate_revision_id: str,
    candidate_bundle_sha256: str,
    baseline_model_receipt: Receipt,
    baseline_external_evidence: EvidenceEnvelope,
    baseline_native_evidence: EvidenceEnvelope,
    taught_model_receipt: Receipt,
    taught_external_evidence: EvidenceEnvelope,
    taught_native_evidence: EvidenceEnvelope,
) -> MatchedCounterfactualPair:
    """Fail closed unless both arms form a complete model→trace→native chain."""

    baseline = _build_arm(
        "baseline",
        baseline_model_receipt,
        baseline_external_evidence,
        baseline_native_evidence,
        candidate_revision_id=candidate_revision_id,
        candidate_bundle_sha256=candidate_bundle_sha256,
    )
    taught = _build_arm(
        "taught",
        taught_model_receipt,
        taught_external_evidence,
        taught_native_evidence,
        candidate_revision_id=candidate_revision_id,
        candidate_bundle_sha256=candidate_bundle_sha256,
    )
    baseline_native = baseline_native_evidence.payload
    taught_native = taught_native_evidence.payload
    identity: dict[str, Any] = {}
    for field in MATCHED_IDENTITY_FIELDS:
        if field not in baseline_native or field not in taught_native:
            raise ContractViolation(f"counterfactual native evidence is missing {field}")
        if baseline_native[field] != taught_native[field]:
            raise ContractViolation(f"counterfactual arms have unmatched {field}")
        identity[field] = baseline_native[field]
    if baseline.campaign_id != taught.campaign_id:
        raise ContractViolation("counterfactual arms have unmatched campaign_id")
    return MatchedCounterfactualPair(
        candidate_id=candidate_id,
        candidate_revision_id=candidate_revision_id,
        candidate_bundle_sha256=candidate_bundle_sha256,
        campaign_id=baseline.campaign_id,
        task_revision_id=str(identity["task_revision_id"]),
        task_source_sha256=str(identity["task_source_sha256"]),
        model_identity=str(identity["model_identity"]),
        native_evaluator_id=str(identity["native_evaluator_id"]),
        execution_config_sha256=str(identity["execution_config_sha256"]),
        baseline=baseline,
        taught=taught,
    )


def _build_arm(
    arm: str,
    model: Receipt,
    external: EvidenceEnvelope,
    native: EvidenceEnvelope,
    *,
    candidate_revision_id: str,
    candidate_bundle_sha256: str,
) -> CounterfactualArmEvidence:
    if model.kind != "model":
        raise ContractViolation(f"{arm} model receipt has kind {model.kind!r}")
    if model.artifact_sha256 != content_sha256(model.payload):
        raise ContractViolation(f"{arm} model receipt artifact does not bind payload")
    _validate_evidence(arm, external, observer_id="external-trace-v1")
    _validate_evidence(arm, native, observer_id="native-v1")
    for label, envelope in (("external", external), ("native", native)):
        if envelope.payload.get("campaign_id") != model.campaign_id:
            raise ContractViolation(f"{arm} {label} campaign does not match model")
        if envelope.payload.get("plan_id") != model.plan_id:
            raise ContractViolation(f"{arm} {label} plan does not match model")
        if envelope.payload.get("model_receipt_id") != model.receipt_id:
            raise ContractViolation(f"{arm} {label} does not reference model receipt")
        if envelope.payload.get("model_artifact_sha256") != model.artifact_sha256:
            raise ContractViolation(f"{arm} {label} does not bind model artifact")
    if external.payload.get("task_revision_id") != native.payload.get(
        "task_revision_id"
    ):
        raise ContractViolation(f"{arm} trace/native task identity mismatch")
    model_identity = (
        f"{model.payload.get('provider')}/{model.payload.get('model')}"
        f"@{model.payload.get('revision')}"
    )
    if native.payload.get("model_identity") != model_identity:
        raise ContractViolation(f"{arm} native model identity mismatch")

    model_prediction = _prediction_sha256(model.payload, label=f"{arm} model")
    external_prediction = _prediction_sha256(
        external.payload, label=f"{arm} external trace"
    )
    native_prediction = _prediction_sha256(
        native.payload, label=f"{arm} native outcome"
    )
    if len({model_prediction, external_prediction, native_prediction}) != 1:
        raise ContractViolation(f"{arm} prediction artifact lineage mismatch")

    consumed = model.payload.get("candidate_consumed")
    revision = model.payload.get("candidate_revision_id")
    bundle = model.payload.get("candidate_bundle_sha256")
    if not isinstance(consumed, bool):
        raise ContractViolation(f"{arm} model candidate_consumed must be boolean")
    for field, expected in (
        ("candidate_consumed", consumed),
        ("candidate_revision_id", revision),
        ("candidate_bundle_sha256", bundle),
    ):
        if external.payload.get(field) != expected:
            raise ContractViolation(
                f"{arm} external candidate lineage does not match model"
            )
    if arm == "baseline":
        if consumed or revision is not None or bundle is not None:
            raise ContractViolation("baseline arm contains candidate lineage")
    elif (
        consumed is not True
        or revision != candidate_revision_id
        or bundle != candidate_bundle_sha256
    ):
        raise ContractViolation("taught arm candidate lineage does not match pair")

    return CounterfactualArmEvidence(
        arm=arm,
        campaign_id=model.campaign_id,
        plan_id=model.plan_id,
        model_receipt_id=model.receipt_id,
        model_receipt_sha256=model.content_sha256,
        model_artifact_sha256=model.artifact_sha256,
        external_trace_evidence_id=external.evidence_id,
        external_trace_evidence_sha256=external.content_sha256,
        external_trace_receipt_id=_only_receipt_id(external, arm, "external"),
        external_trace_artifact_sha256=external.artifact_sha256,
        native_outcome_evidence_id=native.evidence_id,
        native_outcome_evidence_sha256=native.content_sha256,
        native_outcome_receipt_id=_only_receipt_id(native, arm, "native"),
        native_outcome_artifact_sha256=native.artifact_sha256,
        prediction_sha256=model_prediction,
        candidate_consumed=consumed,
        candidate_revision_id=revision if isinstance(revision, str) else None,
        candidate_bundle_sha256=bundle if isinstance(bundle, str) else None,
    )


def _validate_evidence(
    arm: str, envelope: EvidenceEnvelope, *, observer_id: str
) -> None:
    if envelope.observer_id != observer_id:
        raise ContractViolation(f"{arm} evidence is not from {observer_id}")
    if envelope.payload.get("arm") != arm:
        raise ContractViolation(f"{arm} evidence arm identity mismatch")
    receipt_payload = dict(envelope.payload)
    for projected_name in ("campaign_id", "plan_id", "receipt_kind"):
        receipt_payload.pop(projected_name, None)
    if envelope.artifact_sha256 != content_sha256(receipt_payload):
        raise ContractViolation(
            f"{arm} {observer_id} artifact does not bind projected receipt payload"
        )
    _only_receipt_id(envelope, arm, observer_id)


def _only_receipt_id(envelope: EvidenceEnvelope, arm: str, label: str) -> str:
    if len(envelope.receipt_ids) != 1:
        raise ContractViolation(f"{arm} {label} evidence must bind one receipt")
    return envelope.receipt_ids[0]


def _prediction_sha256(payload: Mapping[str, Any], *, label: str) -> str:
    value = payload.get("prediction_sha256", payload.get("patch_sha256"))
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractViolation(f"{label} prediction must be literal SHA-256")
    return value


__all__ = ["build_matched_counterfactual_pair"]
