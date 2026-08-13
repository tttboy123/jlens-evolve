"""Native baseline/taught alignment without weakening evaluator identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from evolve.contracts import EvidenceEnvelope


class AlignmentError(ValueError):
    """Evidence cannot support a matched counterfactual comparison."""


MATCHED_IDENTITY_FIELDS = (
    "task_revision_id",
    "task_source_sha256",
    "model_identity",
    "native_evaluator_id",
    "execution_config_sha256",
)


@dataclass(frozen=True, slots=True)
class MatchedNativePair:
    baseline: EvidenceEnvelope
    taught: EvidenceEnvelope
    matched_identity: Mapping[str, Any]


def align_native_pair(
    baseline: EvidenceEnvelope,
    taught: EvidenceEnvelope,
) -> MatchedNativePair:
    """Fail closed unless two native results differ only by their arm/candidate."""

    for label, envelope in (("baseline", baseline), ("taught", taught)):
        if envelope.observer_id != "native-v1":
            raise AlignmentError(f"{label} is not native-v1 evidence")
        arm = envelope.payload.get("arm")
        if arm != label:
            raise AlignmentError(f"{label} evidence has arm {arm!r}")
        for field in (*MATCHED_IDENTITY_FIELDS, "resolved", "evaluator_error"):
            if field not in envelope.payload:
                raise AlignmentError(f"{label} evidence is missing {field}")

    identity: dict[str, Any] = {}
    for field in MATCHED_IDENTITY_FIELDS:
        baseline_value = baseline.payload[field]
        taught_value = taught.payload[field]
        if baseline_value != taught_value:
            raise AlignmentError(
                f"unmatched {field}: {baseline_value!r} != {taught_value!r}"
            )
        identity[field] = baseline_value
    return MatchedNativePair(
        baseline=baseline,
        taught=taught,
        matched_identity=identity,
    )
