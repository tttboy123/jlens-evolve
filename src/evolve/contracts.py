"""Frozen cross-module contracts for v3.

These value objects are the only types shared by Kernel, Runtime, Observer,
Strategy, Registry, Governance, and Reporting modules.  Strategy-specific terms
such as baseline/taught remain opaque strings to the neutral Kernel and Runtime.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Mapping

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractViolation(ValueError):
    """A caller supplied data that would break an audit or safety invariant."""


class Cohort(StrEnum):
    FEEDBACK = "feedback"
    HOLDOUT = "holdout"
    FINAL_SEALED = "final-sealed"
    BURNED = "burned"


class ClaimGrade(StrEnum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"


class ClaimClassification(StrEnum):
    GAIN = "gain"
    NEUTRAL = "neutral"
    REGRESSION = "regression"
    INFRA_FAILURE = "infra_failure"


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractViolation(f"{name} must be non-empty text")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ContractViolation(f"{name} must be a literal lowercase SHA-256")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name != "content_sha256"
            and not (
                field.metadata.get("omit_if_none")
                and getattr(value, field.name) is None
            )
            and not (
                field.metadata.get("omit_if_empty") and not getattr(value, field.name)
            )
        }
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return the stable representation used by every content hash."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskRevision:
    task_id: str
    revision_id: str
    project: str
    cohort: Cohort
    source_sha256: str
    evaluator_id: str
    source_uri: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "revision_id", "project", "evaluator_id"):
            _require_text(name, getattr(self, name))
        _require_sha256("source_sha256", self.source_sha256)
        if self.cohort is Cohort.BURNED:
            raise ContractViolation("burned task revisions cannot enter execution")

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    provider: str
    model: str
    revision: str

    def __post_init__(self) -> None:
        for name in ("provider", "model", "revision"):
            _require_text(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    max_tokens: int
    max_seconds: int
    max_cost_cny: float

    def __post_init__(self) -> None:
        if self.max_tokens < 0 or self.max_seconds <= 0 or self.max_cost_cny < 0:
            raise ContractViolation("execution limits must be non-negative and bounded")


@dataclass(frozen=True, slots=True)
class MechanismPrediction:
    """A falsifiable internal-effect prediction frozen before model dispatch."""

    prediction_id: str
    candidate_revision_id: str
    mechanism_id: str
    observer_config_sha256: str
    expected_internal_effect_json: str
    expected_internal_effect_sha256: str

    def __post_init__(self) -> None:
        for name in ("prediction_id", "candidate_revision_id", "mechanism_id"):
            _require_text(name, getattr(self, name))
        _require_sha256("observer_config_sha256", self.observer_config_sha256)
        _require_sha256(
            "expected_internal_effect_sha256",
            self.expected_internal_effect_sha256,
        )
        try:
            effect = json.loads(self.expected_internal_effect_json)
        except json.JSONDecodeError as error:
            raise ContractViolation(
                "expected_internal_effect_json must be valid JSON"
            ) from error
        if not isinstance(effect, Mapping) or not effect:
            raise ContractViolation("expected internal effect must be an object")
        if canonical_json(effect) != self.expected_internal_effect_json:
            raise ContractViolation("expected internal effect must use canonical JSON")
        if content_sha256(effect) != self.expected_internal_effect_sha256:
            raise ContractViolation("expected internal effect SHA-256 mismatch")

    @classmethod
    def create(
        cls,
        *,
        prediction_id: str,
        candidate_revision_id: str,
        mechanism_id: str,
        observer_config_sha256: str,
        expected_internal_effect: Mapping[str, Any],
    ) -> MechanismPrediction:
        canonical_effect = canonical_json(expected_internal_effect)
        return cls(
            prediction_id=prediction_id,
            candidate_revision_id=candidate_revision_id,
            mechanism_id=mechanism_id,
            observer_config_sha256=observer_config_sha256,
            expected_internal_effect_json=canonical_effect,
            expected_internal_effect_sha256=content_sha256(expected_internal_effect),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MechanismPrediction:
        required = {
            "prediction_id",
            "candidate_revision_id",
            "mechanism_id",
            "observer_config_sha256",
            "expected_internal_effect",
            "expected_internal_effect_sha256",
        }
        if set(payload) != required:
            raise ContractViolation("mechanism prediction payload fields are invalid")
        effect = payload["expected_internal_effect"]
        if not isinstance(effect, Mapping):
            raise ContractViolation("expected internal effect must be an object")
        return cls(
            prediction_id=str(payload["prediction_id"]),
            candidate_revision_id=str(payload["candidate_revision_id"]),
            mechanism_id=str(payload["mechanism_id"]),
            observer_config_sha256=str(payload["observer_config_sha256"]),
            expected_internal_effect_json=canonical_json(effect),
            expected_internal_effect_sha256=str(
                payload["expected_internal_effect_sha256"]
            ),
        )

    @property
    def expected_internal_effect(self) -> Mapping[str, Any]:
        value = json.loads(self.expected_internal_effect_json)
        if not isinstance(value, dict):  # guaranteed by construction
            raise ContractViolation("expected internal effect must be an object")
        return value

    def as_payload(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "candidate_revision_id": self.candidate_revision_id,
            "mechanism_id": self.mechanism_id,
            "observer_config_sha256": self.observer_config_sha256,
            "expected_internal_effect": dict(self.expected_internal_effect),
            "expected_internal_effect_sha256": self.expected_internal_effect_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    plan_id: str
    campaign_id: str
    strategy_id: str
    task: TaskRevision
    candidate_revision_id: str
    arm: str
    model: ModelIdentity
    context_policy_id: str
    tool_policy_id: str
    observer_policy_ids: tuple[str, ...]
    native_evaluator_id: str
    limits: ExecutionLimits
    holdout_scope: str
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "plan_id",
            "campaign_id",
            "strategy_id",
            "candidate_revision_id",
            "arm",
            "context_policy_id",
            "tool_policy_id",
            "native_evaluator_id",
            "holdout_scope",
        ):
            _require_text(name, getattr(self, name))
        if not self.observer_policy_ids:
            raise ContractViolation("observer_policy_ids must not be empty")
        if (
            self.task.cohort is not Cohort.FEEDBACK
            and self.holdout_scope == "feedback-only"
        ):
            raise ContractViolation(
                "feedback-only plan cannot reference a non-feedback task"
            )

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class Authorization:
    authorization_id: str
    campaign_id: str
    allowed_cohorts: tuple[Cohort, ...]
    max_cost_cny: float
    max_model_calls: int
    expires_at: datetime
    remote_calls_allowed: bool

    def __post_init__(self) -> None:
        _require_text("authorization_id", self.authorization_id)
        _require_text("campaign_id", self.campaign_id)
        if not self.allowed_cohorts:
            raise ContractViolation("allowed_cohorts must not be empty")
        if (
            Cohort.BURNED in self.allowed_cohorts
            or Cohort.FINAL_SEALED in self.allowed_cohorts
        ):
            raise ContractViolation(
                "burned/final-sealed cohorts cannot be authorized here"
            )
        if self.max_cost_cny < 0 or self.max_model_calls < 0:
            raise ContractViolation("authorization budgets must be non-negative")
        if self.expires_at.tzinfo is None:
            raise ContractViolation("expires_at must be timezone-aware")

    def assert_allows(
        self,
        *,
        cohort: Cohort,
        reserved_cost_cny: float,
        reserved_model_calls: int,
        remote: bool,
        now: datetime | None = None,
    ) -> None:
        checked_at = now or datetime.now(UTC)
        if checked_at >= self.expires_at:
            raise ContractViolation("authorization expired")
        if cohort not in self.allowed_cohorts:
            raise ContractViolation(f"cohort {cohort} is not authorized")
        if reserved_cost_cny > self.max_cost_cny:
            raise ContractViolation("cost budget exceeded")
        if reserved_model_calls > self.max_model_calls:
            raise ContractViolation("model call budget exceeded")
        if remote and not self.remote_calls_allowed:
            raise ContractViolation("remote calls are not authorized")


@dataclass(frozen=True, slots=True)
class Receipt:
    receipt_id: str
    campaign_id: str
    plan_id: str
    sequence: int
    kind: str
    created_at: str
    payload: Mapping[str, Any]
    artifact_sha256: str
    supersedes_receipt_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("receipt_id", "campaign_id", "plan_id", "kind", "created_at"):
            _require_text(name, getattr(self, name))
        if self.sequence < 1:
            raise ContractViolation("receipt sequence must start at one")
        _require_sha256("artifact_sha256", self.artifact_sha256)

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    evidence_id: str
    receipt_ids: tuple[str, ...]
    observer_id: str
    grade: ClaimGrade
    payload: Mapping[str, Any]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require_text("evidence_id", self.evidence_id)
        _require_text("observer_id", self.observer_id)
        if not self.receipt_ids:
            raise ContractViolation("evidence must reference at least one receipt")
        _require_sha256("artifact_sha256", self.artifact_sha256)

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class CounterfactualArmEvidence:
    """Immutable references proving one arm from model output to native outcome."""

    arm: str
    campaign_id: str
    plan_id: str
    model_receipt_id: str
    model_receipt_sha256: str
    model_artifact_sha256: str
    external_trace_evidence_id: str
    external_trace_evidence_sha256: str
    external_trace_receipt_id: str
    external_trace_artifact_sha256: str
    native_outcome_evidence_id: str
    native_outcome_evidence_sha256: str
    native_outcome_receipt_id: str
    native_outcome_artifact_sha256: str
    prediction_sha256: str
    prompt_bundle_sha256: str
    candidate_prompt_sha256: str | None
    candidate_consumed: bool
    candidate_revision_id: str | None
    candidate_bundle_sha256: str | None

    def __post_init__(self) -> None:
        if self.arm not in {"baseline", "taught"}:
            raise ContractViolation("counterfactual arm must be baseline or taught")
        for name in (
            "campaign_id",
            "plan_id",
            "model_receipt_id",
            "external_trace_evidence_id",
            "external_trace_receipt_id",
            "native_outcome_evidence_id",
            "native_outcome_receipt_id",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "model_receipt_sha256",
            "model_artifact_sha256",
            "external_trace_evidence_sha256",
            "external_trace_artifact_sha256",
            "native_outcome_evidence_sha256",
            "native_outcome_artifact_sha256",
            "prediction_sha256",
            "prompt_bundle_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.arm == "baseline":
            if (
                self.candidate_consumed
                or self.candidate_revision_id is not None
                or self.candidate_bundle_sha256 is not None
                or self.candidate_prompt_sha256 is not None
            ):
                raise ContractViolation("baseline arm contains candidate lineage")
        else:
            if not self.candidate_consumed:
                raise ContractViolation("taught arm did not consume candidate")
            _require_text("candidate_revision_id", self.candidate_revision_id or "")
            _require_sha256(
                "candidate_bundle_sha256", self.candidate_bundle_sha256 or ""
            )
            _require_sha256(
                "candidate_prompt_sha256", self.candidate_prompt_sha256 or ""
            )

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class MatchedCounterfactualPair:
    """Complete baseline/taught lineage for one causal native comparison."""

    candidate_id: str
    candidate_revision_id: str
    candidate_bundle_sha256: str
    campaign_id: str
    task_revision_id: str
    task_source_sha256: str
    model_identity: str
    native_evaluator_id: str
    execution_config_sha256: str
    baseline: CounterfactualArmEvidence
    taught: CounterfactualArmEvidence

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "candidate_revision_id",
            "campaign_id",
            "task_revision_id",
            "model_identity",
            "native_evaluator_id",
        ):
            _require_text(name, getattr(self, name))
        for name in (
            "candidate_bundle_sha256",
            "task_source_sha256",
            "execution_config_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.baseline.arm != "baseline" or self.taught.arm != "taught":
            raise ContractViolation("counterfactual pair arm ordering is invalid")
        if self.baseline.campaign_id != self.campaign_id:
            raise ContractViolation("baseline campaign identity does not match pair")
        if self.taught.campaign_id != self.campaign_id:
            raise ContractViolation("taught campaign identity does not match pair")
        if self.taught.candidate_revision_id != self.candidate_revision_id:
            raise ContractViolation("taught candidate revision does not match pair")
        if self.taught.candidate_bundle_sha256 != self.candidate_bundle_sha256:
            raise ContractViolation("taught candidate bundle does not match pair")
        if self.baseline.plan_id == self.taught.plan_id:
            raise ContractViolation("counterfactual arms must be separate executions")
        if len(set(self.evidence_ids)) != 4 or len(set(self.receipt_ids)) != 6:
            raise ContractViolation(
                "counterfactual pair contains duplicate evidence or receipts"
            )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return (
            self.baseline.external_trace_evidence_id,
            self.baseline.native_outcome_evidence_id,
            self.taught.external_trace_evidence_id,
            self.taught.native_outcome_evidence_id,
        )

    @property
    def receipt_ids(self) -> tuple[str, ...]:
        return (
            self.baseline.model_receipt_id,
            self.baseline.external_trace_receipt_id,
            self.baseline.native_outcome_receipt_id,
            self.taught.model_receipt_id,
            self.taught.external_trace_receipt_id,
            self.taught.native_outcome_receipt_id,
        )

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    candidate_id: str
    grade: ClaimGrade
    classification: ClaimClassification
    evidence_ids: tuple[str, ...]
    rationale: str
    supersedes_claim_id: str | None
    counterfactual_pair_sha256: str | None = dataclasses.field(
        default=None, metadata={"omit_if_none": True}
    )
    counterfactual_receipt_ids: tuple[str, ...] = dataclasses.field(
        default=(), metadata={"omit_if_empty": True}
    )
    _legacy_read: dataclasses.InitVar[bool] = False

    def __post_init__(self, _legacy_read: bool) -> None:
        _require_text("claim_id", self.claim_id)
        _require_text("candidate_id", self.candidate_id)
        _require_text("rationale", self.rationale)
        if not self.evidence_ids:
            raise ContractViolation("claim must reference evidence")
        if self.counterfactual_pair_sha256 is not None:
            _require_sha256(
                "counterfactual_pair_sha256", self.counterfactual_pair_sha256
            )
        if self.counterfactual_receipt_ids:
            for receipt_id in self.counterfactual_receipt_ids:
                _require_text("counterfactual receipt id", receipt_id)
        if self.grade in {ClaimGrade.E2, ClaimGrade.E3}:
            if self.counterfactual_pair_sha256 is None:
                if not _legacy_read:
                    raise ContractViolation(
                        "E2/E3 claim requires complete counterfactual lineage"
                    )
            elif (
                len(self.evidence_ids) != 4 or len(self.counterfactual_receipt_ids) != 6
            ):
                raise ContractViolation(
                    "E2/E3 claim requires all counterfactual evidence and receipts"
                )

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)
