"""Strict, serializable contracts for loop-local infrastructure."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
FAILURE_REASON_CODES = frozenset(
    {
        "no-diff",
        "bad-header",
        "malformed-hunk",
        "apply-fail",
        "reasoning-only",
        "full-file",
        "wrong-target",
        "eval-infra",
        "native-unresolved",
        "regression",
        "json-malformed",
        "selector-no-match",
        "expression-used-for-statement",
        "no-op",
        "duplicate-file",
        "unrelated-symbol",
        "plan-too-large",
        "unresolved",
        "inconsistent-plan",
        "identifier-drift",
        "boundary-collapse",
        "semantic-oracle-mismatch",
        "non-executable-insertion",
        "schema-invalid",
        "selector-not-enumerated",
        "semantic-overbroad",
        "syntax-invalid",
        "unbound-name",
        "unsafe-empty-sequence",
    }
)


class ContractError(ValueError):
    """Raised before malformed or unauthorized loop state is accepted."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_identifier(label: str, value: str) -> None:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"invalid {label}")


def _require_nonempty(label: str, value: str) -> None:
    if not value.strip():
        raise ContractError(f"{label} must be non-empty")


@dataclass(frozen=True)
class LoopAuthorization:
    """Explicit human authorization and hard parent-call ceiling."""

    schema_version: int
    authorization_id: str
    approved_by: str
    maximum_parent_calls: int
    expires_at: datetime

    _FIELDS = frozenset(
        {
            "schema_version",
            "authorization_id",
            "approved_by",
            "maximum_parent_calls",
            "expires_at",
        }
    )

    @classmethod
    def create(
        cls,
        *,
        authorization_id: str,
        approved_by: str,
        maximum_parent_calls: int,
        expires_at: datetime,
    ) -> LoopAuthorization:
        authorization = cls(
            schema_version=1,
            authorization_id=authorization_id,
            approved_by=approved_by,
            maximum_parent_calls=maximum_parent_calls,
            expires_at=expires_at,
        )
        authorization.validate()
        return authorization

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopAuthorization:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid authorization fields")
        try:
            expires_at = datetime.fromisoformat(str(data["expires_at"]))
        except ValueError as exc:
            raise ContractError("invalid authorization expiry") from exc
        authorization = cls(
            schema_version=data["schema_version"],
            authorization_id=str(data["authorization_id"]),
            approved_by=str(data["approved_by"]),
            maximum_parent_calls=data["maximum_parent_calls"],
            expires_at=expires_at,
        )
        authorization.validate()
        return authorization

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported authorization schema")
        _require_identifier("authorization_id", self.authorization_id)
        _require_nonempty("approved_by", self.approved_by)
        if type(self.maximum_parent_calls) is not int or self.maximum_parent_calls < 1:
            raise ContractError("maximum_parent_calls must be a positive integer")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise ContractError("authorization expiry must be timezone-aware")

    def assert_active(self, *, now: datetime | None = None) -> None:
        instant = now or datetime.now(UTC)
        if instant >= self.expires_at:
            raise ContractError("authorization has expired")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authorization_id": self.authorization_id,
            "approved_by": self.approved_by,
            "maximum_parent_calls": self.maximum_parent_calls,
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass(frozen=True)
class LoopRevision:
    """One immutable loop-local Skill/protocol revision."""

    schema_version: int
    skill_id: str
    revision_id: str
    parent_revision_id: str | None
    source_round: int
    protocol: str
    skill_text: str
    prompt_template: str
    eval_note: str

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "skill_id",
            "revision_id",
            "parent_revision_id",
            "source_round",
            "protocol",
            "skill_text",
            "prompt_template",
            "eval_note",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(cls, **fields: Any) -> LoopRevision:
        revision = cls(schema_version=1, **fields)
        revision.validate()
        return revision

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopRevision:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid loop revision fields")
        revision = cls(
            schema_version=data["schema_version"],
            skill_id=str(data["skill_id"]),
            revision_id=str(data["revision_id"]),
            parent_revision_id=(
                None
                if data["parent_revision_id"] is None
                else str(data["parent_revision_id"])
            ),
            source_round=data["source_round"],
            protocol=str(data["protocol"]),
            skill_text=str(data["skill_text"]),
            prompt_template=str(data["prompt_template"]),
            eval_note=str(data["eval_note"]),
        )
        revision.validate()
        if data["fingerprint"] != revision.fingerprint:
            raise ContractError("loop revision fingerprint mismatch")
        return revision

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported loop revision schema")
        _require_identifier("skill_id", self.skill_id)
        _require_identifier("revision_id", self.revision_id)
        if self.parent_revision_id is not None:
            _require_identifier("parent_revision_id", self.parent_revision_id)
        if type(self.source_round) is not int or self.source_round < 0:
            raise ContractError("source_round must be a non-negative integer")
        for label, value in (
            ("protocol", self.protocol),
            ("skill_text", self.skill_text),
            ("prompt_template", self.prompt_template),
            ("eval_note", self.eval_note),
        ):
            _require_nonempty(label, value)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "revision_id": self.revision_id,
            "parent_revision_id": self.parent_revision_id,
            "source_round": self.source_round,
            "protocol": self.protocol,
            "skill_text": self.skill_text,
            "prompt_template": self.prompt_template,
            "eval_note": self.eval_note,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class FailureEvidence:
    """One classified student-arm failure with content-addressed evidence."""

    task_id: str
    reason_code: str
    diagnostic_summary: str
    raw_output_sha256: str
    extracted_edit_sha256: str | None
    apply_error: str | None

    _FIELDS = frozenset(
        {
            "task_id",
            "reason_code",
            "diagnostic_summary",
            "raw_output_sha256",
            "extracted_edit_sha256",
            "apply_error",
        }
    )

    @classmethod
    def create(cls, **fields: Any) -> FailureEvidence:
        evidence = cls(**fields)
        evidence.validate()
        return evidence

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureEvidence:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid failure evidence fields")
        evidence = cls(
            task_id=str(data["task_id"]),
            reason_code=str(data["reason_code"]),
            diagnostic_summary=str(data["diagnostic_summary"]),
            raw_output_sha256=str(data["raw_output_sha256"]),
            extracted_edit_sha256=data["extracted_edit_sha256"],
            apply_error=data["apply_error"],
        )
        evidence.validate()
        return evidence

    def validate(self) -> None:
        _require_identifier("task_id", self.task_id)
        if self.reason_code not in FAILURE_REASON_CODES:
            raise ContractError("invalid failure reason code")
        _require_nonempty("diagnostic_summary", self.diagnostic_summary)
        if _SHA256.fullmatch(self.raw_output_sha256) is None:
            raise ContractError("invalid raw output sha256")
        if self.extracted_edit_sha256 is not None and (
            _SHA256.fullmatch(self.extracted_edit_sha256) is None
        ):
            raise ContractError("invalid extracted edit sha256")
        if self.apply_error is not None and not self.apply_error.strip():
            raise ContractError("apply_error cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reason_code": self.reason_code,
            "diagnostic_summary": self.diagnostic_summary,
            "raw_output_sha256": self.raw_output_sha256,
            "extracted_edit_sha256": self.extracted_edit_sha256,
            "apply_error": self.apply_error,
        }


@dataclass(frozen=True)
class FeedbackPackage:
    """Versioned failure signal supplied to a parent model."""

    schema_version: int
    current_round: int
    reason_counts: dict[str, int]
    arm_evidence: tuple[FailureEvidence, ...]
    previous_eval_note: str
    no_progress: bool
    rejected_fingerprints: tuple[str, ...]

    _CONTENT_FIELDS = frozenset(
        {
            "schema_version",
            "current_round",
            "reason_counts",
            "arm_evidence",
            "previous_eval_note",
            "no_progress",
            "rejected_fingerprints",
        }
    )
    _FIELDS = _CONTENT_FIELDS | {"fingerprint"}

    @classmethod
    def create(
        cls,
        *,
        current_round: int,
        arm_evidence: list[FailureEvidence],
        previous_eval_note: str,
        no_progress: bool,
        rejected_fingerprints: list[str],
    ) -> FeedbackPackage:
        counts: dict[str, int] = {}
        for evidence in arm_evidence:
            counts[evidence.reason_code] = counts.get(evidence.reason_code, 0) + 1
        package = cls(
            schema_version=1,
            current_round=current_round,
            reason_counts=counts,
            arm_evidence=tuple(arm_evidence),
            previous_eval_note=previous_eval_note,
            no_progress=no_progress,
            rejected_fingerprints=tuple(rejected_fingerprints),
        )
        package.validate()
        return package

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackPackage:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid feedback package fields")
        package = cls(
            schema_version=data["schema_version"],
            current_round=data["current_round"],
            reason_counts=dict(data["reason_counts"]),
            arm_evidence=tuple(
                FailureEvidence.from_dict(row) for row in data["arm_evidence"]
            ),
            previous_eval_note=str(data["previous_eval_note"]),
            no_progress=data["no_progress"],
            rejected_fingerprints=tuple(
                str(row) for row in data["rejected_fingerprints"]
            ),
        )
        package.validate()
        if data["fingerprint"] != package.fingerprint:
            raise ContractError("feedback package fingerprint mismatch")
        return package

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported feedback package schema")
        if type(self.current_round) is not int or self.current_round < 0:
            raise ContractError("current_round must be a non-negative integer")
        if not self.arm_evidence:
            raise ContractError("feedback package requires arm evidence")
        expected: dict[str, int] = {}
        for evidence in self.arm_evidence:
            evidence.validate()
            expected[evidence.reason_code] = expected.get(evidence.reason_code, 0) + 1
        if self.reason_counts != expected:
            raise ContractError("reason_counts do not match arm evidence")
        _require_nonempty("previous_eval_note", self.previous_eval_note)
        if type(self.no_progress) is not bool:
            raise ContractError("no_progress must be boolean")
        if any(_SHA256.fullmatch(row) is None for row in self.rejected_fingerprints):
            raise ContractError("invalid rejected revision fingerprint")
        if len(set(self.rejected_fingerprints)) != len(self.rejected_fingerprints):
            raise ContractError("rejected revision fingerprints must be unique")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "current_round": self.current_round,
            "reason_counts": self.reason_counts,
            "arm_evidence": [row.to_dict() for row in self.arm_evidence],
            "previous_eval_note": self.previous_eval_note,
            "no_progress": self.no_progress,
            "rejected_fingerprints": list(self.rejected_fingerprints),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class ParentModelRequest:
    """Replayable input to a parent model."""

    schema_version: int
    feedback: FeedbackPackage
    current_revision: LoopRevision

    _FIELDS = frozenset({"schema_version", "feedback", "current_revision"})

    @classmethod
    def create(
        cls, *, feedback: FeedbackPackage, current_revision: LoopRevision
    ) -> ParentModelRequest:
        request = cls(
            schema_version=1,
            feedback=feedback,
            current_revision=current_revision,
        )
        request.validate()
        return request

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParentModelRequest:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid parent request fields")
        request = cls(
            schema_version=data["schema_version"],
            feedback=FeedbackPackage.from_dict(data["feedback"]),
            current_revision=LoopRevision.from_dict(data["current_revision"]),
        )
        request.validate()
        return request

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported parent request schema")
        self.feedback.validate()
        self.current_revision.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "feedback": self.feedback.to_dict(),
            "current_revision": self.current_revision.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class ParentModelResponse:
    """Validated parent output; no unknown fields are accepted."""

    schema_version: int
    protocol: str
    skill_text: str
    prompt_template: str
    eval_note: str
    usage: dict[str, Any]

    _FIELDS = frozenset(
        {
            "schema_version",
            "protocol",
            "skill_text",
            "prompt_template",
            "eval_note",
            "usage",
        }
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParentModelResponse:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise ContractError("invalid parent response fields")
        response = cls(
            schema_version=data["schema_version"],
            protocol=str(data["protocol"]),
            skill_text=str(data["skill_text"]),
            prompt_template=str(data["prompt_template"]),
            eval_note=str(data["eval_note"]),
            usage=dict(data["usage"]),
        )
        response.validate()
        return response

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported parent response schema")
        for label, value in (
            ("protocol", self.protocol),
            ("skill_text", self.skill_text),
            ("prompt_template", self.prompt_template),
            ("eval_note", self.eval_note),
        ):
            _require_nonempty(label, value)
        if not isinstance(self.usage, dict):
            raise ContractError("usage must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "skill_text": self.skill_text,
            "prompt_template": self.prompt_template,
            "eval_note": self.eval_note,
            "usage": self.usage,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())
