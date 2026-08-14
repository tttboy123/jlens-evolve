"""Immutable product asset records owned by v3 registries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evolve.contracts import content_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RegistryViolation(ValueError):
    """A registry record or operation violates an immutability invariant."""


def _validate_record(
    identity: tuple[str, ...], artifact_sha256: str, active: bool
) -> None:
    if not all(isinstance(value, str) and value.strip() for value in identity):
        raise RegistryViolation("registry identities must be non-empty text")
    if _SHA256.fullmatch(artifact_sha256) is None:
        raise RegistryViolation("artifact_sha256 must be a literal lowercase SHA-256")
    if active:
        raise RegistryViolation("new registry revisions must be inactive")


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    revision_id: str
    candidate_kind: str
    source_claim_ids: tuple[str, ...]
    artifact_sha256: str
    active: bool = False

    def __post_init__(self) -> None:
        _validate_record(
            (self.candidate_id, self.revision_id, self.candidate_kind),
            self.artifact_sha256,
            self.active,
        )
        if not self.source_claim_ids:
            raise RegistryViolation(
                "candidate must reference at least one source claim"
            )

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate_id, self.revision_id

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    capability_id: str
    revision_id: str
    capability_kind: str
    evidence_claim_ids: tuple[str, ...]
    artifact_sha256: str
    promotion_decision_id: str | None = None
    source_candidate_id: str | None = None
    active: bool = False

    def __post_init__(self) -> None:
        _validate_record(
            (self.capability_id, self.revision_id, self.capability_kind),
            self.artifact_sha256,
            self.active,
        )
        if not self.evidence_claim_ids:
            raise RegistryViolation("capability must reference evidence claims")
        if (
            self.promotion_decision_id is not None
            and not self.promotion_decision_id.strip()
        ):
            raise RegistryViolation("promotion_decision_id must be non-empty")
        if (
            self.source_candidate_id is not None
            and not self.source_candidate_id.strip()
        ):
            raise RegistryViolation("source_candidate_id must be non-empty")

    @property
    def key(self) -> tuple[str, str]:
        return self.capability_id, self.revision_id

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    candidate_id: str
    revision_id: str
    candidate_kind: str
    evidence_claim_ids: tuple[str, ...]
    promotion_decision_id: str
    reason: str
    artifact_sha256: str
    active: bool = False

    def __post_init__(self) -> None:
        _validate_record(
            (self.candidate_id, self.revision_id, self.candidate_kind),
            self.artifact_sha256,
            self.active,
        )
        if not self.evidence_claim_ids:
            raise RegistryViolation("rejected candidate must reference evidence claims")
        if not self.promotion_decision_id.strip() or not self.reason.strip():
            raise RegistryViolation("rejection must reference a decision and reason")

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate_id, self.revision_id

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class AgentProgramRecord:
    program_id: str
    revision_id: str
    parent_revision_id: str | None
    capability_revision_ids: tuple[str, ...]
    artifact_sha256: str
    active: bool = False

    def __post_init__(self) -> None:
        _validate_record(
            (self.program_id, self.revision_id), self.artifact_sha256, self.active
        )
        if self.parent_revision_id == self.revision_id:
            raise RegistryViolation("AgentProgram revision cannot be its own parent")

    @property
    def key(self) -> tuple[str, str]:
        return self.program_id, self.revision_id

    @property
    def content_sha256(self) -> str:
        return content_sha256(self)
