"""Append-only human review ladder for project-local skill candidates (v2.5 T4).

States (never auto-active):
    candidate -> transfer_verified -> reviewed -> active(project-local, human only)
Every human decision is an append-only ledger row with reviewer identity and
an optional evidence ref.  The ladder never mutates a SkillCandidate; it only
records review decisions on top of the frozen registry transitions.

Boundaries (unchanged project invariants):
- weight-frozen: reviewing a skill never trains or changes a model;
- project_local_only / auto_install=false / active=false are registry-level
  hard constraints; "active" here means human-approved project-local enablement,
  recorded out-of-band, never auto-set on a SkillCandidate;
- evidence is append-only; negative decisions are retained;
- no final-sealed opening, no production/global promotion, no global skill install.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,191}")
_REVIEW_DECISIONS = {"reviewed", "active", "rejected"}
_REVIEW_TTL_DAYS = 30


class PromotionLadderError(ValueError):
    """Raised when a review decision violates the ladder contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class ReviewDecision:
    schema_version: int
    skill_id: str
    revision_id: str
    decision: str
    reviewer: str
    reviewed_at_utc: str
    notes: str
    evidence_refs: tuple[dict[str, str], ...]

    _FIELDS = frozenset(
        {
            "schema_version",
            "skill_id",
            "revision_id",
            "decision",
            "reviewer",
            "reviewed_at_utc",
            "notes",
            "evidence_refs",
            "row_fingerprint",
        }
    )

    @classmethod
    def create(
        cls,
        *,
        skill_id: str,
        revision_id: str,
        decision: str,
        reviewer: str,
        notes: str = "",
        evidence_refs: tuple[dict[str, str], ...] = (),
        reviewed_at_utc: str | None = None,
    ) -> ReviewDecision:
        row = cls(
            schema_version=1,
            skill_id=skill_id,
            revision_id=revision_id,
            decision=decision,
            reviewer=reviewer,
            reviewed_at_utc=reviewed_at_utc or _now_utc(),
            notes=notes,
            evidence_refs=evidence_refs,
        )
        row.validate()
        return row

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewDecision:
        if not isinstance(data, dict) or set(data) != cls._FIELDS:
            raise PromotionLadderError("invalid ReviewDecision fields")
        row = cls(
            schema_version=data["schema_version"],
            skill_id=str(data["skill_id"]),
            revision_id=str(data["revision_id"]),
            decision=str(data["decision"]),
            reviewer=str(data["reviewer"]),
            reviewed_at_utc=str(data["reviewed_at_utc"]),
            notes=str(data["notes"]),
            evidence_refs=tuple(dict(row) for row in data["evidence_refs"]),
        )
        row.validate()
        if data["row_fingerprint"] != row.row_fingerprint:
            raise PromotionLadderError("row fingerprint mismatch")
        return row

    @property
    def row_fingerprint(self) -> str:
        import hashlib

        stable = {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "revision_id": self.revision_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewed_at_utc": self.reviewed_at_utc,
            "notes": self.notes,
            "evidence_refs": self.evidence_refs,
        }
        return hashlib.sha256(_canonical_json(stable).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "revision_id": self.revision_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reviewed_at_utc": self.reviewed_at_utc,
            "notes": self.notes,
            "evidence_refs": list(self.evidence_refs),
            "row_fingerprint": self.row_fingerprint,
        }

    def validate(self) -> None:
        if self.schema_version != 1:
            raise PromotionLadderError("unsupported ladder schema")
        if _IDENTIFIER.fullmatch(self.skill_id) is None:
            raise PromotionLadderError("invalid skill_id")
        if _IDENTIFIER.fullmatch(self.revision_id) is None:
            raise PromotionLadderError("invalid revision_id")
        if self.decision not in _REVIEW_DECISIONS:
            raise PromotionLadderError("invalid review decision")
        if not self.reviewer.strip():
            raise PromotionLadderError("reviewer identity is required")
        try:
            datetime.fromisoformat(self.reviewed_at_utc.replace("Z", "+00:00"))
        except ValueError as error:
            raise PromotionLadderError("invalid reviewed_at_utc") from error
        for ref in self.evidence_refs:
            if set(ref) != {"path", "sha256", "role"}:
                raise PromotionLadderError("invalid evidence ref fields")
            if re.fullmatch(r"[0-9a-f]{64}", str(ref["sha256"])) is None:
                raise PromotionLadderError("invalid evidence sha256")


class PromotionLadder:
    """Append-only human review ledger keyed by skill_id + revision_id."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "PROMOTION-LEDGER.jsonl"

    def record(self, decision: ReviewDecision) -> bool:
        decision.validate()
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            existing = {
                json.loads(line)["row_fingerprint"] for line in handle if line.strip()
            }
            if decision.row_fingerprint in existing:
                return False
            handle.seek(0, os.SEEK_END)
            handle.write(_canonical_json(decision.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            return True

    def read_decisions(self) -> list[ReviewDecision]:
        if not self.path.exists():
            return []
        return [
            ReviewDecision.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def latest(self, skill_id: str, revision_id: str) -> ReviewDecision | None:
        matches = [
            row
            for row in self.read_decisions()
            if row.skill_id == skill_id and row.revision_id == revision_id
        ]
        return matches[-1] if matches else None

    def effective_status(
        self, skill_id: str, revision_id: str, *, now_utc: str | None = None
    ) -> str:
        """reviewed / active / rejected / candidate, with 30-day TTL on reviewed."""
        decision = self.latest(skill_id, revision_id)
        if decision is None:
            return "candidate"
        if decision.decision == "active":
            return "active"
        if decision.decision == "rejected":
            return "rejected"
        now = now_utc or _now_utc()
        try:
            reviewed_at = datetime.fromisoformat(
                decision.reviewed_at_utc.replace("Z", "+00:00")
            )
            now_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
        except ValueError as error:
            raise PromotionLadderError("invalid timestamp") from error
        if now_dt - reviewed_at > timedelta(days=_REVIEW_TTL_DAYS):
            return "expired"
        return "reviewed"

    def review_summary(self, skill_id: str, revision_id: str) -> dict[str, Any]:
        return {
            "skill_id": skill_id,
            "revision_id": revision_id,
            "effective_status": self.effective_status(skill_id, revision_id),
            "ttl_days": _REVIEW_TTL_DAYS,
            "latest_decision": (
                self.latest(skill_id, revision_id).to_dict()
                if self.latest(skill_id, revision_id)
                else None
            ),
        }


__all__ = [
    "PromotionLadder",
    "PromotionLadderError",
    "ReviewDecision",
]
