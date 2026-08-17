"""Tests for the v2.5 human review ladder (T4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from promotion_ladder import (
    PromotionLadder,
    PromotionLadderError,
    ReviewDecision,
)


def _tmp_ladder(tmp_path: Path) -> PromotionLadder:
    return PromotionLadder(tmp_path / "ladder")


def test_record_and_latest(tmp_path: Path) -> None:
    ladder = _tmp_ladder(tmp_path)
    decision = ReviewDecision.create(
        skill_id="search-candidate-abc-skills",
        revision_id="search-candidate-abc-skills-123456789abc",
        decision="reviewed",
        reviewer="human-reviewer-1",
        notes="evidence chain reviewed",
    )
    assert ladder.record(decision) is True
    assert ladder.record(decision) is False  # append-only, no duplicates
    latest = ladder.latest(decision.skill_id, decision.revision_id)
    assert latest is not None
    assert latest.decision == "reviewed"
    assert (
        ladder.effective_status(decision.skill_id, decision.revision_id) == "reviewed"
    )


def test_effective_status_flow(tmp_path: Path) -> None:
    ladder = _tmp_ladder(tmp_path)
    skill = "search-candidate-abc-skills"
    rev = "search-candidate-abc-skills-123456789abc"
    assert ladder.effective_status(skill, rev) == "candidate"
    ladder.record(
        ReviewDecision.create(
            skill_id=skill,
            revision_id=rev,
            decision="reviewed",
            reviewer="human-1",
        )
    )
    assert ladder.effective_status(skill, rev) == "reviewed"
    ladder.record(
        ReviewDecision.create(
            skill_id=skill,
            revision_id=rev,
            decision="active",
            reviewer="human-2",
        )
    )
    assert ladder.effective_status(skill, rev) == "active"


def test_reviewed_ttl_expiry(tmp_path: Path) -> None:
    ladder = _tmp_ladder(tmp_path)
    skill = "s-1"
    rev = "s-1-abc"
    ladder.record(
        ReviewDecision.create(
            skill_id=skill,
            revision_id=rev,
            decision="reviewed",
            reviewer="human-1",
            reviewed_at_utc="2026-01-01T00:00:00Z",
        )
    )
    # 40 days later -> expired
    assert (
        ladder.effective_status(skill, rev, now_utc="2026-02-10T00:00:00Z") == "expired"
    )
    # active is not subject to the review TTL
    ladder.record(
        ReviewDecision.create(
            skill_id=skill,
            revision_id=rev,
            decision="active",
            reviewer="human-2",
            reviewed_at_utc="2026-01-01T00:00:00Z",
        )
    )
    assert (
        ladder.effective_status(skill, rev, now_utc="2026-02-10T00:00:00Z") == "active"
    )


def test_rejected_retained(tmp_path: Path) -> None:
    ladder = _tmp_ladder(tmp_path)
    decision = ReviewDecision.create(
        skill_id="s-reject",
        revision_id="s-reject-abc",
        decision="rejected",
        reviewer="human-1",
        notes="fails evidence review",
    )
    ladder.record(decision)
    assert ladder.effective_status("s-reject", "s-reject-abc") == "rejected"


def test_reviewer_required() -> None:
    with pytest.raises(PromotionLadderError):
        ReviewDecision.create(
            skill_id="s",
            revision_id="s-abc",
            decision="active",
            reviewer="",
        )


def test_invalid_decision() -> None:
    with pytest.raises(PromotionLadderError):
        ReviewDecision.create(
            skill_id="s",
            revision_id="s-abc",
            decision="auto_active",  # never auto
            reviewer="human-1",
        )


def test_row_fingerprint_detects_tamper(tmp_path: Path) -> None:
    ladder = _tmp_ladder(tmp_path)
    decision = ReviewDecision.create(
        skill_id="s-fp",
        revision_id="s-fp-abc",
        decision="reviewed",
        reviewer="human-1",
    )
    ladder.record(decision)
    raw = ladder.path.read_text(encoding="utf-8")
    tampered = raw.replace('"reviewer":"human-1"', '"reviewer":"human-2"')
    ladder.path.write_text(tampered, encoding="utf-8")
    with pytest.raises(PromotionLadderError):
        ladder.read_decisions()
