from __future__ import annotations

from pathlib import Path

import pytest

from skill_registry import (
    SkillCandidate,
    SkillContractError,
    SkillEvidenceRef,
    SkillRegistry,
)


def _candidate(source: Path, *, status: str = "candidate") -> SkillCandidate:
    return SkillCandidate.create(
        skill_id="record-cleaning-invariants-v2",
        revision_id=f"record-cleaning-invariants-v2-{status}",
        parent_revision_id=None,
        status=status,
        status_reason="predeclared source distillation",
        task_family="record-cleaning",
        content=(
            "Normalize categorical fields before filtering.",
            "Aggregate validated finite positive values before rounding.",
        ),
        source_task_ids=("transaction-record-cleaning-v1",),
        applicability={
            "required_semantics": [
                "identity",
                "positive_numeric_value",
                "accepted_status",
            ]
        },
        counterexamples=("Do not apply when negative values are meaningful.",),
        known_failure_modes=("Task descriptors can bind the wrong fields.",),
        evidence_refs=(SkillEvidenceRef.from_path(source, role="source_evidence"),),
        project_local_only=True,
        auto_install=False,
        active=False,
    )


def test_candidate_rejects_install_or_active_authority(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    candidate = _candidate(source)

    with pytest.raises(SkillContractError, match="project-local"):
        SkillCandidate.create(
            **{
                **candidate.constructor_fields(),
                "project_local_only": False,
            }
        )
    with pytest.raises(SkillContractError, match="install"):
        SkillCandidate.create(
            **{
                **candidate.constructor_fields(),
                "auto_install": True,
            }
        )
    with pytest.raises(SkillContractError, match="active"):
        SkillCandidate.create(
            **{
                **candidate.constructor_fields(),
                "active": True,
            }
        )


def test_registry_is_append_only_idempotent_and_filters_rejected(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "project-registry")
    candidate = _candidate(source)

    assert registry.append(candidate) is True
    assert registry.append(candidate) is False
    selected = registry.retrieve(
        task_family="record-cleaning",
        semantics={"identity", "positive_numeric_value", "accepted_status"},
    )
    assert [row.skill_id for row in selected] == [candidate.skill_id]

    rejected = registry.transition(
        skill_id=candidate.skill_id,
        new_status="rejected",
        reason="matched transfer regressed",
        evidence_refs=candidate.evidence_refs,
    )
    assert rejected.parent_revision_id == candidate.revision_id
    assert (
        registry.retrieve(
            task_family="record-cleaning",
            semantics={"identity", "positive_numeric_value", "accepted_status"},
        )
        == []
    )
    assert len(registry.read_revisions()) == 2


def test_verified_transition_stays_project_local_and_is_rendered_for_review(
    tmp_path: Path,
):
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "registry")
    candidate = _candidate(source)
    registry.append(candidate)

    verified = registry.transition(
        skill_id=candidate.skill_id,
        new_status="transfer_verified",
        reason="two-task sealed matched A/B passed",
        evidence_refs=candidate.evidence_refs,
    )
    path = registry.render_for_review(verified.skill_id)

    assert verified.active is False
    assert verified.auto_install is False
    assert path.is_relative_to(registry.root)
    assert "status: transfer_verified" in path.read_text(encoding="utf-8")


def test_illegal_terminal_transition_is_rejected(tmp_path: Path):
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "registry")
    registry.append(_candidate(source, status="rejected"))

    with pytest.raises(SkillContractError, match="transition"):
        registry.transition(
            skill_id="record-cleaning-invariants-v2",
            new_status="transfer_verified",
            reason="cannot revive rejected evidence",
            evidence_refs=(),
        )
