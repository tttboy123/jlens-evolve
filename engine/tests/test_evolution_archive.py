from __future__ import annotations

import json
from dataclasses import replace

import pytest

from evolution_archive import ArchiveContractError, CandidateArchive
from mutation_proposer import InactiveChangeSet

ORIGINAL = "a" * 64
PARENT = "b" * 64
CANDIDATE_1 = "c" * 64
CANDIDATE_2 = "d" * 64
EVIDENCE = "e" * 64
PATCH = "f" * 64
ROLLBACK = "1" * 64


def _changeset(
    changeset_id: str = "cs-1",
    *,
    candidate: str = CANDIDATE_1,
    parent: str = PARENT,
) -> InactiveChangeSet:
    return InactiveChangeSet(
        schema_version=1,
        changeset_id=changeset_id,
        status="inactive",
        parent_agent_program_sha256=parent,
        candidate_agent_program_sha256=candidate,
        hypothesis_ids=("pattern-1",),
        surface="prompt",
        operations=({"op": "replace", "path": "AGENTS.md", "after": "new"},),
        rollback_operations=({"op": "replace", "path": "AGENTS.md", "after": "old"},),
        proposer={"platform": "codex", "model": "gpt"},
        native_evaluator_epoch="native-adapters-v2.1.0-frozen",
        native_evaluator_authority="external_fixed",
        auto_apply=False,
        production_promotion_allowed=False,
    )


def test_candidate_blob_is_immutable_and_duplicate_registration_is_idempotent(tmp_path):
    archive = CandidateArchive.create(
        tmp_path / "archive",
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
    )
    proposal = _changeset()

    first = archive.register_candidate(proposal)
    second = archive.register_candidate(proposal)

    assert first == second
    assert archive.candidate_state(CANDIDATE_1) == "inactive"
    assert len(archive.events()) == 1
    with pytest.raises(ArchiveContractError, match="immutable"):
        archive.register_candidate(replace(proposal, hypothesis_ids=("other",)))


def test_archive_keeps_selected_rejected_and_failed_candidates(tmp_path):
    archive = CandidateArchive.create(
        tmp_path / "archive",
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
    )
    archive.register_candidate(_changeset())
    archive.register_candidate(_changeset("cs-2", candidate=CANDIDATE_2))
    failed = "2" * 64
    archive.register_candidate(_changeset("cs-3", candidate=failed))

    archive.transition(
        CANDIDATE_1, "evaluating", reason="scout", evidence_sha256=EVIDENCE
    )
    archive.transition(
        CANDIDATE_1, "selected", reason="winner", evidence_sha256=EVIDENCE
    )
    archive.transition(
        CANDIDATE_2, "evaluating", reason="scout", evidence_sha256=EVIDENCE
    )
    archive.transition(
        CANDIDATE_2, "rejected", reason="quality regression", evidence_sha256=EVIDENCE
    )
    archive.transition(
        failed, "failed", reason="execution error", evidence_sha256=EVIDENCE
    )

    assert archive.candidate_state(CANDIDATE_1) == "selected"
    assert archive.candidate_state(CANDIDATE_2) == "rejected"
    assert archive.candidate_state(failed) == "failed"
    assert set(archive.candidates()) == {CANDIDATE_1, CANDIDATE_2, failed}


def test_lineage_rejects_missing_parent_self_cycle_and_illegal_transition(tmp_path):
    archive = CandidateArchive.create(
        tmp_path / "archive",
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
    )
    with pytest.raises(ArchiveContractError, match="parent"):
        archive.register_candidate(_changeset(parent="9" * 64))
    with pytest.raises(ArchiveContractError, match="cycle"):
        archive.register_candidate(_changeset(parent=CANDIDATE_1))

    archive.register_candidate(_changeset())
    with pytest.raises(ArchiveContractError, match="transition"):
        archive.transition(
            CANDIDATE_1, "selected", reason="skip evaluation", evidence_sha256=EVIDENCE
        )


def test_event_hash_chain_detects_tampering(tmp_path):
    archive = CandidateArchive.create(
        tmp_path / "archive",
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
    )
    archive.register_candidate(_changeset())
    archive.transition(
        CANDIDATE_1, "failed", reason="execution error", evidence_sha256=EVIDENCE
    )
    assert archive.verify()["valid"] is True

    events_path = tmp_path / "archive/events.jsonl"
    rows = events_path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[-1])
    payload["reason"] = "tampered"
    rows[-1] = json.dumps(payload, sort_keys=True)
    events_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert archive.verify()["valid"] is False


def test_search_parent_advance_requires_selection_and_rollback_and_never_updates_production(
    tmp_path,
):
    archive = CandidateArchive.create(
        tmp_path / "archive",
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
    )
    archive.register_candidate(_changeset())
    archive.transition(
        CANDIDATE_1, "evaluating", reason="confirmation", evidence_sha256=EVIDENCE
    )
    archive.transition(
        CANDIDATE_1,
        "selected",
        reason="passed experimental gate",
        evidence_sha256=EVIDENCE,
    )

    with pytest.raises(ArchiveContractError, match="rollback"):
        archive.advance_search_parent(CANDIDATE_1, decision_sha256=EVIDENCE)
    archive.record_rollback(
        CANDIDATE_1,
        forward_patch_sha256=PATCH,
        rollback_patch_sha256=ROLLBACK,
        verified=True,
    )
    decision = archive.advance_search_parent(CANDIDATE_1, decision_sha256=EVIDENCE)

    assert decision["previous_parent_sha256"] == PARENT
    assert decision["search_parent_sha256"] == CANDIDATE_1
    assert archive.search_parent_sha256 == CANDIDATE_1
    assert archive.authority()["production_active_ref"] is None
    with pytest.raises(ArchiveContractError, match="production"):
        archive.advance_search_parent(
            CANDIDATE_1, decision_sha256=EVIDENCE, production=True
        )
