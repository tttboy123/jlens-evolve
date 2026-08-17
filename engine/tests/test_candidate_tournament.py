from __future__ import annotations

import hashlib

import pytest

from candidate_tournament import (
    ArmEvaluation,
    CandidateTournament,
    SearchParentSelector,
    TournamentContractError,
)

ORIGINAL = "a" * 64
PARENT = "b" * 64
CANDIDATES = tuple(char * 64 for char in "cdef")
EVIDENCE = "1" * 64
EPOCH = "native-adapters-v2.1.0-frozen"


def _contract(task_uid: str) -> str:
    return hashlib.sha256(task_uid.encode()).hexdigest()


def _row(
    task_uid: str,
    *,
    role: str,
    agent: str,
    score: float,
    cost: float = 1.0,
    safe: bool = True,
    family: str = "verified",
) -> dict:
    return {
        "schema_version": 1,
        "task_uid": task_uid,
        "benchmark_family": family,
        "role": role,
        "agent_program_sha256": agent,
        "matched_contract_sha256": _contract(task_uid),
        "native_evaluator_epoch": EPOCH,
        "native_score": score,
        "safety_passed": safe,
        "cost_units": cost,
        "evidence_sha256": EVIDENCE,
    }


def _stage_rows(
    task_uids: tuple[str, ...],
    candidate_scores: dict[str, float],
    *,
    parent_score: float = 0.6,
    original_score: float = 0.7,
    candidate_cost: float = 1.0,
) -> list[ArmEvaluation]:
    rows = []
    for task_uid in task_uids:
        rows.extend(
            [
                ArmEvaluation.from_dict(
                    _row(
                        task_uid, role="original", agent=ORIGINAL, score=original_score
                    )
                ),
                ArmEvaluation.from_dict(
                    _row(task_uid, role="parent", agent=PARENT, score=parent_score)
                ),
            ]
        )
        rows.extend(
            ArmEvaluation.from_dict(
                _row(
                    task_uid,
                    role="candidate",
                    agent=candidate,
                    score=score,
                    cost=candidate_cost,
                )
            )
            for candidate, score in candidate_scores.items()
        )
    return rows


def test_arm_contract_rejects_jlens_or_rank_inputs():
    payload = _row("task-1", role="candidate", agent=CANDIDATES[0], score=1.0)
    payload["jlens_score"] = 99
    with pytest.raises(TournamentContractError, match="unknown arm fields"):
        ArmEvaluation.from_dict(payload)


def test_stage_requires_complete_matched_original_parent_and_candidate_arms():
    tasks = tuple(f"task-{index}" for index in range(5))
    rows = _stage_rows(
        tasks,
        {
            candidate: score
            for candidate, score in zip(CANDIDATES, (0.9, 0.8, 0.7, 0.6))
        },
    )
    rows.pop()

    with pytest.raises(TournamentContractError, match="arm inventory"):
        CandidateTournament().evaluate_stage(
            stage="scout",
            evaluations=rows,
            task_uids=tasks,
            original_sha256=ORIGINAL,
            parent_sha256=PARENT,
            candidate_sha256s=CANDIDATES,
            advance_count=2,
        )


def test_stage_rejects_non_matched_contract_hashes():
    tasks = ("task-1",)
    rows = _stage_rows(tasks, {CANDIDATES[0]: 0.9})
    rows[-1] = ArmEvaluation.from_dict(
        {
            **rows[-1].to_dict(),
            "matched_contract_sha256": "9" * 64,
        }
    )

    with pytest.raises(TournamentContractError, match="matched contract"):
        CandidateTournament().evaluate_stage(
            stage="scout",
            evaluations=rows,
            task_uids=tasks,
            original_sha256=ORIGINAL,
            parent_sha256=PARENT,
            candidate_sha256s=(CANDIDATES[0],),
            advance_count=1,
        )


def test_successive_halving_is_deterministic_and_reports_dual_comparisons():
    scout_tasks = tuple(f"scout-{index}" for index in range(5))
    scout = CandidateTournament().evaluate_stage(
        stage="scout",
        evaluations=_stage_rows(
            scout_tasks,
            {
                candidate: score
                for candidate, score in zip(CANDIDATES, (0.9, 0.8, 0.7, 0.6))
            },
        ),
        task_uids=scout_tasks,
        original_sha256=ORIGINAL,
        parent_sha256=PARENT,
        candidate_sha256s=tuple(reversed(CANDIDATES)),
        advance_count=2,
    )
    assert scout.finalists == CANDIDATES[:2]

    semi_tasks = tuple(f"semi-{index}" for index in range(8))
    semifinal = CandidateTournament().evaluate_stage(
        stage="semifinal",
        evaluations=_stage_rows(
            semi_tasks,
            {CANDIDATES[0]: 0.85, CANDIDATES[1]: 0.75},
        ),
        task_uids=semi_tasks,
        original_sha256=ORIGINAL,
        parent_sha256=PARENT,
        candidate_sha256s=scout.finalists,
        advance_count=1,
    )

    assert semifinal.finalists == (CANDIDATES[0],)
    comparison = semifinal.comparisons[CANDIDATES[0]]
    assert comparison["vs_original"]["wins"] == 8
    assert comparison["vs_parent"]["wins"] == 8
    assert "jlens" not in str(semifinal.to_dict()).lower()


def test_confirmation_gate_advances_experimental_parent_only_when_all_rules_pass():
    tasks = tuple(f"confirm-{index}" for index in range(12))
    tournament = CandidateTournament().evaluate_stage(
        stage="confirmation",
        evaluations=_stage_rows(
            tasks,
            {CANDIDATES[0]: 0.82},
            parent_score=0.6,
            original_score=0.7,
            candidate_cost=1.05,
        ),
        task_uids=tasks,
        original_sha256=ORIGINAL,
        parent_sha256=PARENT,
        candidate_sha256s=(CANDIDATES[0],),
        advance_count=1,
    )

    decision = SearchParentSelector().decide(tournament)

    assert decision.advance is True
    assert decision.previous_parent_sha256 == PARENT
    assert decision.search_parent_sha256 == CANDIDATES[0]
    assert decision.scope == "experimental_search_lineage_only"
    assert decision.production_promoted is False


@pytest.mark.parametrize("failure", ["safety", "family", "wins", "cost"])
def test_confirmation_gate_retains_parent_on_any_predeclared_failure(failure):
    tasks = tuple(f"confirm-{index}" for index in range(12))
    rows = _stage_rows(
        tasks,
        {CANDIDATES[0]: 0.8},
        parent_score=0.6,
        original_score=0.7,
        candidate_cost=1.05,
    )
    if failure == "safety":
        target = next(row for row in rows if row.role == "candidate")
        rows[rows.index(target)] = ArmEvaluation.from_dict(
            {**target.to_dict(), "safety_passed": False}
        )
    elif failure == "family":
        for index, row in enumerate(rows):
            if row.role == "candidate" and row.task_uid == tasks[0]:
                rows[index] = ArmEvaluation.from_dict(
                    {**row.to_dict(), "native_score": 0.0, "benchmark_family": "rare"}
                )
            elif row.task_uid == tasks[0]:
                rows[index] = ArmEvaluation.from_dict(
                    {**row.to_dict(), "benchmark_family": "rare"}
                )
    elif failure == "wins":
        for index, row in enumerate(rows):
            if row.role == "candidate" and int(row.task_uid.rsplit("-", 1)[1]) < 7:
                rows[index] = ArmEvaluation.from_dict(
                    {**row.to_dict(), "native_score": 0.5}
                )
    else:
        for index, row in enumerate(rows):
            if row.role == "candidate":
                rows[index] = ArmEvaluation.from_dict(
                    {**row.to_dict(), "cost_units": 1.25, "native_score": 0.65}
                )

    tournament = CandidateTournament().evaluate_stage(
        stage="confirmation",
        evaluations=rows,
        task_uids=tasks,
        original_sha256=ORIGINAL,
        parent_sha256=PARENT,
        candidate_sha256s=(CANDIDATES[0],),
        advance_count=1,
    )
    decision = SearchParentSelector().decide(tournament)

    assert decision.advance is False
    assert decision.search_parent_sha256 == PARENT
    assert decision.reasons
