from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from evolve.agent_program import (
    AgentProgramRevision,
    AgentProgramViolation,
    DeterministicFixtureAgentProgramTransport,
    SearchParentLog,
    TournamentAuthority,
)
from evolve.cli import main
from evolve.contracts import (
    Cohort,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    content_sha256,
)
from evolve.runtime import ExecutionResult
from evolve.strategies import (
    AgentProgramSearchStrategy,
    StrategyContext,
    StrategyStatus,
)


def _revision(
    root: Path,
    *,
    revision_id: str,
    parent_revision_id: str | None,
    score: int,
) -> AgentProgramRevision:
    return AgentProgramRevision.freeze(
        root,
        program_id="repair-agent",
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        program_prompt=f"Repair using revision {revision_id}.",
        context={"fixture_score": score, "analysis_depth": 2},
        tool_policy=("inspect_workspace", "emit_patch"),
        capability_revision_ids=("localize-r1", "patch-r1"),
    )


def test_agent_program_revision_round_trips_and_rejects_artifact_tamper(
    tmp_path: Path,
) -> None:
    frozen = _revision(
        tmp_path / "parent",
        revision_id="program-r1",
        parent_revision_id=None,
        score=1,
    )

    loaded = AgentProgramRevision.load(frozen.root)

    assert loaded == frozen
    assert loaded.program_prompt == "Repair using revision program-r1."
    assert loaded.context == {"analysis_depth": 2, "fixture_score": 1}
    assert loaded.tool_policy == ("inspect_workspace", "emit_patch")
    assert loaded.capability_revision_ids == ("localize-r1", "patch-r1")
    assert len(loaded.bundle_sha256) == 64

    (frozen.root / "PROGRAM-PROMPT.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(AgentProgramViolation, match="artifact hash"):
        AgentProgramRevision.load(frozen.root)


def _plan(
    revision: AgentProgramRevision,
    *,
    arm: str = "candidate",
    tournament_id: str = "tournament-1",
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=f"plan-{revision.revision_id}",
        campaign_id="campaign-program",
        strategy_id="agent-program-search-v3",
        task=TaskRevision(
            task_id="fixture-task",
            revision_id="fixture-task-r1",
            project="fixture-project",
            cohort=Cohort.FEEDBACK,
            source_sha256="a" * 64,
            evaluator_id="fixture-evaluator-v1",
        ),
        candidate_revision_id=revision.revision_id,
        arm=arm,
        model=ModelIdentity("fixture", "deterministic-agent-program", "v1"),
        context_policy_id="fixture-context-v1",
        tool_policy_id="fixture-tools-v1",
        observer_policy_ids=("fixture-observer-v1",),
        native_evaluator_id="fixture-evaluator-v1",
        limits=ExecutionLimits(max_tokens=0, max_seconds=30, max_cost_cny=0),
        holdout_scope="feedback-only",
        metadata={
            "execution_profile": "fixture",
            "program_bundle_sha256": revision.bundle_sha256,
            "program_prompt_sha256": revision.artifact_hash("PROGRAM-PROMPT.txt"),
            "program_context_sha256": revision.artifact_hash("CONTEXT.json"),
            "program_tool_policy_sha256": revision.artifact_hash("TOOL-POLICY.json"),
            "program_capabilities_sha256": revision.artifact_hash(
                "CAPABILITIES.json"
            ),
            "tournament_id": tournament_id,
            "tournament_config_sha256": "b" * 64,
        },
    )


def test_fixture_transport_executes_and_consumes_the_complete_program_revision(
    tmp_path: Path,
) -> None:
    base = _revision(
        tmp_path / "base", revision_id="program-r1", parent_revision_id=None, score=1
    )
    changed = AgentProgramRevision.freeze(
        tmp_path / "changed",
        program_id="repair-agent",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        program_prompt="Use a different complete prompt.",
        context={"fixture_score": 2, "analysis_depth": 3},
        tool_policy=("emit_patch",),
        capability_revision_ids=("different-capability-r2",),
    )
    transport = DeterministicFixtureAgentProgramTransport(
        {base.revision_id: base.root, changed.revision_id: changed.root}
    )
    workspace = {
        "task_revision_id": "fixture-task-r1",
        "task_source_sha256": "a" * 64,
    }

    base_output = transport.infer(_plan(base, arm="search-parent"), workspace)
    changed_output = transport.infer(_plan(changed), workspace)

    assert transport.remote is False
    assert base_output["execution_scope"] == "fixture"
    assert base_output["program_prompt"] == base.program_prompt
    assert base_output["program_context"] == dict(base.context)
    assert base_output["program_tool_policy"] == list(base.tool_policy)
    assert base_output["program_capability_revision_ids"] == list(
        base.capability_revision_ids
    )
    assert base_output["input_tokens"] == base_output["output_tokens"] == 0
    assert base_output["cost_cny"] == 0
    assert base_output["network_calls"] == 0
    assert "resolved" not in base_output
    assert "claim" not in base_output
    assert base_output["patch"] != changed_output["patch"]
    assert base_output["program_projection_sha256"] != changed_output[
        "program_projection_sha256"
    ]


def _execution(
    plan: ExecutionPlan, output: dict[str, object], *, status: str = "completed"
) -> ExecutionResult:
    model = Receipt(
        receipt_id=f"receipt-{plan.plan_id}-model",
        campaign_id=plan.campaign_id,
        plan_id=plan.plan_id,
        sequence=1,
        kind="model",
        created_at="2026-08-15T00:00:00Z",
        payload=output,
        artifact_sha256=content_sha256(output),
    )
    return ExecutionResult(
        status=status,
        receipts=(model,),
        evidence=(),
        replayed=False,
    )


def test_tournament_decision_is_hash_bound_and_parent_wins_a_fixture_tie(
    tmp_path: Path,
) -> None:
    parent = _revision(
        tmp_path / "parent", revision_id="program-r1", parent_revision_id=None, score=2
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        score=2,
    )
    transport = DeterministicFixtureAgentProgramTransport(
        {parent.revision_id: parent.root, candidate.revision_id: candidate.root}
    )
    parent_plan = _plan(parent, arm="search-parent")
    candidate_plan = _plan(candidate)
    workspace = {
        "task_revision_id": "fixture-task-r1",
        "task_source_sha256": "a" * 64,
    }
    parent_output = dict(transport.infer(parent_plan, workspace))
    candidate_output = dict(transport.infer(candidate_plan, workspace))

    decision = TournamentAuthority().decide(
        plans=(parent_plan, candidate_plan),
        executions=(
            _execution(parent_plan, parent_output),
            _execution(candidate_plan, candidate_output),
        ),
    )

    assert decision.execution_scope == "fixture"
    assert decision.parent_revision_id == "program-r1"
    assert decision.participant_revision_ids == ("program-r1", "program-r2")
    assert decision.winner_revision_id == "program-r1"
    assert decision.advanced is False
    assert decision.scores == (("program-r1", 2.0), ("program-r2", 2.0))
    assert len(decision.decision_sha256) == 64
    assert decision.receipt_ids == (
        f"receipt-{parent_plan.plan_id}-model",
        f"receipt-{candidate_plan.plan_id}-model",
    )
    with pytest.raises(AgentProgramViolation, match="hash|outcome"):
        replace(decision, winner_revision_id="program-r2")


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        ("incomplete", "parent and candidates|incomplete"),
        ("partial", "partial"),
        ("duplicate", "duplicate"),
        ("config-drift", "config"),
    ),
)
def test_tournament_rejects_unmatched_or_incomplete_execution_matrices(
    tmp_path: Path, failure: str, message: str
) -> None:
    parent = _revision(
        tmp_path / "parent", revision_id="program-r1", parent_revision_id=None, score=1
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        score=2,
    )
    transport = DeterministicFixtureAgentProgramTransport(
        {parent.revision_id: parent.root, candidate.revision_id: candidate.root}
    )
    parent_plan = _plan(parent, arm="search-parent")
    candidate_plan = _plan(candidate)
    workspace = {
        "task_revision_id": "fixture-task-r1",
        "task_source_sha256": "a" * 64,
    }
    parent_execution = _execution(
        parent_plan, dict(transport.infer(parent_plan, workspace))
    )
    candidate_execution = _execution(
        candidate_plan, dict(transport.infer(candidate_plan, workspace))
    )
    plans: tuple[ExecutionPlan, ...] = (parent_plan, candidate_plan)
    executions: tuple[ExecutionResult, ...] = (
        parent_execution,
        candidate_execution,
    )
    if failure == "incomplete":
        plans, executions = (parent_plan,), (parent_execution,)
    elif failure == "partial":
        executions = (parent_execution, replace(candidate_execution, status="partial"))
    elif failure == "duplicate":
        plans = (*plans, candidate_plan)
        executions = (*executions, candidate_execution)
    else:
        plans = (parent_plan, replace(candidate_plan, context_policy_id="drifted"))

    with pytest.raises(AgentProgramViolation, match=message):
        TournamentAuthority().decide(plans=plans, executions=executions)


def test_search_parent_log_is_idempotent_and_detects_hash_chain_tamper(
    tmp_path: Path,
) -> None:
    parent = _revision(
        tmp_path / "parent", revision_id="program-r1", parent_revision_id=None, score=1
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        score=3,
    )
    transport = DeterministicFixtureAgentProgramTransport(
        {parent.revision_id: parent.root, candidate.revision_id: candidate.root}
    )
    parent_plan = _plan(parent, arm="search-parent")
    candidate_plan = _plan(candidate)
    workspace = {
        "task_revision_id": "fixture-task-r1",
        "task_source_sha256": "a" * 64,
    }
    decision = TournamentAuthority().decide(
        plans=(parent_plan, candidate_plan),
        executions=(
            _execution(parent_plan, dict(transport.infer(parent_plan, workspace))),
            _execution(
                candidate_plan, dict(transport.infer(candidate_plan, workspace))
            ),
        ),
    )
    log_path = tmp_path / "search-parent.jsonl"
    log = SearchParentLog(log_path, program_id="repair-agent")

    assert log.append(decision) is True
    assert log.append(decision) is False
    assert log.current_parent_revision_id() == "program-r2"
    assert len(log.all()) == 1

    row = json.loads(log_path.read_text(encoding="utf-8"))
    row["selected_revision_id"] = "forged-r9"
    log_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(AgentProgramViolation, match="hash"):
        log.current_parent_revision_id()


def test_search_parent_log_rejects_a_valid_but_divergent_parent_fork(
    tmp_path: Path,
) -> None:
    first_parent = _revision(
        tmp_path / "parent-1", revision_id="program-r1", parent_revision_id=None, score=1
    )
    first_winner = _revision(
        tmp_path / "winner-1",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        score=2,
    )
    fork_parent = _revision(
        tmp_path / "fork-parent",
        revision_id="program-r9",
        parent_revision_id=None,
        score=1,
    )
    fork_candidate = _revision(
        tmp_path / "fork-candidate",
        revision_id="program-r10",
        parent_revision_id="program-r9",
        score=2,
    )
    workspace = {
        "task_revision_id": "fixture-task-r1",
        "task_source_sha256": "a" * 64,
    }

    def decide(
        parent: AgentProgramRevision,
        candidate: AgentProgramRevision,
        tournament_id: str,
    ):
        transport = DeterministicFixtureAgentProgramTransport(
            {parent.revision_id: parent.root, candidate.revision_id: candidate.root}
        )
        plans = (
            _plan(parent, arm="search-parent", tournament_id=tournament_id),
            _plan(candidate, tournament_id=tournament_id),
        )
        return TournamentAuthority().decide(
            plans=plans,
            executions=tuple(
                _execution(plan, dict(transport.infer(plan, workspace)))
                for plan in plans
            ),
        )

    log = SearchParentLog(tmp_path / "parents.jsonl", program_id="repair-agent")
    assert log.append(decide(first_parent, first_winner, "tournament-1")) is True

    with pytest.raises(AgentProgramViolation, match="fork"):
        log.append(decide(fork_parent, fork_candidate, "tournament-2"))


def test_fixture_agent_program_strategy_is_live_and_acts_on_tournament_decision(
    tmp_path: Path,
) -> None:
    parent = _revision(
        tmp_path / "parent", revision_id="program-r1", parent_revision_id=None, score=1
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        score=3,
    )
    context = StrategyContext(
        campaign_id="campaign-program",
        task=_plan(parent).task,
        model=_plan(parent).model,
        context_policy_id="fixture-context-v1",
        tool_policy_id="fixture-tools-v1",
        observer_policy_ids=("fixture-observer-v1",),
        limits=ExecutionLimits(max_tokens=0, max_seconds=30, max_cost_cny=0),
        inputs={
            "parent_revision_id": "program-r1",
            "candidate_revision_ids": ("program-r2",),
            "tournament_id": "tournament-1",
            "generation_config": {"temperature": 0},
            "execution_profile": "fixture",
            "revision_roots": {
                "program-r1": str(parent.root),
                "program-r2": str(candidate.root),
            },
        },
    )
    strategy = AgentProgramSearchStrategy(execution_profile="fixture")

    plans = strategy.plan(context)
    transport = DeterministicFixtureAgentProgramTransport(
        {parent.revision_id: parent.root, candidate.revision_id: candidate.root}
    )
    workspace = {
        "task_revision_id": "fixture-task-r1",
        "task_source_sha256": "a" * 64,
    }
    executions = tuple(
        _execution(plan, dict(transport.infer(plan, workspace))) for plan in plans
    )
    decision = TournamentAuthority().decide(plans=plans, executions=executions)
    action = strategy.next_action(context, (), decision=decision)

    assert strategy.status is StrategyStatus.LIVE
    assert [plan.arm for plan in plans] == ["search-parent", "candidate"]
    assert [plan.metadata["program_bundle_sha256"] for plan in plans] == [
        parent.bundle_sha256,
        candidate.bundle_sha256,
    ]
    assert all(plan.metadata["execution_profile"] == "fixture" for plan in plans)
    assert action.action == "advance-search-parent"
    assert decision.winner_revision_id in action.reason
    assert decision.decision_sha256 in action.reason
    assert action.claim_ids == ()


def test_public_cli_runs_fixture_tournament_through_campaign_runtime(
    tmp_path: Path,
) -> None:
    parent = _revision(
        tmp_path / "parent", revision_id="program-r1", parent_revision_id=None, score=1
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
        score=3,
    )
    config = {
        "schema_version": 1,
        "campaign_id": "agent-program-campaign-1",
        "tournament_id": "tournament-1",
        "execution_profile": "fixture",
        "program_id": "repair-agent",
        "parent_revision_root": str(parent.root),
        "candidate_revision_roots": [str(candidate.root)],
        "generation_config": {"temperature": 0, "seed": 7},
        "task": {
            "task_id": "fixture-task",
            "revision_id": "fixture-task-r1",
            "project": "fixture-project",
            "cohort": "feedback",
            "source_sha256": "a" * 64,
            "evaluator_id": "fixture-agent-program-native-v1",
        },
    }
    config_path = tmp_path / "agent-program.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"
    argv = [
        "campaign",
        "run",
        "--strategy",
        "agent-program",
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    result = json.loads((output / "CAMPAIGN-RESULT.json").read_text())
    assert result["status"] == "completed"
    assert result["execution_scope"] == "fixture"
    assert result["selected_parent_revision_id"] == "program-r2"
    assert result["search_parent_advanced"] is True
    assert result["claims"] == []
    assert result["native_gain_claimed"] is False
    assert result["promotion_eligible"] is False
    assert result["capability_active"] is False
    first_receipts = (output / "receipt-store/receipts.jsonl").read_bytes()
    first_parent_log = (output / "SEARCH-PARENT.jsonl").read_bytes()

    assert main(argv) == 0
    assert (output / "receipt-store/receipts.jsonl").read_bytes() == first_receipts
    assert (output / "SEARCH-PARENT.jsonl").read_bytes() == first_parent_log
