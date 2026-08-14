from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from evolve.campaigns import CampaignRunner, CampaignRunStatus, CampaignSpec
from evolve.cli import main
from evolve.contracts import (
    Authorization,
    Cohort,
    ExecutionLimits,
    ModelIdentity,
    Receipt,
    TaskRevision,
)
from evolve.kernel import CampaignController, CampaignStatus
from evolve.runtime import ExecutionResult
from evolve.strategies import (
    AgentProgramSearchStrategy,
    EvolutionStrategy,
    LegacyImportStrategy,
    SkillPairedStrategy,
    StrategyContext,
    StrategyPhase,
    StrategyStatus,
)

NOW = datetime(2026, 8, 14, 3, 0, tzinfo=UTC)
SHA = hashlib.sha256(b"campaign-task").hexdigest()


def _task() -> TaskRevision:
    return TaskRevision(
        task_id="task-1",
        revision_id="task-r1",
        project="project-a",
        cohort=Cohort.FEEDBACK,
        source_sha256=SHA,
        evaluator_id="native-v1",
    )


def _context(**inputs: object) -> StrategyContext:
    return StrategyContext(
        campaign_id="campaign-1",
        task=_task(),
        model=ModelIdentity(provider="local", model="frozen", revision="r1"),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native",),
        limits=ExecutionLimits(max_tokens=128, max_seconds=30, max_cost_cny=0),
        inputs=inputs,
    )


def _authorization() -> Authorization:
    return Authorization(
        authorization_id="auth-1",
        campaign_id="campaign-1",
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=2,
        expires_at=NOW + timedelta(hours=1),
        remote_calls_allowed=False,
    )


def test_all_strategies_implement_the_context_driven_contract() -> None:
    cases = (
        (
            LegacyImportStrategy(),
            _context(
                imported_revision_id="legacy-r1",
                legacy_artifact_sha256=hashlib.sha256(b"legacy").hexdigest(),
                provenance_uri="archive/legacy.json",
            ),
            StrategyStatus.COMPATIBILITY,
        ),
        (
            SkillPairedStrategy(),
            _context(
                baseline_revision_id="baseline-r1",
                taught_revision_id="skill-r2",
            ),
            StrategyStatus.LIVE,
        ),
        (
            AgentProgramSearchStrategy(),
            _context(
                parent_revision_id="program-r1",
                candidate_revision_ids=("program-r2",),
                tournament_id="tournament-1",
            ),
            StrategyStatus.NOT_YET_LIVE,
        ),
    )

    for strategy, context, expected_status in cases:
        assert isinstance(strategy, EvolutionStrategy)
        plans = strategy.plan(context)
        interpreted = strategy.interpret(context, ())
        decision = strategy.next_action(context, ())

        assert plans
        assert all(plan.campaign_id == context.campaign_id for plan in plans)
        assert interpreted.campaign_id == context.campaign_id
        assert interpreted.receipt_ids == ()
        assert decision.status is expected_status
        assert decision.claim_ids == ()


def test_campaign_runner_reports_non_live_strategies_without_dispatch_or_claims() -> (
    None
):
    context = _context(
        imported_revision_id="legacy-r1",
        legacy_artifact_sha256=hashlib.sha256(b"legacy").hexdigest(),
        provenance_uri="archive/legacy.json",
    )
    result = CampaignRunner().run(
        CampaignSpec(
            campaign_id="campaign-1",
            contexts=(context,),
            authorization=_authorization(),
        ),
        LegacyImportStrategy(),
    )

    assert result.status is CampaignRunStatus.COMPATIBILITY
    assert len(result.plans) == 1
    assert result.executions == ()
    assert result.receipts == ()
    assert result.claims == ()
    assert result.decisions[0].status is StrategyStatus.COMPATIBILITY


def test_public_campaign_cli_truthfully_marks_agent_program_not_yet_live(
    tmp_path, capsys
) -> None:
    config = tmp_path / "agent-program.json"
    config.write_text("{}", encoding="utf-8")

    assert main(
        [
            "campaign",
            "run",
            "--strategy",
            "agent-program",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "run"),
        ]
    ) == 2
    assert '"status":"not-yet-live"' in capsys.readouterr().out


def test_campaign_runner_marks_unwired_agent_tournament_not_yet_live() -> None:
    context = _context(
        parent_revision_id="program-r1",
        candidate_revision_ids=("program-r2",),
        tournament_id="tournament-1",
    )

    result = CampaignRunner().run(
        CampaignSpec(
            campaign_id="campaign-1",
            contexts=(context,),
            authorization=_authorization(),
        ),
        AgentProgramSearchStrategy(),
    )

    assert result.status is CampaignRunStatus.NOT_YET_LIVE
    assert [plan.arm for plan in result.plans] == ["search-parent", "candidate"]
    assert result.executions == result.receipts == result.claims == ()


def test_skill_strategy_can_run_a_baseline_only_phase_before_teacher_proposal() -> (
    None
):
    context = StrategyContext(
        campaign_id="campaign-1",
        task=_task(),
        model=ModelIdentity(provider="local", model="frozen", revision="r1"),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native",),
        limits=ExecutionLimits(max_tokens=128, max_seconds=30, max_cost_cny=0),
        phase=StrategyPhase.BASELINE_ONLY,
        inputs={"baseline_revision_id": "empty-harness-r0"},
    )

    plans = SkillPairedStrategy().plan(context)

    assert len(plans) == 1
    assert plans[0].arm == "baseline"
    assert plans[0].candidate_revision_id == "empty-harness-r0"


class _RecordingRuntime:
    def __init__(self) -> None:
        self.plan_ids: list[str] = []

    def execute(self, plan, authorization) -> ExecutionResult:
        assert authorization.campaign_id == plan.campaign_id
        self.plan_ids.append(plan.plan_id)
        payload = {"cost_cny": 0}
        receipt = Receipt(
            receipt_id=f"receipt-{plan.plan_id}",
            campaign_id=plan.campaign_id,
            plan_id=plan.plan_id,
            sequence=1,
            kind="cost",
            created_at="2026-08-14T03:00:00Z",
            payload=payload,
            artifact_sha256=hashlib.sha256(b'{"cost_cny":0}').hexdigest(),
        )
        return ExecutionResult(
            status="completed", receipts=(receipt,), evidence=(), replayed=False
        )


def test_campaign_runner_uses_kernel_and_runtime_authorities_for_live_strategy() -> (
    None
):
    context = _context(
        baseline_revision_id="baseline-r1",
        taught_revision_id="skill-r2",
    )
    authorization = _authorization()
    controller = CampaignController.create(
        campaign_id="campaign-1", authorization=authorization, now=NOW
    )
    runtime = _RecordingRuntime()

    result = CampaignRunner(runtime=runtime, controller=controller).run(
        CampaignSpec(
            campaign_id="campaign-1",
            contexts=(context,),
            authorization=authorization,
        ),
        SkillPairedStrategy(),
    )

    assert result.status is CampaignRunStatus.COMPLETED
    assert len(result.plans) == len(result.executions) == 2
    assert runtime.plan_ids == [plan.plan_id for plan in result.plans]
    assert result.snapshot is not None
    assert result.snapshot.status is CampaignStatus.COMPLETED
    assert result.claims == ()
    assert result.decisions[0].action == "await-authoritative-claims"


def test_campaign_runner_executes_baseline_only_as_an_independent_live_phase() -> (
    None
):
    context = StrategyContext(
        campaign_id="campaign-1",
        task=_task(),
        model=ModelIdentity(provider="local", model="frozen", revision="r1"),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("native",),
        limits=ExecutionLimits(max_tokens=128, max_seconds=30, max_cost_cny=0),
        phase=StrategyPhase.BASELINE_ONLY,
        inputs={"baseline_revision_id": "empty-harness-r0"},
    )
    authorization = _authorization()
    controller = CampaignController.create(
        campaign_id="campaign-1", authorization=authorization, now=NOW
    )
    runtime = _RecordingRuntime()

    result = CampaignRunner(runtime=runtime, controller=controller).run(
        CampaignSpec(
            campaign_id="campaign-1",
            contexts=(context,),
            authorization=authorization,
        ),
        SkillPairedStrategy(),
    )

    assert result.status is CampaignRunStatus.COMPLETED
    assert [plan.arm for plan in result.plans] == ["baseline"]
    assert runtime.plan_ids == [result.plans[0].plan_id]


def test_live_strategy_without_authoritative_dependencies_is_not_yet_live() -> None:
    context = _context(
        baseline_revision_id="baseline-r1",
        taught_revision_id="skill-r2",
    )

    result = CampaignRunner().run(
        CampaignSpec(
            campaign_id="campaign-1",
            contexts=(context,),
            authorization=_authorization(),
        ),
        SkillPairedStrategy(),
    )

    assert result.status is CampaignRunStatus.NOT_YET_LIVE
    assert result.plans
    assert result.executions == ()
    assert "ExecutionRuntime" in (result.reason or "")
