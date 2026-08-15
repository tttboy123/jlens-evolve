from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evolve.agent_program import AgentProgramRevision
from evolve.alignment import align_native_pair
from evolve.campaigns import (
    CampaignRunner,
    CampaignSpec,
    HashVerifiedAgentProgramTransport,
)
from evolve.contracts import (
    Authorization,
    Claim,
    ClaimClassification,
    ClaimGrade,
    Cohort,
    EvidenceEnvelope,
    ExecutionLimits,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
    content_sha256,
)
from evolve.evidence import (
    ClaimEngine,
    EvidenceGradeMachine,
    EvidenceGraph,
    ReceiptStore,
    build_matched_counterfactual_pair,
)
from evolve.governance import (
    GovernanceDecisionAuthority,
    GovernanceService,
    PromotionDecisionLog,
)
from evolve.kernel import CampaignController
from evolve.observers import ExternalTraceObserver, NativeOutcomeObserver, ObserverHub
from evolve.portfolio import (
    CompiledSkillCandidate,
    PortfolioOrchestrator,
    PortfolioRequest,
    PortfolioViolation,
    SkillValidationAuthority,
    TournamentAuthority,
    compiled_bundle_sha256,
)
from evolve.registry import (
    AgentProgramRecord,
    AgentProgramRegistry,
    CandidateRecord,
    CandidateRegistry,
    CapabilityRegistry,
)
from evolve.runtime import ExecutionRuntime
from evolve.strategies import AgentProgramSearchStrategy, StrategyContext

NOW = datetime(2026, 8, 15, 6, 0, tzinfo=UTC)


def _append_receipt(
    store: ReceiptStore,
    *,
    receipt_id: str,
    campaign_id: str,
    plan_id: str,
    sequence: int,
    kind: str,
    payload: dict[str, object],
) -> Receipt:
    artifact = canonical_json(payload).encode()
    receipt = Receipt(
        receipt_id=receipt_id,
        campaign_id=campaign_id,
        plan_id=plan_id,
        sequence=sequence,
        kind=kind,
        created_at="2026-08-15T06:00:00Z",
        payload=payload,
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
    )
    return store.append(receipt, artifact)


def _parent_program(root: Path) -> AgentProgramRevision:
    return AgentProgramRevision.freeze(
        root,
        program_id="repair-agent",
        revision_id="program-r1",
        parent_revision_id=None,
        program_prompt="Repair the feedback task using verified capabilities.",
        context={"analysis_depth": 3},
        tool_policy=("inspect_workspace", "emit_patch"),
        capability_revision_ids=("localize-r1",),
    )


def _failure_authority(
    root: Path, parent: AgentProgramRevision
) -> tuple[ReceiptStore, EvidenceGraph, Claim]:
    store = ReceiptStore(root / "receipts")
    graph = EvidenceGraph(root / "graph")
    model = _append_receipt(
        store,
        receipt_id="receipt-failure-model",
        campaign_id="campaign-failure",
        plan_id="plan-failure-parent",
        sequence=1,
        kind="model",
        payload={
            "revision_id": parent.revision_id,
            "program_bundle_sha256": parent.bundle_sha256,
        },
    )
    native = _append_receipt(
        store,
        receipt_id="receipt-failure-native",
        campaign_id="campaign-failure",
        plan_id="plan-failure-parent",
        sequence=2,
        kind="native_evaluation",
        payload={
            "arm": "search-parent",
            "task_revision_id": "feedback-failure-r1",
            "model_receipt_id": model.receipt_id,
            "model_artifact_sha256": model.artifact_sha256,
            "resolved": False,
            "evaluator_error": None,
        },
    )
    evidence = NativeOutcomeObserver().observe(native)
    assert evidence is not None
    graph.append_evidence(evidence)
    claim = Claim(
        claim_id="claim-agent-program-failure",
        candidate_id=parent.revision_id,
        grade=ClaimGrade.E1,
        classification=ClaimClassification.REGRESSION,
        evidence_ids=(evidence.evidence_id,),
        rationale="authoritative native failure",
        supersedes_claim_id=None,
    )
    graph.append_claim(claim)
    return store, graph, claim


class _Teacher:
    def __init__(self, root: Path, *, active: bool = False) -> None:
        self.root = root
        self.active = active
        self.calls = 0

    def compile(self, gap) -> CompiledSkillCandidate:
        self.calls += 1
        artifact = (
            "Use declaration-bound symbol localization before editing.\n"
        ).encode()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "COMPILED-SKILL.txt").write_bytes(artifact)
        return CompiledSkillCandidate(
            candidate_id="skill-symbol-localization",
            revision_id="skill-symbol-localization-r1",
            bundle_sha256=compiled_bundle_sha256(self.root),
            bundle_root=self.root,
            source_gap_sha256=gap.content_sha256,
            active=self.active,
        )


def _skill_gain_claim(
    *,
    store: ReceiptStore,
    graph: EvidenceGraph,
    candidate: CompiledSkillCandidate,
    task_index: int,
) -> Claim:
    campaign_id = "campaign-skill-validation"
    task_revision_id = f"feedback-task-r{task_index}"
    source_sha256 = hashlib.sha256(task_revision_id.encode()).hexdigest()
    native_by_arm: dict[str, EvidenceEnvelope] = {}
    external_by_arm: dict[str, EvidenceEnvelope] = {}
    model_by_arm: dict[str, Receipt] = {}
    observers = ObserverHub(
        (ExternalTraceObserver(), NativeOutcomeObserver()), graph=graph
    )
    for arm_index, arm in enumerate(("baseline", "taught"), start=1):
        plan_id = f"plan-skill-{task_index}-{arm}"
        patch = f"patch:{task_index}:{arm}"
        prediction_sha256 = hashlib.sha256(patch.encode()).hexdigest()
        candidate_prompt = (
            f"candidate={candidate.revision_id};bundle={candidate.bundle_sha256}"
            if arm == "taught"
            else None
        )
        prompt_texts = (
            [f"COMPILED-CANDIDATE:\n{candidate_prompt}"]
            if arm == "taught"
            else [f"Solve {task_revision_id} without a candidate."]
        )
        model_payload: dict[str, object] = {
            "provider": "local-mlx",
            "model": "qwen",
            "revision": "frozen-r1",
            "patch": patch,
            "patch_sha256": prediction_sha256,
            "prediction_sha256": prediction_sha256,
            "prompt_texts": prompt_texts,
            "prompt_sha256": [
                hashlib.sha256(value.encode()).hexdigest() for value in prompt_texts
            ],
            "candidate_consumed": arm == "taught",
            "candidate_revision_id": (
                candidate.revision_id if arm == "taught" else None
            ),
            "candidate_bundle_sha256": (
                candidate.bundle_sha256 if arm == "taught" else None
            ),
            "candidate_prompt": candidate_prompt,
            "candidate_prompt_sha256": (
                hashlib.sha256(candidate_prompt.encode()).hexdigest()
                if candidate_prompt is not None
                else None
            ),
            "compiled_artifact_sha256": (
                {"skill": candidate.bundle_sha256} if arm == "taught" else {}
            ),
        }
        model = _append_receipt(
            store,
            receipt_id=f"receipt-skill-{task_index}-{arm}-model",
            campaign_id=campaign_id,
            plan_id=plan_id,
            sequence=arm_index * 3 - 2,
            kind="model",
            payload=model_payload,
        )
        model_by_arm[arm] = model
        trace_payload = {
            "model_receipt_id": model.receipt_id,
            "model_artifact_sha256": model.artifact_sha256,
            "arm": arm,
            "task_revision_id": task_revision_id,
            "prediction_sha256": prediction_sha256,
            **{
                name: model_payload[name]
                for name in (
                    "candidate_consumed",
                    "candidate_revision_id",
                    "candidate_bundle_sha256",
                    "prompt_texts",
                    "prompt_sha256",
                    "candidate_prompt",
                    "candidate_prompt_sha256",
                    "compiled_artifact_sha256",
                )
            },
        }
        trace = _append_receipt(
            store,
            receipt_id=f"receipt-skill-{task_index}-{arm}-trace",
            campaign_id=campaign_id,
            plan_id=plan_id,
            sequence=arm_index * 3 - 1,
            kind="external_trace",
            payload=trace_payload,
        )
        emitted = observers.observe(trace)
        external_by_arm[arm] = next(
            row for row in emitted if row.observer_id == "external-trace-v1"
        )
        native_payload = {
            "arm": arm,
            "task_revision_id": task_revision_id,
            "task_source_sha256": source_sha256,
            "model_identity": "local-mlx/qwen@frozen-r1",
            "native_evaluator_id": "official-swebench-v1",
            "execution_config_sha256": "e" * 64,
            "model_receipt_id": model.receipt_id,
            "model_artifact_sha256": model.artifact_sha256,
            "prediction_sha256": prediction_sha256,
            "resolved": arm == "taught",
            "evaluator_error": None,
        }
        native = _append_receipt(
            store,
            receipt_id=f"receipt-skill-{task_index}-{arm}-native",
            campaign_id=campaign_id,
            plan_id=plan_id,
            sequence=arm_index * 3,
            kind="native_evaluation",
            payload=native_payload,
        )
        emitted = observers.observe(native)
        native_by_arm[arm] = next(
            row for row in emitted if row.observer_id == "native-v1"
        )
    pair = build_matched_counterfactual_pair(
        candidate_id=candidate.candidate_id,
        candidate_revision_id=candidate.revision_id,
        candidate_bundle_sha256=candidate.bundle_sha256,
        baseline_model_receipt=model_by_arm["baseline"],
        baseline_external_evidence=external_by_arm["baseline"],
        baseline_native_evidence=native_by_arm["baseline"],
        taught_model_receipt=model_by_arm["taught"],
        taught_external_evidence=external_by_arm["taught"],
        taught_native_evidence=native_by_arm["taught"],
    )
    return ClaimEngine(graph).classify_pair(
        candidate.candidate_id,
        align_native_pair(native_by_arm["baseline"], native_by_arm["taught"]),
        counterfactual_pair=pair,
    )


class _SkillAuthority:
    def __init__(
        self,
        root: Path,
        *,
        governance: GovernanceService,
        decision_log: PromotionDecisionLog,
        capability_artifact_override: str | None = None,
    ) -> None:
        self.root = root
        self.governance = governance
        self.decision_log = decision_log
        self.capability_artifact_override = capability_artifact_override
        self.calls = 0

    def validate(
        self, gap, candidate: CompiledSkillCandidate
    ) -> SkillValidationAuthority:
        self.calls += 1
        store = ReceiptStore(self.root / "receipts")
        graph = EvidenceGraph(self.root / "graph")
        claims = tuple(
            _skill_gain_claim(
                store=store,
                graph=graph,
                candidate=candidate,
                task_index=index,
            )
            for index in range(1, 4)
        )
        state = EvidenceGradeMachine(graph, receipt_store=store).aggregate(
            candidate.candidate_id,
            task_projects={
                "feedback-task-r1": "project-a",
                "feedback-task-r2": "project-a",
                "feedback-task-r3": "project-b",
            },
            mechanism_id="symbol-localization-v1",
        )
        state = replace(
            state,
            grade=ClaimGrade.E3,
            e3_eligible=True,
            prediction_consistent_task_count=3,
            prediction_evidence_ids=tuple(
                f"trusted-observation-{index}" for index in range(6)
            ),
        )
        record = CandidateRecord(
            candidate_id=candidate.candidate_id,
            revision_id=candidate.revision_id,
            candidate_kind="compiled-skill",
            source_claim_ids=tuple(claim.claim_id for claim in claims),
            artifact_sha256=candidate.bundle_sha256,
        )
        decision = self.governance.decide(
            candidate=record,
            evidence=state,
            claims=claims,
            human_approval=True,
            decided_at="2026-08-15T06:05:00Z",
            log=self.decision_log,
        )
        capability = self.governance.to_capability(
            candidate=record,
            decision=decision,
            capability_id="capability-symbol-localization",
        )
        if self.capability_artifact_override is not None:
            capability = replace(
                capability, artifact_sha256=self.capability_artifact_override
            )
        return SkillValidationAuthority(
            candidate=record,
            capability=capability,
            claims=claims,
            receipt_store_root=store.root,
            evidence_graph_root=graph.root,
        )


class _Workspace:
    def materialize(self, plan):
        return {
            "workspace_id": f"workspace-{plan.plan_id}",
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
        }


class _ProgramExecutor:
    remote = False

    def infer_program(self, revision, plan, workspace):
        patch = f"program-output:{revision.revision_id}"
        patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
        return {
            "patch": patch,
            "patch_sha256": patch_sha256,
            "prediction_sha256": patch_sha256,
            "cost_cny": 0,
        }


class _ProgramEvaluator:
    evaluator_id = "official-program-evaluator-v1"

    def __init__(self, winner_revision_id: str | None = None) -> None:
        self.winner_revision_id = winner_revision_id

    def evaluate(self, plan, workspace, model_output):
        return {
            "resolved": plan.candidate_revision_id == self.winner_revision_id,
            "native_valid": True,
            "prediction_sha256": model_output["prediction_sha256"],
        }


class _Tournament:
    def __init__(self, root: Path, *, tie: bool = False) -> None:
        self.root = root
        self.tie = tie
        self.calls = 0

    def run(
        self, parent: AgentProgramRevision, candidate: AgentProgramRevision
    ) -> TournamentAuthority:
        self.calls += 1
        campaign_id = "campaign-program-tournament"
        store = ReceiptStore(self.root / "receipts")
        graph = EvidenceGraph(self.root / "graph")
        task = TaskRevision(
            task_id="portfolio-feedback-task",
            revision_id="portfolio-feedback-task-r1",
            project="project-a",
            cohort=Cohort.FEEDBACK,
            source_sha256="a" * 64,
            evaluator_id=_ProgramEvaluator.evaluator_id,
        )
        context = StrategyContext(
            campaign_id=campaign_id,
            task=task,
            model=ModelIdentity("local", "program-executor", "r1"),
            context_policy_id="program-context-v1",
            tool_policy_id="program-tools-v1",
            observer_policy_ids=("native-v1",),
            limits=ExecutionLimits(128, 60, 0),
            inputs={
                "parent_revision_id": parent.revision_id,
                "candidate_revision_ids": (candidate.revision_id,),
                "tournament_id": "portfolio-tournament-v1",
                "generation_config": {"temperature": 0},
                "execution_profile": "live",
                "revision_roots": {
                    parent.revision_id: str(parent.root),
                    candidate.revision_id: str(candidate.root),
                },
                "claim_evidence_graph_root": str(graph.root),
                "claim_receipt_store_root": str(store.root),
            },
        )
        authorization = Authorization(
            authorization_id="auth-program-tournament",
            campaign_id=campaign_id,
            allowed_cohorts=(Cohort.FEEDBACK,),
            max_cost_cny=0,
            max_model_calls=2,
            expires_at=NOW + timedelta(hours=1),
            remote_calls_allowed=False,
        )
        runtime = ExecutionRuntime(
            model_transport=HashVerifiedAgentProgramTransport(
                {
                    parent.revision_id: parent.root,
                    candidate.revision_id: candidate.root,
                },
                executor=_ProgramExecutor(),
            ),
            workspace_manager=_Workspace(),
            native_evaluator=_ProgramEvaluator(
                None if self.tie else candidate.revision_id
            ),
            observer_hub=ObserverHub((NativeOutcomeObserver(),), graph=graph),
            receipt_sink=store,
            clock=lambda: NOW,
        )
        runner = CampaignRunner(
            runtime=runtime,
            controller=CampaignController.create(
                campaign_id=campaign_id,
                authorization=authorization,
                now=NOW,
            ),
        )
        strategy = AgentProgramSearchStrategy(execution_profile="live")
        first = runner.run(
            CampaignSpec(campaign_id, (context,), authorization), strategy
        )
        evidence_by_plan = {
            row.payload["plan_id"]: row for row in graph.list_evidence()
        }
        claims = tuple(
            Claim(
                claim_id=f"claim-tournament-{index}",
                candidate_id=plan.candidate_revision_id,
                grade=ClaimGrade.E1,
                classification=(
                    ClaimClassification.NEUTRAL
                    if self.tie or plan.arm == "search-parent"
                    else ClaimClassification.GAIN
                ),
                evidence_ids=(evidence_by_plan[plan.plan_id].evidence_id,),
                rationale="external tournament claim authority",
                supersedes_claim_id=None,
            )
            for index, plan in enumerate(first.plans, start=1)
        )
        for claim in claims:
            graph.append_claim(claim)
        result = runner.run(
            CampaignSpec(campaign_id, (context,), authorization, claims=claims),
            strategy,
        )
        return TournamentAuthority(
            result=result,
            receipt_store_root=store.root,
            evidence_graph_root=graph.root,
        )


class _ForgedParentWinTournament(_Tournament):
    def run(
        self, parent: AgentProgramRevision, candidate: AgentProgramRevision
    ) -> TournamentAuthority:
        authority = super().run(parent, candidate)
        forged_claims = tuple(
            replace(claim, classification=ClaimClassification.GAIN)
            if claim.candidate_id == parent.revision_id
            else claim
            for claim in authority.result.claims
        )
        claim_path = authority.evidence_graph_root / "claims.jsonl"
        records = [
            json.loads(line) for line in claim_path.read_text().splitlines()
        ]
        forged_by_id = {claim.claim_id: claim for claim in forged_claims}
        for record in records:
            if record["value"]["candidate_id"] == parent.revision_id:
                record["value"]["classification"] = "gain"
                record["content_sha256"] = forged_by_id[
                    record["value"]["claim_id"]
                ].content_sha256
        claim_path.write_text(
            "".join(canonical_json(record) + "\n" for record in records),
            encoding="utf-8",
        )
        ordered = tuple(sorted(forged_claims, key=lambda row: row.claim_id))
        scores = {parent.revision_id: 2, candidate.revision_id: 2}
        decision_sha256 = content_sha256(
            {
                "tournament_id": authority.result.plans[0].metadata["tournament_id"],
                "execution_scope": "live",
                "parent_revision_id": parent.revision_id,
                "participant_revision_ids": [
                    parent.revision_id,
                    candidate.revision_id,
                ],
                "program_bundle_sha256": [
                    [plan.candidate_revision_id, plan.metadata["program_bundle_sha256"]]
                    for plan in authority.result.plans
                ],
                "claim_ids": [claim.claim_id for claim in ordered],
                "claim_sha256": [
                    [claim.claim_id, claim.content_sha256] for claim in ordered
                ],
                "scores": [
                    [parent.revision_id, scores[parent.revision_id]],
                    [candidate.revision_id, scores[candidate.revision_id]],
                ],
                "winner_revision_id": parent.revision_id,
            }
        )
        forged_decision = replace(
            authority.result.decisions[0],
            action="reject-candidates",
            reason=(
                f"decision={decision_sha256};winner={parent.revision_id};"
                "scope=live;promotion_claimed=false"
            ),
            claim_ids=tuple(claim.claim_id for claim in ordered),
        )
        return TournamentAuthority(
            result=replace(
                authority.result,
                claims=forged_claims,
                decisions=(forged_decision,),
            ),
            receipt_store_root=authority.receipt_store_root,
            evidence_graph_root=authority.evidence_graph_root,
        )


def _system(tmp_path: Path, *, teacher_active: bool = False):
    parent = _parent_program(tmp_path / "parent-program")
    failure_store, failure_graph, failure_claim = _failure_authority(
        tmp_path / "failure", parent
    )
    authority = GovernanceDecisionAuthority(
        key_id="portfolio-governance-key", secret_key=b"p" * 32
    )
    decision_log = PromotionDecisionLog(
        tmp_path / "promotion-decisions.jsonl", authority=authority
    )
    candidate_registry = CandidateRegistry(tmp_path / "candidates.jsonl")
    capability_registry = CapabilityRegistry(
        tmp_path / "capabilities.jsonl", decision_log=decision_log
    )
    program_registry = AgentProgramRegistry(tmp_path / "programs.jsonl")
    program_registry.append(
        AgentProgramRecord(
            program_id=parent.program_id,
            revision_id=parent.revision_id,
            parent_revision_id=None,
            capability_revision_ids=parent.capability_revision_ids,
            artifact_sha256=parent.bundle_sha256,
        )
    )
    orchestrator = PortfolioOrchestrator(
        tmp_path / "portfolio",
        candidate_registry=candidate_registry,
        capability_registry=capability_registry,
        agent_program_registry=program_registry,
        promotion_decision_log=decision_log,
    )
    request = PortfolioRequest(
        round_trip_id="portfolio-round-trip-1",
        parent_program_root=parent.root,
        failure_claim=failure_claim,
        failure_receipt_store_root=failure_store.root,
        failure_evidence_graph_root=failure_graph.root,
    )
    teacher = _Teacher(tmp_path / "compiled-skill", active=teacher_active)
    skill = _SkillAuthority(
        tmp_path / "skill-validation",
        governance=GovernanceService(authority=authority),
        decision_log=decision_log,
    )
    tournament = _Tournament(tmp_path / "tournament")
    return orchestrator, request, teacher, skill, tournament, parent


def test_cross_strategy_round_trip_is_authority_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    orchestrator, request, teacher, skill, tournament, parent = _system(tmp_path)

    result = orchestrator.run(
        request,
        teacher=teacher,
        skill_authority=skill,
        tournament=tournament,
    )
    replay = orchestrator.run(
        request,
        teacher=teacher,
        skill_authority=skill,
        tournament=tournament,
    )

    assert result.replayed is False
    assert replay.replayed is True
    assert replay.decision == result.decision
    assert result.decision.tournament_action == "advance-search-parent"
    assert result.decision.search_parent_revision_id != parent.revision_id
    assert result.program.parent_revision_id == parent.revision_id
    assert result.capability.active is False
    assert result.capability.revision_id in result.program.capability_revision_ids
    assert teacher.calls == skill.calls == tournament.calls == 1
    assert len((tmp_path / "portfolio" / "capability-gaps.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "portfolio" / "portfolio-decisions.jsonl").read_text().splitlines()) == 1


def test_portfolio_rejects_synthetic_failure_claim(tmp_path: Path) -> None:
    orchestrator, request, teacher, skill, tournament, _ = _system(tmp_path)
    forged = replace(request.failure_claim, claim_id="claim-not-in-graph")

    with pytest.raises(PortfolioViolation, match="failure Claim is not authoritative"):
        orchestrator.run(
            replace(request, failure_claim=forged),
            teacher=teacher,
            skill_authority=skill,
            tournament=tournament,
        )


def test_portfolio_rejects_active_teacher_candidate(tmp_path: Path) -> None:
    orchestrator, request, teacher, skill, tournament, _ = _system(
        tmp_path, teacher_active=True
    )

    with pytest.raises(PortfolioViolation, match="inactive"):
        orchestrator.run(
            request,
            teacher=teacher,
            skill_authority=skill,
            tournament=tournament,
        )


def test_portfolio_rejects_capability_bundle_mismatch(tmp_path: Path) -> None:
    orchestrator, request, teacher, skill, tournament, _ = _system(tmp_path)
    skill.capability_artifact_override = "f" * 64

    with pytest.raises(PortfolioViolation, match="capability bundle"):
        orchestrator.run(
            request,
            teacher=teacher,
            skill_authority=skill,
            tournament=tournament,
        )


def test_live_tournament_retains_parent_on_equal_claim_scores(tmp_path: Path) -> None:
    orchestrator, request, teacher, skill, _, parent = _system(tmp_path)
    tournament = _Tournament(tmp_path / "tie-tournament", tie=True)

    result = orchestrator.run(
        request,
        teacher=teacher,
        skill_authority=skill,
        tournament=tournament,
    )

    assert result.decision.tournament_action == "reject-candidates"
    assert result.decision.search_parent_revision_id == parent.revision_id


def test_portfolio_rejects_forged_parent_win_claimed_against_native_receipt(
    tmp_path: Path,
) -> None:
    orchestrator, request, teacher, skill, _, _ = _system(tmp_path)
    tournament = _ForgedParentWinTournament(tmp_path / "forged-tournament")

    with pytest.raises(
        PortfolioViolation,
        match="classification contradicts native outcome",
    ):
        orchestrator.run(
            request,
            teacher=teacher,
            skill_authority=skill,
            tournament=tournament,
        )
