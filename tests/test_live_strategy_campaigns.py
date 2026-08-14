from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evolve.agent_program import AgentProgramRevision
from evolve.campaigns import (
    CampaignRunner,
    CampaignRunStatus,
    CampaignSpec,
    HashVerifiedAgentProgramTransport,
    LegacyImportRuntime,
)
from evolve.contracts import (
    Authorization,
    Claim,
    ClaimClassification,
    ClaimGrade,
    Cohort,
    ContractViolation,
    EvidenceEnvelope,
    ExecutionLimits,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
)
from evolve.evidence import EvidenceGraph, ReceiptStore
from evolve.kernel import CampaignController
from evolve.observers import NativeOutcomeObserver, ObserverHub
from evolve.runtime import ExecutionRuntime
from evolve.strategies import (
    AgentProgramSearchStrategy,
    LegacyImportStrategy,
    StrategyContext,
    StrategyStatus,
)

NOW = datetime(2026, 8, 15, 4, 0, tzinfo=UTC)


def _revision(
    root: Path, *, revision_id: str, parent_revision_id: str | None
) -> AgentProgramRevision:
    return AgentProgramRevision.freeze(
        root,
        program_id="repair-agent",
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        program_prompt=f"Execute complete program {revision_id}.",
        context={"analysis_depth": 3, "revision": revision_id},
        tool_policy=("inspect_workspace", "emit_patch"),
        capability_revision_ids=("localize-r1", "patch-r1"),
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

    def __init__(self) -> None:
        self.consumed: list[tuple[object, ...]] = []

    def infer_program(self, revision, plan, workspace):
        self.consumed.append(
            (
                revision.revision_id,
                revision.program_prompt,
                dict(revision.context),
                revision.tool_policy,
                revision.capability_revision_ids,
            )
        )
        patch = (
            "diff --git a/result.txt b/result.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/result.txt\n"
            f"@@ -0,0 +1 @@\n+{revision.revision_id}\n"
        )
        patch_sha256 = hashlib.sha256(patch.encode()).hexdigest()
        return {
            "plan_id": plan.plan_id,
            "arm": plan.arm,
            "task_revision_id": plan.task.revision_id,
            "task_source_sha256": plan.task.source_sha256,
            "patch": patch,
            "patch_sha256": patch_sha256,
            "prediction_sha256": patch_sha256,
            "structural_valid": True,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_cny": 0,
        }


class _NativeEvaluator:
    evaluator_id = "native-agent-program-v1"

    def evaluate(self, plan, workspace, model_output):
        return {
            "resolved": plan.arm == "candidate",
            "native_valid": True,
            "regressions": [],
            "prediction_sha256": model_output["prediction_sha256"],
        }


def _claim(
    revision_id: str,
    classification: ClaimClassification,
    index: int,
    *,
    evidence_id: str | None = None,
) -> Claim:
    return Claim(
        claim_id=f"claim-{index}",
        candidate_id=revision_id,
        grade=ClaimGrade.E1,
        classification=classification,
        evidence_ids=(evidence_id or f"evidence-{index}",),
        rationale=f"authoritative outcome for {revision_id}",
        supersedes_claim_id=None,
    )


def _append_forged_claim_authority(
    *,
    graph_root: Path,
    plans,
    executions,
    claims: tuple[Claim, ...],
) -> None:
    graph = EvidenceGraph(graph_root)
    by_candidate = {claim.candidate_id: claim for claim in claims}
    for plan, execution in zip(plans, executions, strict=True):
        native = tuple(
            receipt
            for receipt in execution.receipts
            if receipt.kind == "native_evaluation"
        )
        assert len(native) == 1
        claim = by_candidate[plan.candidate_revision_id]
        assert len(claim.evidence_ids) == 1
        graph.append_evidence(
            EvidenceEnvelope(
                evidence_id=claim.evidence_ids[0],
                receipt_ids=(native[0].receipt_id,),
                observer_id="native-v1",
                grade=ClaimGrade.E1,
                payload={
                    "plan_id": plan.plan_id,
                    "candidate_revision_id": plan.candidate_revision_id,
                },
                artifact_sha256=native[0].artifact_sha256,
            )
        )
        graph.append_claim(claim)


def _receipt_with_payload(
    receipt: Receipt, payload: dict[str, object]
) -> tuple[Receipt, bytes]:
    artifact = canonical_json(payload).encode("utf-8")
    return (
        replace(
            receipt,
            payload=payload,
            artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        ),
        artifact,
    )


def test_live_agent_program_runs_complete_revisions_and_advances_from_claims(
    tmp_path: Path,
) -> None:
    parent = _revision(
        tmp_path / "parent", revision_id="program-r1", parent_revision_id=None
    )
    candidate = _revision(
        tmp_path / "candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
    )
    task = TaskRevision(
        task_id="task-1",
        revision_id="task-r1",
        project="project-a",
        cohort=Cohort.FEEDBACK,
        source_sha256="a" * 64,
        evaluator_id=_NativeEvaluator.evaluator_id,
    )
    receipt_store_root = tmp_path / "receipts"
    evidence_graph_root = tmp_path / "claim-evidence-graph"
    context = StrategyContext(
        campaign_id="campaign-live-program",
        task=task,
        model=ModelIdentity("local", "real-program-runtime", "r1"),
        context_policy_id="program-context-v1",
        tool_policy_id="program-tools-v1",
        observer_policy_ids=("receipt-integrity-v1",),
        limits=ExecutionLimits(max_tokens=128, max_seconds=60, max_cost_cny=0),
        inputs={
            "parent_revision_id": parent.revision_id,
            "candidate_revision_ids": (candidate.revision_id,),
            "tournament_id": "tournament-live-1",
            "generation_config": {"temperature": 0},
            "execution_profile": "live",
            "revision_roots": {
                parent.revision_id: str(parent.root),
                candidate.revision_id: str(candidate.root),
            },
            "claim_evidence_graph_root": str(evidence_graph_root),
            "claim_receipt_store_root": str(receipt_store_root),
        },
    )
    authorization = Authorization(
        authorization_id="auth-live-program",
        campaign_id=context.campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=2,
        expires_at=NOW + timedelta(hours=1),
        remote_calls_allowed=False,
    )
    executor = _ProgramExecutor()
    transport = HashVerifiedAgentProgramTransport(
        {
            parent.revision_id: parent.root,
            candidate.revision_id: candidate.root,
        },
        executor=executor,
    )
    graph = EvidenceGraph(evidence_graph_root)
    runtime = ExecutionRuntime(
        model_transport=transport,
        workspace_manager=_Workspace(),
        native_evaluator=_NativeEvaluator(),
        observer_hub=ObserverHub((NativeOutcomeObserver(),), graph=graph),
        receipt_sink=ReceiptStore(receipt_store_root),
        clock=lambda: NOW,
    )
    controller = CampaignController.create(
        campaign_id=context.campaign_id,
        authorization=authorization,
        now=NOW,
    )
    runner = CampaignRunner(runtime=runtime, controller=controller)
    first = runner.run(
        CampaignSpec(
            campaign_id=context.campaign_id,
            contexts=(context,),
            authorization=authorization,
        ),
        AgentProgramSearchStrategy(execution_profile="live"),
    )

    assert first.decisions[0].action == "await-tournament-authority"
    evidence_by_plan = {
        envelope.payload["plan_id"]: envelope
        for envelope in graph.list_evidence()
    }
    assert all(
        "candidate_revision_id" not in envelope.payload
        for envelope in evidence_by_plan.values()
    )
    claims = tuple(
        _claim(
            plan.candidate_revision_id,
            (
                ClaimClassification.GAIN
                if plan.arm == "candidate"
                else ClaimClassification.NEUTRAL
            ),
            index,
            evidence_id=evidence_by_plan[plan.plan_id].evidence_id,
        )
        for index, plan in enumerate(first.plans, start=1)
    )
    for claim in claims:
        graph.append_claim(claim)
    result = runner.run(
        CampaignSpec(
            campaign_id=context.campaign_id,
            contexts=(context,),
            authorization=authorization,
            claims=claims,
        ),
        AgentProgramSearchStrategy(execution_profile="live"),
    )

    assert result.status is CampaignRunStatus.COMPLETED
    assert len(result.executions) == 2
    assert all(execution.replayed for execution in result.executions)
    assert [plan.arm for plan in result.plans] == ["search-parent", "candidate"]
    assert [row[0] for row in executor.consumed] == ["program-r1", "program-r2"]
    assert executor.consumed[1][1:] == (
        candidate.program_prompt,
        dict(candidate.context),
        candidate.tool_policy,
        candidate.capability_revision_ids,
    )
    assert all(
        {"workspace", "model", "native_evaluation", "execution_terminal"}
        <= {receipt.kind for receipt in execution.receipts}
        for execution in result.executions
    )
    interpretation = result.interpretations[0]
    assert interpretation.observations["execution_scope"] == "live"
    assert interpretation.observations["participant_revision_ids"] == (
        "program-r1",
        "program-r2",
    )
    assert interpretation.observations["complete_plan_ids"] == tuple(
        plan.plan_id for plan in result.plans
    )
    decision = result.decisions[0]
    assert decision.status is StrategyStatus.LIVE
    assert decision.action == "advance-search-parent"
    assert "winner=program-r2" in decision.reason
    assert "decision=" in decision.reason
    assert decision.claim_ids == ("claim-1", "claim-2")
    assert result.claims == claims


def test_legacy_import_uses_campaign_runner_without_model_native_or_claims(
    tmp_path: Path,
) -> None:
    artifact = b'{"legacy":"seed-only"}'
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    task = TaskRevision(
        task_id="legacy-task",
        revision_id="legacy-task-r1",
        project="legacy-project",
        cohort=Cohort.FEEDBACK,
        source_sha256="b" * 64,
        evaluator_id="legacy-read-only-v1",
    )
    context = StrategyContext(
        campaign_id="campaign-legacy-import",
        task=task,
        model=ModelIdentity("archive", "legacy-import", "r1"),
        context_policy_id="legacy-read-only-v1",
        tool_policy_id="no-tools-v1",
        observer_policy_ids=("receipt-integrity-v1",),
        limits=ExecutionLimits(max_tokens=0, max_seconds=60, max_cost_cny=0),
        inputs={
            "imported_revision_id": "legacy-r106",
            "legacy_artifact_sha256": artifact_sha256,
            "provenance_uri": "archive/round-106.json",
        },
    )
    authorization = Authorization(
        authorization_id="auth-legacy-import",
        campaign_id=context.campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=0,
        expires_at=NOW + timedelta(hours=1),
        remote_calls_allowed=False,
    )
    store = ReceiptStore(tmp_path / "legacy-receipts")
    controller = CampaignController.create(
        campaign_id=context.campaign_id,
        authorization=authorization,
        now=NOW,
    )

    result = CampaignRunner(
        runtime=LegacyImportRuntime(artifact=artifact, store=store),
        controller=controller,
    ).run(
        CampaignSpec(
            campaign_id=context.campaign_id,
            contexts=(context,),
            authorization=authorization,
        ),
        LegacyImportStrategy(),
    )

    assert result.status is CampaignRunStatus.COMPATIBILITY
    assert [receipt.kind for receipt in result.receipts] == ["legacy_import"]
    assert result.claims == ()
    assert result.decisions[0].action == "use-legacy-replay"
    assert result.interpretations[0].observations == {
        "replayed_receipt_count": 1,
        "imported_revision_id": "legacy-r106",
        "legacy_artifact_sha256": artifact_sha256,
        "provenance_uri": "archive/round-106.json",
        "claims_created": 0,
        "candidates_created": 0,
    }


def test_live_tournament_requires_authoritative_complete_claims_and_parent_wins_tie(
    tmp_path: Path,
) -> None:
    parent = _revision(
        tmp_path / "authority-parent",
        revision_id="program-r1",
        parent_revision_id=None,
    )
    candidate = _revision(
        tmp_path / "authority-candidate",
        revision_id="program-r2",
        parent_revision_id="program-r1",
    )
    receipt_store_root = tmp_path / "authority-receipts"
    evidence_graph_root = tmp_path / "authority-evidence"
    context = StrategyContext(
        campaign_id="campaign-live-authority",
        task=TaskRevision(
            task_id="task-authority",
            revision_id="task-authority-r1",
            project="project-a",
            cohort=Cohort.FEEDBACK,
            source_sha256="c" * 64,
            evaluator_id="native-agent-program-v1",
        ),
        model=ModelIdentity("local", "real-program-runtime", "r1"),
        context_policy_id="program-context-v1",
        tool_policy_id="program-tools-v1",
        observer_policy_ids=("native-v1",),
        limits=ExecutionLimits(max_tokens=128, max_seconds=60, max_cost_cny=0),
        inputs={
            "parent_revision_id": parent.revision_id,
            "candidate_revision_ids": (candidate.revision_id,),
            "tournament_id": "tournament-live-authority",
            "generation_config": {"temperature": 0},
            "execution_profile": "live",
            "revision_roots": {
                parent.revision_id: str(parent.root),
                candidate.revision_id: str(candidate.root),
            },
            "claim_evidence_graph_root": str(evidence_graph_root),
            "claim_receipt_store_root": str(receipt_store_root),
        },
    )
    strategy = AgentProgramSearchStrategy(execution_profile="live")
    plans = strategy.plan(context)
    authorization = Authorization(
        authorization_id="auth-live-authority",
        campaign_id=context.campaign_id,
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=0,
        max_model_calls=2,
        expires_at=NOW + timedelta(hours=1),
        remote_calls_allowed=False,
    )
    store = ReceiptStore(receipt_store_root)
    graph = EvidenceGraph(evidence_graph_root)
    runtime = ExecutionRuntime(
        model_transport=HashVerifiedAgentProgramTransport(
            {
                parent.revision_id: parent.root,
                candidate.revision_id: candidate.root,
            },
            executor=_ProgramExecutor(),
        ),
        workspace_manager=_Workspace(),
        native_evaluator=_NativeEvaluator(),
        observer_hub=ObserverHub((NativeOutcomeObserver(),), graph=graph),
        receipt_sink=store,
        clock=lambda: NOW,
    )
    executions = tuple(runtime.execute(plan, authorization) for plan in plans)
    evidence_by_plan = {
        envelope.payload["plan_id"]: envelope
        for envelope in graph.list_evidence()
    }
    parent_claim = _claim(
        parent.revision_id,
        ClaimClassification.NEUTRAL,
        10,
        evidence_id=evidence_by_plan[plans[0].plan_id].evidence_id,
    )
    candidate_claim = _claim(
        candidate.revision_id,
        ClaimClassification.NEUTRAL,
        11,
        evidence_id=evidence_by_plan[plans[1].plan_id].evidence_id,
    )
    claims = (parent_claim, candidate_claim)
    for claim in claims:
        graph.append_claim(claim)

    tied = strategy.next_action(context, (candidate_claim, parent_claim))
    changed_claim = replace(candidate_claim, rationale="independent rerun evidence")
    changed = strategy.next_action(context, (parent_claim, changed_claim))
    incomplete = strategy.next_action(context, (parent_claim,))
    mismatched = strategy.next_action(
        context,
        (
            parent_claim,
            _claim("program-unknown", ClaimClassification.GAIN, 12),
        ),
    )
    low_grade = strategy.next_action(
        context,
        (
            replace(parent_claim, grade=ClaimGrade.E0),
            replace(candidate_claim, grade=ClaimGrade.E0),
        ),
    )
    duplicate_gain_attack = strategy.next_action(
        context,
        (
            replace(parent_claim, classification=ClaimClassification.GAIN),
            replace(candidate_claim, classification=ClaimClassification.GAIN),
            replace(
                candidate_claim,
                claim_id="claim-13",
                classification=ClaimClassification.GAIN,
                evidence_ids=("evidence-13",),
            ),
        ),
    )

    assert tied.action == "reject-candidates"
    assert "winner=program-r1" in tied.reason
    assert tied.claim_ids == ("claim-10", "claim-11")
    assert changed.action == "await-tournament-authority"
    assert incomplete.action == "await-tournament-authority"
    assert mismatched.action == "await-tournament-authority"
    assert low_grade.action == "await-tournament-authority"
    assert duplicate_gain_attack.action == "await-tournament-authority"

    wrong_plan_graph_root = tmp_path / "wrong-plan-evidence"
    wrong_plan_graph = EvidenceGraph(wrong_plan_graph_root)
    native_receipts = tuple(
        next(
            receipt
            for receipt in execution.receipts
            if receipt.kind == "native_evaluation"
        )
        for execution in executions
    )
    for index, (plan, claim) in enumerate(zip(plans, claims, strict=True)):
        backing = native_receipts[0] if index == 1 else native_receipts[index]
        wrong_plan_graph.append_evidence(
            EvidenceEnvelope(
                evidence_id=claim.evidence_ids[0],
                receipt_ids=(backing.receipt_id,),
                observer_id="native-v1",
                grade=ClaimGrade.E1,
                payload={
                    "plan_id": plan.plan_id,
                    "candidate_revision_id": plan.candidate_revision_id,
                },
                artifact_sha256=backing.artifact_sha256,
            )
        )
        wrong_plan_graph.append_claim(claim)
    wrong_plan_context = replace(
        context,
        inputs={
            **context.inputs,
            "claim_evidence_graph_root": str(wrong_plan_graph_root),
        },
    )
    wrong_plan = strategy.next_action(wrong_plan_context, claims)
    assert wrong_plan.action == "await-tournament-authority"

    for attack in ("wrong-bundle", "wrong-model"):
        forged_store_root = tmp_path / f"{attack}-receipts"
        forged_graph_root = tmp_path / f"{attack}-evidence"
        forged_store = ReceiptStore(forged_store_root)
        rebound_models: dict[str, Receipt] = {}
        for plan, execution in zip(plans, executions, strict=True):
            model = next(row for row in execution.receipts if row.kind == "model")
            model_payload = dict(model.payload)
            if attack == "wrong-bundle" and plan.arm == "candidate":
                model_payload["program_bundle_sha256"] = "f" * 64
            rebound_model, model_artifact = _receipt_with_payload(
                model, model_payload
            )
            forged_store.append(rebound_model, model_artifact)
            rebound_models[plan.plan_id] = rebound_model

        forged_executions = []
        for plan, execution in zip(plans, executions, strict=True):
            native = next(
                row for row in execution.receipts if row.kind == "native_evaluation"
            )
            linked_model = rebound_models[plan.plan_id]
            if attack == "wrong-model" and plan.arm == "candidate":
                linked_model = rebound_models[plans[0].plan_id]
            native_payload = {
                **native.payload,
                "model_receipt_id": linked_model.receipt_id,
                "model_artifact_sha256": linked_model.artifact_sha256,
            }
            rebound_native, native_artifact = _receipt_with_payload(
                native, native_payload
            )
            forged_store.append(rebound_native, native_artifact)
            forged_executions.append(
                replace(
                    execution,
                    receipts=(rebound_models[plan.plan_id], rebound_native),
                )
            )
        _append_forged_claim_authority(
            graph_root=forged_graph_root,
            plans=plans,
            executions=tuple(forged_executions),
            claims=claims,
        )
        forged_context = replace(
            context,
            inputs={
                **context.inputs,
                "claim_evidence_graph_root": str(forged_graph_root),
                "claim_receipt_store_root": str(forged_store_root),
            },
        )
        forged = strategy.next_action(forged_context, claims)
        assert forged.action == "await-tournament-authority", attack

    claims_path = evidence_graph_root / "claims.jsonl"
    claims_path.write_text(
        claims_path.read_text(encoding="utf-8").replace(
            "authoritative outcome for program-r1", "tampered outcome"
        ),
        encoding="utf-8",
    )
    tampered = strategy.next_action(context, claims)
    assert tampered.action == "await-tournament-authority"


def test_live_transport_accepts_an_advanced_search_parent_lineage(tmp_path: Path) -> None:
    parent = _revision(
        tmp_path / "advanced-parent",
        revision_id="program-r2",
        parent_revision_id="program-r1",
    )
    candidate = _revision(
        tmp_path / "advanced-candidate",
        revision_id="program-r3",
        parent_revision_id="program-r2",
    )
    context = StrategyContext(
        campaign_id="campaign-advanced-parent",
        task=TaskRevision(
            task_id="task-advanced",
            revision_id="task-advanced-r1",
            project="project-a",
            cohort=Cohort.FEEDBACK,
            source_sha256="d" * 64,
            evaluator_id="native-agent-program-v1",
        ),
        model=ModelIdentity("local", "real-program-runtime", "r1"),
        context_policy_id="program-context-v1",
        tool_policy_id="program-tools-v1",
        observer_policy_ids=("native-v1",),
        limits=ExecutionLimits(max_tokens=128, max_seconds=60, max_cost_cny=0),
        inputs={
            "parent_revision_id": parent.revision_id,
            "candidate_revision_ids": (candidate.revision_id,),
            "tournament_id": "tournament-advanced-parent",
            "generation_config": {"temperature": 0},
            "execution_profile": "live",
            "revision_roots": {
                parent.revision_id: str(parent.root),
                candidate.revision_id: str(candidate.root),
            },
            "claim_evidence_graph_root": str(tmp_path / "advanced-evidence"),
            "claim_receipt_store_root": str(tmp_path / "advanced-receipts"),
        },
    )
    plans = AgentProgramSearchStrategy(execution_profile="live").plan(context)
    executor = _ProgramExecutor()
    transport = HashVerifiedAgentProgramTransport(
        {parent.revision_id: parent.root, candidate.revision_id: candidate.root},
        executor=executor,
    )
    workspace = {
        "task_revision_id": context.task.revision_id,
        "task_source_sha256": context.task.source_sha256,
    }

    parent_output = transport.infer(plans[0], workspace)
    candidate_output = transport.infer(plans[1], workspace)

    assert parent_output["revision_id"] == "program-r2"
    assert parent_output["parent_revision_id"] == "program-r1"
    assert candidate_output["revision_id"] == "program-r3"
    assert candidate_output["parent_revision_id"] == "program-r2"

    (candidate.root / "PROGRAM-PROMPT.txt").write_text(
        "tampered after planning", encoding="utf-8"
    )
    with pytest.raises(ContractViolation, match="hash verification"):
        transport.infer(plans[1], workspace)
