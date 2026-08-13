from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

from evolve.contracts import (
    Authorization,
    ClaimClassification,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    TaskRevision,
)
from evolve.evidence import ClaimEngine, EvidenceGraph, ReceiptStore
from evolve.kernel import (
    CampaignController,
    CampaignStatus,
    CheckpointManager,
)
from evolve.live_campaign import LiveCampaignSpec, run_skill_paired_campaign
from evolve.observers import (
    CostObserver,
    ExternalTraceObserver,
    NativeOutcomeObserver,
    ObserverHub,
)
from evolve.registry import CapabilityRegistry
from evolve.reporting import AuditVerifier
from evolve.strategies import SkillPairedStrategy

NOW = datetime(2026, 8, 14, 2, 0, tzinfo=UTC)
EVALUATOR_ID = "swebench-native-v1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _tasks(*, second_cohort: Cohort = Cohort.FEEDBACK) -> tuple[TaskRevision, ...]:
    return tuple(
        TaskRevision(
            task_id=task_id,
            revision_id=f"{task_id}-feedback-r1",
            project=project,
            cohort=second_cohort if index == 1 else Cohort.FEEDBACK,
            source_sha256=_sha(f"source:{task_id}"),
            evaluator_id=EVALUATOR_ID,
            source_uri=f"feedback/{task_id}",
        )
        for index, (task_id, project) in enumerate(
            (
                ("sphinx-7757", "sphinx"),
                ("phpspreadsheet-3463", "phpspreadsheet"),
                ("laravel-52684", "laravel"),
            )
        )
    )


def _authorization() -> Authorization:
    return Authorization(
        authorization_id="auth-live-1",
        campaign_id="campaign-live-1",
        allowed_cohorts=(Cohort.FEEDBACK,),
        max_cost_cny=1.2,
        max_model_calls=6,
        expires_at=NOW + timedelta(hours=8),
        remote_calls_allowed=False,
    )


def _spec() -> LiveCampaignSpec:
    return LiveCampaignSpec(
        campaign_id="campaign-live-1",
        baseline_revision_id="prompt-baseline-r1",
        candidate_id="cap-feedback-repair",
        candidate_revision_id="skill-feedback-repair-r2",
        candidate_kind="skill",
        candidate_artifact_sha256=_sha("frozen skill candidate"),
        model=ModelIdentity(
            provider="local-mlx",
            model="qwen3.5-4b",
            revision="frozen-r1",
        ),
        context_policy_id="context-v3",
        tool_policy_id="tools-v3",
        observer_policy_ids=("native-v1", "cost-v1"),
        limits=ExecutionLimits(
            max_tokens=256,
            max_seconds=60,
            max_cost_cny=0.2,
        ),
        final_commit_sha="f" * 40,
        generation_config={"temperature": 0, "seed": 7},
        task_execution_metadata={
            task.revision_id: {
                "base_revision": "1" * 40,
                "benchmark_id": "swe-bench-verified",
                "instance_id": task.task_id,
            }
            for task in _tasks()
        },
    )


class FakeWorkspace:
    def __init__(self) -> None:
        self.calls = 0

    def materialize(self, plan: ExecutionPlan) -> Mapping[str, Any]:
        self.calls += 1
        return {"workspace_id": f"workspace-{plan.plan_id}"}


class FakeTransport:
    remote = False

    def __init__(self) -> None:
        self.calls = 0

    def infer(
        self, plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls += 1
        return {
            "output": f"prediction:{plan.task.task_id}:{plan.arm}",
            "workspace_id": workspace["workspace_id"],
            "cost_cny": 0.1,
            "input_tokens": 10,
            "output_tokens": 5,
        }


class FakeNativeEvaluator:
    evaluator_id = EVALUATOR_ID

    _OUTCOMES = {
        "sphinx-7757": {"baseline": False, "taught": True},
        "phpspreadsheet-3463": {"baseline": False, "taught": False},
        "laravel-52684": {"baseline": True, "taught": False},
    }

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        model_output: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls += 1
        return {
            "resolved": self._OUTCOMES[plan.task.task_id][plan.arm],
            "exit_code": 0,
            "evaluator_id": self.evaluator_id,
            "prediction_sha256": _sha(str(model_output["output"])),
        }


def _dependencies(tmp_path: Path):
    authorization = _authorization()
    checkpoints = CheckpointManager(tmp_path / "checkpoints")
    controller = CampaignController.create(
        campaign_id="campaign-live-1",
        authorization=authorization,
        checkpoint_manager=checkpoints,
        now=NOW,
    )
    receipts = ReceiptStore(tmp_path / "receipt-store")
    graph = EvidenceGraph(tmp_path / "evidence-graph")
    observer = ObserverHub(
        (NativeOutcomeObserver(), CostObserver(), ExternalTraceObserver()),
        graph=graph,
    )
    workspace = FakeWorkspace()
    transport = FakeTransport()
    evaluator = FakeNativeEvaluator()
    capability_registry = CapabilityRegistry(tmp_path / "capabilities.jsonl")
    return {
        "authorization": authorization,
        "checkpoints": checkpoints,
        "controller": controller,
        "receipts": receipts,
        "graph": graph,
        "observer": observer,
        "workspace": workspace,
        "transport": transport,
        "evaluator": evaluator,
        "capability_registry": capability_registry,
    }


def _run(tmp_path: Path, dependencies: dict[str, Any], controller):
    return run_skill_paired_campaign(
        spec=_spec(),
        tasks=_tasks(),
        strategy=SkillPairedStrategy(),
        controller=controller,
        authorization=dependencies["authorization"],
        model_transport=dependencies["transport"],
        workspace_manager=dependencies["workspace"],
        native_evaluator=dependencies["evaluator"],
        receipt_store=dependencies["receipts"],
        observer_hub=dependencies["observer"],
        claim_engine=ClaimEngine(dependencies["graph"]),
        capability_registry=dependencies["capability_registry"],
        report_root=tmp_path / "report",
        clock=lambda: NOW,
    )


def test_runtime_backed_campaign_builds_six_plans_and_projects_auditable_assets(
    tmp_path: Path,
) -> None:
    dependencies = _dependencies(tmp_path)

    result = _run(tmp_path, dependencies, dependencies["controller"])

    assert len(result.plans) == len(result.executions) == 6
    assert [plan.arm for plan in result.plans] == [
        "baseline",
        "taught",
        "baseline",
        "taught",
        "baseline",
        "taught",
    ]
    assert all("base_revision" in plan.metadata for plan in result.plans)
    assert all(execution.status == "completed" for execution in result.executions)
    assert all(execution.replayed is False for execution in result.executions)
    assert dependencies["workspace"].calls == 6
    assert dependencies["transport"].calls == 6
    assert dependencies["evaluator"].calls == 6

    receipts = dependencies["receipts"].list_receipts()
    assert len(receipts) == 36
    assert {receipt.kind for receipt in receipts} == {
        "workspace",
        "model",
        "external_trace",
        "cost",
        "native_evaluation",
        "execution_terminal",
    }
    assert all(
        sum(
            receipt.kind == "native_evaluation"
            for receipt in dependencies["receipts"].receipts_for(plan.plan_id)
        )
        == 1
        for plan in result.plans
    )

    assert [claim.classification for claim in result.claims] == [
        ClaimClassification.GAIN,
        ClaimClassification.NEUTRAL,
        ClaimClassification.REGRESSION,
    ]
    assert len(dependencies["graph"].evidence_by_observer("native-v1")) == 6
    assert len(dependencies["graph"].evidence_by_observer("cost-v1")) == 6
    assert len(dependencies["graph"].evidence_by_observer("external-trace-v1")) == 6

    assert result.capability.active is False
    assert result.capability.evidence_claim_ids == tuple(
        claim.claim_id for claim in result.claims
    )
    assert dependencies["capability_registry"].all() == (result.capability,)

    snapshot = result.snapshot
    assert snapshot.status is CampaignStatus.COMPLETED
    assert len(snapshot.work_items) == 6
    assert all(item.status == "completed" for item in snapshot.work_items)
    assert snapshot.budget.spent_cost_cny == pytest.approx(0.6)
    assert snapshot.budget.spent_model_calls == 6
    assert snapshot.budget.reserved_cost_cny == pytest.approx(0)
    assert snapshot.budget.reserved_model_calls == 0

    assert result.report["receipt_count"] == 36
    assert result.report["claim_count"] == 3
    assert result.report["actual_api_spend_cny"] == pytest.approx(0.6)
    assert result.report["counts"] == {
        "gain": 1,
        "neutral": 1,
        "regression": 1,
    }
    assert AuditVerifier().verify_manifest(
        result.report_paths.manifest_path,
        root=result.report_paths.manifest_path.parent,
    ) == 2


def test_terminal_checkpoint_resume_replays_without_dispatch_or_duplicate_records(
    tmp_path: Path,
) -> None:
    dependencies = _dependencies(tmp_path)
    first = _run(tmp_path, dependencies, dependencies["controller"])
    receipt_log_before = dependencies["receipts"].log_path.read_bytes()
    claims_before = dependencies["graph"].claims_path.read_bytes()
    checkpoint_path = dependencies["checkpoints"].path_for("campaign-live-1")
    checkpoint_before = checkpoint_path.read_bytes()
    report_before = first.report_paths.json_path.read_bytes()

    resumed_controller = CampaignController.from_checkpoint(
        campaign_id="campaign-live-1",
        authorization=dependencies["authorization"],
        checkpoint_manager=dependencies["checkpoints"],
        now=NOW,
    )
    second = _run(tmp_path, dependencies, resumed_controller)

    assert all(execution.replayed for execution in second.executions)
    assert dependencies["workspace"].calls == 6
    assert dependencies["transport"].calls == 6
    assert dependencies["evaluator"].calls == 6
    assert dependencies["receipts"].log_path.read_bytes() == receipt_log_before
    assert dependencies["graph"].claims_path.read_bytes() == claims_before
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert second.report_paths.json_path.read_bytes() == report_before
    assert len(dependencies["capability_registry"].all()) == 1
    assert second.snapshot == first.snapshot


def test_non_feedback_input_is_denied_before_campaign_or_adapter_mutation(
    tmp_path: Path,
) -> None:
    dependencies = _dependencies(tmp_path)
    checkpoint_path = dependencies["checkpoints"].path_for("campaign-live-1")
    checkpoint_before = checkpoint_path.read_bytes()

    with pytest.raises(ContractViolation, match="exactly three feedback"):
        run_skill_paired_campaign(
            spec=_spec(),
            tasks=_tasks(second_cohort=Cohort.HOLDOUT),
            strategy=SkillPairedStrategy(),
            controller=dependencies["controller"],
            authorization=dependencies["authorization"],
            model_transport=dependencies["transport"],
            workspace_manager=dependencies["workspace"],
            native_evaluator=dependencies["evaluator"],
            receipt_store=dependencies["receipts"],
            observer_hub=dependencies["observer"],
            claim_engine=ClaimEngine(dependencies["graph"]),
            capability_registry=dependencies["capability_registry"],
            report_root=tmp_path / "report",
            clock=lambda: NOW,
        )

    assert dependencies["controller"].snapshot().status is CampaignStatus.CREATED
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert dependencies["transport"].calls == 0
    assert dependencies["receipts"].list_receipts() == ()
