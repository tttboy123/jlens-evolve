from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

import pytest

from evolve.contracts import (
    Authorization,
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    Receipt,
    TaskRevision,
    canonical_json,
)
from evolve.kernel import (
    BudgetExceeded,
    CampaignController,
    CampaignStatus,
    CheckpointManager,
    FileLeaseManager,
    LeaseContended,
)
from evolve.runtime import (
    EvaluatorInfrastructureError,
    ExecutionInterrupted,
    ExecutionRuntime,
)

SHA = "a" * 64
NOW = datetime(2026, 8, 14, 1, 0, tzinfo=UTC)


def authorization(
    *,
    campaign_id: str = "campaign-1",
    max_cost_cny: float = 10,
    max_model_calls: int = 3,
    allowed_cohorts: tuple[Cohort, ...] = (Cohort.FEEDBACK,),
) -> Authorization:
    return Authorization(
        authorization_id="auth-1",
        campaign_id=campaign_id,
        allowed_cohorts=allowed_cohorts,
        max_cost_cny=max_cost_cny,
        max_model_calls=max_model_calls,
        expires_at=NOW + timedelta(hours=8),
        remote_calls_allowed=True,
    )


def plan(
    *,
    plan_id: str = "plan-1",
    campaign_id: str = "campaign-1",
    cohort: Cohort = Cohort.FEEDBACK,
    max_cost_cny: float = 2,
    arm: str = "opaque-arm-a",
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        campaign_id=campaign_id,
        strategy_id="strategy-with-opaque-semantics",
        task=TaskRevision(
            task_id="task-1",
            revision_id="task-r1",
            project="project-a",
            cohort=cohort,
            source_sha256=SHA,
            evaluator_id="native@" + SHA,
        ),
        candidate_revision_id="candidate-r1",
        arm=arm,
        model=ModelIdentity(provider="local", model="frozen-model", revision="r1"),
        context_policy_id="context-r1",
        tool_policy_id="tools-r1",
        observer_policy_ids=("outcome-r1",),
        native_evaluator_id="native@" + SHA,
        limits=ExecutionLimits(
            max_tokens=128,
            max_seconds=30,
            max_cost_cny=max_cost_cny,
        ),
        holdout_scope="feedback-only" if cohort is Cohort.FEEDBACK else "holdout",
    )


def test_campaign_lifecycle_is_strategy_neutral_and_budgeted() -> None:
    controller = CampaignController.create(
        campaign_id="campaign-1",
        authorization=authorization(),
        now=NOW,
    )

    controller.start()
    assert controller.submit(plan(), reserved_model_calls=1) is True
    assert controller.submit(plan(), reserved_model_calls=1) is False
    controller.record_result("plan-1", actual_cost_cny=1.25, actual_model_calls=1)
    controller.finalize(CampaignStatus.COMPLETED)

    snapshot = controller.snapshot()
    assert snapshot.status is CampaignStatus.COMPLETED
    assert snapshot.work_items[0].arm == "opaque-arm-a"
    assert snapshot.work_items[0].status == "completed"
    assert snapshot.budget.spent_cost_cny == 1.25
    assert snapshot.budget.spent_model_calls == 1


def test_campaign_rejects_wrong_authorization_and_cumulative_budget_exhaustion() -> (
    None
):
    controller = CampaignController.create(
        campaign_id="campaign-1",
        authorization=authorization(max_cost_cny=2, max_model_calls=1),
        now=NOW,
    )
    controller.start()

    with pytest.raises(ContractViolation, match="campaign"):
        controller.submit(plan(campaign_id="another-campaign"))

    assert controller.submit(plan(max_cost_cny=2)) is True
    with pytest.raises(BudgetExceeded, match="model call"):
        controller.submit(
            plan(plan_id="plan-2", max_cost_cny=0), reserved_model_calls=1
        )


def test_feedback_authorization_denies_holdout_execution() -> None:
    controller = CampaignController.create(
        campaign_id="campaign-1",
        authorization=authorization(),
        now=NOW,
    )
    controller.start()

    with pytest.raises(ContractViolation, match="cohort"):
        controller.submit(plan(cohort=Cohort.HOLDOUT))


def test_checkpoint_resume_is_idempotent_and_preserves_terminal_exclusivity(
    tmp_path,
) -> None:
    checkpoints = CheckpointManager(tmp_path / "checkpoints")
    controller = CampaignController.create(
        campaign_id="campaign-1",
        authorization=authorization(),
        checkpoint_manager=checkpoints,
        now=NOW,
    )
    controller.start()
    controller.submit(plan())
    before = checkpoints.path_for("campaign-1").read_bytes()

    resumed = CampaignController.from_checkpoint(
        campaign_id="campaign-1",
        authorization=authorization(),
        checkpoint_manager=checkpoints,
        now=NOW,
    )
    assert resumed.submit(plan()) is False
    assert checkpoints.path_for("campaign-1").read_bytes() == before

    resumed.mark_partial("worker received SIGTERM")
    assert resumed.snapshot().status is CampaignStatus.PARTIAL
    with pytest.raises(ContractViolation, match="terminal"):
        resumed.finalize(CampaignStatus.CANCELLED)


def test_file_lease_allows_only_one_writer_and_checks_release_token(tmp_path) -> None:
    leases = FileLeaseManager(tmp_path / "leases")
    first = leases.acquire(
        "campaign-1", owner_id="worker-a", now=NOW, ttl=timedelta(minutes=5)
    )

    with pytest.raises(LeaseContended, match="worker-a"):
        leases.acquire(
            "campaign-1", owner_id="worker-b", now=NOW, ttl=timedelta(minutes=5)
        )
    with pytest.raises(ContractViolation, match="token"):
        leases.release("campaign-1", "wrong-token")

    leases.release("campaign-1", first.token)
    second = leases.acquire(
        "campaign-1", owner_id="worker-b", now=NOW, ttl=timedelta(minutes=5)
    )
    assert second.owner_id == "worker-b"


class MemoryReceiptSink:
    def __init__(self) -> None:
        self.receipts: list[Receipt] = []

    def append(self, receipt: Receipt, artifact: bytes) -> Receipt:
        assert hashlib.sha256(artifact).hexdigest() == receipt.artifact_sha256
        matching = [
            item for item in self.receipts if item.receipt_id == receipt.receipt_id
        ]
        if matching:
            assert matching[0] == receipt
            return matching[0]
        self.receipts.append(receipt)
        return receipt

    def receipts_for(self, plan_id: str) -> tuple[Receipt, ...]:
        return tuple(item for item in self.receipts if item.plan_id == plan_id)


class FakeWorkspace:
    def __init__(self) -> None:
        self.calls = 0

    def materialize(self, execution_plan: ExecutionPlan) -> Mapping[str, Any]:
        self.calls += 1
        return {
            "workspace_id": "ws-1",
            "task_sha256": execution_plan.task.source_sha256,
        }


class FakeTransport:
    remote = False

    def __init__(self, *, interrupt: bool = False, cost_cny: float = 0.25) -> None:
        self.calls = 0
        self.interrupt = interrupt
        self.cost_cny = cost_cny

    def infer(
        self, execution_plan: ExecutionPlan, workspace: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self.calls += 1
        if self.interrupt:
            raise ExecutionInterrupted("simulated SIGTERM")
        return {
            "output": "candidate patch",
            "model_revision": execution_plan.model.revision,
            "workspace_id": workspace["workspace_id"],
            "cost_cny": self.cost_cny,
            "input_tokens": 10,
            "output_tokens": 5,
        }


class FakeEvaluator:
    evaluator_id = "native@" + SHA

    def __init__(self, *, infrastructure_error: bool = False) -> None:
        self.calls = 0
        self.infrastructure_error = infrastructure_error

    def evaluate(
        self,
        execution_plan: ExecutionPlan,
        workspace: Mapping[str, Any],
        model_output: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.calls += 1
        if self.infrastructure_error:
            raise EvaluatorInfrastructureError("native harness did not start")
        return {"resolved": True, "exit_code": 0, "evaluator_id": self.evaluator_id}


class FakeObserverHub:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self, receipt: Receipt) -> tuple[()]:
        self.calls += 1
        assert receipt.artifact_sha256
        return ()


def runtime_fixture(*, transport=None, evaluator=None, sink=None):
    workspace = FakeWorkspace()
    actual_transport = transport or FakeTransport()
    actual_evaluator = evaluator or FakeEvaluator()
    observer = FakeObserverHub()
    actual_sink = sink or MemoryReceiptSink()
    runtime = ExecutionRuntime(
        model_transport=actual_transport,
        workspace_manager=workspace,
        native_evaluator=actual_evaluator,
        observer_hub=observer,
        receipt_sink=actual_sink,
        clock=lambda: NOW,
    )
    return runtime, workspace, actual_transport, actual_evaluator, observer, actual_sink


def test_execution_runtime_is_the_single_entry_and_replays_finalized_receipts() -> None:
    runtime, workspace, transport, evaluator, observer, sink = runtime_fixture()

    first = runtime.execute(plan(), authorization())
    second = runtime.execute(plan(), authorization())

    assert first.status == "completed"
    assert first.replayed is False
    assert second.status == "completed"
    assert second.replayed is True
    assert workspace.calls == transport.calls == evaluator.calls == 1
    assert observer.calls == 5
    assert [receipt.kind for receipt in first.receipts] == [
        "workspace",
        "model",
        "external_trace",
        "cost",
        "native_evaluation",
        "execution_terminal",
    ]
    assert all(
        receipt.artifact_sha256
        == hashlib.sha256(canonical_json(receipt.payload).encode()).hexdigest()
        for receipt in sink.receipts
    )
    native = next(item for item in first.receipts if item.kind == "native_evaluation")
    assert native.payload["arm"] == "opaque-arm-a"
    assert native.payload["task_revision_id"] == "task-r1"
    assert native.payload["task_source_sha256"] == SHA
    assert native.payload["model_identity"] == "local/frozen-model@r1"
    assert native.payload["native_evaluator_id"] == "native@" + SHA
    assert native.payload["execution_config_sha256"]
    assert native.payload["evaluator_error"] is None
    model = next(item for item in first.receipts if item.kind == "model")
    assert native.payload["model_receipt_id"] == model.receipt_id
    assert native.payload["model_artifact_sha256"] == model.artifact_sha256


def test_terminal_replay_rejects_a_different_plan_with_the_same_id() -> None:
    runtime, _, _, _, _, _ = runtime_fixture()
    original = plan(plan_id="shared-plan")
    assert runtime.execute(original, authorization()).status == "completed"

    with pytest.raises(ContractViolation, match="plan identity"):
        runtime.execute(
            replace(original, candidate_revision_id="different-candidate"),
            authorization(),
        )


def test_runtime_projects_model_dispatch_as_external_trace_receipt() -> None:
    runtime, _, _, _, _, _ = runtime_fixture()

    result = runtime.execute(plan(), authorization())

    trace = next(
        receipt for receipt in result.receipts if receipt.kind == "external_trace"
    )
    model = next(receipt for receipt in result.receipts if receipt.kind == "model")
    assert trace.payload["model_receipt_id"] == model.receipt_id
    assert trace.payload["model_artifact_sha256"] == model.artifact_sha256


def test_runtime_rejects_ambiguous_duplicate_stage_receipts() -> None:
    sink = MemoryReceiptSink()
    runtime, _, _, _, _, _ = runtime_fixture(sink=sink)
    frozen_plan = plan()
    completed = runtime.execute(frozen_plan, authorization())
    model = next(receipt for receipt in completed.receipts if receipt.kind == "model")
    sink.receipts.insert(2, replace(model, receipt_id="duplicate-model", sequence=3))

    with pytest.raises(ContractViolation, match="duplicate model"):
        runtime.execute(frozen_plan, authorization())


def test_execution_runtime_records_infrastructure_error_without_claiming_result() -> (
    None
):
    evaluator = FakeEvaluator(infrastructure_error=True)
    runtime, _, transport, _, observer, _ = runtime_fixture(evaluator=evaluator)

    result = runtime.execute(plan(), authorization())

    assert result.status == "infra_failure"
    assert transport.calls == evaluator.calls == 1
    assert observer.calls == 5
    assert result.receipts[-2].kind == "native_evaluation"
    assert result.receipts[-2].payload["resolved"] is False
    assert result.receipts[-2].payload["evaluator_error"] == (
        "EvaluatorInfrastructureError: native harness did not start"
    )
    model = next(item for item in result.receipts if item.kind == "model")
    assert result.receipts[-2].payload["model_receipt_id"] == model.receipt_id
    assert (
        result.receipts[-2].payload["model_artifact_sha256"]
        == model.artifact_sha256
    )
    assert result.receipts[-1].payload["status"] == "infra_failure"


def test_execution_runtime_partial_receipt_resumes_without_repeating_completed_stage() -> (
    None
):
    sink = MemoryReceiptSink()
    interrupted_transport = FakeTransport(interrupt=True)
    runtime, workspace, _, _, _, _ = runtime_fixture(
        transport=interrupted_transport, sink=sink
    )
    partial = runtime.execute(plan(), authorization())
    assert partial.status == "partial"
    assert [item.kind for item in partial.receipts] == [
        "workspace",
        "execution_partial",
    ]

    healthy_transport = FakeTransport()
    resumed, resumed_workspace, _, evaluator, observer, _ = runtime_fixture(
        transport=healthy_transport, sink=sink
    )
    completed = resumed.execute(plan(), authorization())

    assert completed.status == "completed"
    assert resumed_workspace.calls == 0
    assert healthy_transport.calls == evaluator.calls == 1
    assert observer.calls == 4
    assert len([item for item in sink.receipts if item.kind == "workspace"]) == 1


def test_execution_runtime_denies_invalid_authorization_before_adapters() -> None:
    runtime, workspace, transport, evaluator, observer, sink = runtime_fixture()

    with pytest.raises(ContractViolation, match="campaign"):
        runtime.execute(plan(), authorization(campaign_id="different"))

    assert workspace.calls == transport.calls == evaluator.calls == observer.calls == 0
    assert sink.receipts == []


def test_execution_runtime_stops_before_native_when_actual_cost_exceeds_plan() -> None:
    transport = FakeTransport(cost_cny=2.01)
    runtime, _, _, evaluator, observer, _ = runtime_fixture(transport=transport)

    result = runtime.execute(plan(max_cost_cny=2), authorization())

    assert result.status == "budget_exhausted"
    assert evaluator.calls == 0
    assert observer.calls == 4
    assert result.receipts[-1].kind == "execution_terminal"
    assert result.receipts[-1].payload["status"] == "budget_exhausted"


def test_checkpoint_refuses_mutated_payload(tmp_path) -> None:
    checkpoints = CheckpointManager(tmp_path / "checkpoints")
    controller = CampaignController.create(
        campaign_id="campaign-1",
        authorization=authorization(),
        checkpoint_manager=checkpoints,
        now=NOW,
    )
    controller.start()
    path = checkpoints.path_for("campaign-1")
    path.write_text(path.read_text().replace('"running"', '"paused"'), encoding="utf-8")

    with pytest.raises(ContractViolation, match="hash mismatch"):
        CampaignController.from_checkpoint(
            campaign_id="campaign-1",
            authorization=authorization(),
            checkpoint_manager=checkpoints,
            now=NOW,
        )
