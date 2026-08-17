from __future__ import annotations

import json

import pytest

from evolution_controller import (
    BudgetContractError,
    ControllerContractError,
    EvolutionAuthorization,
    EvolutionController,
    EvolutionPlan,
)
from evolve_service import run_cli

ORIGINAL = "a" * 64
PARENT = "b" * 64
EVIDENCE = "c" * 64
CANDIDATES = tuple(char * 64 for char in "defg")
EPOCH = "native-adapters-v2.1.0-frozen"


def _tasks() -> tuple[str, ...]:
    return tuple(f"task-{index:03d}" for index in range(100))


def _authorization(**overrides) -> EvolutionAuthorization:
    values = {
        "maximum_unique_search_tasks": 100,
        "maximum_real_codex_calls": 2000,
        "maximum_temporary_cloud_instances": 1,
        "maximum_elapsed_hours": 24.0,
        "maximum_cloud_cost_cny": 30.0,
    }
    values.update(overrides)
    return EvolutionAuthorization(**values)


def test_plan_uses_100_unique_tasks_across_four_generations_and_384_calls():
    plan = EvolutionPlan.build(_tasks())

    assert plan.unique_search_tasks == 100
    assert [len(generation.task_uids) for generation in plan.generations] == [25] * 4
    assert [stage.name for stage in plan.generations[0].stages] == ["observe"]
    assert [stage.expected_agent_calls for stage in plan.generations[0].stages] == [50]
    for generation in plan.generations[1:]:
        assert [stage.name for stage in generation.stages] == [
            "scout",
            "semifinal",
            "confirmation",
        ]
        assert [len(stage.task_uids) for stage in generation.stages] == [5, 8, 12]
        assert [stage.expected_agent_calls for stage in generation.stages] == [
            30,
            32,
            36,
        ]
    assert plan.planned_agent_task_calls == 344
    assert plan.maximum_proposer_calls == 32
    assert plan.maximum_reviewer_calls == 8
    assert plan.planned_real_codex_calls == 384
    assert (
        len({task for generation in plan.generations for task in generation.task_uids})
        == 100
    )


def test_build_resume_plan_uses_91_tasks_with_partial_g0():
    tasks = tuple(f"resume-task-{index:03d}" for index in range(91))
    plan = EvolutionPlan.build_resume(tasks)

    assert plan.unique_search_tasks == 91
    assert [len(generation.task_uids) for generation in plan.generations] == [
        16,
        25,
        25,
        25,
    ]
    assert plan.generations[0].stages[0].name == "observe"
    assert plan.generations[0].stages[0].expected_agent_calls == 32
    assert plan.planned_agent_task_calls == 326
    assert plan.planned_real_codex_calls == 366
    assert EvolutionPlan.from_dict(plan.to_dict()).to_dict() == plan.to_dict()
    with pytest.raises(ControllerContractError):
        EvolutionPlan.build_resume(tasks[:90])


def test_plan_rejects_task_reuse_or_non_100_task_scope():
    with pytest.raises(ControllerContractError, match="exactly 100"):
        EvolutionPlan.build(_tasks()[:99])
    duplicate = list(_tasks())
    duplicate[-1] = duplicate[0]
    with pytest.raises(ControllerContractError, match="unique"):
        EvolutionPlan.build(tuple(duplicate))


def test_claim_opens_content_only_until_all_matched_arms_are_frozen_and_retired(
    tmp_path,
):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    first_task = _tasks()[0]
    assert controller.task_materialization_allowed(first_task) is False

    claim = controller.claim_stage(0, "observe", candidate_sha256s=())
    assert claim.arm_sha256s == (ORIGINAL, PARENT)
    assert controller.task_materialization_allowed(first_task) is True
    with pytest.raises(ControllerContractError, match="frozen native evidence"):
        controller.complete_stage(0, "observe")

    for task_uid in claim.task_uids:
        for arm in claim.arm_sha256s:
            controller.record_arm_evidence(
                generation=0,
                stage="observe",
                task_uid=task_uid,
                arm_sha256=arm,
                evidence_sha256=EVIDENCE,
            )
    controller.complete_stage(0, "observe")

    assert controller.task_materialization_allowed(first_task) is False
    assert controller.task_state(first_task) == "retired"
    assert controller.inspect()["usage"]["real_codex_calls"] == 50


def test_resume_and_duplicate_evidence_are_idempotent_but_conflicts_are_rejected(
    tmp_path,
):
    root = tmp_path / "run"
    controller = EvolutionController.initialize(
        root,
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    claim = controller.claim_stage(0, "observe", candidate_sha256s=())
    controller.record_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=claim.task_uids[0],
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
    )

    resumed = EvolutionController(root)
    resumed.record_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=claim.task_uids[0],
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
    )
    assert resumed.inspect()["usage"]["real_codex_calls"] == 1
    with pytest.raises(ControllerContractError, match="immutable"):
        resumed.record_arm_evidence(
            generation=0,
            stage="observe",
            task_uid=claim.task_uids[0],
            arm_sha256=ORIGINAL,
            evidence_sha256="9" * 64,
        )


def test_real_arm_call_is_reserved_before_dispatch_and_not_double_counted(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    claim = controller.claim_stage(0, "observe", candidate_sha256s=())
    task_uid = claim.task_uids[0]

    first = controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )
    resumed = EvolutionController(tmp_path / "run")
    second = resumed.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    assert first["status"] == "reserved"
    assert first["dispatch_allowed"] is True
    assert second["status"] == "reserved"
    assert second["dispatch_allowed"] is False
    assert resumed.inspect()["usage"]["real_codex_calls"] == 1

    resumed.record_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
        real_codex_calls=1,
    )
    assert (
        resumed.arm_call_status(
            generation=0,
            stage="observe",
            task_uid=task_uid,
            arm_sha256=ORIGINAL,
        )["status"]
        == "completed"
    )
    assert resumed.inspect()["usage"]["real_codex_calls"] == 1


def test_pre_dispatch_abort_is_audited_and_can_be_reserved_again(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    claim = controller.claim_stage(0, "observe", candidate_sha256s=())
    task_uid = claim.task_uids[0]
    controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    controller.abort_arm_call_pre_dispatch(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        reason_code="workspace-source-preflight",
        evidence_sha256=EVIDENCE,
    )
    resumed = EvolutionController(tmp_path / "run")
    retry = resumed.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    assert retry["dispatch_allowed"] is True
    assert retry["reservation_count"] == 2
    assert retry["pre_dispatch_aborts"] == [
        {
            "evidence_sha256": EVIDENCE,
            "reason_code": "workspace-source-preflight",
            "reservation_count": 1,
        }
    ]
    assert resumed.inspect()["usage"]["real_codex_calls"] == 2

    resumed.record_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
        real_codex_calls=1,
    )
    assert (
        resumed.arm_call_status(
            generation=0,
            stage="observe",
            task_uid=task_uid,
            arm_sha256=ORIGINAL,
        )["status"]
        == "completed"
    )
    assert resumed.inspect()["usage"]["real_codex_calls"] == 2


def test_no_service_connection_abort_is_distinct_and_recoverable(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    claim = controller.claim_stage(0, "observe", candidate_sha256s=())
    task_uid = claim.task_uids[0]
    controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    aborted = controller.abort_arm_call_without_service_dispatch(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        reason_code="cloud-egress-no-tcp-connection",
        evidence_sha256=EVIDENCE,
    )
    retry = controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    assert aborted["status"] == "aborted_without_service_dispatch"
    assert retry["dispatch_allowed"] is True
    assert retry["reservation_count"] == 2
    assert retry["non_dispatch_aborts"] == [
        {
            "evidence_sha256": EVIDENCE,
            "reason_code": "cloud-egress-no-tcp-connection",
            "reservation_count": 1,
        }
    ]
    assert controller.inspect()["usage"]["real_codex_calls"] == 2


def test_infrastructure_invalid_call_is_retained_and_recoverable(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    claim = controller.claim_stage(0, "observe", candidate_sha256s=())
    task_uid = claim.task_uids[0]
    controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    invalidated = controller.invalidate_arm_call_infrastructure(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        reason_code="linux-sandbox-precondition-failed",
        evidence_sha256=EVIDENCE,
    )
    retry = controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )

    assert invalidated["status"] == "aborted_infrastructure_invalid"
    assert retry["dispatch_allowed"] is True
    assert retry["reservation_count"] == 2
    assert retry["infrastructure_aborts"] == [
        {
            "evidence_sha256": EVIDENCE,
            "reason_code": "linux-sandbox-precondition-failed",
            "reservation_count": 1,
        }
    ]
    assert controller.inspect()["usage"]["real_codex_calls"] == 2


def test_real_auxiliary_call_is_reserved_before_dispatch_and_completed_once(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    reservation_id = "proposer|g0|mutation-01-prompt|attempt-1"

    first = controller.reserve_auxiliary_call(
        reservation_id=reservation_id, kind="mutation_proposer"
    )
    resumed = EvolutionController(tmp_path / "run")
    duplicate = resumed.reserve_auxiliary_call(
        reservation_id=reservation_id, kind="mutation_proposer"
    )

    assert first["dispatch_allowed"] is True
    assert duplicate["dispatch_allowed"] is False
    assert resumed.inspect()["usage"]["real_codex_calls"] == 1
    assert resumed.inspect()["usage"]["auxiliary_calls"] == 1
    resumed.complete_auxiliary_call(
        reservation_id=reservation_id, evidence_sha256=EVIDENCE
    )
    resumed.complete_auxiliary_call(
        reservation_id=reservation_id, evidence_sha256=EVIDENCE
    )
    assert (
        resumed.inspect()["auxiliary_call_reservations"][reservation_id]["status"]
        == "completed"
    )
    assert resumed.inspect()["usage"]["real_codex_calls"] == 1
    assert resumed.inspect()["usage"]["auxiliary_calls"] == 1


def test_aborted_auxiliary_call_can_redispatch_without_refund(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    controller.reserve_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1", kind="mutation_proposer"
    )
    aborted = controller.abort_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1",
        reason_code="codex-provider-config-ignored",
        evidence_sha256=EVIDENCE,
    )
    assert aborted["status"] == "aborted_without_service_dispatch"
    duplicate = controller.abort_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1",
        reason_code="codex-provider-config-ignored",
        evidence_sha256=EVIDENCE,
    )
    assert duplicate == aborted
    usage_after_abort = controller.inspect()["usage"]
    assert usage_after_abort["real_codex_calls"] == 1
    assert usage_after_abort["auxiliary_calls"] == 1

    retry = controller.reserve_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1", kind="mutation_proposer"
    )
    assert retry["dispatch_allowed"] is True
    assert retry["reservation_count"] == 2
    usage = controller.inspect()["usage"]
    assert usage["real_codex_calls"] == 2
    assert usage["auxiliary_calls"] == 2


def test_completed_auxiliary_call_can_be_explicitly_reconciled_and_redispatch(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    controller.reserve_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1", kind="mutation_proposer"
    )
    controller.complete_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1", evidence_sha256=EVIDENCE
    )
    reconciled = controller.reset_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1",
        reason_code="proposer-schema-invalid",
        evidence_sha256=EVIDENCE,
    )
    assert reconciled["status"] == "aborted_without_service_dispatch"
    assert reconciled["aborts"][-1]["previous_evidence_sha256"] == EVIDENCE
    retry = controller.reserve_auxiliary_call(
        reservation_id="proposer|g0|mutation-01|attempt-1", kind="mutation_proposer"
    )
    assert retry["dispatch_allowed"] is True


def test_completed_infrastructure_invalid_evidence_is_quarantined_without_call_refund(
    tmp_path,
):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    task_uid = _tasks()[0]
    controller.claim_stage(0, "observe", candidate_sha256s=())
    controller.reserve_arm_call(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )
    controller.record_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
    )

    quarantined = controller.quarantine_completed_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
        reason_code="native-evaluator-error",
        incident_sha256="d" * 64,
    )
    duplicate = controller.quarantine_completed_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        evidence_sha256=EVIDENCE,
        reason_code="native-evaluator-error",
        incident_sha256="d" * 64,
    )

    assert duplicate == quarantined
    assert (
        controller.arm_evidence_status(
            generation=0,
            stage="observe",
            task_uid=task_uid,
            arm_sha256=ORIGINAL,
        )
        is None
    )
    reservation = controller.arm_call_status(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
    )
    assert reservation["status"] == "reserved"
    assert reservation["evidence_sha256"] is None
    assert controller.inspect()["usage"] == {
        "real_codex_calls": 1,
        "agent_task_calls": 0,
        "auxiliary_calls": 0,
        "cloud_instance_ids": [],
        "elapsed_hours": 0.0,
        "cloud_cost_cny": 0.0,
    }

    controller.record_arm_evidence(
        generation=0,
        stage="observe",
        task_uid=task_uid,
        arm_sha256=ORIGINAL,
        evidence_sha256="e" * 64,
    )
    usage = controller.inspect()["usage"]
    assert usage["real_codex_calls"] == 1
    assert usage["agent_task_calls"] == 1


def test_budget_is_fail_closed_for_calls_instance_time_and_money(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    controller.record_auxiliary_calls(2000)
    with pytest.raises(BudgetContractError, match="2000"):
        controller.record_auxiliary_calls(1)

    controller.register_cloud_usage(
        instance_id="lhins-authorized-1",
        elapsed_hours=23.0,
        cloud_cost_cny=29.0,
    )
    with pytest.raises(BudgetContractError, match="instance"):
        controller.register_cloud_usage(
            instance_id="lhins-second",
            elapsed_hours=23.0,
            cloud_cost_cny=29.0,
        )
    with pytest.raises(BudgetContractError, match="24"):
        controller.register_cloud_usage(
            instance_id="lhins-authorized-1",
            elapsed_hours=24.1,
            cloud_cost_cny=29.0,
        )
    with pytest.raises(BudgetContractError, match="30"):
        controller.register_cloud_usage(
            instance_id="lhins-authorized-1",
            elapsed_hours=23.0,
            cloud_cost_cny=30.01,
        )


def test_controller_refuses_final_sealed_and_production_promotion(tmp_path):
    controller = EvolutionController.initialize(
        tmp_path / "run",
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    with pytest.raises(ControllerContractError, match="final sealed"):
        controller.open_final_sealed()
    with pytest.raises(ControllerContractError, match="production"):
        controller.promote_production(CANDIDATES[0])


def test_local_cli_plan_inspect_and_verify_have_no_external_actions(tmp_path, capsys):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(json.dumps({"task_uids": list(_tasks())}), encoding="utf-8")

    assert run_cli(["evolution-plan", "--tasks", str(tasks_path)]) == 0
    plan_output = json.loads(capsys.readouterr().out)
    assert plan_output["planned_real_codex_calls"] == 384
    assert plan_output["external_actions"] == 0

    root = tmp_path / "run"
    EvolutionController.initialize(
        root,
        plan=EvolutionPlan.build(_tasks()),
        authorization=_authorization(),
        original_agent_program_sha256=ORIGINAL,
        seed_parent_agent_program_sha256=PARENT,
        native_evaluator_epoch=EPOCH,
    )
    assert run_cli(["evolution-inspect", "--root", str(root)]) == 0
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_output["final_sealed_opened"] is False
    assert inspect_output["production_active_ref"] is None

    assert run_cli(["evolution-verify", "--root", str(root)]) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["valid"] is True
    assert verify_output["external_actions"] == 0
