from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolve.contracts import (
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    TaskRevision,
)
from evolve.proposals import (
    CandidateChangeSet,
    CompiledOperator,
    CompiledRevision,
    CompiledRouter,
    CompiledSkill,
)
from evolve.runtime.qwen_transport import LegacyQwenCellRunner, LegacyQwenPairTransport

SHA = "a" * 64


def _plan(*, arm: str = "baseline", cohort: Cohort = Cohort.FEEDBACK) -> ExecutionPlan:
    task = TaskRevision(
        task_id="sphinx-doc__sphinx-7757",
        revision_id="feedback-sphinx-7757-v1",
        project="sphinx",
        cohort=cohort,
        source_sha256=SHA,
        evaluator_id="official-native-v1",
        source_uri="/tmp/source",
    )
    return ExecutionPlan(
        plan_id=f"plan-{arm}",
        campaign_id="campaign-live",
        strategy_id="skill-paired-v3",
        task=task,
        candidate_revision_id=f"candidate-{arm}",
        arm=arm,
        model=ModelIdentity(provider="local-mlx", model="qwen", revision="frozen"),
        context_policy_id="context-v1",
        tool_policy_id="tools-v1",
        observer_policy_ids=("native",),
        native_evaluator_id=task.evaluator_id,
        limits=ExecutionLimits(max_tokens=1536, max_seconds=900, max_cost_cny=0),
        holdout_scope="feedback-only" if cohort is Cohort.FEEDBACK else "holdout",
    )


class RecordingCellRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def run(self, plan, workspace, output_root: Path):
        self.calls.append((plan.plan_id, plan.arm, workspace["task_revision_id"]))
        artifact_root = output_root / plan.plan_id
        artifact_root.mkdir(parents=True)
        raw_path = artifact_root / "raw-output.txt"
        raw_path.write_text("fresh model output", encoding="utf-8")
        return {
            "patch": "diff --git a/a.py b/a.py\n",
            "patch_sha256": "58796596d75205da0ff5f87d83bc169e9ec91b6f6a6c21cbd47af71bf79be2b0",
            "raw_output_path": str(raw_path),
            "raw_output_sha256": "34a4c0279bcfbbc80397957d648c02a67a2f00c1352c1f7c5516ee8a6da6c5b8",
            "prompt_paths": [],
            "prompt_sha256": [],
            "structural_valid": True,
            "failure_reason": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_cny": 0,
        }


def test_qwen_transport_dispatches_each_feedback_arm_through_cell_runner(
    tmp_path,
) -> None:
    runner = RecordingCellRunner()
    transport = LegacyQwenPairTransport(cell_runner=runner, output_root=tmp_path)
    workspace = {
        "task_revision_id": "feedback-sphinx-7757-v1",
        "checkout": "/tmp/source",
    }

    output = transport.infer(_plan(arm="baseline"), workspace)

    assert transport.remote is False
    assert output["structural_valid"] is True
    assert output["arm"] == "baseline"
    assert output["plan_id"] == "plan-baseline"
    assert output["task_revision_id"] == "feedback-sphinx-7757-v1"
    assert output["task_source_sha256"] == SHA
    assert (
        output["patch_sha256"]
        == "58796596d75205da0ff5f87d83bc169e9ec91b6f6a6c21cbd47af71bf79be2b0"
    )
    assert runner.calls == [("plan-baseline", "baseline", "feedback-sphinx-7757-v1")]


def test_qwen_transport_fails_closed_on_task_or_arm_drift(tmp_path) -> None:
    transport = LegacyQwenPairTransport(
        cell_runner=RecordingCellRunner(), output_root=tmp_path
    )

    with pytest.raises(ContractViolation, match="workspace task"):
        transport.infer(
            _plan(), {"task_revision_id": "another-task", "checkout": "/tmp/source"}
        )
    with pytest.raises(ContractViolation, match="arm"):
        transport.infer(
            _plan(arm="candidate"),
            {
                "task_revision_id": "feedback-sphinx-7757-v1",
                "checkout": "/tmp/source",
            },
        )
    with pytest.raises(ContractViolation, match="feedback"):
        transport.infer(
            _plan(cohort=Cohort.HOLDOUT),
            {
                "task_revision_id": "feedback-sphinx-7757-v1",
                "checkout": "/tmp/source",
            },
        )


def test_legacy_runner_baseline_does_not_read_candidate_and_taught_fails_closed(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    model_root = tmp_path / "model"
    legacy_root.mkdir()
    model_root.mkdir()
    taskset = tmp_path / "TASKSET.json"
    routes = tmp_path / "ROUTES.json"
    taskset.write_text('{"tasks": []}\n', encoding="utf-8")
    routes.write_text('{"routes": {}}\n', encoding="utf-8")
    missing_candidate = tmp_path / "compiled-candidate"
    runner = LegacyQwenCellRunner(
        legacy_root=legacy_root,
        model_path=model_root,
        taskset_path=taskset,
        routes_path=routes,
        compiled_revision_root=missing_candidate,
    )

    assert runner._compiled_for_plan(_plan(arm="baseline")) is None
    assert not missing_candidate.exists()

    with pytest.raises(ContractViolation, match="compiled revision manifest"):
        runner._compiled_for_plan(_plan(arm="taught"))


def _compiled(tmp_path: Path, skill_text: str) -> CompiledRevision:
    change_set = CandidateChangeSet(
        candidate_id="candidate-live",
        revision_id="candidate-taught",
        parent_revision_id="baseline-r1",
        source_candidate_sha256=hashlib.sha256(skill_text.encode()).hexdigest(),
        compile_spec_sha256="b" * 64,
        protocol="inactive_external_agent",
        prompt_template="Return one deterministic plan.",
        skill_text=skill_text,
        eval_note="Evaluate with matched native A/B.",
        operator_id="operator-live",
        operator_instruction="Apply the compiled teaching.",
        routes=(("sphinx-doc__sphinx-7757", "operator-live"),),
    )
    return CompiledRevision(
        root=tmp_path,
        change_set=change_set,
        skill=CompiledSkill(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            parent_revision_id=change_set.parent_revision_id,
            protocol=change_set.protocol,
            prompt_template=change_set.prompt_template,
            skill_text=change_set.skill_text,
        ),
        operator=CompiledOperator(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            operator_id=change_set.operator_id,
            kind="zero-arg",
            arguments=(),
            instruction=change_set.operator_instruction,
        ),
        router=CompiledRouter(
            candidate_id=change_set.candidate_id,
            revision_id=change_set.revision_id,
            routes=change_set.routes,
        ),
        provider="deepseek",
        model="deepseek-v4-flash",
        cost_cny=0.001,
        artifact_sha256=(),
        manifest_path=tmp_path / "COMPILED-REVISION.json",
        bundle_sha256=hashlib.sha256(skill_text.encode()).hexdigest(),
    )


def test_legacy_condition_prompt_changes_only_for_taught_candidate_content(
    tmp_path: Path,
) -> None:
    class Adapter:
        @staticmethod
        def experiment_config():
            return {"temperature": 0}

    def builder(*, taught_skill: str, **_kwargs):
        return (
            SimpleNamespace(
                mechanism="operator",
                teaching="baseline",
                revision=hashlib.sha256(b"baseline-no-candidate").hexdigest(),
            ),
            SimpleNamespace(
                mechanism="operator",
                teaching="taught",
                revision=hashlib.sha256(taught_skill.encode()).hexdigest(),
            ),
        )

    first = _compiled(tmp_path / "first", "Preserve exact declarations.")
    second = _compiled(tmp_path / "second", "Preserve exact field declarations.")
    baseline_first = LegacyQwenCellRunner._condition_for_plan(
        plan=_plan(arm="baseline"),
        mechanism="operator",
        adapter=Adapter(),
        compiled=first,
        builder=builder,
    )
    baseline_second = LegacyQwenCellRunner._condition_for_plan(
        plan=_plan(arm="baseline"),
        mechanism="operator",
        adapter=Adapter(),
        compiled=second,
        builder=builder,
    )
    taught_first = LegacyQwenCellRunner._condition_for_plan(
        plan=_plan(arm="taught"),
        mechanism="operator",
        adapter=Adapter(),
        compiled=first,
        builder=builder,
    )
    taught_second = LegacyQwenCellRunner._condition_for_plan(
        plan=_plan(arm="taught"),
        mechanism="operator",
        adapter=Adapter(),
        compiled=second,
        builder=builder,
    )

    assert baseline_first.revision == baseline_second.revision
    assert taught_first.revision != taught_second.revision
