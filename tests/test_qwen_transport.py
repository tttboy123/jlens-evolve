from __future__ import annotations

from pathlib import Path

import pytest

from evolve.contracts import (
    Cohort,
    ContractViolation,
    ExecutionLimits,
    ExecutionPlan,
    ModelIdentity,
    TaskRevision,
)
from evolve.runtime.qwen_transport import LegacyQwenPairTransport

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


def test_qwen_transport_dispatches_each_feedback_arm_through_cell_runner(tmp_path) -> None:
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
    assert output["patch_sha256"] == "58796596d75205da0ff5f87d83bc169e9ec91b6f6a6c21cbd47af71bf79be2b0"
    assert runner.calls == [
        ("plan-baseline", "baseline", "feedback-sphinx-7757-v1")
    ]


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
