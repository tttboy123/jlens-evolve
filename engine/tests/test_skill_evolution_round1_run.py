from pathlib import Path

import pytest

from skill_evolution_loop.__main__ import _parser
from skill_evolution_loop.contracts import ContractError, canonical_json, sha256_json
from skill_evolution_loop.eval_manifest import EvaluationTask, EvaluationTaskSet
from skill_evolution_loop.round1_feedback import (
    augment_feedback_request_with_catalog_context,
)
from skill_evolution_loop.round1_run import (
    BudgetedModelTransport,
    GenerationCallBudget,
    _freeze_model_transport_metrics,
    _select_round1_task_ids,
    compile_round1_span_skill,
    load_round1_feedback_gain_gate,
)


def test_generation_call_budget_fails_before_delegate_on_fifth_call() -> None:
    calls: list[str] = []

    def delegate(*_args, **_kwargs):
        calls.append("called")
        return "output"

    budget = GenerationCallBudget(delegate=delegate, maximum_calls=4)

    assert [budget() for _ in range(4)] == ["output"] * 4
    with pytest.raises(ContractError, match="generation call budget exhausted"):
        budget()

    assert calls == ["called"] * 4
    assert budget.started_calls == 4


def test_transport_generation_budget_counts_prompt_and_chat_calls() -> None:
    class Delegate:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            return request

        def generate_prompt(self, request):
            self.calls += 1
            return request

        @staticmethod
        def identity():
            return {"kind": "fixture"}

    delegate = Delegate()
    transport = BudgetedModelTransport(delegate=delegate, maximum_calls=2)

    assert transport.generate("chat") == "chat"
    assert transport.generate_prompt("prompt") == "prompt"
    with pytest.raises(ContractError, match="generation call budget exhausted"):
        transport.generate_prompt("blocked")

    assert delegate.calls == 2


def test_remote_transport_metrics_are_frozen_as_append_only_receipt(
    tmp_path: Path,
) -> None:
    metrics = {
        "cache_entries": 2,
        "remote_calls": 2,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "current_process_cache_hits": 0,
        "current_process_cache_misses": 2,
        "current_process_cache_writes": 2,
    }

    first = _freeze_model_transport_metrics(
        evidence_root=tmp_path,
        experiment_projection_sha256="a" * 64,
        transport_identity={"kind": "openai-compatible-chat", "model": "student"},
        metrics=metrics,
    )
    second = _freeze_model_transport_metrics(
        evidence_root=tmp_path,
        experiment_projection_sha256="a" * 64,
        transport_identity={"kind": "openai-compatible-chat", "model": "student"},
        metrics=metrics,
    )

    assert first == second
    assert first["network_calls_performed"] is True
    assert first["model_transport_metrics"]["total_tokens"] == 120
    assert len(list((tmp_path / "model-transport-metrics").glob("*.json"))) == 1

    offline = _freeze_model_transport_metrics(
        evidence_root=tmp_path / "offline",
        experiment_projection_sha256="b" * 64,
        transport_identity={"kind": "openai-compatible-chat", "model": "student"},
        metrics={
            **metrics,
            "current_process_cache_hits": 2,
            "current_process_cache_misses": 0,
            "current_process_cache_writes": 0,
        },
    )
    assert offline["network_calls_performed"] is False


def test_round1_span_skill_is_inactive_action_contract_compilation(
    tmp_path: Path,
) -> None:
    source = Path(
        "artifacts/v2.5.0/v2.5.0-local-jlens/runs/skill-evolution-loop/"
        "p1-r6-operator-pattern-skill-v2/OPERATOR-PATTERN-REVISION.round-006.json"
    )

    report = compile_round1_span_skill(
        operator_skill_path=source,
        output_path=tmp_path / "SPAN-SKILL.json",
    )

    assert report["candidate_status"] == "inactive"
    assert report["auto_activate"] is False
    assert report["new_domain_knowledge_added"] is False
    assert "active: false" in report["skill_text"]
    assert "two files" in report["skill_text"]


def test_round1_holdout_gate_requires_complete_native_feedback_gain(
    tmp_path: Path,
) -> None:
    taskset = EvaluationTaskSet.create(
        taskset_id="round1-holdout-gate",
        tasks=[
            EvaluationTask.create(
                task_id=f"task-{index}",
                instance_id=f"repo__repo-{index}",
                benchmark_id="swe-bench-verified",
                benchmark_base_commit="a" * 40,
                repo="repo/repo",
                source_repository=tmp_path,
                source_revision="a" * 40,
                instruction=f"Fix {index}",
                allowed_targets=["src/example.py"],
                cohort="feedback" if index <= 30 else "holdout",
            )
            for index in range(1, 61)
        ],
    )
    content = {
        "schema_version": 1,
        "evaluation_scope": "round1-feedback-only",
        "status": "complete",
        "taskset_fingerprint": taskset.fingerprint,
        "planned_cells": 60,
        "generated_feedback_cells": 60,
        "completed_cells": 60,
        "native_evaluator_failure_count": 0,
        "cell_evidence_fingerprint": "b" * 64,
        "feedback_gain_count": 1,
        "feedback_gain_gate_passed": True,
        "full_capability_gate_evaluated": False,
        "holdout_cells_opened": False,
        "network_calls_performed": False,
    }
    path = tmp_path / "SUMMARY.json"
    path.write_text(
        canonical_json({**content, "summary_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )

    gate = load_round1_feedback_gain_gate(path, taskset)

    assert gate["feedback_gain_count"] == 1
    content["feedback_gain_count"] = 0
    content["feedback_gain_gate_passed"] = False
    path.write_text(
        canonical_json({**content, "summary_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="feedback gain has not unlocked holdout"):
        load_round1_feedback_gain_gate(path, taskset)

    content["feedback_gain_count"] = 1
    content["feedback_gain_gate_passed"] = True
    content["native_evaluator_failure_count"] = 1
    path.write_text(
        canonical_json({**content, "summary_sha256": sha256_json(content)}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="summary boundary is invalid"):
        load_round1_feedback_gain_gate(path, taskset)


def test_round1_feedback_cli_exposes_explicit_shared_context_mechanism() -> None:
    args = _parser().parse_args(
        [
            "round1-feedback-run",
            "--manifest",
            "TASKSET.json",
            "--routes",
            "ROUTES.json",
            "--target-audit",
            "AUDIT.json",
            "--operator-skill",
            "OPERATOR.json",
            "--span-skill",
            "SPAN.json",
            "--out",
            "evidence",
            "--task",
            "task-feedback-2",
            "--shared-diagnosis-localization",
            "--max-plan-repairs",
            "0",
            "--max-generation-calls",
            "4",
        ]
    )

    assert args.shared_diagnosis_localization is True
    assert args.max_plan_repairs == 0
    assert args.max_generation_calls == 4
    assert args.task_ids == ["task-feedback-2"]


def test_round1_cli_accepts_generic_remote_transport_without_cloud_specific_flags() -> (
    None
):
    args = _parser().parse_args(
        [
            "round1-holdout-run",
            "--manifest",
            "TASKSET.json",
            "--routes",
            "ROUTES.json",
            "--target-audit",
            "AUDIT.json",
            "--operator-skill",
            "OPERATOR.json",
            "--span-skill",
            "SPAN.json",
            "--feedback-gain",
            "GAIN.json",
            "--out",
            "evidence",
            "--transport-base-url",
            "http://127.0.0.1:8000",
            "--transport-model",
            "same-base-cuda",
            "--generation-cache",
            "model-cache",
            "--futility-min-cells",
            "8",
            "--futility-min-structural-rate",
            "0.01",
        ]
    )

    assert args.transport_base_url == "http://127.0.0.1:8000"
    assert args.transport_model == "same-base-cuda"
    assert args.generation_cache == Path("model-cache")
    assert args.futility_min_cells == 8
    assert args.futility_min_structural_rate == 0.01


def test_round1_holdout_cli_accepts_explicit_append_only_shard_tasks() -> None:
    args = _parser().parse_args(
        [
            "round1-holdout-run",
            "--manifest",
            "TASKSET.json",
            "--routes",
            "ROUTES.json",
            "--target-audit",
            "AUDIT.json",
            "--operator-skill",
            "OPERATOR.json",
            "--span-skill",
            "SPAN.json",
            "--feedback-gain",
            "GAIN.json",
            "--out",
            "evidence-shard",
            "--task",
            "task-holdout-2",
        ]
    )

    assert args.task_ids == ["task-holdout-2"]


def test_round1_feedback_audit_cli_is_offline_and_catalog_aware() -> None:
    args = _parser().parse_args(
        [
            "round1-feedback-audit",
            "--request",
            "REQUEST.json",
            "--evolution-catalog",
            "catalog",
            "--out",
            "AUDIT.json",
        ]
    )

    assert args.request == Path("REQUEST.json")
    assert args.evolution_catalog == Path("catalog")
    assert args.out == Path("AUDIT.json")


def test_parent_request_is_augmented_with_catalog_dedup_context(tmp_path: Path) -> None:
    from skill_evolution_loop.evolution_catalog import EvolutionCatalog, EvolutionRecord

    catalog = EvolutionCatalog(tmp_path / "catalog")
    catalog.append(
        EvolutionRecord.create(
            record_type="mechanisms",
            record_id="issue-seeding",
            title="Issue anchored seeding",
            status="implemented",
            capability_tags=("localization",),
            task_tags=("swe-bench",),
            failure_mode_tags=("wrong-target",),
            source_model="deepseek-chat",
            source_runtime="api",
            payload={"round": 73},
            evidence_refs=(),
            cross_model_validations=(),
        )
    )
    request = {
        "request": {"failure_evidence": [{"failure_reason": "wrong-target"}]},
        "request_sha256": "old",
    }

    augmented = augment_feedback_request_with_catalog_context(
        request,
        catalog=catalog,
        capability_tags=("localization",),
        failure_mode_tags=("wrong-target",),
    )

    context = augmented["request"]["evolution_catalog_context"]
    assert context["implemented_mechanisms"][0]["record_id"] == "issue-seeding"
    assert augmented["request_sha256"] != "old"
    content = {
        key: value for key, value in augmented.items() if key != "evidence_sha256"
    }
    assert augmented["evidence_sha256"] == sha256_json(content)


def test_round1_feedback_task_selector_cannot_cross_cohort(tmp_path: Path) -> None:
    taskset = EvaluationTaskSet.create(
        taskset_id="round1-task-selector",
        tasks=[
            EvaluationTask.create(
                task_id=task_id,
                instance_id=f"repo__{task_id}",
                benchmark_id="swe-bench-verified",
                benchmark_base_commit="a" * 40,
                repo="repo/repo",
                source_repository=tmp_path,
                source_revision="a" * 40,
                instruction=task_id,
                allowed_targets=["src/example.py"],
                cohort=cohort,
            )
            for task_id, cohort in (
                ("task-feedback-1", "feedback"),
                ("task-feedback-2", "feedback"),
                ("task-feedback-3", "feedback"),
                ("task-holdout-1", "holdout"),
                ("task-holdout-2", "holdout"),
                ("task-holdout-3", "holdout"),
            )
        ],
    )

    assert _select_round1_task_ids(
        taskset, cohort="feedback", requested=("task-feedback-2",)
    ) == {"task-feedback-2"}
    with pytest.raises(ContractError, match="outside feedback cohort"):
        _select_round1_task_ids(
            taskset, cohort="feedback", requested=("task-holdout-1",)
        )

    assert _select_round1_task_ids(
        taskset, cohort="holdout", requested=("task-holdout-2",)
    ) == {"task-holdout-2"}
    with pytest.raises(ContractError, match="outside holdout cohort"):
        _select_round1_task_ids(
            taskset, cohort="holdout", requested=("task-feedback-1",)
        )
