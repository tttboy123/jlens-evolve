from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolve.autonomous import (
    AutonomousEvolutionConfig,
    AutonomousEvolutionError,
    FeedbackTaskSelector,
    GoalRunStatus,
    GoalStateStore,
    TaskSelectionContext,
)
from evolve.autonomous.state import HashChainIndex


def _task_pool(tmp_path: Path) -> Path:
    rows = []
    for index, project in enumerate(("sphinx", "django", "sympy", "pytest"), 1):
        source = tmp_path / f"source-{index}"
        source.mkdir()
        rows.append(
            {
                "instance_id": f"{project}__task-{index}",
                "project": project,
                "benchmark_id": "swe-bench-verified",
                "cohort": "feedback",
                "source_uri": str(source),
                "base_revision": f"{index:040x}",
                "catalog_fingerprint": f"{index:064x}",
                "estimated_cost": index,
            }
        )
    path = tmp_path / "feedback-tasks.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        (model / name).write_text(name, encoding="utf-8")
    harness = tmp_path / "harness"
    harness.mkdir()
    evaluator = tmp_path / "official-evaluator.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    source_pool = tmp_path / "sources"
    source_pool.mkdir()
    payload = {
        "schema_version": 1,
        "goal": {
            "goal_id": "feedback-skill-evolution",
            "description": "Improve localization without changing model weights.",
            "target_native_gains": 3,
            "max_rounds": 8,
            "no_progress_patience": 3,
        },
        "model": {
            "provider": "local-mlx",
            "model_path": str(model),
            "model_identity_files": [
                "config.json",
                "model.safetensors.index.json",
                "tokenizer_config.json",
            ],
        },
        "swe_bench": {
            "task_pool": str(_task_pool(tmp_path)),
            "source_pool": str(source_pool),
            "official_harness": str(harness),
            "official_evaluator": str(evaluator),
            "cohort": "feedback",
        },
        "teacher": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "endpoint": "https://teacher.invalid/chat/completions",
            "api_key_env": "DEEPSEEK_API_KEY",
            "budget_cny": 10.0,
            "max_output_tokens": 1024,
        },
        "execution": {
            "tasks_per_campaign": 3,
            "qwen_prescreen_count": 1,
            "native_finalist_count": 1,
            "seed": 7,
        },
    }
    path = tmp_path / "AUTONOMOUS-EVOLUTION-CONFIG.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_product_config_requires_only_model_swebench_teacher_goal_and_execution(
    tmp_path: Path,
) -> None:
    config = AutonomousEvolutionConfig.load(_config(tmp_path))

    assert config.goal.goal_id == "feedback-skill-evolution"
    assert config.model.provider == "local-mlx"
    assert config.swe_bench.cohort == "feedback"
    assert config.teacher.budget_cny == 10.0
    assert config.execution.tasks_per_campaign == 3
    assert not hasattr(config, "fresh_campaign_template")
    assert not hasattr(config, "final_commit_sha")


@pytest.mark.parametrize(
    ("field", "value"),
    (("qwen_prescreen_count", 2), ("native_finalist_count", 2)),
)
def test_current_single_candidate_search_rejects_misleading_execution_counts(
    tmp_path: Path, field: str, value: int
) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AutonomousEvolutionError, match="single-candidate"):
        AutonomousEvolutionConfig.load(path)


def test_goal_state_uses_only_product_stop_states_and_resumes_active(
    tmp_path: Path,
) -> None:
    store = GoalStateStore(tmp_path / "EVOLUTION-STATE.json")
    created = store.load_or_create(goal_id="feedback-skill-evolution")

    assert created.status is GoalRunStatus.ACTIVE
    assert store.load_or_create(goal_id="feedback-skill-evolution") == created
    forbidden = {
        "waiting_codex_review",
        "finalized_pending_codex_review",
        "waiting_user_message",
    }
    assert forbidden.isdisjoint(status.value for status in GoalRunStatus)
    assert {
        "max_consecutive_infra_failures",
        "max_same_failure_signature",
        "disk_limit",
    }.issubset(status.value for status in GoalRunStatus)


def test_goal_config_accepts_explicit_autonomous_stop_limits(tmp_path: Path) -> None:
    path = _config(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["goal"].update(
        {
            "max_consecutive_infra_failures": 4,
            "max_same_failure_signature": 6,
            "disk_limit_bytes": 123456,
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    goal = AutonomousEvolutionConfig.load(path).goal

    assert goal.max_consecutive_infra_failures == 4
    assert goal.max_same_failure_signature == 6
    assert goal.disk_limit_bytes == 123456


def test_feedback_selector_rotates_three_tasks_across_two_projects_and_replays(
    tmp_path: Path,
) -> None:
    config = AutonomousEvolutionConfig.load(_config(tmp_path))
    selector = FeedbackTaskSelector(config.swe_bench.task_pool)

    first = selector.select(round_index=0, count=3, prior_claims=())
    replay = selector.select(round_index=0, count=3, prior_claims=())
    second = selector.select(round_index=1, count=3, prior_claims=())

    assert first == replay
    assert first.selection_context["mode"] == "compatibility-prior-claims"
    assert len(first.selection_context_sha256) == 64
    assert first.selection_context_sha256 == replay.selection_context_sha256
    assert first.selection_context_sha256 == second.selection_context_sha256
    assert len(first.selected_task_ids) == 3
    assert len(first.selected_projects) >= 2
    assert first.selected_task_ids != second.selected_task_ids
    assert first.selection_id != second.selection_id
    assert first.tasks
    assert all("task_fingerprint_sha256" in task for task in first.tasks)


def test_feedback_selector_reserves_a_slot_for_a_second_project(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "instance_id": f"{project}__task-{index}",
            "project": project,
            "cohort": "feedback",
            "estimated_cost": cost,
        }
        for index, (project, cost) in enumerate(
            (("alpha", 1), ("alpha", 2), ("alpha", 3), ("beta", 100)), 1
        )
    ]
    path = tmp_path / "clustered-feedback-tasks.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    selection = FeedbackTaskSelector(path).select(
        round_index=0, count=3, prior_claims=()
    )

    assert selection.selected_task_ids == (
        "alpha__task-1",
        "beta__task-4",
        "alpha__task-2",
    )
    assert selection.selected_projects == ("alpha", "beta")


def test_feedback_selector_hash_binds_complete_selection_context(
    tmp_path: Path,
) -> None:
    selector = FeedbackTaskSelector(_task_pool(tmp_path))
    context = TaskSelectionContext(
        historical_claims=(
            {
                "claim_id": "claim-old-gain",
                "task_id": "sphinx__task-1",
                "classification": "gain",
            },
            {
                "claim_id": "claim-current-regression",
                "task_id": "django__task-2",
                "classification": "regression",
            },
        ),
        current_best_revision_id="candidate-best-r0001",
        current_best_supported_task_ids=("sphinx__task-1",),
        failure_signature_counts={"django__task-2": 2, "pytest__task-4": 1},
        goal_gap=2,
        task_selection_counts={
            "sphinx__task-1": 1,
            "django__task-2": 1,
            "sympy__task-3": 2,
            "pytest__task-4": 0,
        },
        repeat_hard_cap=3,
    )

    selected = selector.select(round_index=2, count=3, context=context)
    replayed = selector.select(round_index=2, count=3, context=context)
    different_parent = selector.select(
        round_index=2,
        count=3,
        context=TaskSelectionContext(
            historical_claims=context.historical_claims,
            current_best_revision_id="candidate-best-r0002",
            current_best_supported_task_ids=context.current_best_supported_task_ids,
            failure_signature_counts=context.failure_signature_counts,
            goal_gap=context.goal_gap,
            task_selection_counts=context.task_selection_counts,
            repeat_hard_cap=context.repeat_hard_cap,
        ),
    )

    assert selected == replayed
    assert selected.selected_task_ids == (
        "django__task-2",
        "pytest__task-4",
        "sympy__task-3",
    )
    assert selected.selected_task_ids == different_parent.selected_task_ids
    assert selected.selection_id != different_parent.selection_id
    assert selected.selection_context_sha256 != (
        different_parent.selection_context_sha256
    )
    assert selected.selection_context["mode"] == "stateful-v1"
    assert selected.selection_context["goal_gap"] == 2
    assert selected.selection_context_payload()["historical_claims"] == [
        {
            "claim_id": "claim-old-gain",
            "classification": "gain",
            "task_id": "sphinx__task-1",
        },
        {
            "claim_id": "claim-current-regression",
            "classification": "regression",
            "task_id": "django__task-2",
        },
    ]
    assert "goal-gap=2" in selected.selection_reason
    assert "repeat-hard-cap=3" in selected.selection_reason

    with pytest.raises(TypeError):
        selected.selection_context["goal_gap"] = 99  # type: ignore[index]

    context_variants = (
        {
            **context.identity_payload(),
            "historical_claims": (
                *context.historical_claims,
                {
                    "claim_id": "claim-extra-neutral",
                    "task_id": "sympy__task-3",
                    "classification": "neutral",
                },
            ),
        },
        {**context.identity_payload(), "current_best_revision_id": "best-r-other"},
        {
            **context.identity_payload(),
            "current_best_supported_task_ids": ("sphinx__task-1", "sympy__task-3"),
        },
        {
            **context.identity_payload(),
            "failure_signature_counts": {"django__task-2": 3},
        },
        {**context.identity_payload(), "goal_gap": 1},
        {
            **context.identity_payload(),
            "task_selection_counts": {"sphinx__task-1": 0},
        },
        {**context.identity_payload(), "repeat_hard_cap": 4},
    )
    for variant in context_variants:
        changed = selector.select(
            round_index=2,
            count=3,
            context=TaskSelectionContext(**variant),
        )
        assert changed.selection_context_sha256 != selected.selection_context_sha256
        assert changed.selection_id != selected.selection_id


@pytest.mark.parametrize("estimated_cost", (-1, float("nan"), float("inf"), "1", True))
def test_feedback_selector_rejects_untrusted_costs(
    tmp_path: Path, estimated_cost: object
) -> None:
    path = tmp_path / "invalid-cost-tasks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "instance_id": "alpha__task-1",
                    "project": "alpha",
                    "cohort": "feedback",
                    "estimated_cost": estimated_cost,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AutonomousEvolutionError, match="finite and non-negative"):
        FeedbackTaskSelector(path)


def test_feedback_selector_hash_binds_validated_task_costs(tmp_path: Path) -> None:
    rows = [
        {
            "instance_id": "alpha__task-1",
            "project": "alpha",
            "cohort": "feedback",
            "estimated_cost": 1,
        },
        {
            "instance_id": "beta__task-2",
            "project": "beta",
            "cohort": "feedback",
            "estimated_cost": 2,
        },
    ]
    first_path = tmp_path / "cost-v1.json"
    second_path = tmp_path / "cost-v2.json"
    first_path.write_text(json.dumps(rows), encoding="utf-8")
    rows[1]["estimated_cost"] = 3
    second_path.write_text(json.dumps(rows), encoding="utf-8")
    context = TaskSelectionContext(
        historical_claims=(),
        current_best_revision_id=None,
        current_best_supported_task_ids=(),
        failure_signature_counts={},
        goal_gap=2,
        task_selection_counts={},
        repeat_hard_cap=2,
    )

    first = FeedbackTaskSelector(first_path).select(
        round_index=0, count=2, context=context
    )
    second = FeedbackTaskSelector(second_path).select(
        round_index=0, count=2, context=context
    )

    assert first.selected_task_ids == second.selected_task_ids
    assert first.selection_context_sha256 == second.selection_context_sha256
    assert first.selection_id != second.selection_id


def test_feedback_selector_hard_excludes_tasks_at_repeat_cap(tmp_path: Path) -> None:
    selector = FeedbackTaskSelector(_task_pool(tmp_path))
    context = TaskSelectionContext(
        historical_claims=(
            {
                "claim_id": "claim-regression",
                "task_id": "sphinx__task-1",
                "classification": "regression",
            },
        ),
        current_best_revision_id="candidate-best-r0001",
        current_best_supported_task_ids=(),
        failure_signature_counts={"sphinx__task-1": 10},
        goal_gap=3,
        task_selection_counts={"sphinx__task-1": 3},
        repeat_hard_cap=3,
    )

    selection = selector.select(round_index=3, count=3, context=context)

    assert selection.selected_task_ids == (
        "django__task-2",
        "sympy__task-3",
        "pytest__task-4",
    )
    assert "sphinx__task-1" in selection.excluded


def test_feedback_selector_fails_closed_when_repeat_cap_exhausts_pool(
    tmp_path: Path,
) -> None:
    selector = FeedbackTaskSelector(_task_pool(tmp_path))
    context = TaskSelectionContext(
        historical_claims=(),
        current_best_revision_id=None,
        current_best_supported_task_ids=(),
        failure_signature_counts={},
        goal_gap=3,
        task_selection_counts={
            "sphinx__task-1": 1,
            "django__task-2": 1,
        },
        repeat_hard_cap=1,
    )

    with pytest.raises(AutonomousEvolutionError, match="repeat hard cap"):
        selector.select(round_index=1, count=3, context=context)


def test_feedback_selector_rejects_source_outside_configured_pool(
    tmp_path: Path,
) -> None:
    config = AutonomousEvolutionConfig.load(_config(tmp_path))
    with pytest.raises(AutonomousEvolutionError, match="source pool"):
        FeedbackTaskSelector(
            config.swe_bench.task_pool,
            source_pool=config.swe_bench.source_pool,
        )


def test_feedback_selector_rejects_burned_or_holdout_rows(tmp_path: Path) -> None:
    path = _task_pool(tmp_path)
    rows = json.loads(path.read_text())
    rows[0]["instance_id"] = "r076-burned"
    path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(AutonomousEvolutionError, match="forbidden"):
        FeedbackTaskSelector(path)


def test_autonomous_indexes_replay_idempotently_and_detect_chain_tamper(
    tmp_path: Path,
) -> None:
    index = HashChainIndex(tmp_path / "ROUND-INDEX.jsonl", index_id="rounds-v1")

    assert index.append(event_id="round-0000", payload={"status": "completed"})
    assert not index.append(
        event_id="round-0000", payload={"status": "completed"}
    )
    assert len(HashChainIndex(index.path, index_id="rounds-v1").rows()) == 1

    row = json.loads(index.path.read_text())
    row["payload"]["status"] = "gain"
    index.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(AutonomousEvolutionError, match="chain mismatch"):
        index.rows()
