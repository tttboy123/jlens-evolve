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


def test_feedback_selector_rotates_three_tasks_across_two_projects_and_replays(
    tmp_path: Path,
) -> None:
    config = AutonomousEvolutionConfig.load(_config(tmp_path))
    selector = FeedbackTaskSelector(config.swe_bench.task_pool)

    first = selector.select(round_index=0, count=3, prior_claims=())
    replay = selector.select(round_index=0, count=3, prior_claims=())
    second = selector.select(round_index=1, count=3, prior_claims=())

    assert first == replay
    assert len(first.selected_task_ids) == 3
    assert len(first.selected_projects) >= 2
    assert first.selected_task_ids != second.selected_task_ids
    assert first.selection_id != second.selection_id
    assert first.tasks
    assert all("task_fingerprint_sha256" in task for task in first.tasks)


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
