from __future__ import annotations

from pathlib import Path

from experience_store import ExperienceStore


def test_event_log_is_idempotent_and_lessons_transfer_within_family(tmp_path: Path):
    store = ExperienceStore(tmp_path)
    event = {
        "event_id": "event-1",
        "run_id": "run-a",
        "task_id": "task-a",
        "task_family": "record-cleaning",
        "accepted": True,
        "gained_cases": ["case_normalize"],
        "holdout_verified": True,
        "lesson": "Normalize categorical fields before filtering.",
        "tags": ["normalization", "filtering"],
    }

    assert store.append_event(event)
    assert not store.append_event(event)
    promoted = store.distill_lessons(min_evidence=1)

    assert len(store.read_events()) == 1
    assert promoted == 1
    lessons = store.retrieve_lessons(
        task_family="record-cleaning",
        tags={"normalization"},
        exclude_task_id="task-a",
    )
    assert lessons == []

    second = dict(event, event_id="event-2", run_id="run-b", task_id="task-b")
    assert store.append_event(second)
    assert store.distill_lessons(min_evidence=2) == 1
    lessons = store.retrieve_lessons(
        task_family="record-cleaning",
        tags={"normalization"},
        exclude_task_id="task-c",
    )
    assert lessons[0]["evidence_count"] == 2
    skill_path = store.render_skill_candidate("record-cleaning")
    assert skill_path.read_text(encoding="utf-8").startswith("---\n")


def test_latest_checkpoint_uses_numeric_iteration(tmp_path: Path):
    from evolve_runtime import latest_checkpoint

    checkpoints = tmp_path / "checkpoints"
    (checkpoints / "checkpoint_9").mkdir(parents=True)
    (checkpoints / "checkpoint_100").mkdir()
    (checkpoints / "checkpoint_bad").mkdir()

    assert latest_checkpoint(tmp_path) == checkpoints / "checkpoint_100"


def test_meta_policy_explores_untried_then_uses_best_yield():
    from evolve_runtime import select_meta_policy

    candidates = [{"id": "focused"}, {"id": "exploratory"}]
    assert select_meta_policy(candidates, []) == candidates[0]
    assert (
        select_meta_policy(
            candidates, [{"policy_id": "focused", "improvement_yield": 0.4}]
        )
        == candidates[1]
    )
    assert (
        select_meta_policy(
            candidates,
            [
                {"policy_id": "focused", "improvement_yield": 0.4},
                {"policy_id": "exploratory", "improvement_yield": 0.1},
            ],
        )
        == candidates[0]
    )


def test_resume_compatibility_rejects_changed_evaluator_contract():
    from evolve_runtime import resume_is_compatible

    previous = {"config_hash": "c1", "evaluator_hash": "e1", "initial_hash": "i1"}
    assert resume_is_compatible(previous, dict(previous))
    assert not resume_is_compatible(previous, {**previous, "evaluator_hash": "e2"})


def test_persisted_report_does_not_count_holdout_evidence_as_candidate(tmp_path: Path):
    import json

    from evolve_runtime import build_persisted_report

    store = ExperienceStore(tmp_path)
    store.append_event(
        {
            "event_id": "candidate",
            "event_type": "candidate",
            "run_id": "run-a",
            "parent_id": "seed",
            "iteration": 1,
            "accepted": True,
            "source_hash": "source",
            "ast_hash": "ast",
            "behavior_signature": "behavior",
            "parent_score": 0.1,
            "child_score": 0.2,
        }
    )
    store.append_event(
        {
            "event_id": "candidate-with-missing-parent-link",
            "event_type": "candidate",
            "run_id": "run-a",
            "parent_id": None,
            "iteration": 2,
            "accepted": True,
            "source_hash": "source-2",
            "ast_hash": "ast-2",
            "behavior_signature": "behavior-2",
            "parent_score": 0.2,
            "child_score": 0.3,
        }
    )
    store.append_event(
        {
            "event_id": "candidate:holdout",
            "event_type": "holdout_verification",
            "run_id": "run-a",
            "parent_id": "seed",
            "iteration": 1,
            "accepted": True,
            "holdout_verified": True,
            "source_hash": "source",
            "ast_hash": "ast",
            "behavior_signature": "behavior",
        }
    )
    (tmp_path / "meta_policy_trials.jsonl").write_text("", encoding="utf-8")
    manifest = {
        "manifest_id": "manifest",
        "run_id": "run-a",
        "task_id": "task-a",
        "initial_holdout_score": 0.0,
        "final_holdout_score": 0.2,
        "retrieved_lesson_sources": [],
    }
    (tmp_path / "run_manifests.jsonl").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    report = build_persisted_report(
        state_dir=tmp_path, run_id="run-a", manifest=manifest
    )

    assert report["admission"]["candidate_attempts"] == 2
    assert report["admission"]["accepted"] == 2
    assert report["admission"]["holdout_verified_promotions"] == 1


def test_forced_meta_policy_is_not_reported_as_self_revision():
    from evolve_runtime import operator_revision_label

    previous = {"policy_id": "exploratory-v2", "improvement_yield": 0.0}

    assert (
        operator_revision_label(previous, selected_policy_id="focused-v1", forced=True)
        is None
    )
    assert (
        operator_revision_label(previous, selected_policy_id="focused-v1", forced=False)
        == "exploratory-v2->focused-v1"
    )


def test_experience_modes_make_psi_ab_the_only_variable(tmp_path: Path):
    from evolve_runtime import retrieve_experience

    store = ExperienceStore(tmp_path)
    event = {
        "event_id": "source-1",
        "run_id": "source-run",
        "task_id": "transaction-record-cleaning-v1",
        "task_family": "record-cleaning",
        "accepted": True,
        "holdout_verified": True,
        "lesson": "Normalize categorical fields before filtering.",
        "tags": ["normalization", "filtering"],
    }
    assert store.append_event(event)
    assert store.distill_lessons(min_evidence=1) == 1
    extension = {
        "task_id": "payout-record-cleaning-v1",
        "task_family": "record-cleaning",
        "tags": ["normalization", "filtering"],
        "experience_limit": 3,
    }

    assert retrieve_experience(store, extension, mode="off") == []
    transferred = retrieve_experience(store, extension, mode="cross-task")
    assert transferred[0]["source_task_ids"] == ["transaction-record-cleaning-v1"]


def test_task_core_path_is_part_of_resume_contract(tmp_path: Path):
    from evolve_runtime import build_run_contract

    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    core_a = tmp_path / "core_a.py"
    core_b = tmp_path / "core_b.py"
    config = tmp_path / "config.yaml"
    for path, content in (
        (initial, "def solve(rows): return []\n"),
        (evaluator, "def evaluate(path): return None\n"),
        (core_a, "CASES = ()\n"),
        (core_b, "CASES = ({'id': 'different'},)\n"),
        (config, "random_seed: 1\n"),
    ):
        path.write_text(content, encoding="utf-8")

    first = build_run_contract(
        task_id="task",
        initial_path=initial,
        evaluator_path=evaluator,
        evaluator_core_path=core_a,
        config_path=config,
    )
    second = build_run_contract(
        task_id="task",
        initial_path=initial,
        evaluator_path=evaluator,
        evaluator_core_path=core_b,
        config_path=config,
    )

    assert first["evaluator_hash"] != second["evaluator_hash"]


def test_experiment_seed_is_part_of_resume_contract(tmp_path: Path):
    from evolve_runtime import build_run_contract, resume_is_compatible

    initial = tmp_path / "initial.py"
    evaluator = tmp_path / "evaluator.py"
    core = tmp_path / "core.py"
    config = tmp_path / "config.yaml"
    for path in (initial, evaluator, core, config):
        path.write_text("# fixture\n", encoding="utf-8")

    first = build_run_contract(
        task_id="task",
        initial_path=initial,
        evaluator_path=evaluator,
        evaluator_core_path=core,
        config_path=config,
        experiment_seed=11,
    )
    second = build_run_contract(
        task_id="task",
        initial_path=initial,
        evaluator_path=evaluator,
        evaluator_core_path=core,
        config_path=config,
        experiment_seed=12,
    )

    assert first["experiment_seed"] == 11
    assert resume_is_compatible(first, second) is False


def test_psi_cli_contract_rejects_arm_or_mode_mismatches():
    import pytest

    from evolve_runtime import validate_experiment_args

    validate_experiment_args(
        experience_mode="off", psi_experiment_id="ab", psi_arm="control"
    )
    validate_experiment_args(
        experience_mode="cross-task", psi_experiment_id="ab", psi_arm="transfer"
    )
    with pytest.raises(ValueError, match="requires --psi-arm"):
        validate_experiment_args(
            experience_mode="off", psi_experiment_id="ab", psi_arm=None
        )
    with pytest.raises(ValueError, match="control arm requires"):
        validate_experiment_args(
            experience_mode="auto", psi_experiment_id="ab", psi_arm="control"
        )
