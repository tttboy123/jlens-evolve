from __future__ import annotations

from admission_policy import AdmissionPolicy


def metrics(*passed: str) -> dict[str, float]:
    names = ("a", "b", "c")
    result = {f"case_{name}": float(name in passed) for name in names}
    result["combined_score"] = len(passed) / len(names)
    result["evaluator_valid"] = 1.0
    return result


SEED = "def solve(records):\n    return []\n"


def test_rejects_parent_regression_even_when_scalar_score_ties():
    policy = AdmissionPolicy()
    assert policy.assess_and_register(SEED, metrics("a"), None).accepted

    decision = policy.assess_and_register(
        "def solve(records):\n    return [('new', 1.0)]\n",
        metrics("b"),
        metrics("a"),
    )

    assert not decision.accepted
    assert decision.regressed_cases == ("case_a",)
    assert "parent_regression" in decision.reasons


def test_rejects_exact_and_ast_equivalent_duplicates():
    policy = AdmissionPolicy()
    assert policy.assess_and_register(SEED, metrics("a"), None).accepted

    exact = policy.assess_and_register(SEED, metrics("a"), metrics("a"))
    ast_equivalent = policy.assess_and_register(
        "# formatting only\ndef solve(records):\n    return []\n",
        metrics("a"),
        metrics("a"),
    )

    assert not exact.accepted and "exact_duplicate" in exact.reasons
    assert not ast_equivalent.accepted and "ast_duplicate" in ast_equivalent.reasons


def test_caps_behavior_equivalent_variants_but_accepts_non_regressive_gain():
    policy = AdmissionPolicy(behavior_equivalent_limit=2)
    assert policy.assess_and_register(SEED, metrics("a"), None).accepted
    structurally_distinct = "def solve(records):\n    out = []\n    return out\n"
    assert policy.assess_and_register(
        structurally_distinct, metrics("a"), metrics("a")
    ).accepted

    third_same_behavior = policy.assess_and_register(
        "def solve(records):\n    if records:\n        return []\n    return []\n",
        metrics("a"),
        metrics("a"),
    )
    gained = policy.assess_and_register(
        "def solve(records):\n    return [('ok', 1.0)] if records else []\n",
        metrics("a", "b"),
        metrics("a"),
    )

    assert not third_same_behavior.accepted
    assert "behavior_duplicate_limit" in third_same_behavior.reasons
    assert gained.accepted


def test_guard_keeps_rejected_program_traceable_but_not_selectable(tmp_path):
    from openevolve.config import DatabaseConfig
    from openevolve.database import Program, ProgramDatabase

    from admission_policy import OpenEvolveAdmissionGuard
    from experience_store import ExperienceStore

    config = DatabaseConfig(
        population_size=8,
        archive_size=4,
        num_islands=1,
        feature_dimensions=["case_pass_rate", "ast_complexity"],
        feature_bins=4,
    )
    database = ProgramDatabase(config)
    guard = OpenEvolveAdmissionGuard(
        run_context={
            "run_id": "guard-test",
            "task_id": "task-a",
            "task_family": "tests",
        },
        event_store=ExperienceStore(tmp_path),
    )
    guard.install(database)

    seed_metrics = {
        **metrics("a"),
        "case_pass_rate": 1 / 3,
        "ast_complexity": 5.0,
    }
    seed = Program(id="seed", code=SEED, metrics=seed_metrics)
    database.add(seed, iteration=0, target_island=0)
    regressed = Program(
        id="bad",
        parent_id="seed",
        code="def solve(records):\n    return [('bad', 1.0)]\n",
        metrics={
            **metrics("b"),
            "case_pass_rate": 1 / 3,
            "ast_complexity": 8.0,
        },
    )
    database.add(regressed, iteration=1, target_island=0)

    assert "bad" in database.programs
    assert "bad" not in database.islands[0]
    assert "bad" not in database.archive
    assert database.programs["bad"].metadata["admission"]["accepted"] is False
    assert len(ExperienceStore(tmp_path).read_events()) == 2
