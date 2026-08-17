from __future__ import annotations

from pathlib import Path

from openevolve.config import load_config
from openevolve.prompt.sampler import PromptSampler

from evaluator import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_repaired_search_config_uses_behavioral_map_and_bounded_archive():
    config = load_config(ROOT / "evolve_config.yaml")

    assert config.database.feature_dimensions == ["case_pass_rate", "ast_complexity"]
    assert config.database.archive_size < config.database.population_size
    assert config.prompt.num_top_programs == 1
    assert config.prompt.num_diverse_programs == 0
    assert config.llm.max_tokens >= 512


def test_prompt_includes_concise_history_and_stays_below_4b_budget():
    config = load_config(ROOT / "evolve_config.yaml")
    source = (ROOT / "initial_program.py").read_text(encoding="utf-8")
    evaluation = evaluate(str(ROOT / "initial_program.py"))
    prompt = PromptSampler(config.prompt).build_prompt(
        current_program=source,
        parent_program=source,
        program_metrics=evaluation.metrics,
        previous_programs=[],
        top_programs=[{"id": "seed", "code": source, "metrics": evaluation.metrics}],
        inspirations=[],
        language="python",
        evolution_round=1,
        diff_based_evolution=False,
        program_artifacts=evaluation.artifacts,
        feature_dimensions=config.database.feature_dimensions,
    )

    assert "Accepted reference" in prompt["user"]
    assert "target_failure" in prompt["user"]
    assert "holdout_" not in prompt["user"]
    assert len(prompt["system"] + prompt["user"]) < 12_000


def test_model_comparison_configs_differ_only_in_proposer_identity():
    from evolve_runtime import _search_protocol_hash

    policy = {
        "id": "focused-v1",
        "temperature": 0.85,
        "num_top_programs": 1,
        "num_diverse_programs": 0,
    }

    control_hash = _search_protocol_hash(ROOT / "evolve_config.yaml", policy)
    treatment_hash = _search_protocol_hash(ROOT / "evolve_config.coder7b.yaml", policy)

    assert control_hash == treatment_hash


def test_agent_guidance_is_explicitly_observational():
    from evolve_runtime import append_agent_guidance

    result = append_agent_guidance(
        "Base system prompt.",
        {
            "strategy_id": "jlens-guided",
            "causal_boundary": "observational_not_causal",
            "prompt_guidance": "Make one narrow structural change.",
        },
    )

    assert result.startswith("Base system prompt.")
    assert "jlens-guided" in result
    assert "observational_not_causal" in result
    assert "Make one narrow structural change." in result


def test_staged_manifest_fields_accumulate_iterations_and_policy_chain():
    from evolve_runtime import build_staged_manifest_fields

    fields = build_staged_manifest_fields(
        [
            {
                "iterations_requested": 5,
                "operator_policy_id": "focused-v1",
            }
        ],
        current_iterations=5,
        current_policy_id="jlens-guided-v1",
    )

    assert fields == {
        "iterations_requested_total": 10,
        "operator_policy_schedule": ["focused-v1", "jlens-guided-v1"],
    }


def test_payout_task_uses_same_bounded_search_shape():
    from tasks.payout_cleaning.evaluator_core import CASES

    config = load_config(ROOT / "tasks/payout_cleaning/evolve_config.yaml")

    assert config.database.population_size == 12
    assert config.database.archive_size == 6
    assert config.database.feature_dimensions == [
        "case_pass_rate",
        "ast_complexity",
    ]
    assert config.database.feature_bins["case_pass_rate"] == len(CASES) + 1
    assert "holdout_" not in config.prompt.system_message
    assert Path(config.prompt.template_dir).is_dir()
