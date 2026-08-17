"""Operator student behavior regression tests (A1 + grounded repair).

NOTE: the original 44-test file was accidentally truncated by an operator
script (``open(...,'w')`` truncates before a failed argument evaluation) and
was not recoverable from disk.  This reconstruction covers only the contracts
actually changed/verified by the A1 + grounded-repair work, with fixtures
verified against the module.  The 44 original test names are preserved at the
bottom as a coverage checklist; restore the exact original from any external
copy if available.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from skill_evolution_loop import LoopRevision, StudentTask
from skill_evolution_loop.contracts import ContractError
from skill_evolution_loop.operator_rewrite import (
    OperatorPlan,
    materialize_operator_plan,
)
from skill_evolution_loop.capabilities import StudentCapabilityProfile
from skill_evolution_loop.operator_student import (
    MlxOperatorPlanGenerator,
    OperatorPlanAdapter,
    _grounded_repair_detail,
    _ground_operator_selector,
)


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "fixture"],
        cwd=path,
        check=True,
    )


# --------------------------------------------------------------------------
# A1: auto-correcting an expression operator on a statement selector
# --------------------------------------------------------------------------

def test_a1_autocorrects_statement_selector_with_real_replacement(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    module = checkout / "src/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("def compute():\n    xy = xy_0\n    return xy\n", encoding="utf-8")
    _git_init(checkout)
    task = StudentTask.create(
        task_id="operator-fixture",
        checkout=checkout,
        instruction="Copy the offset before mutating.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )
    plan = OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "src/example.py",
            "symbol": "compute",
            "intent": {"defect": "d", "trigger": "t", "desired_boundary": "b"},
            "operations": [
                {
                    "operator": "replace_expression",
                    "selector": {"source": "xy = xy_0", "occurrence": 0},
                    "arguments": {"new_expression": "xy = xy_0.copy()"},
                }
            ],
            "diagnostic": "copy the offset",
        }
    )
    result = materialize_operator_plan(module.read_text(encoding="utf-8"), plan)
    assert result.accepted is True
    assert "xy = xy_0.copy()" in result.after
    assert result.after != module.read_text(encoding="utf-8")


def test_a1_noop_statement_selector_still_fail_closed(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    module = checkout / "src/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("def compute():\n    raise ValueError('x')\n", encoding="utf-8")
    _git_init(checkout)
    task = StudentTask.create(
        task_id="operator-fixture",
        checkout=checkout,
        instruction="Change the error message.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )
    plan = OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "src/example.py",
            "symbol": "compute",
            "intent": {"defect": "d", "trigger": "t", "desired_boundary": "b"},
            "operations": [
                {
                    "operator": "replace_expression",
                    "selector": {"source": "raise ValueError('x')", "occurrence": 0},
                    "arguments": {"new_expression": "raise ValueError('x')"},
                }
            ],
            "diagnostic": "message fix",
        }
    )
    result = materialize_operator_plan(module.read_text(encoding="utf-8"), plan)
    assert result.accepted is False
    assert result.failure_reason in {"no-op", "apply-fail"}


# --------------------------------------------------------------------------
# Grounded repair
# --------------------------------------------------------------------------

def test_grounded_repair_detail_includes_real_symbol_lines(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    src = checkout / "src/example.py"
    src.parent.mkdir(parents=True)
    src.write_text(
        "class Widget:\n"
        "    def build(self):\n"
        "        xy = xy_real\n"
        "        return xy\n",
        encoding="utf-8",
    )
    _git_init(checkout)
    task = StudentTask.create(
        task_id="grounded-repair-fixture",
        checkout=checkout,
        instruction="Fix the widget layout offset.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )
    plan = OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "src/example.py",
            "symbol": "Widget",
            "intent": {"defect": "d", "trigger": "t", "desired_boundary": "b"},
            "operations": [
                {
                    "operator": "replace_expression",
                    "selector": {"source": "xy = xy_0", "occurrence": 0},
                    "arguments": {"new_expression": "xy = xy_0.copy()"},
                }
            ],
            "diagnostic": "phantom selector",
        }
    )
    detail = _grounded_repair_detail(
        task, plan, ContractError("operator selector did not resolve; matches=0")
    )
    assert "xy_real" in detail
    assert "Grounded source" in detail


def test_grounded_repair_detail_noop_hint_path() -> None:
    detail = _grounded_repair_detail(
        None, None, ContractError("operator materialization gate failed: no-op")
    )
    assert "no-op" in detail


def test_grounded_repair_detail_handles_missing_symbol_gracefully(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    src = checkout / "src/example.py"
    src.parent.mkdir(parents=True)
    src.write_text("def compute():\n    return 0\n", encoding="utf-8")
    _git_init(checkout)
    task = StudentTask.create(
        task_id="operator-fixture",
        checkout=checkout,
        instruction="Fix a bug.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )
    plan = OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "src/example.py",
            "symbol": "MissingSymbol",
            "intent": {"defect": "d", "trigger": "t", "desired_boundary": "b"},
            "operations": [
                {
                    "operator": "replace_expression",
                    "selector": {"source": "x + 1", "occurrence": 0},
                    "arguments": {"new_expression": "x + 2"},
                }
            ],
            "diagnostic": "fix",
        }
    )
    detail = _grounded_repair_detail(
        task, plan, ContractError("operator selector did not resolve; matches=0")
    )
    assert "selector did not resolve" in detail


# --------------------------------------------------------------------------
# Original 44-test coverage checklist (restore exact bodies from a backup if
# one exists; this file reconstructs only the A1 + grounded-repair contracts).
# --------------------------------------------------------------------------
_ORIGINAL_TEST_NAMES = (
    "test_ast_selector_harvest_includes_root_ternary_literal_and_header_predicate",
    "test_boundary_oracle_corrects_wrong_new_condition",
    "test_condition_header_output_does_not_rewrite_mixed_statement_kinds",
    "test_condition_header_output_is_canonicalized_to_typed_condition_edit",
    "test_empty_boundary_query_prioritizes_condition_rewrite_candidates",
    "test_grounded_repair_detail_includes_real_symbol_lines",
    "test_grounded_repair_detail_noop_hint_path",
    "test_harvest_includes_statement_level_edit_sites",
    "test_issue_symbol_localizer_prioritizes_negated_named_boundary",
    "test_issue_symbol_localizer_prioritizes_qualified_name_term_overlap",
    "test_missing_operator_is_inferred_from_unique_argument_shape",
    "test_numbered_teaching_projection_keeps_baseline_text_unchanged",
    "test_numbered_teaching_projection_keeps_interface_mandatory_rules",
    "test_numbered_teaching_projection_preserves_read_only_shared_context",
    "test_numbered_teaching_projection_selects_task_relevant_rules",
    "test_operator_adapter_accepts_explicit_unresolved_abstention",
    "test_operator_adapter_classifies_expression_used_for_statement",
    "test_operator_adapter_fails_closed_on_selector_miss",
    "test_operator_adapter_materializes_plan_and_records_receipt",
    "test_operator_adapter_preserves_unbound_name_failure_taxonomy",
    "test_operator_adapter_rejects_mapping_get_as_empty_sequence_fix",
    "test_operator_adapter_rejects_plan_over_r010_budget",
    "test_operator_conditions_vary_only_teaching_content",
    "test_operator_generator_does_not_retry_deterministic_noop_failure",
    "test_operator_generator_keeps_user_prompt_fixed_between_arms",
    "test_operator_generator_labels_nested_anchor_with_qualified_symbol",
    "test_operator_generator_runs_one_same_diagnosis_clause_repair",
    "test_operator_generator_uses_frozen_shared_symbol_as_its_only_context",
    "test_operator_skill_compiles_only_strategy_requirements",
    "test_r019_rejects_identifier_typo_not_grounded_in_selector_or_intent",
    "test_r019_rejects_unresolved_diagnostic_with_nonempty_operations",
    "test_r020_allows_grounded_length_predicate_for_empty_boundary",
    "test_r020_rejects_truthiness_only_rewrite_for_empty_vs_absent_boundary",
    "test_r023_accepts_condition_matching_minimal_boundary_oracle",
    "test_r023_generator_repairs_semantic_oracle_mismatch_before_returning",
    "test_r023_rejects_condition_that_violates_minimal_boundary_oracle",
    "test_replace_condition_full_header_selector_is_reduced_to_predicate",
    "test_selected_pattern_card_constrains_the_operator_catalog",
    "test_selected_single_operator_replaces_a_freeform_operation_shape",
    "test_shared_selector_candidates_harvest_full_file_not_symbol_excerpt",
    "test_single_allowed_operator_is_canonicalized_without_semantic_inference",
    "test_symbol_localizer_prefers_absolute_overlap_over_short_decoy",
    "test_teaching_without_matching_card_falls_back_to_clean_baseline",
    "test_truncated_symbol_excerpt_yields_no_selector_candidates",
)


# --------------------------------------------------------------------------
# B: model capability profile wiring (operator plan gate + prompt budget)
# --------------------------------------------------------------------------


class _LongOutputGenerator:
    """Adapter-side stub: returns a long raw plan and carries a profile."""

    def __init__(self, profile: StudentCapabilityProfile, raw: str) -> None:
        self.profile = profile
        self._raw = raw

    def __call__(self, task, revision) -> str:
        return self._raw


def test_operator_adapter_plan_too_large_uses_profile_max_plan_chars(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    module = checkout / "src/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("def compute():\n    return 1\n", encoding="utf-8")
    _git_init(checkout)
    task = StudentTask.create(
        task_id="operator-fixture",
        checkout=checkout,
        instruction="Bump the return value.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="operator-fixture-skill",
        revision_id="operator-baseline",
        parent_revision_id=None,
        source_round=8,
        protocol="python-typed-operator-plan-v1",
        skill_text="No additional domain teaching is provided.",
        prompt_template="Return exactly one operator plan JSON object.",
        eval_note="fixture",
    )
    long_raw = "x" * 500
    generator = _LongOutputGenerator(
        StudentCapabilityProfile(model_id="tiny", max_plan_chars=100),
        long_raw,
    )
    adapter = OperatorPlanAdapter(generator=generator)
    attempt = adapter.run(task, revision)
    assert attempt.structural_valid is False
    assert attempt.failure_reason == "plan-too-large"
    assert "100 characters" in attempt.detail


def test_operator_generator_prompt_follows_profile_plan_chars(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    module = checkout / "src/example.py"
    module.parent.mkdir(parents=True)
    module.write_text("def compute():\n    return 1\n", encoding="utf-8")
    _git_init(checkout)
    task = StudentTask.create(
        task_id="operator-fixture",
        checkout=checkout,
        instruction="Bump the return value.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )
    revision = LoopRevision.create(
        skill_id="operator-fixture-skill",
        revision_id="operator-baseline",
        parent_revision_id=None,
        source_round=8,
        protocol="python-typed-operator-plan-v1",
        skill_text="No additional domain teaching is provided.",
        prompt_template="Return exactly one operator plan JSON object.",
        eval_note="fixture",
    )
    rendered: list[list[dict[str, str]]] = []

    class Tokenizer:
        @staticmethod
        def apply_chat_template(
            messages, *, add_generation_prompt, enable_thinking, tokenize=False
        ):
            assert tokenize is False
            rendered.append(messages)
            return "\n".join(row["content"] for row in messages)

    generator = MlxOperatorPlanGenerator(
        model_path="models/Qwen3.5-4B-mlx-4bit",
        max_tokens=512,
        loader=lambda _path: (object(), Tokenizer()),
        text_generator=lambda *_args, **_kwargs: '{"schema_version":1}',
        profile=StudentCapabilityProfile(
            model_id="custom", max_plan_chars=777
        ),
    )
    try:
        generator(task, revision)
    except Exception:
        # Prompt capture is what matters here; invalid plan output may raise.
        pass
    assert rendered, "prompt must have been rendered"
    system_content = rendered[0][0]["content"]
    assert "under 777 characters" in system_content
    assert generator.generation_config()["max_plan_chars"] == 777


# --------------------------------------------------------------------------
# Selector grounding: deterministic nearest-candidate re-pointing
# --------------------------------------------------------------------------


def _grounding_task(checkout: Path) -> StudentTask:
    module = checkout / "src/example.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "def compute(value):\n    result = value + 1\n    return result\n",
        encoding="utf-8",
    )
    _git_init(checkout)
    return StudentTask.create(
        task_id="ground-fixture",
        checkout=checkout,
        instruction="Bump the offset from one to two.",
        allowed_targets=["src/example.py"],
        cohort="feedback",
    )


def _grounding_plan(selector_source: str) -> OperatorPlan:
    return OperatorPlan.from_dict(
        {
            "schema_version": 1,
            "file": "src/example.py",
            "symbol": "compute",
            "intent": {
                "defect": "wrong offset",
                "trigger": "any value",
                "desired_boundary": "value plus two",
            },
            "operations": [
                {
                    "operator": "replace_expression",
                    "selector": {"source": selector_source, "occurrence": 0},
                    "arguments": {"new_expression": "value + 2"},
                }
            ],
            "diagnostic": "bump the offset",
        }
    )


def test_ground_operator_selector_repoints_zero_match_to_nearest_candidate(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    task = _grounding_task(checkout)
    source = (checkout / "src/example.py").read_text(encoding="utf-8")
    plan = _grounding_plan("value + 2")  # does not exist; nearest is value + 1

    grounded, reason = _ground_operator_selector(
        task, plan, {"src/example.py": source}
    )

    assert reason is None
    assert grounded is not None
    selector = grounded.operations[0].selector["source"]
    assert selector == "value + 1"
    assert source.count(selector) == 1
    result = materialize_operator_plan(source, grounded)
    assert result.accepted is True
    assert "value + 2" in result.after


def test_ground_operator_selector_rejects_below_threshold(tmp_path: Path) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    task = _grounding_task(checkout)
    source = (checkout / "src/example.py").read_text(encoding="utf-8")
    plan = _grounding_plan("unrelated_symbol_xyz")

    grounded, reason = _ground_operator_selector(
        task, plan, {"src/example.py": source}
    )

    assert grounded is None
    assert reason is not None
    assert "overlap" in reason


def test_ground_operator_selector_leaves_resolving_selector_untouched(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "repo"
    checkout.mkdir()
    task = _grounding_task(checkout)
    source = (checkout / "src/example.py").read_text(encoding="utf-8")
    plan = _grounding_plan("value + 1")  # resolves exactly once

    grounded, reason = _ground_operator_selector(
        task, plan, {"src/example.py": source}
    )

    assert grounded is None
    assert reason == "no selector required grounding"
