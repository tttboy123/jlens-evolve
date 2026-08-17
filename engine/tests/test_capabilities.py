"""Model-agnostic capability profile registry tests."""

from __future__ import annotations

from skill_evolution_loop.capabilities import (
    DEFAULT_PROFILE,
    StudentCapabilityProfile,
    profile_for,
)


def test_profile_for_unknown_model_returns_conservative_default() -> None:
    profile = profile_for("models/SomeBrandNew-13B-q4")
    assert profile is DEFAULT_PROFILE
    assert profile.model_id == "default"
    assert profile.max_plan_repairs == 1
    assert profile.max_span_repairs == 1
    assert profile.show_roles_in_prompt is False
    assert profile.bundle_output_chars == 1200
    assert profile.recipe_output_chars == 600
    assert profile.max_plan_chars == 1500
    assert profile.grounded_repair is True


def test_profile_for_matches_weak_4b_model_case_insensitively() -> None:
    profile = profile_for("/Users/lune/models/Qwen3.5-4B-mlx-4bit")
    assert profile.model_id == "qwen3.5-4b-mlx-4bit"
    assert profile.show_roles_in_prompt is False
    assert profile.max_plan_repairs == 1
    assert profile.max_span_repairs == 1
    assert profile.bundle_output_chars == 1200
    assert profile.recipe_output_chars == 600
    assert profile.max_plan_chars == 1500
    assert profile.grounded_repair is True


def test_profile_for_matches_7b_coder_instruct() -> None:
    profile = profile_for("models/Qwen2.5-Coder-7B-Instruct-4bit")
    assert profile.model_id == "qwen2.5-coder-7b-instruct-4bit"
    # A stronger model can see role labels and gets a slightly larger budget.
    assert profile.show_roles_in_prompt is True
    assert profile.max_plan_repairs == 0
    assert profile.max_span_repairs == 0
    assert profile.bundle_output_chars == 1400
    assert profile.recipe_output_chars == 800
    assert profile.max_plan_chars == 1600


def test_default_profile_is_frozen_and_slots() -> None:
    try:
        DEFAULT_PROFILE.model_id = "mutated"  # type: ignore[misc]
    except AttributeError:
        pass
    else:  # pragma: no cover - only reached when frozenness breaks
        raise AssertionError("profile must be frozen")
    assert DEFAULT_PROFILE.model_id == "default"


def test_custom_profile_overrides_registry() -> None:
    custom = StudentCapabilityProfile(
        model_id="custom",
        show_roles_in_prompt=True,
        bundle_output_chars=2000,
        recipe_output_chars=900,
        max_plan_chars=2500,
        max_plan_repairs=3,
        max_span_repairs=2,
    )
    assert custom.show_roles_in_prompt is True
    assert custom.bundle_output_chars == 2000
    assert custom.max_plan_chars == 2500
    assert custom.max_span_repairs == 2
