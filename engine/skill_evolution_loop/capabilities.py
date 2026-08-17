"""Model-agnostic capability profiles for the Student harness.

The harness-side fixes (deterministic materializer auto-correction, grounded
repair, role-label hygiene, output-length contracts) are model-agnostic
capabilities.  Different Student models fail differently, so instead of
hard-coding per-model behavior in each adapter, a ``StudentCapabilityProfile``
parameterizes the harness.  Any model plugs in by name; unknown models get a
conservative default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StudentCapabilityProfile:
    """Harness knobs that adapt to a Student model's reliability."""

    model_id: str
    # Number of replan attempts after a structural-gate failure (0 = fail fast).
    max_plan_repairs: int = 1
    max_span_repairs: int = 1
    # B1: whether the candidate role labels are shown to the model.  Weak models
    # over-index on these evidence labels and hallucinate about them, so the
    # default is to keep them internal (they still drive typed-action
    # derivation inside the harness).
    show_roles_in_prompt: bool = False
    # B2: output-length contracts (characters).
    bundle_output_chars: int = 1200
    recipe_output_chars: int = 600
    # Operator plan JSON gate (characters).
    max_plan_chars: int = 1_500
    # Grounded repair: feed real line-numbered source on selector-miss / no-op
    # failures so the replan can pick a real target.
    grounded_repair: bool = True


DEFAULT_PROFILE = StudentCapabilityProfile(model_id="default")

# Registry keyed by a case-insensitive substring of the model path/id.
_PROFILES: tuple[tuple[str, StudentCapabilityProfile], ...] = (
    (
        "qwen3.5-4b-mlx-4bit",
        StudentCapabilityProfile(
            model_id="qwen3.5-4b-mlx-4bit",
            max_plan_repairs=1,
            max_span_repairs=1,
            show_roles_in_prompt=False,
            bundle_output_chars=1200,
            recipe_output_chars=600,
            max_plan_chars=1500,
            grounded_repair=True,
        ),
    ),
    (
        "qwen2.5-coder-7b-instruct-4bit",
        StudentCapabilityProfile(
            model_id="qwen2.5-coder-7b-instruct-4bit",
            max_plan_repairs=0,
            max_span_repairs=0,
            show_roles_in_prompt=True,
            bundle_output_chars=1400,
            recipe_output_chars=800,
            max_plan_chars=1600,
            grounded_repair=True,
        ),
    ),
)


def profile_for(model_path: str) -> StudentCapabilityProfile:
    """Look up a capability profile by model path/id substring."""
    lower = str(model_path).casefold()
    for key, profile in _PROFILES:
        if key in lower:
            return profile
    return DEFAULT_PROFILE
