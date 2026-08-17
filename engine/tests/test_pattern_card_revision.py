from __future__ import annotations

import pytest

from skill_evolution_loop.contracts import ContractError, LoopRevision
from skill_evolution_loop.p1_operator import load_frozen_operator_skill_revision
from skill_evolution_loop.pattern_card_revision import refine_pattern_card_symptoms


def _parent() -> LoopRevision:
    return LoopRevision.create(
        skill_id="feedback-patterns",
        revision_id="feedback-patterns-r006",
        parent_revision_id="feedback-patterns-r005",
        source_round=6,
        protocol="structured-search-replace-v1",
        skill_text=(
            "---\nactive: false\nauto_install: false\n---\n\n"
            "## Pattern cards\n"
            "1. Symptom: first old symptom. Transformation: first transform. "
            "Validation: first validation.\n"
            "2. Symptom: second old symptom. Transformation: second transform. "
            "Validation: second validation.\n\n"
            "## Commit gate\nPreserve unrelated behavior."
        ),
        prompt_template="Return one JSON object.",
        eval_note="inactive fixture",
    )


def test_refinement_changes_only_selected_symptom_and_preserves_inactive_boundary() -> (
    None
):
    wrapper = refine_pattern_card_symptoms(
        parent=_parent(),
        replacements={2: "a mocked inherited class displays a truncated base name"},
        source_evidence_sha256="a" * 64,
        revision_id="feedback-patterns-r007-trigger-anchors",
    )

    revision = LoopRevision.from_dict(wrapper["next_revision"])
    assert "1. Symptom: first old symptom." in revision.skill_text
    assert (
        "2. Symptom: a mocked inherited class displays a truncated base name. "
        "Transformation: second transform. Validation: second validation."
        in revision.skill_text
    )
    assert revision.parent_revision_id == "feedback-patterns-r006"
    assert revision.source_round == 7
    assert wrapper["candidate_status"] == "inactive"
    assert wrapper["auto_activate"] is False
    assert wrapper["network_calls_performed"] is False
    assert wrapper["holdout_task_ids_included"] is False


def test_operator_pattern_card_refinement_is_loadable_as_inactive_skill(
    tmp_path,
) -> None:
    parent = LoopRevision.create(
        skill_id="p1-local-qwen-operator-skill",
        revision_id="operator-r006",
        parent_revision_id="operator-r005",
        source_round=6,
        protocol="python-typed-operator-plan-v1",
        skill_text=(
            "---\nactive: false\nauto_install: false\n---\n\n"
            "## Pattern cards\n"
            "1. Symptom: first old symptom. Transformation: first transform. "
            "Validation: first validation."
        ),
        prompt_template="Return exactly one typed operator plan JSON object.",
        eval_note="inactive fixture",
    )
    wrapper = refine_pattern_card_symptoms(
        parent=parent,
        replacements={1: "a mocked inherited class displays a truncated base name"},
        source_evidence_sha256="a" * 64,
        revision_id="operator-r007-trigger-anchors",
    )
    path = tmp_path / "OPERATOR-SKILL.json"
    from skill_evolution_loop.contracts import canonical_json

    path.write_text(canonical_json(wrapper) + "\n", encoding="utf-8")

    revision = load_frozen_operator_skill_revision(path)

    assert revision.revision_id == "operator-r007-trigger-anchors"
    assert "active: false" in revision.skill_text


@pytest.mark.parametrize(
    ("replacements", "message"),
    [
        ({3: "missing"}, "outside the available PatternCards"),
        ({2: "p1-sphinx-9658"}, "task IDs or file paths"),
        ({2: "change src/example.py"}, "task IDs or file paths"),
        ({2: ""}, "non-empty"),
    ],
)
def test_refinement_rejects_unbounded_or_task_specific_changes(
    replacements: dict[int, str], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        refine_pattern_card_symptoms(
            parent=_parent(),
            replacements=replacements,
            source_evidence_sha256="a" * 64,
            revision_id="feedback-patterns-r007-trigger-anchors",
        )
