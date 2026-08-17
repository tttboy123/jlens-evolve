"""Tests for the v2 teaching PatternCard router (lexical + structural anchors)."""

from __future__ import annotations

from skill_evolution_loop.mlx_student import (
    _select_pattern_card,
    _select_pattern_cards_v2,
)


def _five_card_skill() -> str:
    return """## Pattern cards
1. Symptom: a default disappears when two positional categories share one trailing defaults vector. Transformation: count both categories, left-pad the defaults vector with empty sentinels to the total positional count, then index both category loops against that single aligned vector. Validation: cover no default, a default in each category, and unchanged annotations.
2. Symptom: mocked inherited classes are documented with a truncated base name. Transformation: store the generated leaf name in the class attributes and initialize each instance qualified name from that stored name; never replace the display identity with an empty string. Validation: a nested generated chain ends in its leaf class name.
3. Symptom: physical newlines become visible spaces inside an inline TeX wrapper. Transformation: put a percent sentinel immediately before both wrapper-boundary newlines and trim the complete newline-plus-trailer sequence before closing. Validation: inline output contains percent-newline boundaries while block output stays unchanged.
4. Symptom: a property index is rendered with empty parentheses after the property name. Transformation: remove the trailing empty parentheses from the property index text. Validation: index entry keeps the property name without parentheses.
5. Symptom: a variable typed-field entry is rendered with rolename obj when it should be plain. Transformation: remove the explicit obj rolename from the variable field. Validation: variable entry has no obj role marker.

## Commit gate
Copy an exact unique search span. Make one minimal replacement.
"""


def test_top_k_ranked_and_empty_semantics() -> None:
    skill = _five_card_skill()
    top = _select_pattern_cards_v2(
        skill, "The default value for a positional only argument has vanished"
    )
    assert top, "strong match must not abstain"
    assert top[0].startswith("1. Symptom:")
    assert len(top) <= 3

    empty = _select_pattern_cards_v2(
        skill, "A Callable FileField storage is omitted during deconstruction"
    )
    assert empty == []


def test_8595_like_all_none_instruction_abstains() -> None:
    # Regression: __all__ boundary issue must not pull any unrelated card.
    skill = _five_card_skill()
    instruction = (
        "When __all__ is None, the module documentation still lists every name "
        "as if it were exported."
    )
    assert _select_pattern_cards_v2(skill, instruction) == []
    assert _select_pattern_card(skill, instruction) is None


def test_7757_like_positional_defaults_hits_first_card() -> None:
    skill = _five_card_skill()
    instruction = "The default value for a positional only argument has vanished"
    card = _select_pattern_card(skill, instruction)
    assert card is not None
    assert card.startswith("1. Symptom:")


def test_positional_hard_negatives_abstain_from_defaults_card() -> None:
    skill = _five_card_skill()

    for instruction in (
        "A positional argument name is displayed incorrectly",
        "A positional argument should not be linked as an obj role",
        "The default label for a positional argument is rendered incorrectly",
        "A positional argument default should not be linked as an obj role",
        "Document positional parameters by default",
    ):
        assert _select_pattern_card(skill, instruction) is None


def test_9698_like_property_parens_hits_property_card() -> None:
    skill = _five_card_skill()
    instruction = (
        "Sphinx property index entries show a trailing empty pair of parentheses"
    )
    cards = _select_pattern_cards_v2(skill, instruction)
    assert cards
    assert any(card.startswith("4. Symptom:") for card in cards)
    assert _select_pattern_card(skill, instruction).startswith("4. Symptom:")


def test_8638_like_obj_role_hits_variable_card() -> None:
    skill = _five_card_skill()
    instruction = (
        "Variable descriptions render with an explicit obj role instead of plain text"
    )
    cards = _select_pattern_cards_v2(skill, instruction)
    assert cards
    assert any(card.startswith("5. Symptom:") for card in cards)
    assert _select_pattern_card(skill, instruction).startswith("5. Symptom:")


def test_ties_keep_source_order() -> None:
    skill = _five_card_skill()
    # Weak generic overlap without any card's distinctive anchor must abstain;
    # weak lexical matches are exactly what caused HTTP-header "default" to
    # false-match the Python trailing-defaults card.
    instruction = "generated class names and defaults are truncated"
    cards = _select_pattern_cards_v2(skill, instruction)
    assert cards == []
