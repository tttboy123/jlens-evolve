from __future__ import annotations

import pytest

from skill_evolution_loop.pattern_card_calibration import PatternCardLabel
from skill_evolution_loop.pattern_card_calibration import (
    calibrate_pattern_card_router as calibrate,
)

SKILL = """## Pattern cards
1. Symptom: positional arguments lose defaults from one defaults vector. Transformation: align defaults across both argument loops. Validation: cover each category.
2. Symptom: physical newlines become visible spaces inside an inline TeX wrapper. Transformation: add percent sentinels at both boundaries. Validation: block output stays unchanged.
3. Symptom: generated subclasses display a truncated base name. Transformation: preserve the generated leaf name. Validation: nested chains retain the leaf class name.
"""


def test_calibration_counts_hard_negative_as_true_negative() -> None:
    labels = (
        PatternCardLabel(
            task_id="positional",
            instruction="The default for a positional only argument has vanished",
            applicable_card_numbers=(1,),
        ),
        PatternCardLabel(
            task_id="latex",
            instruction="Inline TeX highlighting adds whitespace at both ends",
            applicable_card_numbers=(2,),
        ),
        PatternCardLabel(
            task_id="unrelated",
            instruction="Callable storage is omitted during deconstruction",
            applicable_card_numbers=(),
        ),
        PatternCardLabel(
            task_id="ambiguous-fp",
            instruction="A positional argument and defaults vector are unrelated here",
            applicable_card_numbers=(),
        ),
    )

    report = calibrate(
        skill_text=SKILL,
        labels=labels,
        minimum_positive_pairs=2,
        minimum_negative_pairs=10,
    )

    assert report["decision_counts"] == {"tp": 2, "fp": 0, "tn": 10, "fn": 0}
    assert report["task_counts"] == {
        "total": 4,
        "selected": 2,
        "abstained": 2,
        "correct_selection": 2,
    }
    assert report["metrics"]["false_positive_rate"] == 0.0
    low, high = report["metrics"]["false_positive_rate_wilson_95"]
    assert low == 0.0
    assert 0.0 < high <= 1.0
    assert report["sample_gate"]["calibrated"] is True
    assert report["sample_gate"]["fpr_calibrated"] is True


def test_calibration_refuses_conclusion_when_explicit_labels_are_insufficient() -> None:
    report = calibrate(
        skill_text=SKILL,
        labels=(
            PatternCardLabel(
                task_id="latex",
                instruction="Inline TeX highlighting adds whitespace",
                applicable_card_numbers=(2,),
            ),
        ),
    )

    assert report["sample_gate"]["calibrated"] is False
    assert report["sample_gate"]["fpr_calibrated"] is False
    assert report["sample_gate"]["status"] == "insufficient_explicit_labels"
    assert report["conclusion"] is None


def test_fpr_can_be_calibrated_without_overclaiming_positive_metrics() -> None:
    labels = (
        PatternCardLabel(
            task_id="latex",
            instruction="Inline TeX highlighting adds whitespace",
            applicable_card_numbers=(2,),
        ),
        *(
            PatternCardLabel(
                task_id=f"negative-{index}",
                instruction=f"Unrelated callable storage behavior {index}",
                applicable_card_numbers=(),
            )
            for index in range(7)
        ),
    )
    report = calibrate(
        skill_text=SKILL,
        labels=labels,
        minimum_positive_pairs=20,
        minimum_negative_pairs=20,
    )

    assert report["sample_gate"]["fpr_calibrated"] is True
    assert report["sample_gate"]["overall_calibrated"] is False
    assert report["sample_gate"]["status"] == (
        "fpr_calibrated_positive_labels_insufficient"
    )
    assert report["conclusion"]["observed_false_positive_rate"] == 0.0
    assert report["conclusion"]["precision"] is None


def test_calibration_rejects_duplicate_tasks_and_invalid_card_labels() -> None:
    duplicate = PatternCardLabel(
        task_id="same",
        instruction="Inline TeX highlighting adds whitespace",
        applicable_card_numbers=(2,),
    )
    with pytest.raises(ValueError, match="duplicate task_id"):
        calibrate(skill_text=SKILL, labels=(duplicate, duplicate))

    with pytest.raises(ValueError, match="outside the available PatternCards"):
        calibrate(
            skill_text=SKILL,
            labels=(
                PatternCardLabel(
                    task_id="bad",
                    instruction="Bad label",
                    applicable_card_numbers=(4,),
                ),
            ),
        )


def test_calibration_requires_explicit_labels_and_multiple_cards() -> None:
    with pytest.raises(ValueError, match="at least one explicit label"):
        calibrate(skill_text=SKILL, labels=())
    with pytest.raises(ValueError, match="at least two PatternCards"):
        calibrate(
            skill_text="## Pattern cards\n1. Symptom: one. Transformation: two.",
            labels=(
                PatternCardLabel(
                    task_id="one",
                    instruction="one",
                    applicable_card_numbers=(1,),
                ),
            ),
        )
