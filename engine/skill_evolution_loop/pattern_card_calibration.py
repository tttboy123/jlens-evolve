"""Explicit-label calibration for the inactive PatternCard lexical router.

The router is observational infrastructure.  This module never infers labels
from evaluator results, task names, or PatternCard prose: callers must provide
the applicable cards for every calibration task explicitly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .mlx_student import _pattern_cards, _select_pattern_card

_CARD_NUMBER = re.compile(r"^(\d+)\.\s+Symptom:")


@dataclass(frozen=True)
class PatternCardLabel:
    """A reviewer-supplied task-to-card applicability label."""

    task_id: str
    instruction: str
    applicable_card_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not self.instruction.strip():
            raise ValueError("instruction must be non-empty")
        if any(
            type(number) is not int or number < 1
            for number in self.applicable_card_numbers
        ):
            raise ValueError("applicable card numbers must be positive integers")
        if len(set(self.applicable_card_numbers)) != len(self.applicable_card_numbers):
            raise ValueError("applicable card numbers must be unique")


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _wilson_95(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    observed = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (observed + z_squared / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            observed * (1 - observed) / trials + z_squared / (4 * trials * trials)
        )
        / denominator
    )
    return [
        round(max(0.0, center - half_width), 6),
        round(min(1.0, center + half_width), 6),
    ]


def _selected_card_number(selected: str | None) -> int | None:
    if selected is None:
        return None
    match = _CARD_NUMBER.match(selected)
    if match is None:
        raise ValueError("selected PatternCard has no stable number")
    return int(match.group(1))


def calibrate_pattern_card_router(
    *,
    skill_text: str,
    labels: tuple[PatternCardLabel, ...],
    minimum_positive_pairs: int = 20,
    minimum_negative_pairs: int = 20,
) -> dict[str, Any]:
    """Measure pairwise FPR with reviewer-supplied applicability labels.

    Every task/card combination is one binary decision.  Selecting the wrong
    card for a positive task therefore records both a false positive for the
    selected card and a false negative for the applicable card.
    """

    cards = _pattern_cards(skill_text)
    if len(cards) < 2:
        raise ValueError("calibration requires at least two PatternCards")
    if not labels:
        raise ValueError("calibration requires at least one explicit label")
    if (
        type(minimum_positive_pairs) is not int
        or type(minimum_negative_pairs) is not int
        or minimum_positive_pairs < 1
        or minimum_negative_pairs < 1
    ):
        raise ValueError("calibration sample minima must be positive integers")

    task_ids = [label.task_id for label in labels]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task_id in PatternCard labels")
    available = set(range(1, len(cards) + 1))
    if any(set(label.applicable_card_numbers) - available for label in labels):
        raise ValueError("label is outside the available PatternCards")

    tp = fp = tn = fn = 0
    selected_tasks = correct_selections = 0
    decisions: list[dict[str, Any]] = []
    for label in labels:
        selected = _selected_card_number(
            _select_pattern_card(skill_text, label.instruction)
        )
        applicable = set(label.applicable_card_numbers)
        if selected is not None:
            selected_tasks += 1
        if selected is not None and selected in applicable:
            correct_selections += 1
        for card_number in sorted(available):
            actual = card_number in applicable
            predicted = card_number == selected
            if actual and predicted:
                tp += 1
            elif not actual and predicted:
                fp += 1
            elif actual and not predicted:
                fn += 1
            else:
                tn += 1
        decisions.append(
            {
                "task_id": label.task_id,
                "applicable_card_numbers": sorted(applicable),
                "selected_card_number": selected,
            }
        )

    positive_pairs = tp + fn
    negative_pairs = fp + tn
    fpr_calibrated = negative_pairs >= minimum_negative_pairs
    overall_calibrated = positive_pairs >= minimum_positive_pairs and fpr_calibrated
    metrics = {
        "coverage": _rate(selected_tasks, len(labels)),
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "false_positive_rate": _rate(fp, negative_pairs),
        "false_positive_rate_wilson_95": _wilson_95(fp, negative_pairs),
    }
    return {
        "schema_version": 1,
        "router": "lexical-symptom-overlap-v2",
        "label_boundary": "explicit_reviewer_labels_only",
        "causal_boundary": "observational_not_causal",
        "card_count": len(cards),
        "decision_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "task_counts": {
            "total": len(labels),
            "selected": selected_tasks,
            "abstained": len(labels) - selected_tasks,
            "correct_selection": correct_selections,
        },
        "metrics": metrics,
        "sample_gate": {
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs,
            "minimum_positive_pairs": minimum_positive_pairs,
            "minimum_negative_pairs": minimum_negative_pairs,
            "fpr_calibrated": fpr_calibrated,
            "overall_calibrated": overall_calibrated,
            # Kept as the conservative all-metric gate for schema v1 consumers.
            "calibrated": overall_calibrated,
            "status": (
                "calibrated"
                if overall_calibrated
                else (
                    "fpr_calibrated_positive_labels_insufficient"
                    if fpr_calibrated
                    else "insufficient_explicit_labels"
                )
            ),
        },
        "conclusion": (
            {
                "observed_false_positive_rate": metrics["false_positive_rate"],
                "observed_false_positive_rate_wilson_95": metrics[
                    "false_positive_rate_wilson_95"
                ],
                "precision": metrics["precision"] if overall_calibrated else None,
                "recall": metrics["recall"] if overall_calibrated else None,
            }
            if fpr_calibrated
            else None
        ),
        "decisions": decisions,
    }
