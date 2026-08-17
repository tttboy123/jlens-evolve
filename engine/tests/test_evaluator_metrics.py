from __future__ import annotations

from pathlib import Path

import pytest

from evaluator import evaluate
from evaluator_core import CASES


def test_openevolve_metrics_are_case_granular_and_case_count_dominates(tmp_path: Path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "def solve(records):\n"
        "    return [(row['user'].strip().lower(), round(row['amount'], 2)) "
        "for row in records if isinstance(row, dict) and row.get('status') == 'paid']\n",
        encoding="utf-8",
    )

    result = evaluate(str(candidate))

    assert all(f"case_{case['id']}" in result.metrics for case in CASES)
    assert result.metrics["passed_cases"] == pytest.approx(3.0)
    assert result.metrics["combined_score"] == pytest.approx(
        (result.metrics["passed_cases"] + result.metrics["weighted_score"])
        / (len(CASES) + 1)
    )
    assert result.metrics["ast_complexity"] > 0
    assert "behavior_signature" in result.artifacts


def test_evaluator_does_not_leak_hidden_holdout_ids(tmp_path: Path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def solve(records):\n    return []\n", encoding="utf-8")

    result = evaluate(str(candidate))
    serialized = repr(result.metrics) + repr(result.artifacts)

    assert "holdout" not in serialized.lower()
