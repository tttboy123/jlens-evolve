from __future__ import annotations

import itertools

from epistasis.operators import (
    OPERATOR_IDS,
    apply_operator,
    apply_operators,
    coverage_of,
    postcondition_ok,
)
from epistasis.schema import REFUND_SCHEMA
from epistasis.tasks import generate_synthetic_task, load_task


def test_each_operator_fixes_its_own_probe():
    task = generate_synthetic_task(
        task_id="syn", mechanisms=("status", "amount", "identity", "empty"), seed=0
    )
    for op in OPERATOR_IDS:
        result = apply_operator(task.initial_source, op, task.schema)
        assert result.changed is True, op
        assert result.postcondition_ok is True, op


def test_operator_on_real_payout_increases_score():
    task = load_task("payout")
    baseline = task.score_source(task.initial_source)["passed_cases"]
    for op in ("canonicalize_status", "reject_invalid_amounts"):
        result = apply_operator(task.initial_source, op, task.schema)
        assert result.changed is True
        assert result.postcondition_ok is True
        assert task.score_source(result.source)["passed_cases"] >= baseline


def test_composition_reaches_full_score_on_coupled_synthetic():
    task = generate_synthetic_task(
        task_id="syn",
        mechanisms=("status", "amount", "identity", "empty"),
        coupling=2,
        seed=3,
    )
    total = len(task.public_cases)
    initial = task.score_source(task.initial_source)["passed_cases"]
    composed = apply_operators(task.initial_source, OPERATOR_IDS, task.schema)
    assert composed.ok is True
    assert task.score_source(composed.source)["passed_cases"] == total
    assert total > initial


def test_composition_is_order_insensitive():
    task = generate_synthetic_task(
        task_id="syn",
        mechanisms=("status", "amount", "identity", "empty"),
        coupling=2,
        seed=3,
    )
    scores = set()
    for perm in itertools.permutations(OPERATOR_IDS):
        composed = apply_operators(task.initial_source, perm, task.schema)
        assert composed.ok is True
        scores.add(task.score_source(composed.source)["passed_cases"])
    assert len(scores) == 1


def test_noop_when_mechanism_already_satisfied():
    task = load_task("payout")
    result = apply_operator(task.initial_source, "canonicalize_status", task.schema)
    assert result.changed is True
    second = apply_operator(result.source, "canonicalize_status", task.schema)
    assert second.changed is False
    assert second.postcondition_ok is True


def test_postcondition_ok_is_probe_specific():
    # Structural strip/lower from status normalization must NOT satisfy the
    # identity postcondition (this was the original bug).
    task = load_task("refund")
    status_result = apply_operator(
        task.initial_source, "canonicalize_status", task.schema
    )
    assert status_result.postcondition_ok is True
    assert (
        postcondition_ok(status_result.source, "normalize_identity", task.schema)
        is False
    )


def test_refund_schema_parameterization():
    task = load_task("refund")
    assert task.schema == REFUND_SCHEMA
    for op in ("canonicalize_status", "normalize_identity", "drop_empty_identity"):
        result = apply_operator(task.initial_source, op, task.schema)
        assert result.changed is True
        assert result.postcondition_ok is True


def test_coverage_of_mechanisms():
    assert coverage_of(("canonicalize_status",), ("status", "amount")) == 0.5
    assert (
        coverage_of(
            ("canonicalize_status", "reject_invalid_amounts"), ("status", "amount")
        )
        == 1.0
    )
    assert coverage_of((), ("status", "amount")) == 0.0
