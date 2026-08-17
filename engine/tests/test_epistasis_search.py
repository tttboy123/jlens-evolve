from __future__ import annotations

from epistasis.model_transport import StubTransport
from epistasis.operators import OPERATOR_IDS
from epistasis.search import (
    cell_key,
    run_cell,
    run_lineage_cell,
)
from epistasis.tasks import generate_synthetic_task


def _coupled_task():
    return generate_synthetic_task(
        task_id="syn",
        mechanisms=("status", "amount", "identity", "empty"),
        coupling=2,
        seed=3,
    )


def test_single_operator_cell_stalls_at_one_gain():
    task = _coupled_task()
    result = run_cell(task, ("canonicalize_status",), seed=1, budget=6)
    assert result.initial_passed == 1
    assert result.best_passed == 2  # only the status case gained
    assert result.improved_count == 1
    assert result.no_op_count >= 4


def test_composed_cell_reaches_full_score():
    task = _coupled_task()
    result = run_cell(task, OPERATOR_IDS, seed=1, budget=6)
    assert result.best_passed == len(task.public_cases)
    assert result.yield_ > 0


def test_lineage_reaches_same_score_as_composed():
    task = _coupled_task()
    lineage = run_lineage_cell(task, OPERATOR_IDS, seed=1, budget=12)
    composed = run_cell(task, OPERATOR_IDS, seed=1, budget=12)
    assert lineage.best_passed == composed.best_passed == len(task.public_cases)
    assert lineage.accepted_count >= 4


def test_llm_mode_with_stub_transport_uses_scaffold():
    task = _coupled_task()
    # A stub that echoes the scaffold back (extraction fallback keeps scaffold).
    model = StubTransport(repair_source="")
    result = run_cell(
        task, ("canonicalize_status",), seed=1, budget=3, mode="llm", model=model
    )
    assert result.best_passed >= 2
    assert result.invalid_count == 0


def test_deterministic_search_is_seed_deterministic():
    task = _coupled_task()
    a = run_cell(task, OPERATOR_IDS, seed=5, budget=4)
    b = run_cell(task, OPERATOR_IDS, seed=5, budget=4)
    assert a.best_source == b.best_source
    assert [e.source_hash for e in a.events] == [e.source_hash for e in b.events]


def test_cell_key_uniqueness():
    assert cell_key("t", 1, ("a",), "deterministic", 4) != cell_key(
        "t", 1, ("a", "b"), "deterministic", 4
    )
    assert cell_key("t", 1, ("a",), "deterministic", 4) == cell_key(
        "t", 1, ("a",), "deterministic", 4
    )
