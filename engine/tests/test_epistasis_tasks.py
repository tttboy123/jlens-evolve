from __future__ import annotations

from epistasis.tasks import build_task_matrix, generate_synthetic_task, load_task


def test_real_tasks_load_and_initial_programs_are_weak():
    for task_id in ("paid", "payout", "refund"):
        task = load_task(task_id)
        assert len(task.public_cases) >= 10
        assert len(task.holdout_cases) >= 3
        assert task.initial_source.strip()
        assert task.score_source(task.initial_source)["passed_cases"] < len(
            task.public_cases
        )


def test_synthetic_task_is_deterministic():
    a = generate_synthetic_task(task_id="s", mechanisms=("status", "amount"), seed=7)
    b = generate_synthetic_task(task_id="s", mechanisms=("status", "amount"), seed=7)
    assert a.public_cases == b.public_cases
    assert a.initial_source == b.initial_source
    assert (
        a.score_source(a.initial_source)["passed_cases"]
        == b.score_source(b.initial_source)["passed_cases"]
    )


def test_synthetic_mechanisms_are_decoupled():
    # A task with a single mechanism should be fixed by exactly that operator.
    from epistasis.operators import OPERATOR_IDS

    mapping = {
        "status": "canonicalize_status",
        "amount": "reject_invalid_amounts",
        "identity": "normalize_identity",
        "empty": "drop_empty_identity",
    }
    for mechanism in ("status", "amount", "identity", "empty"):
        task = generate_synthetic_task(task_id="s", mechanisms=(mechanism,), seed=2)
        initial = task.score_source(task.initial_source)["passed_cases"]
        for op in OPERATOR_IDS:
            from epistasis.operators import apply_operator

            result = apply_operator(task.initial_source, op, task.schema)
            score = task.score_source(result.source)["passed_cases"]
            if op == mapping[mechanism]:
                assert score > initial, mechanism
            else:
                assert score == initial, (mechanism, op)


def test_coupled_case_requires_two_mechanisms():
    task = generate_synthetic_task(
        task_id="s", mechanisms=("status", "amount"), coupling=1, seed=1
    )
    from epistasis.operators import apply_operator

    initial = task.score_source(task.initial_source)["passed_cases"]
    status_only = apply_operator(
        task.initial_source, "canonicalize_status", task.schema
    )
    amount_only = apply_operator(
        task.initial_source, "reject_invalid_amounts", task.schema
    )
    assert task.score_source(status_only.source)["passed_cases"] == initial + 1
    assert task.score_source(amount_only.source)["passed_cases"] == initial + 1
    # Coupled case needs both -> full set passes everything.
    from epistasis.operators import apply_operators

    both = apply_operators(
        task.initial_source,
        ("canonicalize_status", "reject_invalid_amounts"),
        task.schema,
    )
    assert task.score_source(both.source)["passed_cases"] == len(task.public_cases)


def test_build_task_matrix_counts():
    tasks = build_task_matrix(real_tasks=("paid", "payout"), synthetic=3)
    assert len(tasks) == 5
    assert [t.task_id for t in tasks[:2]] == ["paid", "payout"]
    assert all(t.task_id.startswith("synthetic-") for t in tasks[2:])
