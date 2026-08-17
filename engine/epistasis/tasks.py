"""Task registry: real record-cleaning tasks plus deterministic synthetic variants.

Scoring reuses the project's shared sandboxed evaluator core
(``evaluator_core.load_candidate`` / ``score_cases``) so real and synthetic
tasks share one loader.  Synthetic tasks are generated from a parametric family
so the coverage / epistasis / cross-task experiments can run at scale without
any model call.
"""

from __future__ import annotations

import importlib.util
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluator_core import CASE_GROUPS, GROUP_WEIGHTS, load_candidate, score_cases

from .schema import PAID_SCHEMA, PAYOUT_SCHEMA, REFUND_SCHEMA, TaskSchema

ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "tasks"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    task_family: str
    schema: TaskSchema
    initial_source: str
    public_cases: tuple[dict[str, Any], ...]
    holdout_cases: tuple[dict[str, Any], ...]
    mechanisms: tuple[str, ...] = field(
        default=("status", "amount", "identity", "empty")
    )

    def score_source(self, source: str) -> dict[str, Any]:
        return score_task_source(source, self.public_cases)

    def score_holdout(self, source: str) -> dict[str, Any]:
        return score_task_source(source, self.holdout_cases)

    @property
    def passed_cases(self, metrics: dict[str, Any]) -> int:
        return int(metrics.get("passed_cases", 0))


def score_task_source(source: str, cases: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Score candidate source against a case tuple via the shared sandbox."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as handle:
        handle.write(source)
        handle.flush()
        solve, reasons = load_candidate(handle.name)
    if solve is None:
        return {
            **{group: 0.0 for group in CASE_GROUPS},
            "weighted_score": 0.0,
            "passed_cases": 0,
            "total_cases": len(cases),
            "case_results": [],
            "rejection_reasons": reasons,
            "evaluator_valid": 0.0,
        }
    metrics = score_cases(
        solve, cases, case_groups=CASE_GROUPS, group_weights=GROUP_WEIGHTS
    )
    metrics["evaluator_valid"] = 1.0
    return metrics


def _load_task_cases(
    path: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    spec = importlib.util.spec_from_file_location(f"_epistasis_task_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load task core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in ("CASES", "HOLDOUT_CASES"):
        if not hasattr(module, required):
            raise TypeError(f"task core missing {required}: {path}")
    return tuple(module.CASES), tuple(module.HOLDOUT_CASES)


def _initial_program(path: Path | None, schema: TaskSchema) -> str:
    if path is not None and path.is_file():
        return path.read_text(encoding="utf-8")
    return default_initial_program(schema)


def default_initial_program(schema: TaskSchema) -> str:
    """A deliberately weak starter: exact status match, no normalization/aggregation."""
    identity = schema.identity_field
    value = schema.value_field
    status = schema.status_field
    accepted = schema.accepted_status
    currency_clause = ""
    if schema.currency_field and schema.accepted_currency:
        currency_clause = f"\n            and row.get({schema.currency_field!r}) == {schema.accepted_currency!r}"
    return (
        f"# EVOLVE-BLOCK-START\n"
        f"def solve(records):\n"
        f"    output = []\n"
        f"    for row in records:\n"
        f"        if (\n"
        f"            isinstance(row, dict)\n"
        f"            and row.get({status!r}) == {accepted!r}{currency_clause}\n"
        f"        ):\n"
        f"            output.append((row[{identity!r}], round(row[{value!r}], 2)))\n"
        f"    return output\n"
        f"# EVOLVE-BLOCK-END\n"
    )


REAL_TASKS: dict[str, tuple[Path, Path | None, TaskSchema, tuple[str, ...]]] = {
    "paid": (
        ROOT / "evaluator_core.py",
        ROOT / "initial_program.py",
        PAID_SCHEMA,
        ("status", "amount", "identity"),
    ),
    "payout": (
        TASKS_ROOT / "payout_cleaning" / "evaluator_core.py",
        TASKS_ROOT / "payout_cleaning" / "initial_program.py",
        PAYOUT_SCHEMA,
        ("status", "amount", "identity", "empty"),
    ),
    "refund": (
        TASKS_ROOT / "refund_cleaning" / "evaluator_core.py",
        None,
        REFUND_SCHEMA,
        ("status", "amount", "identity", "empty"),
    ),
}


def load_task(task_id: str) -> TaskSpec:
    if task_id in REAL_TASKS:
        core_path, initial_path, schema, mechanisms = REAL_TASKS[task_id]
        public, holdout = _load_task_cases(core_path)
        return TaskSpec(
            task_id=task_id,
            task_family=task_id,
            schema=schema,
            initial_source=_initial_program(initial_path, schema),
            public_cases=public,
            holdout_cases=holdout,
            mechanisms=mechanisms,
        )
    raise ValueError(f"unknown task id: {task_id} (available: {sorted(REAL_TASKS)})")


# --------------------------------------------------------------------------
# Synthetic task generator
# --------------------------------------------------------------------------

_BASIC_CASE = {
    "id": "basic_rows",
    "group": "basic",
    "records": [
        {"account": "bob", "value": 3, "state": "settled", "currency": "USD"},
        {"account": "alice", "value": 2, "state": "settled", "currency": "USD"},
    ],
    "expected": [("bob", 3.0), ("alice", 2.0)],
}


def _renamed_case(case: dict[str, Any], schema: TaskSchema) -> dict[str, Any]:
    mapping = {
        "account": schema.identity_field,
        "value": schema.value_field,
        "state": schema.status_field,
        "currency": schema.currency_field or "currency",
    }
    out = {"id": case["id"], "group": case.get("group", "basic")}
    records = []
    for row in case["records"]:
        if row is None:
            records.append(None)
        else:
            records.append({mapping.get(k, k): v for k, v in row.items()})
    expected = [(name.lower(), float(v)) for name, v in case["expected"]]
    out["records"] = records
    out["expected"] = expected
    return out


def _case(
    *,
    case_id: str,
    group: str,
    schema: TaskSchema,
    records: list[Any],
    expected: list[tuple[str, float]],
) -> dict[str, Any]:
    mapping = {
        "account": schema.identity_field,
        "value": schema.value_field,
        "state": schema.status_field,
        "currency": schema.currency_field or "currency",
    }
    rows = []
    for row in records:
        if row is None:
            rows.append(None)
        else:
            rows.append({mapping.get(k, k): v for k, v in row.items()})
    return {"id": case_id, "group": group, "records": rows, "expected": expected}


def _v(schema: TaskSchema) -> str:
    return schema.value_field


def _s(schema: TaskSchema) -> str:
    return schema.status_field


def _i(schema: TaskSchema) -> str:
    return schema.identity_field


def generate_synthetic_task(
    *,
    task_id: str,
    mechanisms: tuple[str, ...],
    coupling: int = 0,
    uncovered_extra: int = 0,
    seed: int = 0,
    schema: TaskSchema = TaskSchema(currency_field=None, accepted_currency=None),
    n_independent_per_mechanism: int = 1,
) -> TaskSpec:
    """Generate a deterministic record-cleaning task.

    ``mechanisms`` selects which of the four operator-fixable failure families
    appear; ``coupling`` adds cases that require several mechanisms *in the same
    rows* (the epistasis driver); ``uncovered_extra`` adds cases no operator in
    the universe fixes (realistic noise / coverage ladder).
    """
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    accepted = schema.accepted_status
    identity = schema.identity_field
    value = schema.value_field
    status = schema.status_field

    cases.append(_renamed_case(_BASIC_CASE, schema))

    def add(case: dict[str, Any]) -> None:
        if not any(existing["id"] == case["id"] for existing in cases):
            cases.append(case)

    if "status" in mechanisms:
        for k in range(n_independent_per_mechanism):
            add(
                _case(
                    case_id=f"status_norm_{k}",
                    group="filtering",
                    schema=schema,
                    records=[
                        {identity: "alice", value: 2, status: f" {accepted.upper()} "},
                        {identity: "bob", value: 1, status: accepted.capitalize()},
                    ],
                    expected=[("alice", 2.0), ("bob", 1.0)],
                )
            )
    if "amount" in mechanisms:
        for k in range(n_independent_per_mechanism):
            add(
                _case(
                    case_id=f"amount_reject_{k}",
                    group="validation",
                    schema=schema,
                    records=[
                        {identity: "a", value: True, status: accepted},
                        {identity: "b", value: "9", status: accepted},
                        {identity: "c", value: 0, status: accepted},
                        {identity: "d", value: -2, status: accepted},
                        {identity: "ok", value: 3, status: accepted},
                    ],
                    expected=[("ok", 3.0)],
                )
            )
    if "identity" in mechanisms:
        for k in range(n_independent_per_mechanism):
            add(
                _case(
                    case_id=f"identity_norm_{k}",
                    group="normalization",
                    schema=schema,
                    records=[
                        {identity: " Alice ", value: 2, status: accepted},
                        {identity: "bob", value: 3, status: accepted},
                    ],
                    expected=[("alice", 2.0), ("bob", 3.0)],
                )
            )
    if "empty" in mechanisms:
        for k in range(n_independent_per_mechanism):
            add(
                _case(
                    case_id=f"empty_identity_{k}",
                    group="normalization",
                    schema=schema,
                    records=[
                        {identity: " ", value: 30, status: accepted},
                        {identity: "bob", value: 4, status: accepted},
                    ],
                    expected=[("bob", 4.0)],
                )
            )

    # Coupled cases: multiple mechanisms required on the same rows.
    if coupling >= 1 and "status" in mechanisms and "amount" in mechanisms:
        add(
            _case(
                case_id="coupled_status_amount",
                group="validation",
                schema=schema,
                records=[
                    {identity: "a", value: 3.0, status: f" {accepted.upper()} "},
                    {identity: "a", value: True, status: accepted.capitalize()},
                ],
                expected=[("a", 3.0)],
            )
        )
    if coupling >= 1 and "identity" in mechanisms and "amount" in mechanisms:
        add(
            _case(
                case_id="coupled_identity_amount",
                group="normalization",
                schema=schema,
                records=[
                    {identity: " Alice ", value: True, status: accepted},
                    {identity: "ALICE", value: 4, status: accepted},
                ],
                expected=[("alice", 4.0)],
            )
        )
    if coupling >= 2 and "identity" in mechanisms and "status" in mechanisms:
        add(
            _case(
                case_id="coupled_identity_status",
                group="filtering",
                schema=schema,
                records=[
                    {identity: " Alice ", value: 2, status: f" {accepted.upper()} "},
                    {identity: "ALICE", value: 3, status: "rejected"},
                ],
                expected=[("alice", 2.0)],
            )
        )

    # Uncovered noise cases (no operator in the universe fixes them).
    for k in range(uncovered_extra):
        noise = [
            {identity: "a", value: 1, status: accepted},
            {identity: "b", value: 2, status: accepted},
        ]
        rng.shuffle(noise)
        add(
            _case(
                case_id=f"uncovered_noise_{k}",
                group="sorting",
                schema=schema,
                records=noise,
                expected=[("b", 2.0), ("a", 1.0)],
            )
        )

    holdout = tuple(
        _renamed_case(
            {
                "id": f"holdout_{case_id}",
                "group": "robustness",
                "records": [
                    {identity: " Bob ", value: 3, status: accepted},
                    {identity: "bob", value: True, status: f" {accepted.upper()} "},
                ],
                "expected": [("bob", 3.0)],
            },
            schema,
        )
        for case_id, _ in enumerate(range(coupling))
    ) or tuple(_renamed_case(_BASIC_CASE, schema))

    return TaskSpec(
        task_id=task_id,
        task_family=task_id.rsplit("-", 1)[0] if "-" in task_id else task_id,
        schema=schema,
        initial_source=default_initial_program(schema),
        public_cases=tuple(cases),
        holdout_cases=holdout,
        mechanisms=mechanisms,
    )


def build_task_matrix(
    *,
    real_tasks: tuple[str, ...] = ("paid", "payout", "refund"),
    synthetic: int = 0,
    synthetic_seed: int = 0,
    coupling: int = 1,
    uncovered_extra: int = 1,
    synthetic_mechanism_sets: tuple[tuple[str, ...], ...] | None = None,
) -> tuple[TaskSpec, ...]:
    """Build the task matrix used by experiment runs (real + synthetic scale)."""
    tasks: list[TaskSpec] = []
    for task_id in real_tasks:
        tasks.append(load_task(task_id))
    if synthetic_mechanism_sets is None:
        synthetic_mechanism_sets = (
            ("status", "amount"),
            ("status", "amount", "identity"),
            ("status", "amount", "identity", "empty"),
        )
    for index in range(synthetic):
        mechanism_set = synthetic_mechanism_sets[index % len(synthetic_mechanism_sets)]
        tasks.append(
            generate_synthetic_task(
                task_id=f"synthetic-{index:03d}",
                mechanisms=mechanism_set,
                coupling=coupling,
                uncovered_extra=uncovered_extra,
                seed=synthetic_seed + index,
            )
        )
    return tuple(tasks)
