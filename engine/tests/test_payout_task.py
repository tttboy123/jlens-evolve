from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.payout_cleaning.evaluator_core import (
    score_callable,
    score_holdout_callable,
)

ROOT = Path(__file__).resolve().parents[1]


def reference_solution(records):
    import math

    totals = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        account = row.get("account")
        state = row.get("state")
        currency = row.get("currency")
        value = row.get("value")
        if not all(isinstance(item, str) for item in (account, state, currency)):
            continue
        account = account.strip().lower()
        if (
            not account
            or state.strip().lower() != "settled"
            or currency.strip().upper() != "USD"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            continue
        totals[account] = totals.get(account, 0.0) + float(value)
    return sorted(
        ((account, round(total, 2)) for account, total in totals.items()),
        key=lambda item: (-item[1], item[0]),
    )


def test_payout_reference_solution_passes_public_and_hidden_partitions():
    public = score_callable(reference_solution)
    hidden = score_holdout_callable(reference_solution)

    assert public["passed_cases"] == public["total_cases"]
    assert hidden["passed_cases"] == hidden["total_cases"]


def test_payout_initial_program_is_a_nontrivial_search_seed():
    from tasks.payout_cleaning.evaluator_core import score_program_path

    result = score_program_path(ROOT / "tasks/payout_cleaning/initial_program.py")

    assert 0 < result["passed_cases"] < result["total_cases"]


def test_payout_evaluator_artifacts_do_not_leak_holdout_cases():
    import json

    from tasks.payout_cleaning.evaluator import evaluate

    result = evaluate(str(ROOT / "tasks/payout_cleaning/initial_program.py"))

    assert "holdout" not in json.dumps(result.artifacts).lower()


def test_payout_evaluator_imports_with_worker_path_precedence():
    command = """
import importlib.util
import sys
from pathlib import Path
root = Path(sys.argv[1])
task_dir = root / 'tasks/payout_cleaning'
sys.path.insert(0, str(task_dir))
sys.path.insert(1, str(root))
spec = importlib.util.spec_from_file_location('worker_evaluator', task_dir / 'evaluator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.evaluate)
"""

    result = subprocess.run(
        [sys.executable, "-c", command, str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_preflight_accepts_payout_worker_evaluator():
    from evolve_runtime import preflight_evaluator_import

    preflight_evaluator_import(
        ROOT / "tasks/payout_cleaning/evaluator.py",
        ROOT,
    )
