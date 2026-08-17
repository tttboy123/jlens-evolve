from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from backend_poc import read_persisted_run, run_backend_poc

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "initial_program.py"
OBSERVATION = ROOT / "analysis/agent-baseline/agent_strategy.json"


def test_backend_persists_process_convergence_and_evidence(tmp_path):
    db_path = tmp_path / "evolve.sqlite3"
    result = run_backend_poc(
        program_path=PROGRAM,
        observation_path=OBSERVATION,
        output_dir=tmp_path / "artifacts",
        db_path=db_path,
        run_id="run-one",
    )

    assert result["status"] == "completed"
    assert result["baseline"]["public_passed"] == 3
    assert result["final"]["public_passed"] == 6
    assert result["convergence"] == {
        "converged": True,
        "reason": "operator_space_exhausted",
        "task_solved": False,
    }
    assert [row["operator_id"] for row in result["iterations"]] == [
        "canonicalize_before_predicate",
        "finite_numeric_guard",
    ]
    assert all(row["decision"] == "accepted" for row in result["iterations"])
    assert result["search_protocol"]["holdout_used_for_search"] is False
    assert result["admission"]["holdout_evaluations"] == 2
    assert result["evidence"]["diagnosis"]["primary_cause"] == (
        "operator_coverage_exhausted"
    )

    persisted = read_persisted_run(db_path, "run-one")
    assert persisted["run"]["status"] == "completed"
    assert len(persisted["iterations"]) == 2
    assert persisted["result"]["outcome_fingerprint"] == result["outcome_fingerprint"]

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT phase, partition FROM evaluations "
            "WHERE run_id = ? ORDER BY ordinal",
            ("run-one",),
        ).fetchall()
    assert rows == [
        ("search", "public"),
        ("search", "public"),
        ("search", "public"),
        ("post_search_audit", "holdout"),
        ("post_search_audit", "holdout"),
    ]

    artifact_dir = Path(result["artifact_dir"])
    assert (artifact_dir / "candidate.py").is_file()
    assert (artifact_dir / "result.json").is_file()
    assert (artifact_dir / "evidence.json").is_file()


def test_three_runs_have_one_deterministic_outcome_fingerprint(tmp_path):
    db_path = tmp_path / "evolve.sqlite3"
    fingerprints = {
        run_backend_poc(
            program_path=PROGRAM,
            observation_path=OBSERVATION,
            output_dir=tmp_path / "artifacts",
            db_path=db_path,
            run_id=f"stable-{index}",
        )["outcome_fingerprint"]
        for index in range(3)
    }

    assert len(fingerprints) == 1
    with sqlite3.connect(db_path) as connection:
        completed = connection.execute(
            "SELECT COUNT(*) FROM runs WHERE status = 'completed'"
        ).fetchone()[0]
    assert completed == 3


def test_holdout_cannot_change_search_path(monkeypatch, tmp_path):
    import backend_poc

    phases: list[str] = []
    real_public = backend_poc._score_public_source
    real_holdout = backend_poc._score_holdout_source

    def public(source: str):
        phases.append("public")
        return real_public(source)

    def holdout(source: str):
        phases.append("holdout")
        return real_holdout(source)

    monkeypatch.setattr(backend_poc, "_score_public_source", public)
    monkeypatch.setattr(backend_poc, "_score_holdout_source", holdout)

    result = run_backend_poc(
        program_path=PROGRAM,
        observation_path=OBSERVATION,
        output_dir=tmp_path / "artifacts",
        db_path=tmp_path / "evolve.sqlite3",
        run_id="partition-order",
    )

    assert phases == ["public", "public", "public", "holdout", "holdout"]
    assert [row["operator_id"] for row in result["iterations"]] == [
        "canonicalize_before_predicate",
        "finite_numeric_guard",
    ]


def test_cli_runs_and_inspects_the_persisted_run(tmp_path):
    db_path = tmp_path / "evolve.sqlite3"
    output_dir = tmp_path / "artifacts"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend_poc.py"),
            "run",
            "--db",
            str(db_path),
            "--output",
            str(output_dir),
            "--run-id",
            "cli-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "completed"' in run.stdout
    assert '"reason": "operator_space_exhausted"' in run.stdout

    inspect = subprocess.run(
        [
            sys.executable,
            str(ROOT / "backend_poc.py"),
            "inspect",
            "--db",
            str(db_path),
            "--run-id",
            "cli-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"run_id": "cli-run"' in inspect.stdout
    assert '"operator_id": "finite_numeric_guard"' in inspect.stdout


def test_run_id_cannot_escape_the_artifact_directory(tmp_path):
    with pytest.raises(ValueError, match="run_id must start"):
        run_backend_poc(
            program_path=PROGRAM,
            observation_path=OBSERVATION,
            output_dir=tmp_path / "artifacts",
            db_path=tmp_path / "evolve.sqlite3",
            run_id="../escape",
        )
