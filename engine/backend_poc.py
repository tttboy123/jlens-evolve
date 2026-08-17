"""Stable local backend POC for a minimal supervised evolution run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluator_core
from structured_mutation import apply_mutation_plan, derive_fallback_plan

ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRAM = ROOT / "initial_program.py"
DEFAULT_OBSERVATION = ROOT / "analysis/agent-baseline/agent_strategy.json"
DEFAULT_OUTPUT = ROOT / "runs/backend-poc"
DEFAULT_DB = DEFAULT_OUTPUT / "evolve.sqlite3"
TASK_ID = "order-summary-cleaning-minimal-v1"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    program_sha256 TEXT NOT NULL,
    observation_sha256 TEXT NOT NULL,
    protocol_sha256 TEXT NOT NULL,
    outcome_fingerprint TEXT,
    convergence_reason TEXT,
    decision TEXT,
    artifact_dir TEXT,
    result_json TEXT,
    evidence_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    ordinal INTEGER NOT NULL,
    phase TEXT NOT NULL,
    partition TEXT NOT NULL,
    candidate_role TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    passed_cases INTEGER NOT NULL,
    total_cases INTEGER NOT NULL,
    score REAL NOT NULL,
    details_json TEXT NOT NULL,
    PRIMARY KEY (run_id, ordinal)
);
CREATE TABLE IF NOT EXISTS iterations (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration INTEGER NOT NULL,
    public_failure TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    parent_sha256 TEXT NOT NULL,
    candidate_sha256 TEXT NOT NULL,
    decision TEXT NOT NULL,
    public_gain INTEGER NOT NULL,
    gained_cases_json TEXT NOT NULL,
    lost_cases_json TEXT NOT NULL,
    gate_json TEXT NOT NULL,
    PRIMARY KEY (run_id, iteration)
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _score_source(source: str, *, holdout: bool) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="evolve-backend-poc-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        if not holdout:
            return evaluator_core.score_program_path(path)
        solve, reasons = evaluator_core.load_candidate(path)
        if solve is None:
            raise ValueError(f"candidate rejected by evaluator: {reasons}")
        return evaluator_core.score_holdout_callable(solve)


def _score_public_source(source: str) -> dict[str, Any]:
    return _score_source(source, holdout=False)


def _score_holdout_source(source: str) -> dict[str, Any]:
    return _score_source(source, holdout=True)


def _passed_ids(metrics: dict[str, Any]) -> set[str]:
    return {
        str(row["id"]) for row in metrics.get("case_results", []) if row.get("passed")
    }


def _failed_ids(metrics: dict[str, Any]) -> list[str]:
    return [
        str(row["id"])
        for row in metrics.get("case_results", [])
        if not row.get("passed")
    ]


def _public_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "public_passed": int(metrics["passed_cases"]),
        "public_total": int(metrics["total_cases"]),
        "public_score": float(metrics["combined_score"]),
        "passed_cases": sorted(_passed_ids(metrics)),
        "failed_cases": _failed_ids(metrics),
    }


def _holdout_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "holdout_passed": int(metrics["passed_cases"]),
        "holdout_total": int(metrics["total_cases"]),
        "holdout_score": float(metrics["combined_score"]),
        "passed_cases": sorted(_passed_ids(metrics)),
        "failed_cases": _failed_ids(metrics),
    }


class BackendStore:
    """Small SQLite event store for one-process POC runs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(_SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def create_run(self, manifest: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO runs "
            "(run_id, task_id, status, started_at, program_sha256, "
            "observation_sha256, protocol_sha256) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                manifest["run_id"],
                manifest["task_id"],
                "running",
                manifest["started_at"],
                manifest["program_sha256"],
                manifest["observation_sha256"],
                manifest["protocol_sha256"],
            ),
        )
        self.connection.commit()

    def append_evaluation(
        self,
        run_id: str,
        *,
        phase: str,
        partition: str,
        candidate_role: str,
        candidate_sha256: str,
        metrics: dict[str, Any],
    ) -> None:
        ordinal = self.connection.execute(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM evaluations WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        self.connection.execute(
            "INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                ordinal,
                phase,
                partition,
                candidate_role,
                candidate_sha256,
                int(metrics["passed_cases"]),
                int(metrics["total_cases"]),
                float(metrics["combined_score"]),
                _json(metrics),
            ),
        )
        self.connection.commit()

    def append_iteration(self, run_id: str, row: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT INTO iterations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                row["iteration"],
                row["public_failure"],
                row["operator_id"],
                row["parent_sha256"],
                row["candidate_sha256"],
                row["decision"],
                row["public_gain"],
                _json(row["gained_public_cases"]),
                _json(row["lost_public_cases"]),
                _json(row["checks"]),
            ),
        )
        self.connection.commit()

    def complete(self, run_id: str, result: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE runs SET status = 'completed', completed_at = ?, "
            "outcome_fingerprint = ?, convergence_reason = ?, decision = ?, "
            "artifact_dir = ?, result_json = ?, evidence_json = ? WHERE run_id = ?",
            (
                result["completed_at"],
                result["outcome_fingerprint"],
                result["convergence"]["reason"],
                result["admission"]["decision"],
                result["artifact_dir"],
                _json(result),
                _json(result["evidence"]),
                run_id,
            ),
        )
        self.connection.commit()

    def fail(self, run_id: str, error: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status = 'failed', completed_at = ?, error = ? "
            "WHERE run_id = ?",
            (_now(), error, run_id),
        )
        self.connection.commit()


def read_persisted_run(db_path: Path, run_id: str | None = None) -> dict[str, Any]:
    """Read one completed or failed run, including its process rows."""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        if run_id is None:
            row = connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, rowid DESC LIMIT 1"
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id or 'latest'}")
        selected_run_id = str(row["run_id"])
        iterations = connection.execute(
            "SELECT * FROM iterations WHERE run_id = ? ORDER BY iteration",
            (selected_run_id,),
        ).fetchall()
        evaluations = connection.execute(
            "SELECT * FROM evaluations WHERE run_id = ? ORDER BY ordinal",
            (selected_run_id,),
        ).fetchall()
    run = dict(row)
    result_json = run.pop("result_json")
    evidence_json = run.pop("evidence_json")
    parsed_iterations = []
    for item in iterations:
        parsed = dict(item)
        parsed["gained_public_cases"] = json.loads(parsed.pop("gained_cases_json"))
        parsed["lost_public_cases"] = json.loads(parsed.pop("lost_cases_json"))
        parsed["checks"] = json.loads(parsed.pop("gate_json"))
        parsed_iterations.append(parsed)
    evaluation_summaries = []
    for item in evaluations:
        parsed = dict(item)
        parsed.pop("details_json")
        evaluation_summaries.append(parsed)
    return {
        "run": run,
        "iterations": parsed_iterations,
        "evaluations": evaluation_summaries,
        "result": json.loads(result_json) if result_json else None,
        "evidence": json.loads(evidence_json) if evidence_json else None,
    }


def _select_target(
    metrics: dict[str, Any], attempted_failures: set[str]
) -> tuple[str, Any] | None:
    for failure in _failed_ids(metrics):
        if failure in attempted_failures:
            continue
        plan = derive_fallback_plan(failure)
        if plan.structured:
            return failure, plan
    return None


def _gate(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    changed: bool,
    postcondition_valid: bool,
) -> dict[str, Any]:
    parent_passed = _passed_ids(parent)
    candidate_passed = _passed_ids(candidate)
    gained = sorted(candidate_passed - parent_passed)
    lost = sorted(parent_passed - candidate_passed)
    gain = int(candidate["passed_cases"] - parent["passed_cases"])
    checks = {
        "source_changed": changed,
        "operator_postcondition": postcondition_valid,
        "public_gain_positive": gain > 0,
        "no_public_regression": not lost,
    }
    return {
        "decision": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "public_gain": gain,
        "gained_public_cases": gained,
        "lost_public_cases": lost,
    }


def _protocol(max_iterations: int) -> dict[str, Any]:
    protocol = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "search_partition": "public",
        "admission_partition": "holdout",
        "holdout_used_for_search": False,
        "target_selection": "first_failed_case_with_allowlisted_operator",
        "operators": [
            "canonicalize_before_predicate",
            "finite_numeric_guard",
        ],
        "max_iterations": max_iterations,
        "evaluator_sha256": _sha256(
            (ROOT / "evaluator_core.py").read_text(encoding="utf-8")
        ),
        "mutation_engine_sha256": _sha256(
            (ROOT / "structured_mutation.py").read_text(encoding="utf-8")
        ),
        "runner_sha256": _sha256(Path(__file__).read_text(encoding="utf-8")),
    }
    return {**protocol, "protocol_sha256": _sha256(_json(protocol))}


def _build_evidence(
    *,
    observation: dict[str, Any],
    iterations: list[dict[str, Any]],
    convergence: dict[str, Any],
    final_public: dict[str, Any],
    baseline_holdout: dict[str, Any],
    final_holdout: dict[str, Any],
) -> dict[str, Any]:
    observer_evidence = observation.get("evidence", {})
    accepted = [row for row in iterations if row["decision"] == "accepted"]
    if convergence["reason"] == "operator_space_exhausted":
        primary_cause = "operator_coverage_exhausted"
        cause_zh = (
            "当前白名单只覆盖状态标准化和金额合法性；剩余公开失败没有可用的"
            "结构化 operator，所以搜索停止在 6/13，而不是任务已经解决。"
        )
    elif convergence["reason"] == "iteration_budget_exhausted":
        primary_cause = "iteration_budget_exhausted"
        cause_zh = "搜索预算耗尽时仍存在可尝试的 operator。"
    else:
        primary_cause = "none_public_target_solved"
        cause_zh = "公开任务目标已经全部通过。"
    holdout_gain = final_holdout["holdout_passed"] - baseline_holdout["holdout_passed"]
    return {
        "schema_version": 1,
        "claim_boundary": "diagnostic_and_bounded_intervention_not_model_training",
        "observations": [
            {
                "id": "jlens_prior_search_repetition",
                "kind": "observational",
                "value": float(
                    observer_evidence.get("repeated_transition_fraction", 0.0)
                ),
                "trace_edges": int(observer_evidence.get("trace_edges", 0)),
                "unique_transitions": int(
                    observer_evidence.get("unique_transitions", 0)
                ),
                "jlens_incremental_supported": bool(
                    observer_evidence.get("jlens_incremental_supported", False)
                ),
                "interpretation_zh": (
                    "历史搜索存在重复迁移，支持尝试有界 mutation；这不是因果证据，"
                    "也不参与候选准入。"
                ),
            }
        ],
        "operator_effects": [
            {
                "iteration": row["iteration"],
                "operator_id": row["operator_id"],
                "parent_sha256": row["parent_sha256"],
                "candidate_sha256": row["candidate_sha256"],
                "public_gain": row["public_gain"],
                "gained_public_cases": row["gained_public_cases"],
                "lost_public_cases": row["lost_public_cases"],
                "evidence_kind": "deterministic_parent_child_intervention",
            }
            for row in accepted
        ],
        "generalization_audit": {
            "phase": "post_search_audit",
            "used_for_search": False,
            "baseline_holdout_passed": baseline_holdout["holdout_passed"],
            "final_holdout_passed": final_holdout["holdout_passed"],
            "holdout_gain": holdout_gain,
            "conclusion": (
                "no_hidden_generalization_gain"
                if holdout_gain <= 0
                else "positive_hidden_generalization_gain"
            ),
        },
        "diagnosis": {
            "primary_cause": primary_cause,
            "explanation_zh": cause_zh,
            "remaining_public_failures": final_public["failed_cases"],
            "limitations": [
                "JLens 证据仅为观察性，未证明独立因果增益。",
                "当前只有一个确定性任务和两个 operator。",
                "sealed holdout 仅用于搜索结束后的审计，不会触发继续搜索。",
            ],
        },
    }


def _outcome_fingerprint(result: dict[str, Any]) -> str:
    stable = {
        "protocol_sha256": result["search_protocol"]["protocol_sha256"],
        "program_sha256": result["inputs"]["program_sha256"],
        "observation_sha256": result["inputs"]["observation_sha256"],
        "baseline": result["baseline"],
        "iterations": [
            {
                key: row[key]
                for key in (
                    "iteration",
                    "public_failure",
                    "operator_id",
                    "parent_sha256",
                    "candidate_sha256",
                    "decision",
                    "public_gain",
                    "gained_public_cases",
                    "lost_public_cases",
                )
            }
            for row in result["iterations"]
        ],
        "final": result["final"],
        "convergence": result["convergence"],
        "admission": result["admission"],
        "diagnosis": result["evidence"]["diagnosis"],
    }
    return _sha256(_json(stable))


def _render_summary(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    final = result["final"]
    audit = result["admission"]
    diagnosis = result["evidence"]["diagnosis"]
    return f"""# 后台 Evolve POC 运行结果

- Run：`{result["run_id"]}`
- 状态：`{result["status"]}`
- Public：`{baseline["public_passed"]}/{baseline["public_total"]}` → `{final["public_passed"]}/{final["public_total"]}`
- Holdout：`{audit["baseline_holdout_passed"]}/{audit["holdout_total"]}` → `{audit["final_holdout_passed"]}/{audit["holdout_total"]}`
- 搜索收敛：`{result["convergence"]["reason"]}`
- 任务解决：`{str(result["convergence"]["task_solved"]).lower()}`
- 晋升决定：`{audit["decision"]}`
- 稳定结果指纹：`{result["outcome_fingerprint"]}`

## 原因

{diagnosis["explanation_zh"]}

剩余公开失败：{", ".join(diagnosis["remaining_public_failures"])}

## 证据边界

JLens 只提供历史搜索诊断；逐步 Gate 只读取 public evaluator。Holdout 在搜索停止后
评测 baseline 与最终候选，不会触发新的 mutation。当前结果不是模型训练，也不是
生产就绪结论。
"""


def run_backend_poc(
    *,
    program_path: Path,
    observation_path: Path,
    output_dir: Path,
    db_path: Path,
    max_iterations: int = 8,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run the fixed minimal search, persist events, and audit after convergence."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    source = program_path.read_text(encoding="utf-8")
    observation_text = observation_path.read_text(encoding="utf-8")
    observation = json.loads(observation_text)
    if observation.get("causal_boundary") != "observational_not_causal":
        raise ValueError("observation must retain observational_not_causal boundary")
    protocol = _protocol(max_iterations)
    selected_run_id = run_id or f"poc-{uuid.uuid4().hex[:12]}"
    if _RUN_ID.fullmatch(selected_run_id) is None:
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, digits, dot, underscore, or hyphen (max 128 characters)"
        )
    started_at = _now()
    artifact_dir = (output_dir / selected_run_id).resolve()
    manifest = {
        "run_id": selected_run_id,
        "task_id": TASK_ID,
        "started_at": started_at,
        "program_sha256": _sha256(source),
        "observation_sha256": _sha256(observation_text),
        "protocol_sha256": protocol["protocol_sha256"],
    }
    store = BackendStore(db_path)
    store.create_run(manifest)
    try:
        baseline_public_metrics = _score_public_source(source)
        store.append_evaluation(
            selected_run_id,
            phase="search",
            partition="public",
            candidate_role="baseline",
            candidate_sha256=_sha256(source),
            metrics=baseline_public_metrics,
        )
        current_source = source
        current_metrics = baseline_public_metrics
        attempted_failures: set[str] = set()
        iterations: list[dict[str, Any]] = []
        convergence_reason = "iteration_budget_exhausted"

        for iteration in range(1, max_iterations + 1):
            if int(current_metrics["passed_cases"]) == int(
                current_metrics["total_cases"]
            ):
                convergence_reason = "public_target_solved"
                break
            selected = _select_target(current_metrics, attempted_failures)
            if selected is None:
                convergence_reason = "operator_space_exhausted"
                break
            public_failure, mutation_plan = selected
            attempted_failures.add(public_failure)
            parent_sha256 = _sha256(current_source)
            mutation = apply_mutation_plan(current_source, mutation_plan)
            candidate_sha256 = _sha256(mutation.source)
            candidate_metrics = _score_public_source(mutation.source)
            store.append_evaluation(
                selected_run_id,
                phase="search",
                partition="public",
                candidate_role=f"candidate_{iteration}",
                candidate_sha256=candidate_sha256,
                metrics=candidate_metrics,
            )
            gate = _gate(
                current_metrics,
                candidate_metrics,
                changed=mutation.changed,
                postcondition_valid=mutation.postcondition_valid,
            )
            row = {
                "iteration": iteration,
                "public_failure": public_failure,
                "operator_id": mutation.operator_id,
                "parent_sha256": parent_sha256,
                "candidate_sha256": candidate_sha256,
                "before": _public_summary(current_metrics),
                "after": _public_summary(candidate_metrics),
                **gate,
            }
            iterations.append(row)
            store.append_iteration(selected_run_id, row)
            if gate["decision"] == "accepted":
                current_source = mutation.source
                current_metrics = candidate_metrics

        baseline_holdout_metrics = _score_holdout_source(source)
        store.append_evaluation(
            selected_run_id,
            phase="post_search_audit",
            partition="holdout",
            candidate_role="baseline",
            candidate_sha256=_sha256(source),
            metrics=baseline_holdout_metrics,
        )
        final_holdout_metrics = _score_holdout_source(current_source)
        store.append_evaluation(
            selected_run_id,
            phase="post_search_audit",
            partition="holdout",
            candidate_role="final",
            candidate_sha256=_sha256(current_source),
            metrics=final_holdout_metrics,
        )

        baseline = _public_summary(baseline_public_metrics)
        final = _public_summary(current_metrics)
        baseline_holdout = _holdout_summary(baseline_holdout_metrics)
        final_holdout = _holdout_summary(final_holdout_metrics)
        convergence = {
            "converged": convergence_reason
            in {"operator_space_exhausted", "public_target_solved"},
            "reason": convergence_reason,
            "task_solved": bool(
                final["public_passed"] == final["public_total"]
                and final_holdout["holdout_passed"] == final_holdout["holdout_total"]
            ),
        }
        public_gain = final["public_passed"] - baseline["public_passed"]
        holdout_gain = (
            final_holdout["holdout_passed"] - baseline_holdout["holdout_passed"]
        )
        admission = {
            "decision": (
                "poc_candidate_accepted"
                if public_gain > 0 and holdout_gain >= 0
                else "poc_candidate_rejected"
            ),
            "public_gain": public_gain,
            "baseline_holdout_passed": baseline_holdout["holdout_passed"],
            "final_holdout_passed": final_holdout["holdout_passed"],
            "holdout_total": final_holdout["holdout_total"],
            "holdout_gain": holdout_gain,
            "holdout_evaluations": 2,
            "production_ready": False,
        }
        evidence = _build_evidence(
            observation=observation,
            iterations=iterations,
            convergence=convergence,
            final_public=final,
            baseline_holdout=baseline_holdout,
            final_holdout=final_holdout,
        )
        result = {
            "schema_version": 1,
            "run_id": selected_run_id,
            "task_id": TASK_ID,
            "status": "completed",
            "started_at": started_at,
            "completed_at": _now(),
            "inputs": {
                "program": str(program_path.resolve()),
                "program_sha256": manifest["program_sha256"],
                "observation": str(observation_path.resolve()),
                "observation_sha256": manifest["observation_sha256"],
            },
            "search_protocol": protocol,
            "baseline": baseline,
            "iterations": iterations,
            "final": final,
            "convergence": convergence,
            "admission": admission,
            "evidence": evidence,
            "artifact_dir": str(artifact_dir),
        }
        result["outcome_fingerprint"] = _outcome_fingerprint(result)
        _atomic_text(artifact_dir / "candidate.py", current_source)
        _atomic_text(
            artifact_dir / "evidence.json",
            json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        _atomic_text(artifact_dir / "summary.md", _render_summary(result))
        _atomic_text(
            artifact_dir / "result.json",
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )
        store.complete(selected_run_id, result)
        return result
    except Exception as exc:
        store.fail(selected_run_id, f"{type(exc).__name__}: {exc}")
        raise
    finally:
        store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run one deterministic evolve POC")
    run_parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    run_parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--max-iterations", type=int, default=8)

    inspect_parser = commands.add_parser(
        "inspect", help="read a run and its process rows from SQLite"
    )
    inspect_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    inspect_parser.add_argument("--run-id")

    args = parser.parse_args()
    if args.command == "inspect":
        persisted = read_persisted_run(args.db, args.run_id)
        result = persisted["result"] or {}
        print(
            json.dumps(
                {
                    "run": persisted["run"],
                    "iterations": persisted["iterations"],
                    "evaluations": persisted["evaluations"],
                    "convergence": result.get("convergence"),
                    "admission": result.get("admission"),
                    "evidence": persisted["evidence"],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0

    result = run_backend_poc(
        program_path=args.program,
        observation_path=args.observation,
        output_dir=args.output,
        db_path=args.db,
        max_iterations=args.max_iterations,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "public": {
                    "baseline": result["baseline"]["public_passed"],
                    "final": result["final"]["public_passed"],
                    "total": result["final"]["public_total"],
                },
                "holdout": {
                    "baseline": result["admission"]["baseline_holdout_passed"],
                    "final": result["admission"]["final_holdout_passed"],
                    "total": result["admission"]["holdout_total"],
                },
                "convergence": result["convergence"],
                "decision": result["admission"]["decision"],
                "outcome_fingerprint": result["outcome_fingerprint"],
                "db": str(args.db.resolve()),
                "artifact_dir": result["artifact_dir"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
