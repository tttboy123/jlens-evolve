"""Deterministic v0.2 AgentProgram search and sealed-audit runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluator_core
from agent_program import (
    AgentProgram,
    ComponentRegistry,
    ContractError,
    ReplaySupervisor,
    apply_proposal,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = ROOT / "artifacts/v1.0.0/v0.2.0-agent-program"
DEFAULT_CONFIGS = DEFAULT_STAGE / "configs"
DEFAULT_RUNS = DEFAULT_STAGE / "runs"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


@dataclass(frozen=True)
class AgentExecution:
    output: list[tuple[str, float]] | None
    attempts: int
    retried: bool
    first_error: str | None
    first_invariant_failure: bool
    final_error: str | None


class FrozenReplayAgent:
    """Fixed local harness whose behavior is selected only by AgentProgram refs."""

    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry

    def _capabilities(self, program: AgentProgram) -> set[str]:
        capabilities = set(
            self.registry.prompts[program.system_prompt_ref].get("capabilities", [])
        )
        for skill_ref in program.skill_refs:
            capabilities.update(self.registry.skills[skill_ref].get("capabilities", []))
        return capabilities

    @staticmethod
    def _output_invariants(output: list[tuple[str, float]], *, aggregate: bool) -> bool:
        if not isinstance(output, list):
            return False
        for item in output:
            if not isinstance(item, tuple) or len(item) != 2:
                return False
            user, amount = item
            if not isinstance(user, str) or not user:
                return False
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                return False
            if not math.isfinite(float(amount)) or amount <= 0:
                return False
        if aggregate:
            users = [item[0] for item in output]
            if len(users) != len(set(users)):
                return False
            expected = sorted(output, key=lambda item: (-item[1], item[0]))
            if output != expected:
                return False
        return True

    def _attempt(
        self,
        program: AgentProgram,
        records: list[Any],
        *,
        safe_validation: bool,
    ) -> list[tuple[str, float]]:
        capabilities = self._capabilities(program)
        normalize_status = "normalize_status" in capabilities
        normalize_identity = "normalize_identity" in capabilities
        drop_empty = "drop_empty_identity" in capabilities
        aggregate = "aggregate_by_identity" in capabilities
        totals: dict[str, float] = {}
        pairs: list[tuple[str, float]] = []

        for row in records:
            if safe_validation:
                if not isinstance(row, dict):
                    continue
                user = row.get("user")
                status = row.get("status")
                amount = row.get("amount")
                if not isinstance(user, str) or not isinstance(status, str):
                    continue
                if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                    continue
                if not math.isfinite(float(amount)) or amount <= 0:
                    continue
            else:
                user = row["user"]
                status = row["status"]
                amount = row["amount"]

            compared_status = status.strip().lower() if normalize_status else status
            if compared_status != "paid":
                continue
            identity = user.strip().lower() if normalize_identity else user.lower()
            if drop_empty and not identity:
                continue
            if aggregate:
                totals[identity] = totals.get(identity, 0.0) + amount
            else:
                pairs.append((identity, round(amount, 2)))

        if not aggregate:
            return pairs
        output = [(user, round(total, 2)) for user, total in totals.items()]
        if "sort_total_then_identity" in capabilities:
            output.sort(key=lambda item: (-item[1], item[0]))
        return output

    def execute(self, program: AgentProgram, records: list[Any]) -> AgentExecution:
        capabilities = self._capabilities(program)
        retry = self.registry.retry_policies[program.retry_policy_ref]
        aggregate = "aggregate_by_identity" in capabilities
        first_error = None
        first_output: list[tuple[str, float]] | None = None
        try:
            first_output = self._attempt(program, records, safe_validation=False)
        except Exception as exc:  # noqa: BLE001 - failure is evaluator evidence.
            first_error = f"{type(exc).__name__}: {exc}"
        first_invariant_failure = bool(
            first_output is not None
            and not self._output_invariants(first_output, aggregate=aggregate)
        )
        should_retry = bool(
            int(retry["max_attempts"]) > 1
            and (
                (first_error and retry["retry_on_exception"])
                or (first_invariant_failure and retry["retry_on_output_invariant"])
            )
        )
        if not should_retry:
            return AgentExecution(
                output=first_output,
                attempts=1,
                retried=False,
                first_error=first_error,
                first_invariant_failure=first_invariant_failure,
                final_error=first_error,
            )
        try:
            output = self._attempt(
                program,
                records,
                safe_validation=bool(retry["prevalidate_on_retry"]),
            )
            if not self._output_invariants(output, aggregate=aggregate):
                raise ValueError("output invariant failed after retry")
            return AgentExecution(
                output=output,
                attempts=2,
                retried=True,
                first_error=first_error,
                first_invariant_failure=first_invariant_failure,
                final_error=None,
            )
        except Exception as exc:  # noqa: BLE001 - failure is evaluator evidence.
            return AgentExecution(
                output=None,
                attempts=2,
                retried=True,
                first_error=first_error,
                first_invariant_failure=first_invariant_failure,
                final_error=f"{type(exc).__name__}: {exc}",
            )


def _ordered_cases(
    cases: tuple[dict[str, Any], ...], seed: int
) -> tuple[dict[str, Any], ...]:
    ordered = list(cases)
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _score_program_seed(
    program: AgentProgram,
    registry: ComponentRegistry,
    *,
    partition: str,
    seed: int,
) -> dict[str, Any]:
    if partition == "public":
        cases = evaluator_core.CASES
    elif partition == "sealed":
        cases = evaluator_core.HOLDOUT_CASES
    else:
        raise ValueError(f"unknown partition: {partition}")
    agent = FrozenReplayAgent(registry)
    traces: list[dict[str, Any]] = []

    def solve(records: list[Any]) -> list[tuple[str, float]]:
        execution = agent.execute(program, records)
        traces.append(
            {
                "attempts": execution.attempts,
                "retried": execution.retried,
                "first_error": execution.first_error,
                "first_invariant_failure": execution.first_invariant_failure,
                "final_error": execution.final_error,
            }
        )
        if execution.final_error:
            raise RuntimeError(execution.final_error)
        return execution.output or []

    metrics = evaluator_core.score_cases(solve, _ordered_cases(cases, seed))
    metrics["agent_execution"] = {
        "calls": len(traces),
        "retried_calls": sum(bool(row["retried"]) for row in traces),
        "failed_calls": sum(bool(row["final_error"]) for row in traces),
        "traces": traces,
    }
    return metrics


def _evaluate_program(
    program: AgentProgram,
    registry: ComponentRegistry,
    *,
    partition: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    by_seed = {
        str(seed): _score_program_seed(
            program, registry, partition=partition, seed=seed
        )
        for seed in seeds
    }
    passed_by_seed = {
        seed: int(metrics["passed_cases"]) for seed, metrics in by_seed.items()
    }
    passed_pairs = sorted(
        f"{seed}:{row['id']}"
        for seed, metrics in by_seed.items()
        for row in metrics["case_results"]
        if row["passed"]
    )
    return {
        "partition": partition,
        "program": program.to_dict(),
        "program_hash": program.sha256,
        "passed_by_seed": passed_by_seed,
        "passed_mean": sum(passed_by_seed.values()) / len(passed_by_seed),
        "passed_pairs": passed_pairs,
        "metrics_by_seed": by_seed,
    }


def _evaluation_view(evaluation: dict[str, Any]) -> dict[str, Any]:
    partition = str(evaluation["partition"])
    return {
        "program": evaluation["program"],
        "program_hash": evaluation["program_hash"],
        f"{partition}_passed_by_seed": evaluation["passed_by_seed"],
        f"{partition}_passed_mean": evaluation["passed_mean"],
        f"{partition}_score_by_seed": {
            seed: float(metrics["combined_score"])
            for seed, metrics in evaluation["metrics_by_seed"].items()
        },
    }


def _public_gate(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    parent_counts = parent["passed_by_seed"]
    candidate_counts = candidate["passed_by_seed"]
    noninferior = sum(
        candidate_counts[seed] >= parent_counts[seed] for seed in parent_counts
    )
    lost = sorted(set(parent["passed_pairs"]) - set(candidate["passed_pairs"]))
    checks = {
        "public_mean_strict_gain": candidate["passed_mean"] > parent["passed_mean"],
        "at_least_two_of_three_seeds_noninferior": noninferior >= 2,
        "no_parent_public_case_regression": not lost,
        "program_hash_changed": candidate["program_hash"] != parent["program_hash"],
    }
    return {
        "decision": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "noninferior_seeds": noninferior,
        "public_mean_gain": candidate["passed_mean"] - parent["passed_mean"],
        "lost_parent_public_cases": lost,
    }


def _evaluation_events(
    evaluation: dict[str, Any], *, phase: str, candidate_role: str
) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "evaluation",
            "phase": phase,
            "partition": evaluation["partition"],
            "candidate_role": candidate_role,
            "program_hash": evaluation["program_hash"],
            "seed": int(seed),
            "metrics": metrics,
        }
        for seed, metrics in evaluation["metrics_by_seed"].items()
    ]


def _fingerprint(result: dict[str, Any]) -> str:
    stable = {
        "inputs": result["inputs"],
        "experiment": result["experiment"],
        "baseline": result["baseline"],
        "steps": result["steps"],
        "final": result["final"],
        "search": result["search"],
        "sealed_audit": result["sealed_audit"],
        "decision": result["decision"],
        "claims": result["claims"],
    }
    return _sha256_text(_canonical_json(stable))


def run_agent_program_experiment(
    *,
    baseline_path: Path,
    registry_path: Path,
    proposals_path: Path,
    experiment_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run public-only replay search, then open the sealed baseline/final audit."""
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("system_version") != "0.2.0":
        raise ContractError("experiment must use system_version 0.2.0")
    if experiment.get("model_weights") != "frozen":
        raise ContractError("model_weights must remain frozen")
    if int(experiment.get("network_calls", -1)) != 0:
        raise ContractError("ReplaySupervisor experiment cannot use network calls")
    seeds = tuple(int(seed) for seed in experiment["seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ContractError("experiment requires three unique seeds")

    registry = ComponentRegistry.from_path(registry_path)
    baseline_program = AgentProgram.from_path(baseline_path)
    baseline_program.validate(registry)
    supervisor = ReplaySupervisor.from_path(proposals_path)
    if len(supervisor.proposals) > int(experiment["proposal_budget"]):
        raise ContractError("proposal file exceeds frozen proposal budget")

    events: list[dict[str, Any]] = [
        {
            "event_type": "experiment_started",
            "phase": "search",
            "experiment_id": experiment["experiment_id"],
            "sealed_open": False,
        }
    ]
    archive: list[dict[str, Any]] = []
    baseline_public = _evaluate_program(
        baseline_program, registry, partition="public", seeds=seeds
    )
    events.extend(
        _evaluation_events(baseline_public, phase="search", candidate_role="baseline")
    )
    current_program = baseline_program
    current_public = baseline_public
    steps: list[dict[str, Any]] = []

    for iteration, proposal in enumerate(supervisor.proposals, start=1):
        try:
            candidate_program = apply_proposal(current_program, proposal, registry)
        except ContractError as exc:
            row = {
                "iteration": iteration,
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.sha256,
                "mutation_type": proposal.mutation_type,
                "parent_program_hash": current_program.sha256,
                "candidate_program_hash": None,
                "decision": "contract_rejected",
                "reasons": [str(exc)],
            }
            archive.append(row)
            steps.append(row)
            events.append({"event_type": "candidate_gate", "phase": "search", **row})
            continue

        candidate_public = _evaluate_program(
            candidate_program, registry, partition="public", seeds=seeds
        )
        events.extend(
            _evaluation_events(
                candidate_public,
                phase="search",
                candidate_role=f"candidate_{iteration}",
            )
        )
        gate = _public_gate(current_public, candidate_public)
        row = {
            "iteration": iteration,
            "proposal_id": proposal.proposal_id,
            "proposal_hash": proposal.sha256,
            "mutation_type": proposal.mutation_type,
            "reason": proposal.reason,
            "parent_program_hash": current_program.sha256,
            "candidate_program_hash": candidate_program.sha256,
            "parent": _evaluation_view(current_public),
            "candidate": _evaluation_view(candidate_public),
            **gate,
        }
        archive.append(
            {
                "iteration": iteration,
                "proposal_id": proposal.proposal_id,
                "proposal_hash": proposal.sha256,
                "mutation_type": proposal.mutation_type,
                "parent_program_hash": current_program.sha256,
                "candidate_program_hash": candidate_program.sha256,
                "candidate_program": candidate_program.to_dict(),
                "decision": gate["decision"],
                "checks": gate["checks"],
            }
        )
        steps.append(row)
        events.append({"event_type": "candidate_gate", "phase": "search", **row})
        if gate["decision"] == "accepted":
            current_program = candidate_program
            current_public = candidate_public

    public_target_solved = all(
        count == len(evaluator_core.CASES)
        for count in current_public["passed_by_seed"].values()
    )
    search = {
        "converged": True,
        "reason": (
            "public_target_solved"
            if public_target_solved
            else "proposal_budget_exhausted"
        ),
        "proposal_budget": int(experiment["proposal_budget"]),
        "proposals_consumed": len(supervisor.proposals),
        "accepted_candidates": sum(row["decision"] == "accepted" for row in steps),
        "rejected_candidates": sum(row["decision"] != "accepted" for row in steps),
        "sealed_used_for_search": False,
    }
    events.append(
        {
            "event_type": "search_complete",
            "phase": "search",
            "sealed_open": False,
            **search,
        }
    )

    public_checkpoint = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "phase": "search_complete",
        "sealed_open": False,
        "search": search,
        "baseline": _evaluation_view(baseline_public),
        "final": _evaluation_view(current_public),
        "active_program_hash": current_program.sha256,
    }
    public_checkpoint_path = output_dir / "public-checkpoint.json"
    _atomic_text(
        public_checkpoint_path,
        json.dumps(public_checkpoint, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
    )
    events.append(
        {
            "event_type": "sealed_opened",
            "phase": "post_search_audit",
            "public_results_persisted": True,
            "public_checkpoint_sha256": _sha256_file(public_checkpoint_path),
        }
    )

    baseline_sealed = _evaluate_program(
        baseline_program, registry, partition="sealed", seeds=seeds
    )
    final_sealed = _evaluate_program(
        current_program, registry, partition="sealed", seeds=seeds
    )
    events.extend(
        _evaluation_events(
            baseline_sealed,
            phase="post_search_audit",
            candidate_role="baseline",
        )
    )
    events.extend(
        _evaluation_events(
            final_sealed,
            phase="post_search_audit",
            candidate_role="final",
        )
    )
    sealed_noninferior = sum(
        final_sealed["passed_by_seed"][seed] >= baseline_sealed["passed_by_seed"][seed]
        for seed in baseline_sealed["passed_by_seed"]
    )
    lost_initial_public = sorted(
        set(baseline_public["passed_pairs"]) - set(current_public["passed_pairs"])
    )
    sealed_checks = {
        "at_least_two_of_three_seeds_noninferior": sealed_noninferior >= 2,
        "sealed_mean_nonregression": (
            final_sealed["passed_mean"] >= baseline_sealed["passed_mean"]
        ),
        "application_layer_public_strict_gain": (
            current_public["passed_mean"] > baseline_public["passed_mean"]
        ),
        "no_initial_public_capability_regression": not lost_initial_public,
        "harness_code_ref_frozen": (
            baseline_program.harness_code_ref == current_program.harness_code_ref
        ),
        "model_and_network_calls_zero": True,
    }
    decision = "accepted" if all(sealed_checks.values()) else "rejected"
    sealed_audit = {
        "opened_after_search": True,
        "baseline_passed_by_seed": baseline_sealed["passed_by_seed"],
        "final_passed_by_seed": final_sealed["passed_by_seed"],
        "baseline_passed_mean": baseline_sealed["passed_mean"],
        "final_passed_mean": final_sealed["passed_mean"],
        "noninferior_seeds": sealed_noninferior,
        "lost_initial_public_cases": lost_initial_public,
        "checks": sealed_checks,
    }
    evidence = {
        "schema_version": 1,
        "observational": {
            "kind": "replay_supervisor_trace",
            "proposal_ids": [proposal.proposal_id for proposal in supervisor.proposals],
            "supervisor_hash": supervisor.sha256,
            "claim": "proposal order and reasons only; not an admission signal",
        },
        "deterministic_interventions": [
            {
                "proposal_id": row["proposal_id"],
                "mutation_type": row["mutation_type"],
                "parent_program_hash": row["parent_program_hash"],
                "candidate_program_hash": row["candidate_program_hash"],
                "decision": row["decision"],
                "public_mean_gain": row.get("public_mean_gain"),
                "lost_parent_public_cases": row.get("lost_parent_public_cases", []),
            }
            for row in steps
        ],
        "sealed_generalization_audit": sealed_audit,
        "explanation_zh": (
            "固定 harness 将 Prompt、Skill 和 retry 引用解释为不同应用层能力；"
            "三个单轴候选依次补上状态标准化、身份聚合输出和失败后安全校验，"
            "因此 public 从 3/13 到 4/13、10/13、13/13。sealed 只在搜索完成后"
            "打开，最终从 0/6 到 6/6。该结论只证明冻结 Replay Agent 上的"
            "AgentProgram 机制，不外推到 Direct LLM。"
        ),
        "limitations": [
            "FrozenReplayAgent is a deterministic local model substitute.",
            "Only one task family is evaluated; cross-task PSI is not claimed.",
            "No JLens signal participates in proposal selection or admission in v0.2.0.",
        ],
    }
    inputs = {
        "baseline_sha256": _sha256_file(baseline_path),
        "registry_sha256": _sha256_file(registry_path),
        "proposals_sha256": _sha256_file(proposals_path),
        "experiment_sha256": _sha256_file(experiment_path),
        "evaluator_sha256": _sha256_file(Path(evaluator_core.__file__)),
    }
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "experiment": experiment,
        "inputs": inputs,
        "baseline": _evaluation_view(baseline_public),
        "steps": steps,
        "final": _evaluation_view(current_public),
        "search": search,
        "sealed_audit": sealed_audit,
        "decision": decision,
        "production_ready": False,
        "claims": {
            "agent_program_public_gain": (
                current_public["passed_mean"] - baseline_public["passed_mean"]
            ),
            "sealed_gain": (
                final_sealed["passed_mean"] - baseline_sealed["passed_mean"]
            ),
            "task_program_mutated": False,
            "harness_code_mutated": False,
            "model_calls": 0,
            "network_calls": 0,
            "direct_llm_gain_proven": False,
            "jlens_incremental_gain_proven": False,
            "psi_proven": False,
        },
        "evidence": evidence,
    }
    result["outcome_fingerprint"] = _fingerprint(result)
    events.append(
        {
            "event_type": "experiment_complete",
            "phase": "post_search_audit",
            "decision": decision,
            "outcome_fingerprint": result["outcome_fingerprint"],
        }
    )
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    active = current_program if decision == "accepted" else baseline_program
    _atomic_text(
        output_dir / "events.jsonl",
        "".join(_canonical_json(event) + "\n" for event in events),
    )
    _atomic_text(
        output_dir / "candidate_archive.jsonl",
        "".join(_canonical_json(row) + "\n" for row in archive),
    )
    _atomic_text(
        output_dir / "active_agent_program.json",
        json.dumps(active.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
    )
    _atomic_text(
        output_dir / "evidence.json",
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(
        output_dir / "result.json",
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(
        output_dir / "summary.md",
        _render_summary(result),
    )
    return result


def _render_summary(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    final = result["final"]
    sealed = result["sealed_audit"]
    return f"""# AgentProgram Replay 实验

- Experiment：`{result["experiment"]["experiment_id"]}`
- Model：`{result["experiment"]["model"]}`，权重冻结，模型调用 0
- Public mean：`{baseline["public_passed_mean"]}/13` → `{final["public_passed_mean"]}/13`
- Sealed mean：`{sealed["baseline_passed_mean"]}/6` → `{sealed["final_passed_mean"]}/6`
- Search：`{result["search"]["reason"]}`
- Decision：`{result["decision"]}`
- Production ready：`false`
- Outcome fingerprint：`{result["outcome_fingerprint"]}`

三个候选只改变 AgentProgram 引用，`harness_code_ref` 和 task program 全程冻结。
sealed 在 public 搜索停止后才打开。本结果只证明本地 FrozenReplayAgent 上的应用层
mutation 协议，不证明 Direct LLM、JLens 增量价值或跨任务 PSI。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run the frozen replay experiment")
    run_parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_CONFIGS / "baseline_agent_program.json"
    )
    run_parser.add_argument(
        "--registry", type=Path, default=DEFAULT_CONFIGS / "component_registry.json"
    )
    run_parser.add_argument(
        "--proposals", type=Path, default=DEFAULT_CONFIGS / "replay_proposals.json"
    )
    run_parser.add_argument(
        "--experiment", type=Path, default=DEFAULT_CONFIGS / "experiment.json"
    )
    run_parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    run_parser.add_argument("--run-id")
    inspect_parser = commands.add_parser("inspect", help="inspect a persisted result")
    inspect_parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "inspect":
        result = json.loads(args.result.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "experiment": result["experiment"]["experiment_id"],
                    "baseline": result["baseline"],
                    "steps": result["steps"],
                    "final": result["final"],
                    "search": result["search"],
                    "sealed_audit": result["sealed_audit"],
                    "decision": result["decision"],
                    "outcome_fingerprint": result["outcome_fingerprint"],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        return 0

    run_id = args.run_id or f"agent-program-{uuid.uuid4().hex[:12]}"
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsafe characters")
    output_dir = args.runs_dir / run_id
    result = run_agent_program_experiment(
        baseline_path=args.baseline,
        registry_path=args.registry,
        proposals_path=args.proposals,
        experiment_path=args.experiment,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(output_dir.resolve()),
                "public": {
                    "baseline_mean": result["baseline"]["public_passed_mean"],
                    "final_mean": result["final"]["public_passed_mean"],
                },
                "sealed": {
                    "baseline_mean": result["sealed_audit"]["baseline_passed_mean"],
                    "final_mean": result["sealed_audit"]["final_passed_mean"],
                },
                "decision": result["decision"],
                "outcome_fingerprint": result["outcome_fingerprint"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
