"""Assemble and run the authorized real multi-generation Codex evolution search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_arm_runner import profile_tree_hash
from codex_mutation_caller import CodexMutationCaller
from continuous_ab import BaselineContract
from evolution_controller import (
    EvolutionAuthorization,
    EvolutionController,
    EvolutionPlan,
)
from evolution_runtime import run_evolution_search
from official_patch_evaluator import OfficialPatchEvaluator
from real_evolution_bridge import RealEvolutionAdapters, load_search_task_schedule
from real_mutation_proposer import RealMutationProposerAdapter
from real_workspace_factory import GitWorkspaceFactory
from trace_observer import FrozenTrajectoryObserver


def _write_partial_result(*, output_dir: Path, run_root: Path, reason: str) -> Path:
    """Atomically write a partial terminal record from the persisted controller state."""
    output_dir = Path(output_dir)
    run_root = Path(run_root)
    state_path = run_root / "controller/STATE.json"
    digest_path = run_root / "controller/STATE.sha256"
    state: dict[str, Any] = {}
    digest: str | None = None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        digest = digest_path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    task_states = state.get("task_states", {})
    task_counts = {
        "retired": sum(
            1 for row in task_states.values() if row.get("state") == "retired"
        ),
        "claimed": sum(
            1 for row in task_states.values() if row.get("state") == "claimed"
        ),
        "unopened": sum(
            1 for row in task_states.values() if row.get("state") == "unopened"
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "partial",
        "terminal_state": "partial",
        "stop_reason": {
            "code": "signal",
            "signal": reason,
            "action": "graceful_partial_record_on_termination",
        },
        "controller_state_sha256": digest,
        "usage": state.get("usage"),
        "claims": {
            key: row.get("status") for key, row in state.get("claims", {}).items()
        },
        "task_counts": task_counts,
        "arm_evidence_count": len(state.get("arm_evidence", {})),
        "final_sealed_opened": state.get("final_sealed_opened"),
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "RESULT.partial.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(json_path)
    (output_dir / "RESULT.partial.zh-CN.md").write_text(
        "# real-search partial（信号终止）\n\n"
        f"- terminal_state: partial（stop_reason={reason}）\n"
        f"- controller STATE 摘要：{digest or 'unavailable'}\n"
        f"- usage: {json.dumps(state.get('usage', {}), ensure_ascii=False)}\n"
        "证据仍在本地 run root；恢复时以 controller/STATE.json 为准。\n",
        encoding="utf-8",
    )
    return json_path


def install_signal_partial_writer(*, run_root: Path) -> None:
    """TERM/INT -> write RESULT.partial atomically, then exit with the signal code."""
    run_root = Path(run_root)

    def _handler(signum: int, _frame: Any) -> None:
        try:
            _write_partial_result(
                output_dir=run_root, run_root=run_root, reason=f"signal_{signum}"
            )
        except Exception:  # noqa: S110, BLE001 - signal handler must not raise
            pass
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


class RealRunContractError(ValueError):
    """Raised before dispatch when the real run is not exactly frozen."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != payload:
            raise RealRunContractError("real run contract is immutable")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_baseline_authority(path: Path) -> BaselineContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("freeze_count") != 1
        or payload.get("immutable") is not True
    ):
        raise RealRunContractError("permanent baseline authority is invalid")
    baseline = BaselineContract.from_dict(payload["baseline"])
    if payload.get("baseline_contract_sha256") != baseline.contract_sha256:
        raise RealRunContractError("permanent baseline authority was tampered")
    return baseline


def _load_authorization(path: Path) -> tuple[dict[str, Any], EvolutionAuthorization]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "AUTHORIZED_WITH_HARD_CAPS":
        raise RealRunContractError("real execution authorization is missing")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise RealRunContractError("real execution authorization scope is invalid")
    authorization = EvolutionAuthorization.from_dict(scope)
    return payload, authorization


@dataclass(frozen=True)
class RealEvolutionAssembly:
    run_root: Path
    plan: EvolutionPlan
    authorization: EvolutionAuthorization
    baseline: BaselineContract
    seed_parent_agent_program_sha256: str
    controller: EvolutionController
    adapters: RealEvolutionAdapters
    worker_count: int = 1
    skill_registry: Path | None = None
    auto_gate: bool = False

    def execute(self) -> dict[str, Any]:
        return run_evolution_search(
            output_dir=self.run_root,
            plan=self.plan,
            authorization=self.authorization,
            original_agent_program_sha256=self.baseline.agent_program_sha256,
            seed_parent_agent_program_sha256=self.seed_parent_agent_program_sha256,
            native_evaluator_epoch=self.baseline.evaluator_epoch,
            adapters=self.adapters,
            controller=self.controller,
            worker_count=self.worker_count,
        )

    def finalize_skills(self, result: dict[str, Any]) -> dict[str, Any]:
        from search_skill_bridge import finalize_skill_bridge

        if self.skill_registry is None:
            return {"compiled": 0, "applied_gates": []}
        return finalize_skill_bridge(
            result=result,
            run_root=self.run_root,
            registry_root=self.skill_registry,
            auto_gate=self.auto_gate,
        )


def assemble_real_evolution_run(
    *,
    run_root: Path,
    schedule_path: Path,
    pool_root: Path,
    baseline_authority_path: Path,
    authorization_path: Path,
    original_profile_root: Path,
    seed_profile_root: Path,
    codex_executable: Path,
    swe_python: Path,
    multi_python: Path,
    swe_harness_root: Path,
    multi_harness_root: Path,
    project_root: Path,
    worker_count: int = 1,
    skill_registry: Path | None = None,
    auto_gate: bool = False,
) -> RealEvolutionAssembly:
    """Freeze every real adapter and budget without opening any task content."""

    paths = {
        "run_root": run_root.resolve(),
        "schedule": schedule_path.resolve(),
        "pool": pool_root.resolve(),
        "baseline_authority": baseline_authority_path.resolve(),
        "authorization": authorization_path.resolve(),
        "original_profile": original_profile_root.resolve(),
        "seed_profile": seed_profile_root.resolve(),
        "codex": codex_executable.resolve(),
        # Preserve venv launcher paths. Resolving these symlinks executes the
        # system interpreter and silently drops the harness environment.
        "swe_python": swe_python.absolute(),
        "multi_python": multi_python.absolute(),
        "swe_harness": swe_harness_root.resolve(),
        "multi_harness": multi_harness_root.resolve(),
        "project_root": project_root.resolve(),
    }
    for key in (
        "schedule",
        "baseline_authority",
        "authorization",
        "codex",
        "swe_python",
        "multi_python",
    ):
        if not paths[key].is_file():
            raise RealRunContractError(f"required real run file is missing: {key}")
    for key in (
        "pool",
        "original_profile",
        "seed_profile",
        "swe_harness",
        "multi_harness",
        "project_root",
    ):
        if not paths[key].is_dir():
            raise RealRunContractError(f"required real run directory is missing: {key}")

    schedule = load_search_task_schedule(paths["schedule"])
    is_resume = schedule["schema_version"] == "1.1-resume"
    task_uids = tuple(row["task_uid"] for row in schedule["tasks"])
    plan = (
        EvolutionPlan.build_resume(task_uids)
        if is_resume
        else EvolutionPlan.build(task_uids)
    )
    baseline = _load_baseline_authority(paths["baseline_authority"])
    authorization_payload, authorization = _load_authorization(paths["authorization"])
    authorization.validate(plan)
    original_hash = profile_tree_hash(paths["original_profile"])
    seed_hash = profile_tree_hash(paths["seed_profile"])
    if original_hash != baseline.agent_program_sha256:
        raise RealRunContractError("original profile differs from permanent baseline")
    if seed_hash == original_hash:
        raise RealRunContractError("seed parent must differ from permanent original")

    paths["run_root"].mkdir(parents=True, exist_ok=True)
    controller = EvolutionController.initialize(
        paths["run_root"] / "controller",
        plan=plan,
        authorization=authorization,
        original_agent_program_sha256=original_hash,
        seed_parent_agent_program_sha256=seed_hash,
        native_evaluator_epoch=baseline.evaluator_epoch,
    )
    profile_roots = {
        original_hash: paths["original_profile"],
        seed_hash: paths["seed_profile"],
    }
    model_caller = CodexMutationCaller(
        codex_executable=paths["codex"],
        output_root=paths["run_root"] / "proposer-calls",
        working_directory=paths["project_root"],
        model=baseline.model,
        reasoning=baseline.reasoning,
        timeout_seconds=baseline.timeout_seconds,
    )
    proposer = RealMutationProposerAdapter(
        profile_roots=profile_roots,
        output_root=paths["run_root"] / "proposals",
        provider={"platform": "codex", "model": baseline.model},
        model_call=model_caller,
        reserve_call=lambda reservation_id: controller.reserve_auxiliary_call(
            reservation_id=reservation_id, kind="mutation_proposer"
        )["dispatch_allowed"],
        complete_call=lambda reservation_id, response_sha256: (
            controller.complete_auxiliary_call(
                reservation_id=reservation_id, evidence_sha256=response_sha256
            )
        ),
    )
    native_evaluator = OfficialPatchEvaluator(
        swe_python=paths["swe_python"],
        multi_python=paths["multi_python"],
        swe_harness_root=paths["swe_harness"],
        multi_harness_root=paths["multi_harness"],
        pool_root=paths["pool"],
        output_root=paths["run_root"] / "native-evaluator",
    )
    observer = FrozenTrajectoryObserver()
    adapters = RealEvolutionAdapters(
        controller=controller,
        schedule_path=paths["schedule"],
        pool_root=paths["pool"],
        baseline=baseline,
        profile_roots=profile_roots,
        run_root=paths["run_root"],
        codex_executable=paths["codex"],
        authorization=authorization_payload,
        workspace_factory=GitWorkspaceFactory(root=paths["run_root"] / "workspaces"),
        native_evaluator=native_evaluator,
        observer=observer,
        proposer=proposer,
    )
    contract = {
        "schema_version": 1,
        "status": "frozen_before_real_task_dispatch",
        "schedule": {
            "path": str(paths["schedule"]),
            "sha256": _sha256_file(paths["schedule"]),
            "semantic_fingerprint": schedule["semantic_fingerprint"],
            "unique_search_tasks": len(task_uids),
        },
        "resume": schedule.get("resume"),
        "baseline": {
            "authority_path": str(paths["baseline_authority"]),
            "authority_sha256": _sha256_file(paths["baseline_authority"]),
            "contract_sha256": baseline.contract_sha256,
            "original_agent_program_sha256": original_hash,
            "model": baseline.model,
            "reasoning": baseline.reasoning,
            "native_evaluator_epoch": baseline.evaluator_epoch,
        },
        "seed_parent_agent_program_sha256": seed_hash,
        "authorization": authorization_payload,
        "planned_agent_task_calls": plan.planned_agent_task_calls,
        "planned_real_codex_calls": plan.planned_real_codex_calls,
        "observer": {
            "kind": observer.observer_kind,
            "hidden_state_access": observer.hidden_state_access,
            "admission_gate_allowed": observer.admission_gate_allowed,
        },
        "model_weights_frozen": True,
        "final_sealed_opened": False,
        "production_promotion_allowed": False,
        "global_skill_install_allowed": False,
    }
    contract["semantic_fingerprint"] = hashlib.sha256(
        _canonical_json(contract).encode()
    ).hexdigest()
    _write_immutable_json(paths["run_root"] / "RUN-CONTRACT.json", contract)
    return RealEvolutionAssembly(
        run_root=paths["run_root"],
        plan=plan,
        authorization=authorization,
        baseline=baseline,
        seed_parent_agent_program_sha256=seed_hash,
        controller=controller,
        adapters=adapters,
        worker_count=worker_count,
        skill_registry=skill_registry,
        auto_gate=auto_gate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--baseline-authority", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--original-profile", type=Path, required=True)
    parser.add_argument("--seed-profile", type=Path, required=True)
    parser.add_argument("--codex-executable", type=Path, required=True)
    parser.add_argument("--swe-python", type=Path, required=True)
    parser.add_argument("--multi-python", type=Path, required=True)
    parser.add_argument("--swe-harness-root", type=Path, required=True)
    parser.add_argument("--multi-harness-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="candidate/task execution concurrency inside each claimed stage (1-4; 2vCPU instance: use 2)",
    )
    parser.add_argument(
        "--skill-registry",
        type=Path,
        default=None,
        help="after the search, compile selected candidates into this project-local skill registry",
    )
    parser.add_argument(
        "--auto-gate",
        action="store_true",
        help="with --skill-registry, apply the cross-task transfer gate using confirmation-stage paired evidence",
    )
    args = parser.parse_args(argv)
    install_signal_partial_writer(run_root=args.run_root)
    assembly = assemble_real_evolution_run(
        run_root=args.run_root,
        schedule_path=args.schedule,
        pool_root=args.pool_root,
        baseline_authority_path=args.baseline_authority,
        authorization_path=args.authorization,
        original_profile_root=args.original_profile,
        seed_profile_root=args.seed_profile,
        codex_executable=args.codex_executable,
        swe_python=args.swe_python,
        multi_python=args.multi_python,
        swe_harness_root=args.swe_harness_root,
        multi_harness_root=args.multi_harness_root,
        project_root=args.project_root,
        worker_count=args.worker_count,
        skill_registry=args.skill_registry,
        auto_gate=args.auto_gate,
    )
    if args.preflight_only:
        payload = {
            "status": "preflight_complete",
            "run_root": str(assembly.run_root),
            "planned_real_codex_calls": assembly.plan.planned_real_codex_calls,
            "final_sealed_opened": False,
        }
    else:
        result = assembly.execute()
        skills = assembly.finalize_skills(result)
        payload = {
            "status": result["status"],
            "run_root": str(assembly.run_root),
            "real_codex_calls": result["usage"]["real_codex_calls"],
            "final_sealed_opened": result["final_sealed_opened"],
            "skill_candidates_compiled": skills.get("compiled", 0)
            if isinstance(skills, dict)
            else 0,
            "skill_gates_applied": len(skills.get("applied_gates", []))
            if isinstance(skills, dict)
            else 0,
        }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
