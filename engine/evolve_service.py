"""Stable v1 CLI for local Agent-layer self-evolution and evidence inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from artifact_verifier import DEFAULT_MANIFEST, verify_manifest
from codex_evolution_runtime import DEFAULT_CONTRACT as DEFAULT_CODEX_CONTRACT
from codex_evolution_runtime import DEFAULT_PROFILE as DEFAULT_CODEX_PROFILE
from codex_evolution_runtime import run_codex_evolution
from evolution_controller import EvolutionController, EvolutionPlan
from meta_evolution_runtime import DEFAULT_PROGRAMS as DEFAULT_META_PROGRAMS
from meta_evolution_runtime import run_meta_evolution
from multi_model_eval import probe_model_registry
from multi_model_eval import run_suite as run_multi_model_suite
from release_candidate import DEFAULT_CONFIG as DEFAULT_RC_CONFIG
from release_candidate import run_release_candidate
from swe_bench_adapter import probe_environment as probe_swe_environment

ROOT = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/v1.0.0"
DEFAULT_BENCHMARK_ROOT = (
    ROOT / "artifacts/v2.0.0/v2.0.0-meta-evolution/benchmark-suite-001/configs"
)
DEFAULT_MODEL_REGISTRY = DEFAULT_BENCHMARK_ROOT / "model-registry.json"
DEFAULT_EVAL_SUITE = DEFAULT_BENCHMARK_ROOT / "eval-suite.json"


class ServiceCLIError(ValueError):
    """Raised when a CLI request violates a stable local service contract."""


def inspect_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ServiceCLIError(f"result does not exist: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = result.get("contract_checks", {})
    if not isinstance(checks, dict):
        raise ServiceCLIError("result contract_checks must be a mapping")
    return {
        "path": str(path.resolve()),
        "schema_version": result.get("schema_version"),
        "stage": result.get("stage"),
        "decision": result.get("decision"),
        "experiment_fingerprint": result.get("experiment_fingerprint"),
        "checks_passed": sum(value is True for value in checks.values()),
        "checks_total": len(checks),
        "claims": result.get("claims", {}),
    }


def rollback_plan(*, kind: str, evidence_root: Path) -> dict[str, Any]:
    root = evidence_root.resolve()
    specs = {
        "agent-program": {
            "paths": [
                "v0.2.0-agent-program/DECISION.json",
                "v0.2.0-agent-program/runs/agent-program-final-pass3-3/active_agent_program.json",
            ],
            "steps": [
                "select an archived parent AgentProgram hash",
                "create a new durable operation id",
                "rerun public and post-search sealed admission before publishing active ref",
            ],
        },
        "agent-code": {
            "paths": [
                "v0.5.0-agent-code-mutation/DECISION.json",
                "v0.5.0-agent-code-mutation/runs/code-pass3-3/archive/active-transitions.jsonl",
            ],
            "steps": [
                "select the verified parent source hash from lineage",
                "create a new rollback transition",
                "replay capability public sealed checks before active publication",
            ],
        },
        "skill": {
            "paths": [
                "v0.4.0-psi-skill-library/DECISION.json",
                "v0.4.0-psi-skill-library/runs/psi-pass3-3/registry/registry.jsonl",
            ],
            "steps": [
                "keep the candidate inactive or append a rejected revision",
                "preserve negative-transfer evidence",
                "require a new matched cross-task audit before any future promotion",
            ],
        },
        "evaluator": {
            "paths": [
                "v0.6.0-evaluator-shadow/DECISION.json",
                "v0.6.0-evaluator-shadow/runs/shadow-pass3-3/review-proposal.json",
            ],
            "steps": [
                "keep the current anchor epoch unchanged",
                "prepare an epoch-boundary reverse proposal referencing the old anchor hash",
                "require human review and cross-play before any switch",
            ],
        },
    }
    if kind not in specs:
        raise ServiceCLIError(f"unsupported rollback kind: {kind}")
    evidence_paths = [(root / relative).resolve() for relative in specs[kind]["paths"]]
    if any(
        not path.is_relative_to(root) or not path.is_file() for path in evidence_paths
    ):
        raise ServiceCLIError(f"rollback evidence is incomplete for {kind}")
    return {
        "kind": kind,
        "applied": False,
        "requires_new_operation": True,
        "requires_human_review": kind == "evaluator",
        "evidence_paths": [str(path) for path in evidence_paths],
        "steps": specs[kind]["steps"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one full v0.9 RC matrix")
    run.add_argument("--config", type=Path, default=DEFAULT_RC_CONFIG)
    run.add_argument("--output", type=Path, required=True)
    codex_run = commands.add_parser(
        "codex-run", help="run one real-history Codex application-layer evolution"
    )
    codex_run.add_argument("--contract", type=Path, default=DEFAULT_CODEX_CONTRACT)
    codex_run.add_argument("--profile", type=Path, default=DEFAULT_CODEX_PROFILE)
    codex_run.add_argument("--sessions-root", type=Path, default=None)
    codex_run.add_argument("--output", type=Path, required=True)
    meta_run = commands.add_parser(
        "meta-run", help="run bounded v2 MetaProgram evolution"
    )
    meta_run.add_argument("--contract", type=Path, default=DEFAULT_CODEX_CONTRACT)
    meta_run.add_argument("--profile", type=Path, default=DEFAULT_CODEX_PROFILE)
    meta_run.add_argument("--sessions-root", type=Path, default=None)
    meta_run.add_argument("--programs", type=Path, default=DEFAULT_META_PROGRAMS)
    meta_run.add_argument("--output", type=Path, required=True)
    model_probe = commands.add_parser(
        "model-probe",
        help="probe configured local model adapters without loading weights",
    )
    model_probe.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    benchmark_run = commands.add_parser(
        "benchmark-run", help="run the frozen local multi-model diagnostic matrix"
    )
    benchmark_run.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    benchmark_run.add_argument("--suite", type=Path, default=DEFAULT_EVAL_SUITE)
    benchmark_run.add_argument("--output", type=Path, required=True)
    swe_probe = commands.add_parser(
        "swe-probe", help="probe official SWE-bench harness runtime prerequisites"
    )
    swe_probe.add_argument("--path", type=Path, default=ROOT)
    inspect = commands.add_parser("inspect", help="inspect an existing result")
    inspect.add_argument("--result", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify the artifact manifest")
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    rollback = commands.add_parser(
        "rollback-plan", help="produce a non-mutating rollback plan"
    )
    rollback.add_argument(
        "--kind",
        required=True,
        choices=["agent-program", "agent-code", "skill", "evaluator"],
    )
    rollback.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    evolution_plan = commands.add_parser(
        "evolution-plan",
        help="inspect the frozen 100-task multi-generation plan without external actions",
    )
    evolution_plan.add_argument("--tasks", type=Path, required=True)
    evolution_inspect = commands.add_parser(
        "evolution-inspect", help="inspect one local multi-generation controller"
    )
    evolution_inspect.add_argument("--root", type=Path, required=True)
    evolution_verify = commands.add_parser(
        "evolution-verify", help="verify one local multi-generation controller"
    )
    evolution_verify.add_argument("--root", type=Path, required=True)
    skill_candidates = commands.add_parser(
        "skill-candidates",
        help=(
            "compile v2.2 search results into project-local candidate skills and "
            "optionally apply the cross-task transfer gate"
        ),
    )
    skill_candidates.add_argument("--run-root", type=Path, required=True)
    skill_candidates.add_argument("--registry", type=Path, required=True)
    skill_candidates.add_argument(
        "--paired-evidences",
        type=Path,
        default=None,
        help=(
            "optional JSON: {skill_id: {evals: [...], expected_contract_sha256: ..., "
            "expected_evaluator_epoch: ...}}; applies the transfer gate per skill"
        ),
    )
    ladder = commands.add_parser(
        "skill-ladder",
        help=(
            "v2.5 human review ladder: record review/active/rejected decisions "
            "for a skill candidate (append-only, never auto-active)"
        ),
    )
    ladder.add_argument("--registry", type=Path, required=True)
    ladder.add_argument("--skill-id", required=True)
    ladder.add_argument("--revision-id", required=True)
    ladder.add_argument(
        "--decision", choices=("reviewed", "active", "rejected"), required=True
    )
    ladder.add_argument("--reviewer", required=True)
    ladder.add_argument("--notes", default="")
    ladder_status = commands.add_parser(
        "skill-ladder-status",
        help="show effective promotion status of one skill candidate",
    )
    ladder_status.add_argument("--registry", type=Path, required=True)
    ladder_status.add_argument("--skill-id", required=True)
    ladder_status.add_argument("--revision-id", required=True)
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        if args.output.exists() and any(args.output.iterdir()):
            raise ServiceCLIError(f"refusing non-empty output: {args.output}")
        result = run_release_candidate(config_path=args.config, output_dir=args.output)
        payload = {
            "decision": result["decision"],
            "experiment_fingerprint": result["experiment_fingerprint"],
            "output": str(args.output.resolve()),
        }
        code = 0 if result["decision"] == "accepted" else 1
    elif args.command == "codex-run":
        result = run_codex_evolution(
            contract_path=args.contract,
            baseline_root=args.profile,
            output_dir=args.output,
            sessions_root=args.sessions_root,
        )
        payload = {
            "decision": result["decision"],
            "experiment_fingerprint": result["experiment_fingerprint"],
            "output": str(args.output.resolve()),
            "report": str((args.output / "REPORT.zh-CN.md").resolve()),
            "rollback_patch": str((args.output / "changeset/rollback.patch").resolve()),
        }
        code = 0 if result["decision"] == "accepted" else 1
    elif args.command == "meta-run":
        result = run_meta_evolution(
            contract_path=args.contract,
            baseline_root=args.profile,
            programs_path=args.programs,
            output_dir=args.output,
            sessions_root=args.sessions_root,
        )
        payload = {
            "release_decision": result["release_decision"],
            "agentic_rsi_decision": result["agentic_rsi_decision"],
            "experiment_fingerprint": result["experiment_fingerprint"],
            "output": str(args.output.resolve()),
            "report": str((args.output / "REPORT.zh-CN.md").resolve()),
        }
        code = 0 if result["release_decision"] == "accepted" else 1
    elif args.command == "model-probe":
        models = probe_model_registry(args.registry)
        blocked_required = [
            model
            for model in models
            if model["required"] and model["enabled"] and model["status"] != "available"
        ]
        payload = {"models": models, "blocked_required": blocked_required}
        code = 1 if blocked_required else 0
    elif args.command == "benchmark-run":
        result = run_multi_model_suite(
            registry_path=args.registry,
            suite_path=args.suite,
            output_root=args.output,
        )
        payload = {
            "summary": result["summary"],
            "cells": len(result["cells"]),
            "output": str(args.output.resolve()),
            "report": str((args.output / "REPORT.zh-CN.md").resolve()),
        }
        code = 0
    elif args.command == "swe-probe":
        payload = probe_swe_environment(args.path)
        code = 0 if payload["ready"] else 1
    elif args.command == "inspect":
        payload = inspect_result(args.result)
        code = 0 if payload["decision"] == "accepted" else 1
    elif args.command == "verify":
        payload = verify_manifest(args.manifest)
        code = 0 if payload["valid"] else 1
    elif args.command == "rollback-plan":
        payload = rollback_plan(kind=args.kind, evidence_root=args.evidence_root)
        code = 0
    elif args.command == "evolution-plan":
        task_payload = json.loads(args.tasks.read_text(encoding="utf-8"))
        if not isinstance(task_payload, dict) or set(task_payload) != {"task_uids"}:
            raise ServiceCLIError("evolution task file must contain only task_uids")
        plan = EvolutionPlan.build(tuple(task_payload["task_uids"]))
        payload = {**plan.to_dict(), "external_actions": 0}
        code = 0
    elif args.command == "evolution-inspect":
        payload = EvolutionController(args.root).inspect()
        code = 0
    elif args.command == "skill-candidates":
        from search_skill_bridge import (
            apply_transfer_gate,
            compile_candidate_skills,
            evaluate_transfer_gate,
        )
        from skill_registry import SkillRegistry

        result = json.loads((args.run_root / "RESULT.json").read_text(encoding="utf-8"))
        candidates = compile_candidate_skills(result=result, run_root=args.run_root)
        registry = SkillRegistry(args.registry)
        for candidate in candidates:
            registry.append(candidate)
        gate_payload: dict[str, Any] = {}
        if args.paired_evidences is not None:
            gate_payload = json.loads(args.paired_evidences.read_text(encoding="utf-8"))
        applied = []
        for candidate in candidates:
            entry = gate_payload.get(candidate.skill_id)
            if entry is None:
                continue
            gate = evaluate_transfer_gate(
                paired_evals=entry.get("evals", []),
                expected_contract_sha256=entry.get("expected_contract_sha256"),
                expected_evaluator_epoch=entry.get("expected_evaluator_epoch"),
            )
            gate_path = args.run_root / f"TRANSFER-GATE-{candidate.skill_id}.json"
            gate_path.write_text(
                json.dumps(gate.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            terminal = apply_transfer_gate(
                registry=registry,
                candidate=candidate,
                gate_result=gate,
                gate_evidence_path=gate_path,
            )
            registry.render_for_review(candidate.skill_id)
            applied.append(terminal.to_dict())
        payload = {
            "compiled": [candidate.to_dict() for candidate in candidates],
            "applied_gates": applied,
        }
        (args.run_root / "CANDIDATE-SKILLS.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        code = 0
    elif args.command == "skill-ladder":
        from promotion_ladder import PromotionLadder, ReviewDecision

        ladder = PromotionLadder(args.registry)
        decision = ReviewDecision.create(
            skill_id=args.skill_id,
            revision_id=args.revision_id,
            decision=args.decision,
            reviewer=args.reviewer,
            notes=args.notes,
        )
        ladder.record(decision)
        payload = {
            "recorded": decision.to_dict(),
            "effective_status": ladder.effective_status(
                args.skill_id, args.revision_id
            ),
        }
        code = 0
    elif args.command == "skill-ladder-status":
        from promotion_ladder import PromotionLadder

        ladder = PromotionLadder(args.registry)
        payload = ladder.review_summary(args.skill_id, args.revision_id)
        code = 0
    else:
        payload = EvolutionController(args.root).verify()
        code = 0 if payload["valid"] else 1
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return code


def main() -> int:
    try:
        return run_cli()
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
