"""Command-line diagnostics for loop infrastructure."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def doctor(root: Path) -> dict[str, Any]:
    """Run deterministic, offline checks required before model execution."""
    resolved = root.resolve()
    checks = {
        "python_supported": (3, 12) <= sys.version_info < (3, 14),
        "root_exists": resolved.is_dir(),
        "root_writable": resolved.is_dir() and os.access(resolved, os.W_OK),
        "dependency_manifest": (resolved / "pyproject.toml").is_file(),
        "package_importable": True,
    }
    return {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "root": str(resolved),
        "python": sys.version.split()[0],
        "network_calls_performed": False,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m skill_evolution_loop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor", help="check offline infrastructure prerequisites"
    )
    doctor_parser.add_argument("--root", type=Path, default=Path.cwd())
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")
    smoke_parser = subparsers.add_parser(
        "offline-smoke", help="run the complete loop with zero-network fixtures"
    )
    smoke_parser.add_argument("--out", type=Path, required=True)
    smoke_parser.add_argument("--json", action="store_true", dest="as_json")
    cuda_replay_parser = subparsers.add_parser(
        "cuda-calibration-replay",
        help="replay frozen rendered prompts through a remote model transport",
    )
    cuda_replay_parser.add_argument("--manifest", type=Path, required=True)
    cuda_replay_parser.add_argument("--out", type=Path, required=True)
    cuda_replay_parser.add_argument("--transport-base-url", required=True)
    cuda_replay_parser.add_argument("--transport-model", required=True)
    cuda_replay_parser.add_argument("--transport-api-key-env", default=None)
    cuda_replay_parser.add_argument("--max-tokens", type=int, default=1536)
    cuda_replay_parser.add_argument("--json", action="store_true", dest="as_json")
    cuda_collect_parser = subparsers.add_parser(
        "cuda-calibration-collect",
        help="collect formal runner cells and verify prompt identity",
    )
    cuda_collect_parser.add_argument("--manifest", type=Path, required=True)
    cuda_collect_parser.add_argument("--experiment", type=Path, required=True)
    cuda_collect_parser.add_argument("--out", type=Path, required=True)
    cuda_collect_parser.add_argument("--json", action="store_true", dest="as_json")
    cuda_evaluate_parser = subparsers.add_parser(
        "cuda-calibration-evaluate",
        help="evaluate the frozen three-task MLX-versus-CUDA gate",
    )
    cuda_evaluate_parser.add_argument("--manifest", type=Path, required=True)
    cuda_evaluate_parser.add_argument("--cuda-results", type=Path, required=True)
    cuda_evaluate_parser.add_argument("--out", type=Path, required=True)
    cuda_evaluate_parser.add_argument("--json", action="store_true", dest="as_json")
    convergence_project_parser = subparsers.add_parser(
        "convergence-project",
        help="project normalized convergence over frozen paired comparisons",
    )
    convergence_project_parser.add_argument("--source", type=Path, required=True)
    convergence_project_parser.add_argument("--out", type=Path, required=True)
    convergence_project_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    safety_project_parser = subparsers.add_parser(
        "independent-safety-project",
        help="aggregate four independent Agent safety probe receipts",
    )
    safety_project_parser.add_argument("--subject-sha256", required=True)
    safety_project_parser.add_argument(
        "--probe", action="append", dest="probes", type=Path, required=True
    )
    safety_project_parser.add_argument("--out", type=Path, required=True)
    safety_project_parser.add_argument("--json", action="store_true", dest="as_json")
    statistical_project_parser = subparsers.add_parser(
        "statistical-capability-project",
        help="evaluate the strict statistical Skill-transfer completion gate",
    )
    statistical_project_parser.add_argument("--feedback", type=Path, required=True)
    statistical_project_parser.add_argument("--holdout", type=Path, required=True)
    statistical_project_parser.add_argument(
        "--independent-safety", type=Path, required=True
    )
    statistical_project_parser.add_argument(
        "--runtime-identity", type=Path, required=True
    )
    statistical_project_parser.add_argument("--cost-receipt", type=Path, required=True)
    statistical_project_parser.add_argument("--catalog-audit", type=Path, required=True)
    statistical_project_parser.add_argument("--out", type=Path, required=True)
    statistical_project_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    runtime_identity_parser = subparsers.add_parser(
        "runtime-identity-project",
        help="freeze one complete, immutable student runtime identity",
    )
    runtime_identity_parser.add_argument("--source", type=Path, required=True)
    runtime_identity_parser.add_argument("--out", type=Path, required=True)
    runtime_identity_parser.add_argument("--json", action="store_true", dest="as_json")
    accelerator_preflight_parser = subparsers.add_parser(
        "accelerator-preflight-project",
        help="freeze a provider-neutral accelerator readiness receipt",
    )
    accelerator_preflight_parser.add_argument("--source", type=Path, required=True)
    accelerator_preflight_parser.add_argument("--out", type=Path, required=True)
    accelerator_preflight_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    freeze_parser = subparsers.add_parser(
        "freeze-local-p1-taskset",
        help="freeze the existing local 3+3 Qwen pilot taskset",
    )
    freeze_parser.add_argument("--root", type=Path, default=Path.cwd())
    freeze_parser.add_argument("--out", type=Path, required=True)
    freeze_parser.add_argument("--json", action="store_true", dest="as_json")
    freeze_v2_parser = subparsers.add_parser(
        "freeze-local-p1-v2",
        help="freeze the target-qualified local 3+3 taskset and provenance",
    )
    freeze_v2_parser.add_argument("--root", type=Path, default=Path.cwd())
    freeze_v2_parser.add_argument("--taskset-out", type=Path, required=True)
    freeze_v2_parser.add_argument("--target-selection-out", type=Path, required=True)
    freeze_v2_parser.add_argument("--json", action="store_true", dest="as_json")
    freeze_v3_parser = subparsers.add_parser(
        "freeze-local-p1-v3",
        help="freeze the single-file-capacity-qualified local 3+3 bundle",
    )
    freeze_v3_parser.add_argument("--root", type=Path, default=Path.cwd())
    freeze_v3_parser.add_argument("--taskset-out", type=Path, required=True)
    freeze_v3_parser.add_argument("--target-selection-out", type=Path, required=True)
    freeze_v3_parser.add_argument("--json", action="store_true", dest="as_json")
    freeze_v4_parser = subparsers.add_parser(
        "freeze-local-p1-v4",
        help="freeze the dual-mechanism-capacity-qualified local 3+3 bundle",
    )
    freeze_v4_parser.add_argument("--root", type=Path, default=Path.cwd())
    freeze_v4_parser.add_argument("--taskset-out", type=Path, required=True)
    freeze_v4_parser.add_argument("--target-selection-out", type=Path, required=True)
    freeze_v4_parser.add_argument("--json", action="store_true", dest="as_json")
    preflight_parser = subparsers.add_parser(
        "taskset-preflight", help="validate a frozen evaluation taskset"
    )
    preflight_parser.add_argument("--manifest", type=Path, required=True)
    preflight_parser.add_argument("--json", action="store_true", dest="as_json")
    target_audit_parser = subparsers.add_parser(
        "taskset-target-audit",
        help="run an evaluator-only reference target coverage audit",
    )
    target_audit_parser.add_argument("--manifest", type=Path, required=True)
    target_audit_parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="repeatable TASK_ID=FIX_PATCH_PATH evaluator-only reference",
    )
    target_audit_parser.add_argument("--out", type=Path, required=True)
    target_audit_parser.add_argument("--json", action="store_true", dest="as_json")
    capacity_parser = subparsers.add_parser(
        "taskset-mechanism-audit",
        help="audit reference fixes against structured and hunk output limits",
    )
    capacity_parser.add_argument("--manifest", type=Path, required=True)
    capacity_parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="repeatable TASK_ID=FIX_PATCH_PATH evaluator-only reference",
    )
    capacity_parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("models/Qwen3.5-4B-mlx-4bit/tokenizer.json"),
    )
    capacity_parser.add_argument("--structured-max-tokens", type=int, default=512)
    capacity_parser.add_argument("--out", type=Path, required=True)
    capacity_parser.add_argument("--json", action="store_true", dest="as_json")
    p1_parser = subparsers.add_parser(
        "p1-run", help="run or resume the real local-Qwen paired pilot"
    )
    p1_parser.add_argument("--manifest", type=Path, required=True)
    skill_group = p1_parser.add_mutually_exclusive_group(required=True)
    skill_group.add_argument("--skill", type=Path)
    skill_group.add_argument("--skill-revision", type=Path)
    p1_parser.add_argument("--target-selection", type=Path, default=None)
    p1_parser.add_argument("--out", type=Path, required=True)
    p1_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/private/tmp/jlens-p1-workspaces"),
    )
    p1_parser.add_argument(
        "--model", type=Path, default=Path("models/Qwen3.5-4B-mlx-4bit")
    )
    p1_parser.add_argument("--max-cells", type=int, default=None)
    p1_parser.add_argument("--structured-max-tokens", type=int, default=768)
    p1_parser.add_argument("--structured-context-chars", type=int, default=80_000)
    p1_parser.add_argument("--hunk-max-tokens", type=int, default=512)
    p1_parser.add_argument("--task", action="append", dest="tasks")
    p1_parser.add_argument("--condition", action="append", dest="conditions")
    p1_parser.add_argument("--json", action="store_true", dest="as_json")
    symbol_parser = subparsers.add_parser(
        "p1-symbol-run",
        help="run or resume the AST-anchored local-Qwen symbol pilot",
    )
    symbol_parser.add_argument("--manifest", type=Path, required=True)
    symbol_parser.add_argument("--skill-revision", type=Path, required=True)
    symbol_parser.add_argument("--target-selection", type=Path, required=True)
    symbol_parser.add_argument("--out", type=Path, required=True)
    symbol_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/private/tmp/jlens-p1-symbol-workspaces"),
    )
    symbol_parser.add_argument(
        "--model", type=Path, default=Path("models/Qwen3.5-4B-mlx-4bit")
    )
    symbol_parser.add_argument("--max-cells", type=int, default=None)
    symbol_parser.add_argument("--max-tokens", type=int, default=1536)
    symbol_parser.add_argument("--context-chars", type=int, default=8_000)
    symbol_parser.add_argument("--task", action="append", dest="tasks")
    symbol_parser.add_argument("--condition", action="append", dest="conditions")
    symbol_parser.add_argument("--json", action="store_true", dest="as_json")
    block_parser = subparsers.add_parser(
        "p1-block-run",
        help="run or resume the numbered line-block local-Qwen pilot",
    )
    block_parser.add_argument("--manifest", type=Path, required=True)
    block_parser.add_argument("--skill-revision", type=Path, required=True)
    block_parser.add_argument("--target-selection", type=Path, required=True)
    block_parser.add_argument("--out", type=Path, required=True)
    block_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/private/tmp/jlens-p1-block-workspaces"),
    )
    block_parser.add_argument(
        "--model", type=Path, default=Path("models/Qwen3.5-4B-mlx-4bit")
    )
    block_parser.add_argument("--max-cells", type=int, default=None)
    block_parser.add_argument("--max-tokens", type=int, default=768)
    block_parser.add_argument("--context-chars", type=int, default=8_000)
    block_parser.add_argument("--task", action="append", dest="tasks")
    block_parser.add_argument("--condition", action="append", dest="conditions")
    block_parser.add_argument("--json", action="store_true", dest="as_json")
    native_parser = subparsers.add_parser(
        "p1-native", help="run or resume official native evaluation for P1 cells"
    )
    native_parser.add_argument("--manifest", type=Path, required=True)
    native_parser.add_argument("--experiment", type=Path, required=True)
    native_parser.add_argument("--out", type=Path, required=True)
    native_parser.add_argument("--official-out", type=Path, required=True)
    native_parser.add_argument("--pool-root", type=Path, required=True)
    native_parser.add_argument("--swe-python", type=Path, required=True)
    native_parser.add_argument("--multi-python", type=Path, required=True)
    native_parser.add_argument("--swe-harness", type=Path, required=True)
    native_parser.add_argument("--multi-harness", type=Path, required=True)
    native_parser.add_argument("--native-assets", type=Path, default=None)
    native_parser.add_argument("--timeout-seconds", type=int, default=7200)
    native_parser.add_argument("--max-cells", type=int, default=None)
    native_parser.add_argument("--json", action="store_true", dest="as_json")
    native_cell_parser = subparsers.add_parser(
        "p1-native-cell",
        help="run an official native judge for one frozen feedback cell",
    )
    native_cell_parser.add_argument("--manifest", type=Path, required=True)
    native_cell_parser.add_argument("--experiment", type=Path, required=True)
    native_cell_parser.add_argument("--out", type=Path, required=True)
    native_cell_parser.add_argument("--official-out", type=Path, required=True)
    native_cell_parser.add_argument("--pool-root", type=Path, required=True)
    native_cell_parser.add_argument("--swe-python", type=Path, required=True)
    native_cell_parser.add_argument("--multi-python", type=Path, required=True)
    native_cell_parser.add_argument("--swe-harness", type=Path, required=True)
    native_cell_parser.add_argument("--multi-harness", type=Path, required=True)
    native_cell_parser.add_argument("--native-assets", type=Path, default=None)
    native_cell_parser.add_argument("--timeout-seconds", type=int, default=7200)
    native_cell_parser.add_argument("--task", required=True)
    native_cell_parser.add_argument("--condition", required=True)
    native_cell_parser.add_argument("--json", action="store_true", dest="as_json")
    round1_feedback_run_parser = subparsers.add_parser(
        "round1-feedback-run",
        help="run or resume the leakage-gated 30-task Round 1 feedback cohort",
    )
    round1_feedback_run_parser.add_argument("--manifest", type=Path, required=True)
    round1_feedback_run_parser.add_argument("--routes", type=Path, required=True)
    round1_feedback_run_parser.add_argument("--target-audit", type=Path, required=True)
    round1_feedback_run_parser.add_argument(
        "--operator-skill", type=Path, required=True
    )
    round1_feedback_run_parser.add_argument("--span-skill", type=Path, required=True)
    round1_feedback_run_parser.add_argument("--out", type=Path, required=True)
    round1_feedback_run_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/private/tmp/jlens-round1-workspaces"),
    )
    round1_feedback_run_parser.add_argument(
        "--model", type=Path, default=Path("models/Qwen3.5-4B-mlx-4bit")
    )
    round1_feedback_run_parser.add_argument("--max-cells", type=int, default=None)
    round1_feedback_run_parser.add_argument(
        "--task", action="append", dest="task_ids", default=None
    )
    round1_feedback_run_parser.add_argument(
        "--realization-candidates", type=int, default=1
    )
    round1_feedback_run_parser.add_argument(
        "--max-plan-repairs", type=int, choices=range(0, 4), default=None
    )
    round1_feedback_run_parser.add_argument(
        "--max-generation-calls", type=int, default=None
    )
    round1_feedback_run_parser.add_argument(
        "--shared-diagnosis-localization", action="store_true"
    )
    round1_feedback_run_parser.add_argument("--transport-base-url", default=None)
    round1_feedback_run_parser.add_argument("--transport-model", default=None)
    round1_feedback_run_parser.add_argument("--transport-api-key-env", default=None)
    round1_feedback_run_parser.add_argument(
        "--generation-cache", type=Path, default=None
    )
    round1_feedback_run_parser.add_argument(
        "--futility-min-cells", type=int, default=None
    )
    round1_feedback_run_parser.add_argument(
        "--futility-min-structural-rate", type=float, default=0.0
    )
    round1_feedback_run_parser.add_argument(
        "--shared-context-source", type=Path, default=None
    )
    round1_feedback_run_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_holdout_run_parser = subparsers.add_parser(
        "round1-holdout-run",
        help="run/resume the 30-task holdout only after a native feedback gain",
    )
    round1_holdout_run_parser.add_argument("--manifest", type=Path, required=True)
    round1_holdout_run_parser.add_argument("--routes", type=Path, required=True)
    round1_holdout_run_parser.add_argument("--target-audit", type=Path, required=True)
    round1_holdout_run_parser.add_argument("--operator-skill", type=Path, required=True)
    round1_holdout_run_parser.add_argument("--span-skill", type=Path, required=True)
    round1_holdout_run_parser.add_argument("--feedback-gain", type=Path, required=True)
    round1_holdout_run_parser.add_argument("--out", type=Path, required=True)
    round1_holdout_run_parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/private/tmp/jlens-round1-holdout-workspaces"),
    )
    round1_holdout_run_parser.add_argument(
        "--model", type=Path, default=Path("models/Qwen3.5-4B-mlx-4bit")
    )
    round1_holdout_run_parser.add_argument("--max-cells", type=int, default=None)
    round1_holdout_run_parser.add_argument(
        "--task", action="append", dest="task_ids", default=None
    )
    round1_holdout_run_parser.add_argument(
        "--realization-candidates", type=int, default=1
    )
    round1_holdout_run_parser.add_argument(
        "--max-plan-repairs", type=int, choices=range(0, 4), default=None
    )
    round1_holdout_run_parser.add_argument(
        "--max-generation-calls", type=int, default=None
    )
    round1_holdout_run_parser.add_argument(
        "--shared-diagnosis-localization", action="store_true"
    )
    round1_holdout_run_parser.add_argument("--transport-base-url", default=None)
    round1_holdout_run_parser.add_argument("--transport-model", default=None)
    round1_holdout_run_parser.add_argument("--transport-api-key-env", default=None)
    round1_holdout_run_parser.add_argument(
        "--generation-cache", type=Path, default=None
    )
    round1_holdout_run_parser.add_argument(
        "--futility-min-cells", type=int, default=None
    )
    round1_holdout_run_parser.add_argument(
        "--futility-min-structural-rate", type=float, default=0.0
    )
    round1_holdout_run_parser.add_argument(
        "--shared-context-source", type=Path, default=None
    )
    round1_holdout_run_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_feedback_native_parser = subparsers.add_parser(
        "round1-feedback-native",
        help="run or resume official native judges for Round 1 feedback only",
    )
    round1_feedback_native_parser.add_argument("--manifest", type=Path, required=True)
    round1_feedback_native_parser.add_argument("--routes", type=Path, required=True)
    round1_feedback_native_parser.add_argument("--experiment", type=Path, required=True)
    round1_feedback_native_parser.add_argument("--out", type=Path, required=True)
    round1_feedback_native_parser.add_argument(
        "--official-out", type=Path, required=True
    )
    round1_feedback_native_parser.add_argument("--pool-root", type=Path, required=True)
    round1_feedback_native_parser.add_argument("--swe-python", type=Path, required=True)
    round1_feedback_native_parser.add_argument(
        "--multi-python", type=Path, required=True
    )
    round1_feedback_native_parser.add_argument(
        "--swe-harness", type=Path, required=True
    )
    round1_feedback_native_parser.add_argument(
        "--multi-harness", type=Path, required=True
    )
    round1_feedback_native_parser.add_argument(
        "--native-assets", type=Path, default=None
    )
    round1_feedback_native_parser.add_argument(
        "--timeout-seconds", type=int, default=7200
    )
    round1_feedback_native_parser.add_argument("--max-cells", type=int, default=None)
    round1_feedback_native_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_holdout_native_parser = subparsers.add_parser(
        "round1-holdout-native",
        help="run official holdout judges and close the full capability gate",
    )
    round1_holdout_native_parser.add_argument("--manifest", type=Path, required=True)
    round1_holdout_native_parser.add_argument("--routes", type=Path, required=True)
    round1_holdout_native_parser.add_argument("--experiment", type=Path, required=True)
    round1_holdout_native_parser.add_argument(
        "--feedback-gain", type=Path, required=True
    )
    round1_holdout_native_parser.add_argument("--out", type=Path, required=True)
    round1_holdout_native_parser.add_argument(
        "--official-out", type=Path, required=True
    )
    round1_holdout_native_parser.add_argument("--pool-root", type=Path, required=True)
    round1_holdout_native_parser.add_argument("--swe-python", type=Path, required=True)
    round1_holdout_native_parser.add_argument(
        "--multi-python", type=Path, required=True
    )
    round1_holdout_native_parser.add_argument("--swe-harness", type=Path, required=True)
    round1_holdout_native_parser.add_argument(
        "--multi-harness", type=Path, required=True
    )
    round1_holdout_native_parser.add_argument(
        "--native-assets", type=Path, default=None
    )
    round1_holdout_native_parser.add_argument(
        "--timeout-seconds", type=int, default=7200
    )
    round1_holdout_native_parser.add_argument("--max-cells", type=int, default=None)
    round1_holdout_native_parser.add_argument(
        "--task", action="append", dest="task_ids", default=None
    )
    round1_holdout_native_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_holdout_project_parser = subparsers.add_parser(
        "round1-holdout-native-project",
        help="recompute a versioned summary over immutable holdout native cells",
    )
    round1_holdout_project_parser.add_argument("--manifest", type=Path, required=True)
    round1_holdout_project_parser.add_argument("--routes", type=Path, required=True)
    round1_holdout_project_parser.add_argument("--experiment", type=Path, required=True)
    round1_holdout_project_parser.add_argument(
        "--feedback-gain", type=Path, required=True
    )
    round1_holdout_project_parser.add_argument("--native", type=Path, required=True)
    round1_holdout_project_parser.add_argument("--out", type=Path, required=True)
    round1_holdout_project_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_holdout_retry_parser = subparsers.add_parser(
        "round1-holdout-native-retry",
        help="retry infrastructure-failed holdout cells in a new evidence root",
    )
    round1_holdout_retry_parser.add_argument("--manifest", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--routes", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--experiment", type=Path, required=True)
    round1_holdout_retry_parser.add_argument(
        "--feedback-gain", type=Path, required=True
    )
    round1_holdout_retry_parser.add_argument(
        "--source-native", type=Path, required=True
    )
    round1_holdout_retry_parser.add_argument("--retry-out", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--out", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--official-out", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--pool-root", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--swe-python", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--multi-python", type=Path, required=True)
    round1_holdout_retry_parser.add_argument("--swe-harness", type=Path, required=True)
    round1_holdout_retry_parser.add_argument(
        "--multi-harness", type=Path, required=True
    )
    round1_holdout_retry_parser.add_argument("--native-assets", type=Path, default=None)
    round1_holdout_retry_parser.add_argument(
        "--timeout-seconds", type=int, default=7200
    )
    round1_holdout_retry_parser.add_argument("--max-cells", type=int, default=None)
    round1_holdout_retry_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_feedback_request_parser = subparsers.add_parser(
        "round1-feedback-request",
        help="freeze the complete holdout-free Round 1 teacher request",
    )
    round1_feedback_request_parser.add_argument("--manifest", type=Path, required=True)
    round1_feedback_request_parser.add_argument("--routes", type=Path, required=True)
    round1_feedback_request_parser.add_argument(
        "--experiment", type=Path, required=True
    )
    round1_feedback_request_parser.add_argument("--native", type=Path, required=True)
    round1_feedback_request_parser.add_argument("--out", type=Path, required=True)
    round1_feedback_request_parser.add_argument(
        "--post-holdout-projection", action="store_true"
    )
    round1_feedback_request_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_feedback_audit_parser = subparsers.add_parser(
        "round1-feedback-audit",
        help="audit complete feedback A/B evidence before another Skill proposal",
    )
    round1_feedback_audit_parser.add_argument("--request", type=Path, required=True)
    round1_feedback_audit_parser.add_argument(
        "--evolution-catalog", type=Path, required=True
    )
    round1_feedback_audit_parser.add_argument("--out", type=Path, required=True)
    round1_feedback_audit_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_realization_feedback_request_parser = subparsers.add_parser(
        "round1-realization-feedback-request",
        help="freeze completed feedback-only structural failures for the teacher",
    )
    round1_realization_feedback_request_parser.add_argument(
        "--manifest", type=Path, required=True
    )
    round1_realization_feedback_request_parser.add_argument(
        "--routes", type=Path, required=True
    )
    round1_realization_feedback_request_parser.add_argument(
        "--experiment", type=Path, required=True
    )
    round1_realization_feedback_request_parser.add_argument(
        "--out", type=Path, required=True
    )
    round1_realization_feedback_request_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_targeted_feedback_request_parser = subparsers.add_parser(
        "round1-targeted-native-feedback-request",
        help="freeze selected completed feedback A/B native pairs for the teacher",
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--manifest", type=Path, required=True
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--routes", type=Path, required=True
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--experiment", type=Path, required=True
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--native", type=Path, required=True
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--task", action="append", dest="task_ids", required=True
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--prior-request",
        action="append",
        dest="prior_requests",
        type=Path,
        default=[],
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--out", type=Path, required=True
    )
    round1_targeted_feedback_request_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_feedback_authorize_parser = subparsers.add_parser(
        "round1-feedback-authorize",
        help="bind the 3M DeepSeek campaign grant to one feedback request",
    )
    round1_feedback_authorize_parser.add_argument("--request", type=Path, required=True)
    round1_feedback_authorize_parser.add_argument(
        "--campaign-checkpoint", type=Path, required=True
    )
    round1_feedback_authorize_parser.add_argument(
        "--evolution-catalog", type=Path, required=True
    )
    round1_feedback_authorize_parser.add_argument(
        "--expires-at", required=True, help="timezone-aware ISO-8601 timestamp"
    )
    round1_feedback_authorize_parser.add_argument(
        "--maximum-output-tokens", type=int, default=256_000
    )
    round1_feedback_authorize_parser.add_argument("--out", type=Path, required=True)
    round1_feedback_authorize_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_feedback_strategy_parser = subparsers.add_parser(
        "round1-feedback-strategy",
        help="dispatch one replay-safe DeepSeek feedback strategy call",
    )
    round1_feedback_strategy_parser.add_argument("--request", type=Path, required=True)
    round1_feedback_strategy_parser.add_argument(
        "--authorization", type=Path, required=True
    )
    round1_feedback_strategy_parser.add_argument(
        "--campaign-checkpoint", type=Path, required=True
    )
    round1_feedback_strategy_parser.add_argument("--ledger", type=Path, required=True)
    round1_feedback_strategy_parser.add_argument("--out", type=Path, required=True)
    round1_feedback_strategy_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    round1_feedback_compile_parser = subparsers.add_parser(
        "round1-feedback-compile",
        help="compile one feedback strategy into inactive operator/span Skills",
    )
    round1_feedback_compile_parser.add_argument("--strategy", type=Path, required=True)
    round1_feedback_compile_parser.add_argument(
        "--parent-operator-skill", type=Path, required=True
    )
    round1_feedback_compile_parser.add_argument(
        "--operator-out", type=Path, required=True
    )
    round1_feedback_compile_parser.add_argument("--span-out", type=Path, required=True)
    round1_feedback_compile_parser.add_argument(
        "--source-round", type=int, required=True
    )
    round1_feedback_compile_parser.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    scale_parser = subparsers.add_parser(
        "round1-scale-preflight",
        help="freeze 60-task readiness without opening sealed partitions",
    )
    scale_parser.add_argument("--task-pool", type=Path, required=True)
    scale_parser.add_argument("--out", type=Path, required=True)
    scale_parser.add_argument("--target-tasks", type=int, default=60)
    scale_parser.add_argument("--partition", default="search")
    scale_parser.add_argument(
        "--renderer-language", action="append", dest="renderer_languages"
    )
    scale_parser.add_argument(
        "--opened-taskset", action="append", type=Path, default=[]
    )
    scale_parser.add_argument("--json", action="store_true", dest="as_json")
    compose_parser = subparsers.add_parser(
        "p1-compose",
        help="compose compatible feedback and holdout evidence without rerunning",
    )
    compose_parser.add_argument("--manifest", type=Path, required=True)
    compose_parser.add_argument("--target-selection", type=Path, required=True)
    for cohort in ("feedback", "holdout"):
        compose_parser.add_argument(f"--{cohort}-experiment", type=Path, required=True)
        compose_parser.add_argument(f"--{cohort}-manifest", type=Path, required=True)
        compose_parser.add_argument(
            f"--{cohort}-target-selection", type=Path, required=True
        )
    compose_parser.add_argument("--out", type=Path, required=True)
    compose_parser.add_argument("--json", action="store_true", dest="as_json")
    feedback_parser = subparsers.add_parser(
        "p1-feedback-request",
        help="freeze a holdout-free parent request from completed P1 evidence",
    )
    feedback_parser.add_argument("--composition", type=Path, required=True)
    feedback_parser.add_argument("--semantic-review", type=Path, required=True)
    feedback_parser.add_argument("--condition", default="structured-taught")
    feedback_parser.add_argument("--out", type=Path, required=True)
    feedback_parser.add_argument("--json", action="store_true", dest="as_json")
    round_feedback_parser = subparsers.add_parser(
        "p1-round-feedback-request",
        help="freeze a holdout-free request from a later paired P1 round",
    )
    round_feedback_parser.add_argument("--experiment", type=Path, required=True)
    round_feedback_parser.add_argument("--semantic-review", type=Path, required=True)
    round_feedback_parser.add_argument("--condition", default="structured-taught")
    round_feedback_parser.add_argument(
        "--rejected-fingerprint", action="append", default=[]
    )
    round_feedback_parser.add_argument("--out", type=Path, required=True)
    round_feedback_parser.add_argument("--json", action="store_true", dest="as_json")
    parent_preflight_parser = subparsers.add_parser(
        "p1-parent-preflight",
        help="check the P1 DeepSeek authorization boundary without dispatching",
    )
    parent_preflight_parser.add_argument("--request", type=Path, required=True)
    parent_preflight_parser.add_argument("--authorization", type=Path, default=None)
    parent_preflight_parser.add_argument("--out", type=Path, required=True)
    parent_preflight_parser.add_argument("--json", action="store_true", dest="as_json")
    parent_call_parser = subparsers.add_parser(
        "p1-parent-call",
        help="dispatch one exactly authorized DeepSeek Skill revision call",
    )
    parent_call_parser.add_argument("--request", type=Path, required=True)
    parent_call_parser.add_argument("--authorization", type=Path, required=True)
    parent_call_parser.add_argument("--ledger", type=Path, required=True)
    parent_call_parser.add_argument("--registry", type=Path, required=True)
    parent_call_parser.add_argument("--out", type=Path, required=True)
    parent_call_parser.add_argument("--call-id", default="p1-skill-round-001")
    parent_call_parser.add_argument(
        "--prior-response", type=Path, action="append", default=[]
    )
    parent_call_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _official_evaluator_from_args(args: argparse.Namespace) -> Any:
    from official_patch_evaluator import (
        OfficialPatchEvaluator,
        freeze_official_harness_runtime,
    )

    freeze_official_harness_runtime(
        swe_python=args.swe_python,
        multi_python=args.multi_python,
        swe_harness_root=args.swe_harness,
        multi_harness_root=args.multi_harness,
        output_root=args.official_out,
        native_assets_path=args.native_assets,
    )
    return OfficialPatchEvaluator(
        swe_python=args.swe_python,
        multi_python=args.multi_python,
        swe_harness_root=args.swe_harness,
        multi_harness_root=args.multi_harness,
        pool_root=args.pool_root,
        output_root=args.official_out,
        timeout_seconds=args.timeout_seconds,
    )


def main() -> int:
    args = _parser().parse_args()
    if args.command == "doctor":
        report = doctor(args.root)
    elif args.command == "offline-smoke":
        from .smoke import run_offline_smoke

        report = run_offline_smoke(args.out)
    elif args.command == "cuda-calibration-replay":
        from .cuda_calibration import (
            load_calibration_evidence,
            replay_cuda_calibration,
        )
        from .model_transport import OpenAICompatibleTransport

        manifest = load_calibration_evidence(args.manifest, "CUDA calibration manifest")
        report = replay_cuda_calibration(
            manifest=manifest,
            evidence_root=args.out,
            transport=OpenAICompatibleTransport(
                base_url=args.transport_base_url,
                model=args.transport_model,
                api_key_env=args.transport_api_key_env,
            ),
            max_tokens=args.max_tokens,
        )
    elif args.command == "cuda-calibration-collect":
        from .cuda_calibration import (
            collect_cuda_calibration,
            load_calibration_evidence,
        )

        report = collect_cuda_calibration(
            manifest=load_calibration_evidence(
                args.manifest, "CUDA calibration manifest"
            ),
            experiment_root=args.experiment,
            output_path=args.out,
        )
    elif args.command == "cuda-calibration-evaluate":
        from .contracts import canonical_json
        from .cuda_calibration import evaluate_calibration, load_calibration_evidence

        report = evaluate_calibration(
            manifest=load_calibration_evidence(
                args.manifest, "CUDA calibration manifest"
            ),
            cuda_results=load_calibration_evidence(
                args.cuda_results, "CUDA calibration results"
            ),
        )
        output = args.out.resolve()
        if output.exists():
            if json.loads(output.read_text(encoding="utf-8")) != report:
                raise ValueError("frozen CUDA calibration report changed")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    elif args.command == "convergence-project":
        from .control_gate_projection import freeze_convergence_projection

        report = freeze_convergence_projection(
            source_path=args.source, output_path=args.out
        )
    elif args.command == "independent-safety-project":
        from .control_gate_projection import freeze_independent_safety_projection

        report = freeze_independent_safety_projection(
            subject_sha256=args.subject_sha256,
            receipt_paths=tuple(args.probes),
            output_path=args.out,
        )
    elif args.command == "statistical-capability-project":
        from .control_gate_projection import (
            freeze_statistical_capability_projection,
        )

        report = freeze_statistical_capability_projection(
            feedback_path=args.feedback,
            holdout_path=args.holdout,
            independent_safety_path=args.independent_safety,
            runtime_identity_path=args.runtime_identity,
            cost_receipt_path=args.cost_receipt,
            catalog_audit_path=args.catalog_audit,
            output_path=args.out,
        )
    elif args.command == "runtime-identity-project":
        from .control_gate_projection import freeze_runtime_identity_projection

        report = freeze_runtime_identity_projection(
            source_path=args.source, output_path=args.out
        )
    elif args.command == "accelerator-preflight-project":
        from .control_gate_projection import (
            freeze_accelerator_preflight_projection,
        )

        report = freeze_accelerator_preflight_projection(
            source_path=args.source, output_path=args.out
        )
    elif args.command == "freeze-local-p1-taskset":
        from .p1_taskset import freeze_local_p1_taskset

        taskset = freeze_local_p1_taskset(args.root, args.out)
        report = {
            "status": "ready",
            "taskset": taskset.to_dict(),
            "preflight": taskset.preflight().to_dict(),
            "network_calls_performed": False,
        }
    elif args.command == "freeze-local-p1-v2":
        from .p1_taskset import freeze_local_p1_v2_bundle

        taskset, selection = freeze_local_p1_v2_bundle(
            args.root,
            args.taskset_out,
            args.target_selection_out,
        )
        report = {
            "status": "ready",
            "taskset": taskset.to_dict(),
            "target_selection": selection.to_dict(),
            "preflight": taskset.preflight().to_dict(),
            "network_calls_performed": False,
        }
    elif args.command == "freeze-local-p1-v3":
        from .p1_taskset import freeze_local_p1_v3_bundle

        taskset, selection = freeze_local_p1_v3_bundle(
            args.root,
            args.taskset_out,
            args.target_selection_out,
        )
        report = {
            "status": "ready",
            "taskset": taskset.to_dict(),
            "target_selection": selection.to_dict(),
            "preflight": taskset.preflight().to_dict(),
            "network_calls_performed": False,
        }
    elif args.command == "freeze-local-p1-v4":
        from .p1_taskset import freeze_local_p1_v4_bundle

        taskset, selection = freeze_local_p1_v4_bundle(
            args.root,
            args.taskset_out,
            args.target_selection_out,
        )
        report = {
            "status": "ready",
            "taskset": taskset.to_dict(),
            "target_selection": selection.to_dict(),
            "preflight": taskset.preflight().to_dict(),
            "network_calls_performed": False,
        }
    elif args.command == "taskset-preflight":
        from .eval_manifest import EvaluationTaskSet

        taskset = EvaluationTaskSet.from_dict(
            json.loads(args.manifest.read_text(encoding="utf-8"))
        )
        preflight = taskset.preflight()
        report = {
            "status": "ready" if preflight.ready else "blocked",
            "taskset_id": taskset.taskset_id,
            "taskset_fingerprint": taskset.fingerprint,
            "cohort_counts": taskset.cohort_counts,
            "preflight": preflight.to_dict(),
            "network_calls_performed": False,
        }
    elif args.command == "taskset-target-audit":
        from .contracts import ContractError, canonical_json
        from .eval_manifest import EvaluationTaskSet
        from .target_audit import GoldPatchReference, audit_target_coverage

        taskset = EvaluationTaskSet.from_dict(
            json.loads(args.manifest.read_text(encoding="utf-8"))
        )
        references = []
        for value in args.reference:
            task_id, separator, patch_path = value.partition("=")
            if not separator or not task_id or not patch_path:
                raise ContractError(
                    "target audit reference must be TASK_ID=FIX_PATCH_PATH"
                )
            references.append(
                GoldPatchReference(task_id=task_id, patch_path=Path(patch_path))
            )
        audit = audit_target_coverage(taskset, references)
        report = {
            "status": "ready" if audit.ready else "blocked",
            **audit.to_dict(),
        }
        output = args.out.resolve()
        if output.exists():
            raise ContractError(f"target audit evidence already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    elif args.command == "taskset-mechanism-audit":
        from .contracts import ContractError, canonical_json
        from .eval_manifest import EvaluationTaskSet
        from .target_audit import (
            GoldPatchReference,
            MechanismCapacityPolicy,
            audit_mechanism_capacity,
        )

        taskset = EvaluationTaskSet.from_dict(
            json.loads(args.manifest.read_text(encoding="utf-8"))
        )
        references = []
        for value in args.reference:
            task_id, separator, patch_path = value.partition("=")
            if not separator or not task_id or not patch_path:
                raise ContractError(
                    "mechanism audit reference must be TASK_ID=FIX_PATCH_PATH"
                )
            references.append(
                GoldPatchReference(task_id=task_id, patch_path=Path(patch_path))
            )
        audit = audit_mechanism_capacity(
            taskset,
            references,
            policy=MechanismCapacityPolicy(
                structured_max_tokens=args.structured_max_tokens,
                tokenizer_path=args.tokenizer,
            ),
        )
        report = {
            "status": "ready" if audit.ready else "blocked",
            **audit.to_dict(),
        }
        output = args.out.resolve()
        if output.exists():
            raise ContractError(f"mechanism audit evidence already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    elif args.command == "p1-run":
        from .p1_experiment import run_local_qwen_p1

        report = run_local_qwen_p1(
            manifest_path=args.manifest,
            skill_path=args.skill,
            skill_revision_path=args.skill_revision,
            evidence_root=args.out,
            workspace_root=args.workspace,
            model_path=args.model,
            max_cells=args.max_cells,
            target_selection_path=args.target_selection,
            structured_max_tokens=args.structured_max_tokens,
            structured_context_chars=args.structured_context_chars,
            hunk_max_tokens=args.hunk_max_tokens,
            task_ids=set(args.tasks) if args.tasks else None,
            condition_ids=set(args.conditions) if args.conditions else None,
        )
    elif args.command == "p1-symbol-run":
        from .p1_symbol import run_local_qwen_symbol_p1

        report = run_local_qwen_symbol_p1(
            manifest_path=args.manifest,
            skill_revision_path=args.skill_revision,
            target_selection_path=args.target_selection,
            evidence_root=args.out,
            workspace_root=args.workspace,
            model_path=args.model,
            max_cells=args.max_cells,
            max_tokens=args.max_tokens,
            context_chars=args.context_chars,
            task_ids=set(args.tasks) if args.tasks else None,
            condition_ids=set(args.conditions) if args.conditions else None,
        )
    elif args.command == "p1-block-run":
        from .p1_block import run_local_qwen_block_p1

        report = run_local_qwen_block_p1(
            manifest_path=args.manifest,
            skill_revision_path=args.skill_revision,
            target_selection_path=args.target_selection,
            evidence_root=args.out,
            workspace_root=args.workspace,
            model_path=args.model,
            max_cells=args.max_cells,
            max_tokens=args.max_tokens,
            context_chars=args.context_chars,
            task_ids=set(args.tasks) if args.tasks else None,
            condition_ids=set(args.conditions) if args.conditions else None,
        )
    elif args.command == "p1-native":
        from .p1_native import evaluate_p1_experiment_native

        evaluator = _official_evaluator_from_args(args)
        report = evaluate_p1_experiment_native(
            manifest_path=args.manifest,
            experiment_root=args.experiment,
            evidence_root=args.out,
            pool_root=args.pool_root,
            evaluator=evaluator,
            max_cells=args.max_cells,
        )
    elif args.command == "p1-native-cell":
        from .p1_native import evaluate_p1_feedback_cell_native

        evaluator = _official_evaluator_from_args(args)
        report = evaluate_p1_feedback_cell_native(
            manifest_path=args.manifest,
            experiment_root=args.experiment,
            evidence_root=args.out,
            pool_root=args.pool_root,
            evaluator=evaluator,
            task_id=args.task,
            condition_id=args.condition,
        )
    elif args.command == "round1-feedback-run":
        from .round1_run import run_round1_feedback

        report = run_round1_feedback(
            taskset_path=args.manifest,
            routes_path=args.routes,
            target_audit_path=args.target_audit,
            operator_skill_path=args.operator_skill,
            span_skill_path=args.span_skill,
            model_path=args.model,
            evidence_root=args.out,
            workspace_root=args.workspace,
            max_cells=args.max_cells,
            realization_candidates=args.realization_candidates,
            max_plan_repairs=args.max_plan_repairs,
            maximum_generation_calls=args.max_generation_calls,
            shared_diagnosis_localization=args.shared_diagnosis_localization,
            task_ids=tuple(args.task_ids) if args.task_ids else None,
            transport_base_url=args.transport_base_url,
            transport_model=args.transport_model,
            transport_api_key_env=args.transport_api_key_env,
            generation_cache_root=args.generation_cache,
            futility_min_cells_per_mechanism=args.futility_min_cells,
            futility_min_structural_rate=args.futility_min_structural_rate,
            shared_context_source_root=args.shared_context_source,
        )
    elif args.command == "round1-holdout-run":
        from .round1_run import run_round1_holdout

        report = run_round1_holdout(
            taskset_path=args.manifest,
            routes_path=args.routes,
            target_audit_path=args.target_audit,
            operator_skill_path=args.operator_skill,
            span_skill_path=args.span_skill,
            feedback_gain_path=args.feedback_gain,
            model_path=args.model,
            evidence_root=args.out,
            workspace_root=args.workspace,
            max_cells=args.max_cells,
            realization_candidates=args.realization_candidates,
            max_plan_repairs=args.max_plan_repairs,
            maximum_generation_calls=args.max_generation_calls,
            shared_diagnosis_localization=args.shared_diagnosis_localization,
            task_ids=tuple(args.task_ids) if args.task_ids else None,
            transport_base_url=args.transport_base_url,
            transport_model=args.transport_model,
            transport_api_key_env=args.transport_api_key_env,
            generation_cache_root=args.generation_cache,
            futility_min_cells_per_mechanism=args.futility_min_cells,
            futility_min_structural_rate=args.futility_min_structural_rate,
            shared_context_source_root=args.shared_context_source,
        )
    elif args.command == "round1-feedback-native":
        from .round1_native import run_round1_feedback_native

        evaluator = _official_evaluator_from_args(args)
        report = run_round1_feedback_native(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            evidence_root=args.out,
            pool_root=args.pool_root,
            evaluator=evaluator,
            max_cells=args.max_cells,
        )
    elif args.command == "round1-holdout-native":
        from .round1_native import run_round1_holdout_native

        evaluator = _official_evaluator_from_args(args)
        report = run_round1_holdout_native(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            feedback_gain_path=args.feedback_gain,
            evidence_root=args.out,
            pool_root=args.pool_root,
            evaluator=evaluator,
            max_cells=args.max_cells,
            task_ids=tuple(args.task_ids) if args.task_ids else None,
        )
    elif args.command == "round1-holdout-native-project":
        from .round1_native import project_round1_holdout_native_summary

        report = project_round1_holdout_native_summary(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            feedback_gain_path=args.feedback_gain,
            evidence_root=args.native,
            output_path=args.out,
        )
    elif args.command == "round1-holdout-native-retry":
        from .round1_native import retry_round1_holdout_native_failures

        evaluator = _official_evaluator_from_args(args)
        report = retry_round1_holdout_native_failures(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            feedback_gain_path=args.feedback_gain,
            source_evidence_root=args.source_native,
            retry_evidence_root=args.retry_out,
            pool_root=args.pool_root,
            evaluator=evaluator,
            output_path=args.out,
            max_cells=args.max_cells,
        )
    elif args.command == "round1-feedback-request":
        from .round1_feedback import freeze_round1_feedback_request

        report = freeze_round1_feedback_request(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            native_root=args.native,
            output_path=args.out,
            post_holdout_projection=args.post_holdout_projection,
        )
    elif args.command == "round1-realization-feedback-request":
        from .round1_feedback import freeze_round1_realization_feedback_request

        report = freeze_round1_realization_feedback_request(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            output_path=args.out,
        )
    elif args.command == "round1-feedback-audit":
        from .evolution_catalog import EvolutionCatalog
        from .feedback_audit import audit_feedback_request

        report = audit_feedback_request(
            request_path=args.request,
            catalog=EvolutionCatalog(args.evolution_catalog),
            output_path=args.out,
        )
    elif args.command == "round1-targeted-native-feedback-request":
        from .round1_feedback import freeze_round1_targeted_native_feedback_request

        report = freeze_round1_targeted_native_feedback_request(
            taskset_path=args.manifest,
            routes_path=args.routes,
            experiment_root=args.experiment,
            native_root=args.native,
            task_ids=tuple(args.task_ids),
            prior_request_paths=tuple(args.prior_requests),
            output_path=args.out,
        )
    elif args.command == "round1-feedback-authorize":
        from datetime import datetime

        from .contracts import canonical_json
        from .evolution_catalog import EvolutionCatalog
        from .round1_feedback import (
            augment_feedback_request_with_catalog_context,
            create_round1_feedback_authorization,
        )

        source_request = json.loads(args.request.read_text(encoding="utf-8"))
        augmented = augment_feedback_request_with_catalog_context(
            source_request,
            catalog=EvolutionCatalog(args.evolution_catalog),
            capability_tags=("localization", "patch-realization"),
            task_tags=("swe-bench",),
            failure_mode_tags=(
                "wrong-target",
                "selector-no-match",
                "native-unresolved",
            ),
        )
        catalog_request = args.out.with_name(f"{args.out.stem}.catalog-request.json")
        if catalog_request.exists():
            if json.loads(catalog_request.read_text(encoding="utf-8")) != augmented:
                raise ValueError(
                    "frozen catalog-augmented request does not match replay"
                )
        else:
            catalog_request.parent.mkdir(parents=True, exist_ok=True)
            catalog_request.write_text(
                canonical_json(augmented) + "\n", encoding="utf-8"
            )

        report = create_round1_feedback_authorization(
            request_path=catalog_request,
            campaign_checkpoint_path=args.campaign_checkpoint,
            output_path=args.out,
            expires_at=datetime.fromisoformat(args.expires_at),
            maximum_output_tokens=args.maximum_output_tokens,
        )
    elif args.command == "round1-feedback-strategy":
        from .round1_feedback import dispatch_round1_feedback_strategy

        frozen_request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(
            frozen_request.get("request", {}).get("evolution_catalog_context"),
            dict,
        ):
            raise ValueError("DeepSeek request must include frozen catalog context")

        report = dispatch_round1_feedback_strategy(
            request_path=args.request,
            authorization_path=args.authorization,
            campaign_checkpoint_path=args.campaign_checkpoint,
            ledger_path=args.ledger,
            output_path=args.out,
        )
    elif args.command == "round1-feedback-compile":
        from .round1_feedback import compile_round1_feedback_skills

        report = compile_round1_feedback_skills(
            strategy_response_path=args.strategy,
            parent_operator_skill_path=args.parent_operator_skill,
            operator_output_path=args.operator_out,
            span_output_path=args.span_out,
            source_round=args.source_round,
        )
    elif args.command == "round1-scale-preflight":
        from .scale_readiness import freeze_round1_scale_readiness

        report = freeze_round1_scale_readiness(
            task_pool_path=args.task_pool,
            output_path=args.out,
            target_tasks=args.target_tasks,
            partition=args.partition,
            renderer_languages=frozenset(args.renderer_languages or {"python"}),
            opened_taskset_paths=tuple(args.opened_taskset),
        )
    elif args.command == "p1-compose":
        from .composition import (
            ExperimentEvidenceSource,
            compose_experiment_evidence,
        )

        report = compose_experiment_evidence(
            taskset_path=args.manifest,
            target_selection_path=args.target_selection,
            sources=[
                ExperimentEvidenceSource(
                    cohort=cohort,
                    experiment_root=getattr(args, f"{cohort}_experiment"),
                    taskset_path=getattr(args, f"{cohort}_manifest"),
                    target_selection_path=getattr(args, f"{cohort}_target_selection"),
                )
                for cohort in ("feedback", "holdout")
            ],
            output_path=args.out,
        )
    elif args.command == "p1-feedback-request":
        from .p1_feedback import freeze_p1_parent_request

        report = freeze_p1_parent_request(
            composition_path=args.composition,
            semantic_review_path=args.semantic_review,
            output_path=args.out,
            condition_id=args.condition,
        )
    elif args.command == "p1-round-feedback-request":
        from .p1_feedback import freeze_p1_round_parent_request

        report = freeze_p1_round_parent_request(
            experiment_root=args.experiment,
            semantic_review_path=args.semantic_review,
            output_path=args.out,
            condition_id=args.condition,
            rejected_fingerprints=args.rejected_fingerprint,
        )
    elif args.command == "p1-parent-preflight":
        from .p1_parent import freeze_p1_parent_preflight

        report = freeze_p1_parent_preflight(
            request_evidence_path=args.request,
            authorization_path=args.authorization,
            output_path=args.out,
        )
    else:
        from .p1_parent import dispatch_p1_parent_call

        report = dispatch_p1_parent_call(
            request_evidence_path=args.request,
            authorization_path=args.authorization,
            ledger_path=args.ledger,
            registry_root=args.registry,
            output_path=args.out,
            call_id=args.call_id,
            prior_response_paths=args.prior_response,
        )
    if args.as_json:
        print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    else:
        if args.command == "doctor":
            print(f"Skill evolution infrastructure: {report['status']}")
            for name, passed in report["checks"].items():
                print(f"- {name}: {'ok' if passed else 'failed'}")
        elif args.command == "offline-smoke":
            print(f"Offline loop smoke: {report['result']['status']}")
        elif args.command == "convergence-project":
            print(
                "Normalized convergence projection: "
                f"{report['metrics']['normalized_mean_abs_delta']}"
            )
        elif args.command == "independent-safety-project":
            print(
                "Independent safety suite: "
                f"{'passed' if report['suite_passed'] else 'failed'}"
            )
        elif args.command == "statistical-capability-project":
            print(
                "Statistical Skill-transfer gate: "
                f"{'passed' if report['gate_passed'] else 'failed'}"
            )
        elif args.command == "runtime-identity-project":
            print(
                "Replayable runtime identity: "
                f"{'complete' if report['complete'] else 'incomplete'}"
            )
        elif args.command == "accelerator-preflight-project":
            print(
                "Accelerator runtime preflight: "
                f"{'ready' if report['ready'] else 'blocked'}"
            )
        elif args.command in {
            "freeze-local-p1-taskset",
            "freeze-local-p1-v2",
            "freeze-local-p1-v3",
            "freeze-local-p1-v4",
            "taskset-preflight",
            "taskset-target-audit",
            "taskset-mechanism-audit",
        }:
            print(f"Evaluation taskset: {report['status']}")
        elif args.command == "p1-compose":
            print(
                f"P1 evidence composition: {report['status']} "
                f"({report['completed_cells']}/{report['planned_cells']})"
            )
        elif args.command in {"p1-feedback-request", "p1-round-feedback-request"}:
            print(
                "P1 parent request: ready "
                f"({report['feedback_task_count']} feedback tasks)"
            )
        elif args.command == "p1-parent-preflight":
            print(f"P1 parent preflight: {report['status']}")
            for error in report["errors"]:
                print(f"- {error}")
        elif args.command == "p1-parent-call":
            print(
                "P1 parent call: completed "
                f"({report['next_revision']['revision_id']}, inactive)"
            )
        elif args.command == "p1-native":
            print(
                f"P1 native evaluation: {report['status']} "
                f"({report['completed_cells']}/{report['planned_cells']})"
            )
        elif args.command == "p1-native-cell":
            print(
                "P1 feedback native cell: "
                f"{'resolved' if report['outcome']['resolved'] else 'unresolved'} "
                f"({report['task_id']}/{report['condition_id']})"
            )
        elif args.command == "round1-feedback-run":
            print(
                f"Round 1 feedback experiment: {report['status']} "
                f"({report['completed_cells']}/{report['planned_cells']})"
            )
        elif args.command == "round1-holdout-run":
            print(
                f"Round 1 holdout experiment: {report['status']} "
                f"({report['completed_cells']}/{report['planned_cells']})"
            )
        elif args.command == "round1-feedback-native":
            print(
                f"Round 1 feedback native: {report['status']} "
                f"({report['completed_cells']}/{report['planned_cells']}, "
                f"gains={report['feedback_gain_count']})"
            )
        elif args.command == "round1-holdout-native":
            print(
                f"Round 1 full capability: {report['status']} "
                f"({report['completed_holdout_cells']}/"
                f"{report['planned_holdout_cells']}, "
                f"regressions={report['holdout_regression_count']})"
            )
        elif args.command == "round1-feedback-request":
            print(
                "Round 1 feedback request: ready "
                f"({report['feedback_task_count']} tasks/"
                f"{report['feedback_cell_count']} cells)"
            )
        elif args.command == "round1-feedback-authorize":
            print("Round 1 feedback authorization: ready")
        elif args.command == "round1-feedback-strategy":
            print(
                "Round 1 feedback strategy: completed "
                f"({report['tokens_charged']} tokens)"
            )
        elif args.command == "round1-feedback-compile":
            print(f"Round 1 feedback Skills: inactive (r{report['source_round']:03d})")
        elif args.command == "round1-scale-preflight":
            print(
                f"Round 1 scale readiness: {report['status']} "
                f"({report['fully_compatible_tasks']}/{report['target_tasks']})"
            )
        else:
            print(
                f"P1 local-Qwen experiment: {report['status']} "
                f"({report['completed_cells']}/{report['planned_cells']})"
            )
    return 1 if report.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
