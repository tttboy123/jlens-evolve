"""Resumable, guarded OpenEvolve runtime with persistent experience reuse."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from agent_optimizer import load_policy_agent_strategy
from proposal_controller import (
    load_policy_proposal_controller,
    verify_proposal_controller_endpoint,
)

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)$")


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the numerically latest valid OpenEvolve checkpoint directory."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def select_meta_policy(
    candidates: list[dict[str, Any]], trials: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Explore every bounded operator once, then exploit best mean yield."""
    if not candidates:
        raise ValueError("at least one meta-policy candidate is required")
    trial_rows = list(trials)
    tried = {str(row.get("policy_id")) for row in trial_rows}
    for candidate in candidates:
        if str(candidate["id"]) not in tried:
            return candidate
    yields: dict[str, list[float]] = {str(row["id"]): [] for row in candidates}
    for row in trial_rows:
        policy_id = str(row.get("policy_id"))
        if policy_id in yields:
            yields[policy_id].append(float(row.get("improvement_yield", 0.0)))
    ranked = sorted(
        candidates,
        key=lambda row: (
            -sum(yields[str(row["id"])]) / max(1, len(yields[str(row["id"])])),
            str(row["id"]),
        ),
    )
    return ranked[0]


def operator_revision_label(
    previous_trial: dict[str, Any] | None,
    *,
    selected_policy_id: str,
    forced: bool,
) -> str | None:
    """Return a self-revision label only for autonomous policy selection."""
    if forced or not previous_trial:
        return None
    previous_policy_id = str(previous_trial.get("policy_id", ""))
    if not previous_policy_id or previous_policy_id == selected_policy_id:
        return None
    return f"{previous_policy_id}->{selected_policy_id}"


def append_agent_guidance(system_message: str, strategy: dict[str, Any]) -> str:
    """Append bounded observer-derived guidance without granting JLens authority."""
    return (
        f"{system_message.rstrip()}\n\n"
        f"[Agent strategy {strategy['strategy_id']}; "
        f"boundary={strategy['causal_boundary']}]\n"
        f"{str(strategy['prompt_guidance']).strip()}"
    )


def build_staged_manifest_fields(
    prior_manifests: Iterable[dict[str, Any]],
    *,
    current_iterations: int,
    current_policy_id: str,
    current_controller_id: str | None = None,
) -> dict[str, Any]:
    """Accumulate an auditable policy schedule across resumed executions."""
    prior = list(prior_manifests)
    prior_iterations = sum(int(row.get("iterations_requested", 0)) for row in prior)
    schedule = [
        str(row["operator_policy_id"]) for row in prior if row.get("operator_policy_id")
    ]
    schedule.append(current_policy_id)
    fields: dict[str, Any] = {
        "iterations_requested_total": prior_iterations + current_iterations,
        "operator_policy_schedule": schedule,
    }
    if current_controller_id is not None:
        fields["proposal_controller_schedule"] = [
            row.get("proposal_controller_id") for row in prior
        ] + [current_controller_id]
    return fields


def resume_is_compatible(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Require stable task, initial program, evaluator, and search configuration."""
    keys = ["task_id", "config_hash", "evaluator_hash", "initial_hash"]
    if "experiment_seed" in previous or "experiment_seed" in current:
        keys.append("experiment_seed")
    return all(previous.get(key) == current.get(key) for key in keys)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _combined_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_run_contract(
    *,
    task_id: str,
    initial_path: Path,
    evaluator_path: Path,
    evaluator_core_path: Path,
    config_path: Path,
    experiment_seed: int | None = None,
) -> dict[str, Any]:
    """Build the exact resume contract, including the task-specific core."""
    contract: dict[str, Any] = {
        "task_id": task_id,
        "config_hash": _sha256_file(config_path),
        "evaluator_hash": _combined_hash([evaluator_path, evaluator_core_path]),
        "initial_hash": _sha256_file(initial_path),
    }
    if experiment_seed is not None:
        contract["experiment_seed"] = experiment_seed
    return contract


def _search_protocol_hash(config_path: Path, selected_policy: dict[str, Any]) -> str:
    """Hash search settings while excluding proposer identity for model A/B."""
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("OpenEvolve config must be a mapping")
    normalized = json.loads(json.dumps(data))
    llm = normalized.get("llm", {})
    if isinstance(llm, dict):
        llm.pop("primary_model", None)
        llm.pop("api_key", None)
    payload = {"config_without_model": normalized, "meta_policy": selected_policy}
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _load_task_core(path: Path) -> Any:
    module_name = f"_evolve_task_core_{_sha256_file(path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator core: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in ("load_candidate", "score_holdout_callable"):
        if not callable(getattr(module, required, None)):
            raise TypeError(f"evaluator core missing callable {required}: {path}")
    return module


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _append_jsonl_unique(path: Path, row: dict[str, Any], id_key: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        ids = {
            parsed.get(id_key)
            for line in handle
            if line.strip()
            for parsed in (json.loads(line),)
        }
        if row[id_key] in ids:
            return False
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return True


def _score_holdout_source(source: str, evaluator_core_path: Path) -> dict[str, Any]:
    core = _load_task_core(evaluator_core_path)
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8") as handle:
        handle.write(source)
        handle.flush()
        solve, reasons = core.load_candidate(handle.name)
    if solve is None:
        return {"holdout_pass_rate": 0.0, "rejection_reasons": reasons}
    return core.score_holdout_callable(solve)


def retrieve_experience(
    store: Any, extension: dict[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    """Retrieve lessons under an explicit experimental experience policy."""
    if mode == "off":
        return []
    if mode not in {"auto", "cross-task"}:
        raise ValueError(f"unknown experience mode: {mode}")
    return store.retrieve_lessons(
        task_family=extension["task_family"],
        tags=set(extension.get("tags", [])),
        exclude_task_id=(extension["task_id"] if mode == "cross-task" else None),
        limit=int(extension.get("experience_limit", 3)),
    )


def validate_experiment_args(
    *, experience_mode: str, psi_experiment_id: str | None, psi_arm: str | None
) -> None:
    """Reject partially specified or confounded PSI A/B arms."""
    if psi_experiment_id and not psi_arm:
        raise ValueError("--psi-experiment-id requires --psi-arm")
    if psi_arm and not psi_experiment_id:
        raise ValueError("--psi-arm requires --psi-experiment-id")
    if psi_arm == "control" and experience_mode != "off":
        raise ValueError("PSI control arm requires --experience-mode off")
    if psi_arm == "transfer" and experience_mode != "cross-task":
        raise ValueError("PSI transfer arm requires --experience-mode cross-task")


def preflight_evaluator_import(evaluator_path: Path, project_root: Path) -> None:
    """Reproduce worker path precedence before spending any search iterations."""
    command = """
import importlib.util
import sys
from pathlib import Path
evaluator_path = Path(sys.argv[1])
project_root = Path(sys.argv[2])
sys.path.insert(0, str(evaluator_path.parent))
sys.path.insert(1, str(project_root))
spec = importlib.util.spec_from_file_location('_evolve_preflight_evaluator', evaluator_path)
if spec is None or spec.loader is None:
    raise ImportError(f'cannot create evaluator spec: {evaluator_path}')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if not callable(getattr(module, 'evaluate', None)):
    raise TypeError(f'evaluator has no callable evaluate: {evaluator_path}')
"""
    result = subprocess.run(
        [sys.executable, "-c", command, str(evaluator_path), str(project_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"evaluator worker preflight failed: {detail}")


def _lesson_for_gains(gained_cases: Iterable[str]) -> tuple[str, list[str]]:
    case_names = [case.removeprefix("case_") for case in gained_cases]
    tags: set[str] = set()
    clauses: list[str] = []
    for name in case_names:
        if "status" in name or "filter" in name:
            tags.update(("filtering", "normalization"))
            clauses.append("normalize categorical fields before filtering")
        elif "user" in name or "identity" in name:
            tags.add("normalization")
            clauses.append("canonicalize identities before grouping")
        elif "aggregate" in name:
            tags.add("aggregation")
            clauses.append("aggregate validated values before rounding")
        elif "amount" in name or "positive" in name:
            tags.add("validation")
            clauses.append(
                "reject booleans, non-finite, and non-positive numeric inputs"
            )
        elif "sort" in name:
            tags.add("sorting")
            clauses.append("apply deterministic tie-breaking after aggregation")
        else:
            tags.add("non-regression")
            clauses.append(
                "make one evidence-targeted change while preserving passing behavior"
            )
    unique = list(dict.fromkeys(clauses))
    return "; ".join(unique).capitalize() + ".", sorted(tags)


def _load_extension(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("evolve extension config must use schema_version: 1")
    for required in ("task_id", "task_family", "state_dir"):
        if not data.get(required):
            raise ValueError(f"missing extension config field: {required}")
    return data


def build_persisted_report(
    *, state_dir: Path, run_id: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Rebuild a run report from append-only state without invoking a model."""
    from experience_store import ExperienceStore
    from self_improvement_eval import evaluate_psi, evaluate_psi_ab, evaluate_rsi

    store = ExperienceStore(state_dir)
    all_events = store.read_events()
    candidates = [
        row
        for row in all_events
        if row.get("run_id") == run_id
        and row.get("event_type") == "candidate"
        and int(row.get("iteration", 0)) > 0
        and "source_hash" in row
    ]
    verifications = [
        row
        for row in all_events
        if row.get("run_id") == run_id and row.get("holdout_verified")
    ]
    trials = [
        row
        for row in _read_jsonl(state_dir / "meta_policy_trials.jsonl")
        if row.get("run_id") == run_id
    ]
    if manifest.get("meta_policy_forced"):
        trials = [
            {
                **row,
                "operator_revision": None,
                "pre_revision_yield": 0.0,
                "selection_mode": "forced",
            }
            for row in trials
        ]
    manifests = _read_jsonl(state_dir / "run_manifests.jsonl")
    iterations = {int(row.get("iteration", 0)) for row in candidates}
    accepted = [row for row in candidates if row.get("accepted")]
    run_seed_and_candidates = [
        row
        for row in all_events
        if row.get("run_id") == run_id and "source_hash" in row
    ]
    rejection_reasons = Counter(
        reason for row in candidates for reason in row.get("admission_reasons", [])
    )
    reusable_lessons = [
        row
        for row in store.read_lessons()
        if row.get("task_family") == manifest.get("task_family")
    ]
    psi = evaluate_psi(manifests)
    psi_experiment_id = manifest.get("psi_experiment_id")
    if psi_experiment_id:
        psi_ab = evaluate_psi_ab(manifests, experiment_id=str(psi_experiment_id))
        psi["matched_ab"] = psi_ab
        psi["cross_task_transfer_pass"] = psi_ab["psi_ab_pass"]
        psi["cross_task_transfer_observed"] = psi_ab["cross_task_provenance"]
        psi["cross_task_holdout_gain"] = psi_ab["transfer_holdout_gain"]
        psi["psi_pass"] = bool(psi["same_search_resume_pass"] and psi_ab["psi_ab_pass"])
    return {
        "run": manifest,
        "admission": {
            "candidate_attempts": len(candidates),
            "search_iterations": len(iterations),
            "extra_replayed_attempts": len(candidates) - len(iterations),
            "accepted": sum(bool(row.get("accepted")) for row in candidates),
            "rejected": sum(not bool(row.get("accepted")) for row in candidates),
            "accept_rate": len(accepted) / max(1, len(candidates)),
            "accepted_parent_regressions": sum(
                bool(row.get("accepted")) and bool(row.get("regressed_cases"))
                for row in candidates
            ),
            "accepted_exact_duplicates": sum(
                bool(row.get("accepted"))
                and "exact_duplicate" in row.get("admission_reasons", [])
                for row in candidates
            ),
            "accepted_ast_duplicates": sum(
                bool(row.get("accepted"))
                and "ast_duplicate" in row.get("admission_reasons", [])
                for row in candidates
            ),
            "unique_source_hashes": len({row["source_hash"] for row in candidates}),
            "unique_ast_hashes": len({row["ast_hash"] for row in candidates}),
            "unique_behavior_signatures": len(
                {row["behavior_signature"] for row in candidates}
            ),
            "accepted_unique_ast_hashes": len({row["ast_hash"] for row in accepted}),
            "accepted_unique_behavior_signatures": len(
                {row["behavior_signature"] for row in accepted}
            ),
            "best_passed_cases": max(
                (
                    float(row.get("metrics", {}).get("passed_cases", 0.0))
                    for row in run_seed_and_candidates
                ),
                default=0.0,
            ),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "holdout_verified_promotions": len(verifications),
        },
        "experience": {
            "archive_events": len(all_events),
            "run_candidate_events": len(candidates),
            "reusable_lessons_for_family": len(reusable_lessons),
            "retrieved_lesson_sources": manifest.get("retrieved_lesson_sources", []),
            "skill_candidate": manifest.get("skill_candidate"),
        },
        "rsi": evaluate_rsi(candidates + trials),
        "psi": psi,
    }


def _rebuild_existing_report(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    extension = _load_extension((project_root / args.extension_config).resolve())
    state_dir = (project_root / extension["state_dir"]).resolve()
    output_dir = (project_root / args.output).resolve()
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = build_persisted_report(
        state_dir=state_dir,
        run_id=str(manifest["run_id"]),
        manifest=manifest,
    )
    _atomic_json(output_dir / "self_improvement_report.json", report)
    return report


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    from openevolve.config import load_config
    from openevolve.controller import OpenEvolve

    from admission_policy import OpenEvolveAdmissionGuard
    from experience_store import ExperienceStore

    validate_experiment_args(
        experience_mode=args.experience_mode,
        psi_experiment_id=args.psi_experiment_id,
        psi_arm=args.psi_arm,
    )
    started_monotonic = time.monotonic()

    project_root = Path(__file__).resolve().parent
    initial_path = (project_root / args.initial).resolve()
    evaluator_path = (project_root / args.evaluator).resolve()
    config_path = (project_root / args.config).resolve()
    extension_path = (project_root / args.extension_config).resolve()
    output_dir = (project_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_evaluator_import(evaluator_path, project_root)

    extension = _load_extension(extension_path)
    evaluator_core_path = (
        project_root / extension.get("evaluator_core", "evaluator_core.py")
    ).resolve()
    if not evaluator_core_path.is_file():
        raise FileNotFoundError(f"evaluator core not found: {evaluator_core_path}")
    state_dir = (project_root / extension["state_dir"]).resolve()
    store = ExperienceStore(state_dir)
    manifests_path = state_dir / "run_manifests.jsonl"
    policy_trials_path = state_dir / "meta_policy_trials.jsonl"

    contract = build_run_contract(
        task_id=extension["task_id"],
        initial_path=initial_path,
        evaluator_path=evaluator_path,
        evaluator_core_path=evaluator_core_path,
        config_path=config_path,
        experiment_seed=args.experiment_seed,
    )
    existing_manifest_path = output_dir / "run_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.exists()
        else None
    )

    checkpoint: Path | None = None
    if args.resume == "auto":
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint and not existing_manifest:
            raise RuntimeError("refusing auto-resume: checkpoint has no run manifest")
        if checkpoint and not resume_is_compatible(existing_manifest, contract):
            raise RuntimeError(
                "refusing auto-resume: task/evaluator/config contract changed"
            )
    elif args.resume != "none":
        checkpoint = Path(args.resume).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")

    now = datetime.now(UTC)
    run_id = (
        str(existing_manifest["run_id"])
        if checkpoint and existing_manifest
        else args.run_id or now.strftime("%Y%m%dT%H%M%SZ")
    )
    execution_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    prior_run_manifests = [
        row
        for row in _read_jsonl(manifests_path)
        if row.get("run_id") == run_id and row.get("output_dir") == str(output_dir)
    ]

    retrieved = retrieve_experience(store, extension, mode=args.experience_mode)
    retrieved_path = output_dir / "retrieved_lessons.json"
    _atomic_json(retrieved_path, retrieved)
    os.environ["EVOLVE_RETRIEVED_LESSONS_FILE"] = str(retrieved_path)

    config = load_config(config_path)
    if args.experiment_seed is not None:
        config.random_seed = args.experiment_seed
        config.database.random_seed = args.experiment_seed
        config.llm.random_seed = args.experiment_seed
        config.llm.update_model_params(
            {"random_seed": args.experiment_seed}, overwrite=True
        )
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_id = str(raw_config.get("llm", {}).get("primary_model", "unknown"))
    meta_config = extension.get("meta_policy", {})
    candidates = list(meta_config.get("candidates", []))
    trials = [
        row
        for row in _read_jsonl(policy_trials_path)
        if row.get("task_family", extension["task_family"]) == extension["task_family"]
    ]
    if args.meta_policy:
        selected_policy = next(
            (row for row in candidates if row.get("id") == args.meta_policy),
            None,
        )
        if selected_policy is None:
            raise ValueError(f"unknown meta-policy: {args.meta_policy}")
    else:
        selected_policy = (
            select_meta_policy(candidates, trials)
            if meta_config.get("enabled") and candidates
            else {"id": "fixed"}
        )
    agent_strategy, agent_strategy_path, agent_strategy_hash = (
        load_policy_agent_strategy(project_root, selected_policy)
    )
    proposal_controller, proposal_controller_path, proposal_controller_hash = (
        load_policy_proposal_controller(project_root, selected_policy)
    )
    proposal_controller_descriptor = None
    if proposal_controller:
        proposal_controller_descriptor = verify_proposal_controller_endpoint(
            str(config.llm.api_base),
            proposal_controller,
            str(proposal_controller_hash),
            implementation_sha256=_sha256_file(project_root / "novelty_proxy.py"),
        )
    if agent_strategy:
        config.prompt.system_message = append_agent_guidance(
            str(config.prompt.system_message), agent_strategy
        )
    if "temperature" in selected_policy:
        config.llm.update_model_params(
            {"temperature": float(selected_policy["temperature"])}
        )
    if selected_policy.get("stochastic_llm"):
        # Keep database.random_seed fixed for reproducible selection while not
        # sending the same seed with every LLM request (which collapses mutation).
        config.random_seed = None
        config.llm.random_seed = None
        for model in config.llm.models:
            model.random_seed = None
    for name in ("num_top_programs", "num_diverse_programs"):
        if name in selected_policy:
            setattr(config.prompt, name, int(selected_policy[name]))
    effective_policy = dict(selected_policy)
    if args.experiment_seed is not None:
        effective_policy["experiment_seed"] = args.experiment_seed
    if agent_strategy_hash:
        effective_policy["agent_strategy_sha256"] = agent_strategy_hash
    if proposal_controller_hash:
        effective_policy["proposal_controller_sha256"] = proposal_controller_hash
    search_protocol_hash = _search_protocol_hash(config_path, effective_policy)
    config.evolution_trace.output_path = str(output_dir / "evolution_trace.jsonl")

    initial_source = initial_path.read_text(encoding="utf-8")
    initial_holdout = _score_holdout_source(initial_source, evaluator_core_path)[
        "holdout_pass_rate"
    ]
    pre_resume_best = None
    if checkpoint and (checkpoint / "best_program.py").exists():
        pre_resume_best = _score_holdout_source(
            (checkpoint / "best_program.py").read_text(encoding="utf-8"),
            evaluator_core_path,
        )["holdout_pass_rate"]

    controller = OpenEvolve(
        initial_program_path=str(initial_path),
        evaluation_file=str(evaluator_path),
        config=config,
        output_dir=str(output_dir),
    )
    guard = OpenEvolveAdmissionGuard(
        run_context={
            "run_id": run_id,
            "execution_id": execution_id,
            "task_id": extension["task_id"],
            "task_family": extension["task_family"],
            "operator_policy_id": selected_policy["id"],
        },
        event_store=store,
        behavior_equivalent_limit=int(extension.get("behavior_equivalent_limit", 2)),
    )
    guard.install(controller.database)
    best = await controller.run(
        iterations=args.iterations,
        checkpoint_path=str(checkpoint) if checkpoint else None,
    )
    if best is None:
        raise RuntimeError("OpenEvolve returned no best program")

    final_holdout = _score_holdout_source(best.code, evaluator_core_path)[
        "holdout_pass_rate"
    ]
    execution_events = [
        row for row in store.read_events() if row.get("execution_id") == execution_id
    ]
    mutation_events = [row for row in execution_events if row.get("parent_id")]
    improved_events = [
        row
        for row in mutation_events
        if row.get("accepted") and row.get("gained_cases")
    ]
    improvement_yield = len(improved_events) / max(1, len(mutation_events))

    previous_trial = trials[-1] if trials else None
    policy_trial = {
        "trial_id": execution_id,
        "run_id": run_id,
        "execution_id": execution_id,
        "policy_id": selected_policy["id"],
        "agent_strategy_id": (
            agent_strategy.get("strategy_id") if agent_strategy else None
        ),
        "agent_strategy_sha256": agent_strategy_hash,
        "proposal_controller_id": (
            proposal_controller.get("controller_id") if proposal_controller else None
        ),
        "proposal_controller_mode": (
            proposal_controller.get("mode") if proposal_controller else None
        ),
        "proposal_controller_sha256": proposal_controller_hash,
        "selection_mode": "forced" if args.meta_policy else "autonomous",
        "task_family": extension["task_family"],
        "improvement_yield": improvement_yield,
        "mutation_count": len(mutation_events),
        "accepted_improvement_count": len(improved_events),
        "operator_revision": operator_revision_label(
            previous_trial,
            selected_policy_id=str(selected_policy["id"]),
            forced=bool(args.meta_policy),
        ),
        "pre_revision_yield": (
            float(previous_trial.get("improvement_yield", 0.0))
            if previous_trial and not args.meta_policy
            else 0.0
        ),
        "post_revision_yield": improvement_yield,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _append_jsonl_unique(policy_trials_path, policy_trial, "trial_id")

    verified_promotions = 0
    for event in improved_events:
        parent = controller.database.programs.get(event.get("parent_id"))
        if parent is None:
            continue
        parent_holdout = _score_holdout_source(parent.code, evaluator_core_path)[
            "holdout_pass_rate"
        ]
        child_holdout = _score_holdout_source(event["code"], evaluator_core_path)[
            "holdout_pass_rate"
        ]
        if child_holdout < parent_holdout:
            continue
        lesson, tags = _lesson_for_gains(event.get("gained_cases", []))
        verified = {
            **{
                key: event.get(key)
                for key in (
                    "run_id",
                    "execution_id",
                    "task_id",
                    "task_family",
                    "program_id",
                    "parent_id",
                    "iteration",
                    "accepted",
                    "gained_cases",
                    "source_hash",
                    "ast_hash",
                    "behavior_signature",
                )
            },
            "event_id": f"{event['event_id']}:holdout",
            "event_type": "holdout_verification",
            "holdout_verified": True,
            "parent_holdout_score": parent_holdout,
            "child_holdout_score": child_holdout,
            "lesson": lesson,
            "tags": tags,
        }
        verified_promotions += int(store.append_event(verified))

    distilled = store.distill_lessons(
        min_evidence=int(extension.get("lesson_min_evidence", 2))
    )
    skill_path = store.render_skill_candidate(extension["task_family"])
    lesson_sources = [
        {"lesson_id": lesson["lesson_id"], "task_id": task_id}
        for lesson in retrieved
        for task_id in lesson.get("source_task_ids", [])
    ]
    manifest = {
        **contract,
        "schema_version": 1,
        "manifest_id": execution_id,
        "run_id": run_id,
        "execution_id": execution_id,
        "task_family": extension["task_family"],
        "model_id": model_id,
        "search_protocol_hash": search_protocol_hash,
        "iterations_requested": args.iterations,
        **build_staged_manifest_fields(
            prior_run_manifests,
            current_iterations=args.iterations,
            current_policy_id=str(selected_policy["id"]),
            current_controller_id=(
                str(proposal_controller["controller_id"])
                if proposal_controller
                else None
            ),
        ),
        "experience_mode": args.experience_mode,
        "psi_experiment_id": args.psi_experiment_id,
        "psi_arm": args.psi_arm,
        "started_at": now.isoformat(),
        "output_dir": str(output_dir),
        "resumed_from": str(checkpoint) if checkpoint else None,
        "resume_parent_run_id": existing_manifest.get("run_id")
        if checkpoint and existing_manifest
        else None,
        "pre_resume_best_score": pre_resume_best,
        "initial_holdout_score": initial_holdout,
        "final_holdout_score": final_holdout,
        "best_public_score": float(best.metrics.get("combined_score", 0.0)),
        "best_program_id": best.id,
        "operator_policy_id": selected_policy["id"],
        "agent_strategy_id": (
            agent_strategy.get("strategy_id") if agent_strategy else None
        ),
        "agent_strategy_status": (
            agent_strategy.get("status") if agent_strategy else None
        ),
        "agent_strategy_sha256": agent_strategy_hash,
        "agent_strategy_path": (
            str(agent_strategy_path) if agent_strategy_path else None
        ),
        "proposal_controller_id": (
            proposal_controller.get("controller_id") if proposal_controller else None
        ),
        "proposal_controller_mode": (
            proposal_controller.get("mode") if proposal_controller else None
        ),
        "proposal_controller_sha256": proposal_controller_hash,
        "proposal_controller_path": (
            str(proposal_controller_path) if proposal_controller_path else None
        ),
        "proposal_controller_calls_per_request": (
            proposal_controller.get("calls_per_request")
            if proposal_controller
            else None
        ),
        "proposal_controller_endpoint_verified": bool(proposal_controller_descriptor),
        "proposal_controller_implementation_sha256": (
            proposal_controller_descriptor.get("implementation_sha256")
            if proposal_controller_descriptor
            else None
        ),
        "meta_policy_forced": bool(args.meta_policy),
        "improvement_yield": improvement_yield,
        "retrieved_lesson_sources": lesson_sources,
        "verified_promotion_events": verified_promotions,
        "distilled_lessons": distilled,
        "skill_candidate": str(skill_path),
        "duration_seconds": time.monotonic() - started_monotonic,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(existing_manifest_path, manifest)
    _append_jsonl_unique(manifests_path, manifest, "manifest_id")

    report = build_persisted_report(
        state_dir=state_dir,
        run_id=run_id,
        manifest=manifest,
    )
    _atomic_json(output_dir / "self_improvement_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default="initial_program.py")
    parser.add_argument("--evaluator", default="evaluator.py")
    parser.add_argument("--config", default="evolve_config.yaml")
    parser.add_argument("--extension-config", default="evolve_extension.yaml")
    parser.add_argument("--output", default="runs/repaired")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--resume", default="auto", help="auto, none, or checkpoint path"
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--meta-policy", default=None)
    parser.add_argument("--experiment-seed", type=int, default=None)
    parser.add_argument(
        "--experience-mode",
        choices=("auto", "off", "cross-task"),
        default="auto",
    )
    parser.add_argument("--psi-experiment-id", default=None)
    parser.add_argument("--psi-arm", choices=("control", "transfer"), default=None)
    parser.add_argument("--report-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = (
        _rebuild_existing_report(args) if args.report_only else asyncio.run(_run(args))
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
