"""Deterministic two-target PSI matched A/B over a project-local skill registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from statistics import mean
from typing import Any

from skill_registry import SkillCandidate, SkillEvidenceRef, SkillRegistry

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = ROOT / "artifacts/v1.0.0/v0.4.0-psi-skill-library"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _load_module(path: Path) -> Any:
    module_name = f"_psi_task_{_sha256_file(path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load task evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("CASES", "HOLDOUT_CASES", "score_callable", "score_holdout_callable"):
        if not hasattr(module, name):
            raise TypeError(f"evaluator missing {name}: {path}")
    return module


def _normalize(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _build_solver(
    schema: dict[str, Any], *, skill_enabled: bool
) -> Callable[[list[Any]], list[tuple[str, float]]]:
    identity_field = str(schema["identity_field"])
    value_field = str(schema["value_field"])
    status_field = str(schema["status_field"])
    accepted_status = str(schema["accepted_status"])
    currency_field = schema.get("currency_field")
    accepted_currency = schema.get("accepted_currency")

    def solve(records: list[Any]) -> list[tuple[str, float]]:
        valid: list[tuple[str, float]] = []
        for row in records:
            if not isinstance(row, Mapping):
                continue
            identity = _normalize(row.get(identity_field))
            value = row.get(value_field)
            if identity is None or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            amount = float(value)
            if amount <= 0:
                continue
            if skill_enabled:
                if _normalize(row.get(status_field)) != _normalize(accepted_status):
                    continue
                if currency_field is not None and (
                    _normalize(row.get(str(currency_field)))
                    != _normalize(accepted_currency)
                ):
                    continue
            else:
                if row.get(status_field) != accepted_status:
                    continue
                if (
                    currency_field is not None
                    and row.get(str(currency_field)) != accepted_currency
                ):
                    continue
            valid.append((identity, amount))

        if not skill_enabled:
            return sorted(
                [(identity, round(amount, 2)) for identity, amount in valid],
                key=lambda item: (-item[1], item[0]),
            )
        totals: dict[str, float] = {}
        for identity, amount in valid:
            totals[identity] = totals.get(identity, 0.0) + amount
        return sorted(
            [(identity, round(amount, 2)) for identity, amount in totals.items()],
            key=lambda item: (-item[1], item[0]),
        )

    return solve


def _score_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed_cases": int(result["passed_cases"]),
        "total_cases": int(result["total_cases"]),
        "pass_rate": float(result["passed_cases"]) / float(result["total_cases"]),
        "failed_case_ids": sorted(
            str(row["id"]) for row in result["case_results"] if not row["passed"]
        ),
    }


def _task_contract(
    *,
    task: dict[str, Any],
    evaluator_path: Path,
    seed: int,
    runtime_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "task_schema": task["schema"],
        "evaluator_sha256": _sha256_file(evaluator_path),
        "seed": seed,
        "runtime_id": "frozen-skill-replay-agent-v1",
        "runtime_sha256": runtime_hash,
        "experiment_config_sha256": config_hash,
        "budget": 1,
    }


def _aggregate_targets(
    *,
    public_rows: list[dict[str, Any]],
    sealed_rows: list[dict[str, Any]],
    task_ids: list[str],
) -> tuple[dict[str, Any], bool, bool]:
    targets: dict[str, Any] = {}
    public_all_noninferior = True
    sealed_all_noninferior = True
    strict_benefit = False
    for task_id in task_ids:
        public_control = [
            row["score"]["pass_rate"]
            for row in public_rows
            if row["task_id"] == task_id and row["arm"] == "control"
        ]
        public_transfer = [
            row["score"]["pass_rate"]
            for row in public_rows
            if row["task_id"] == task_id and row["arm"] == "transfer"
        ]
        sealed_control = [
            row["score"]["pass_rate"]
            for row in sealed_rows
            if row["task_id"] == task_id and row["arm"] == "control"
        ]
        sealed_transfer = [
            row["score"]["pass_rate"]
            for row in sealed_rows
            if row["task_id"] == task_id and row["arm"] == "transfer"
        ]
        public_deltas = [
            transfer - control
            for control, transfer in zip(public_control, public_transfer, strict=True)
        ]
        sealed_deltas = [
            transfer - control
            for control, transfer in zip(sealed_control, sealed_transfer, strict=True)
        ]
        public_noninferior = sum(delta >= -1e-12 for delta in public_deltas)
        sealed_noninferior = sum(delta >= -1e-12 for delta in sealed_deltas)
        public_mean_delta = mean(public_transfer) - mean(public_control)
        sealed_mean_delta = mean(sealed_transfer) - mean(sealed_control)
        public_all_noninferior &= public_noninferior == len(public_deltas)
        sealed_all_noninferior &= (
            sealed_noninferior >= 2 and sealed_mean_delta >= -1e-12
        )
        strict_benefit |= public_mean_delta > 1e-12 or sealed_mean_delta > 1e-12
        targets[task_id] = {
            "public_control": public_control,
            "public_transfer": public_transfer,
            "public_mean_delta": public_mean_delta,
            "public_noninferior_seeds": public_noninferior,
            "sealed_control": sealed_control,
            "sealed_transfer": sealed_transfer,
            "sealed_mean_delta": sealed_mean_delta,
            "sealed_noninferior_seeds": sealed_noninferior,
            "strict_transfer_benefit": (
                public_mean_delta > 1e-12 or sealed_mean_delta > 1e-12
            ),
        }
    return targets, public_all_noninferior and sealed_all_noninferior, strict_benefit


def run_psi_experiment(
    *, config_path: Path, candidate_path: Path, output_dir: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("system_version") != "0.4.0":
        raise ValueError("PSI experiment must use system_version 0.4.0")
    if config.get("model_calls") != 0 or config.get("network_calls") != 0:
        raise ValueError("deterministic PSI replay cannot call models or network")
    if config.get("global_skill_installs") != 0:
        raise ValueError("PSI stage cannot install global skills")
    if config.get("observer_used_for_admission") is not False:
        raise ValueError("Observer cannot be used for PSI admission")
    if config["arms"] != {
        "control": {"skill_refs": []},
        "transfer": {"skill_refs": ["record-cleaning-invariants-v2"]},
    }:
        raise ValueError("matched PSI arms must differ only by the candidate skill")

    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = [SkillCandidate.from_dict(row) for row in candidate_data["candidates"]]
    candidate = next(
        row for row in candidates if row.skill_id == "record-cleaning-invariants-v2"
    )
    legacy = next(
        row for row in candidates if row.skill_id == "legacy-record-cleaning-bundle-v1"
    )
    registry = SkillRegistry(output_dir / "registry")
    for row in candidates:
        registry.append(row)
    semantics = {
        "accepted_status",
        "deterministic_ranking",
        "identity",
        "positive_numeric_value",
    }
    retrieved = registry.retrieve(task_family="record-cleaning", semantics=semantics)
    if [row.skill_id for row in retrieved] != [candidate.skill_id]:
        raise RuntimeError(
            "registry retrieval did not select only the predeclared candidate"
        )
    _atomic_json(
        output_dir / "retrieved-skills.json", [row.to_dict() for row in retrieved]
    )

    config_hash = _sha256_file(config_path)
    candidate_hash = _sha256_file(candidate_path)
    runtime_hash = _sha256_file(Path(__file__))
    seeds = [int(seed) for seed in config["seeds"]]
    public_rows: list[dict[str, Any]] = []
    sealed_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    modules: dict[str, Any] = {}
    evaluator_paths: dict[str, Path] = {}
    contracts: dict[tuple[str, int, str], dict[str, Any]] = {}

    for task in config["target_tasks"]:
        task_id = str(task["task_id"])
        evaluator_path = (ROOT / task["evaluator_core"]).resolve()
        module = _load_module(evaluator_path)
        modules[task_id] = module
        evaluator_paths[task_id] = evaluator_path
        for seed in seeds:
            for arm in ("control", "transfer"):
                skill_refs = list(config["arms"][arm]["skill_refs"])
                contract = _task_contract(
                    task=task,
                    evaluator_path=evaluator_path,
                    seed=seed,
                    runtime_hash=runtime_hash,
                    config_hash=config_hash,
                )
                contracts[(task_id, seed, arm)] = contract
                solver = _build_solver(task["schema"], skill_enabled=bool(skill_refs))
                score = _score_summary(module.score_callable(solver))
                row = {
                    "task_id": task_id,
                    "seed": seed,
                    "arm": arm,
                    "skill_refs": skill_refs,
                    "contract": contract,
                    "contract_hash": hashlib.sha256(
                        _canonical_json(contract).encode("utf-8")
                    ).hexdigest(),
                    "score": score,
                }
                public_rows.append(row)
                events.append(
                    {"event_type": "evaluation", "partition": "public", **row}
                )

    public_path = output_dir / "public-results.json"
    _atomic_json(public_path, public_rows)
    events.append(
        {
            "event_type": "sealed_opened",
            "public_results_persisted": public_path.is_file(),
            "public_results_sha256": _sha256_file(public_path),
        }
    )

    sealed_ids: list[str] = []
    for task in config["target_tasks"]:
        task_id = str(task["task_id"])
        module = modules[task_id]
        sealed_ids.extend(str(row["id"]) for row in module.HOLDOUT_CASES)
        for seed in seeds:
            for arm in ("control", "transfer"):
                skill_refs = list(config["arms"][arm]["skill_refs"])
                solver = _build_solver(task["schema"], skill_enabled=bool(skill_refs))
                score = _score_summary(module.score_holdout_callable(solver))
                row = {
                    "task_id": task_id,
                    "seed": seed,
                    "arm": arm,
                    "skill_refs": skill_refs,
                    "contract": contracts[(task_id, seed, arm)],
                    "contract_hash": hashlib.sha256(
                        _canonical_json(contracts[(task_id, seed, arm)]).encode("utf-8")
                    ).hexdigest(),
                    "score": score,
                }
                sealed_rows.append(row)
                events.append(
                    {"event_type": "evaluation", "partition": "sealed", **row}
                )

    sealed_path = output_dir / "sealed-results.json"
    _atomic_json(sealed_path, sealed_rows)
    _atomic_text(
        output_dir / "events.jsonl",
        "".join(_canonical_json(row) + "\n" for row in events),
    )

    matched = all(
        contracts[(task_id, seed, "control")] == contracts[(task_id, seed, "transfer")]
        for task_id in modules
        for seed in seeds
    )
    candidate_text = candidate_path.read_text(encoding="utf-8")
    no_sealed_leak = all(case_id not in candidate_text for case_id in sealed_ids)
    contract_checks = {
        "all_target_seed_arms_present": len(public_rows)
        == len(config["target_tasks"]) * len(seeds) * 2
        == len(sealed_rows),
        "only_skill_refs_changed": matched,
        "sealed_after_public_persistence": public_path.is_file()
        and events[len(public_rows)]["event_type"] == "sealed_opened",
        "target_sealed_ids_absent_from_candidate": no_sealed_leak,
        "project_local_registry_only": registry.root.is_relative_to(
            output_dir.resolve()
        )
        and config["global_skill_installs"] == 0,
        "legacy_negative_preserved": legacy.status == "rejected",
    }
    targets, capability_gate, strict_benefit = _aggregate_targets(
        public_rows=public_rows,
        sealed_rows=sealed_rows,
        task_ids=list(modules),
    )

    source_module = _load_module(
        (ROOT / config["source_task"]["evaluator_core"]).resolve()
    )
    source_schema = {
        "identity_field": "user",
        "value_field": "amount",
        "status_field": "status",
        "accepted_status": "paid",
        "currency_field": None,
        "accepted_currency": None,
    }
    source_solver = _build_solver(source_schema, skill_enabled=True)
    source_public = _score_summary(source_module.score_callable(source_solver))
    source_sealed = _score_summary(source_module.score_holdout_callable(source_solver))
    source_replay = {
        "public_passed": source_public["passed_cases"],
        "public_total": source_public["total_cases"],
        "sealed_passed": source_sealed["passed_cases"],
        "sealed_total": source_sealed["total_cases"],
    }
    source_gate = source_replay == {
        "public_passed": 13,
        "public_total": 13,
        "sealed_passed": 6,
        "sealed_total": 6,
    }
    accepted = all(contract_checks.values()) and capability_gate and source_gate
    transition_ref = SkillEvidenceRef.from_path(
        sealed_path, role="two_target_sealed_matched_ab"
    )
    terminal = registry.transition(
        skill_id=candidate.skill_id,
        new_status="transfer_verified" if accepted else "rejected",
        reason=(
            "Two-target, three-seed public/sealed matched A/B passed without source regression."
            if accepted
            else "Predeclared two-target PSI gate failed; candidate retained as rejected."
        ),
        evidence_refs=(transition_ref,),
    )
    review_path = registry.render_for_review(candidate.skill_id)

    stable = {
        "config_sha256": config_hash,
        "candidate_config_sha256": candidate_hash,
        "runtime_sha256": runtime_hash,
        "contract_checks": contract_checks,
        "targets": targets,
        "source_replay": source_replay,
        "candidate_status": terminal.status,
        "legacy_candidate_status": legacy.status,
    }
    result = {
        "schema_version": 1,
        "stage": "v0.4.0-psi-skill-library",
        **stable,
        "psi_capability": "passed" if accepted else "not_passed",
        "strict_transfer_benefit": strict_benefit,
        "decision": "accepted" if accepted else "rejected",
        "registry_path": str(registry.path),
        "review_skill_path": str(review_path),
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
        "claims": {
            "global_skill_installs": 0,
            "model_calls": 0,
            "network_calls": 0,
            "observer_used_for_admission": False,
            "model_weights_frozen": True,
            "production_ready": False,
        },
    }
    _atomic_json(
        output_dir / "evidence.json",
        {
            "contract_checks": contract_checks,
            "historical_negative": legacy.to_dict(),
            "targets": targets,
            "source_replay": source_replay,
            "sealed_results_sha256": _sha256_file(sealed_path),
        },
    )
    _atomic_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_STAGE / "configs/experiment.json"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_STAGE / "configs/skill-candidates.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_psi_experiment(
        config_path=args.config,
        candidate_path=args.candidates,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "psi_capability": result["psi_capability"],
                "strict_transfer_benefit": result["strict_transfer_benefit"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "candidate_status": result["candidate_status"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
