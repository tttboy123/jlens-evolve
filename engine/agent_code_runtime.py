"""Bounded Agent harness code-mutation experiment with rollback evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from agent_code_mutation import MutationArchive, validate_source
from sandbox_runner import run_candidate

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = ROOT / "artifacts/v1.0.0/v0.5.0-agent-code-mutation"


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


def _score_pair(runs: list[dict[str, Any]]) -> list[int]:
    unique = {(row["passed_cases"], row["total_cases"]) for row in runs}
    if len(unique) != 1:
        raise RuntimeError("seed replay produced inconsistent deterministic scores")
    passed, total = unique.pop()
    return [int(passed), int(total)]


def _passed_case_ids(run: dict[str, Any]) -> set[str]:
    return {str(row["id"]) for row in run["case_results"] if row["passed"]}


def _run_partition(
    *,
    candidate_id: str,
    source: str,
    cases: list[dict[str, Any]],
    partition: str,
    seeds: list[int],
    limits: dict[str, Any],
    sandbox_parent: Path,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        run = run_candidate(
            source=source,
            cases=cases,
            limits=limits,
            sandbox_parent=sandbox_parent,
        )
        events.append(
            {
                "event_type": "sandbox_execution",
                "candidate_id": candidate_id,
                "partition": partition,
                "seed": seed,
                "status": run["status"],
                "sandbox": run["sandbox"],
            }
        )
        events.append(
            {
                "event_type": "evaluation",
                "candidate_id": candidate_id,
                "partition": partition,
                "seed": seed,
                "passed_cases": run["passed_cases"],
                "total_cases": run["total_cases"],
                "case_results": run["case_results"],
            }
        )
        if run["status"] != "completed":
            raise RuntimeError(
                f"sandbox failed for allowlisted candidate: {candidate_id}"
            )
        rows.append({"seed": seed, **run})
    return rows


def run_agent_code_experiment(
    *,
    config_path: Path,
    evaluator_path: Path,
    candidate_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluator = json.loads(evaluator_path.read_text(encoding="utf-8"))
    manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if config.get("system_version") != "0.5.0":
        raise ValueError("code mutation experiment must use system_version 0.5.0")
    if config.get("model_calls") != 0 or config.get("network_calls") != 0:
        raise ValueError("bounded code mutation cannot call model or network")
    if config.get("observer_used_for_admission") is not False:
        raise ValueError("Observer cannot be used for code admission")
    candidates = list(manifest["candidates"])
    if len(candidates) != int(config["mutation_budget"]):
        raise ValueError("candidate count does not match frozen mutation budget")
    stage_root = candidate_manifest_path.resolve().parents[1]
    seeds = [int(seed) for seed in config["seeds"]]
    limits = dict(config["limits"])
    hashes_before = {
        "config": _sha256_file(config_path),
        "evaluator": _sha256_file(evaluator_path),
        "candidate_manifest": _sha256_file(candidate_manifest_path),
    }
    parent_meta = manifest["parent"]
    parent_path = stage_root / parent_meta["path"]
    parent_source = parent_path.read_text(encoding="utf-8")
    parent_hash = _sha256_file(parent_path)
    if parent_hash != parent_meta["source_sha256"]:
        raise ValueError("parent source hash mismatch")
    parent_static = validate_source(parent_source, limits=limits)
    if not parent_static["allowed"]:
        raise ValueError("frozen parent does not pass capability gate")

    archive = MutationArchive(output_dir / "archive")
    archive.initialize_active(
        candidate_id=parent_meta["candidate_id"], source_sha256=parent_hash
    )
    active_before_candidates = archive.read_active()
    events: list[dict[str, Any]] = []
    sandbox_parent = output_dir / "sandbox-work"
    public_cases = list(evaluator["public_cases"])
    sealed_cases = list(evaluator["sealed_cases"])
    parent_public_runs = _run_partition(
        candidate_id=parent_meta["candidate_id"],
        source=parent_source,
        cases=public_cases,
        partition="public",
        seeds=seeds,
        limits=limits,
        sandbox_parent=sandbox_parent,
        events=events,
    )
    parent_public = _score_pair(parent_public_runs)
    parent_passed_ids = _passed_case_ids(parent_public_runs[0])
    capability_reports: dict[str, Any] = {parent_meta["candidate_id"]: parent_static}
    summaries: dict[str, dict[str, Any]] = {}
    public_payload: dict[str, Any] = {parent_meta["candidate_id"]: parent_public_runs}
    public_winners: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []

    for metadata in candidates:
        candidate_id = str(metadata["candidate_id"])
        path = stage_root / metadata["path"]
        source = path.read_text(encoding="utf-8")
        source_hash = _sha256_file(path)
        if source_hash != metadata["source_sha256"]:
            raise ValueError(f"candidate source hash mismatch: {candidate_id}")
        if metadata["parent_source_sha256"] != parent_hash:
            raise ValueError(f"candidate parent hash mismatch: {candidate_id}")
        report = validate_source(source, limits=limits)
        capability_reports[candidate_id] = report
        events.append(
            {
                "event_type": "static_gate",
                "candidate_id": candidate_id,
                "allowed": report["allowed"],
                "reasons": report["reasons"],
            }
        )
        if not report["allowed"]:
            summary = {
                "status": "rejected_static",
                "executed": False,
                "public": None,
                "sealed": None,
                "source_sha256": source_hash,
                "reasons": report["reasons"],
            }
            summaries[candidate_id] = summary
            archive.append_record(
                {
                    "candidate_id": candidate_id,
                    "source_sha256": source_hash,
                    "parent_source_sha256": parent_hash,
                    "status": summary["status"],
                    "static_report": report,
                    "executed": False,
                }
            )
            continue
        public_runs = _run_partition(
            candidate_id=candidate_id,
            source=source,
            cases=public_cases,
            partition="public",
            seeds=seeds,
            limits=limits,
            sandbox_parent=sandbox_parent,
            events=events,
        )
        public_payload[candidate_id] = public_runs
        public_score = _score_pair(public_runs)
        preserves_parent = all(
            parent_passed_ids.issubset(_passed_case_ids(run)) for run in public_runs
        )
        public_gate = public_score[0] > parent_public[0] and preserves_parent
        if not public_gate:
            summary = {
                "status": "rejected_public",
                "executed": True,
                "public": public_score,
                "sealed": None,
                "source_sha256": source_hash,
                "reasons": ["public_not_strictly_better_or_parent_case_regression"],
            }
            summaries[candidate_id] = summary
            archive.append_record(
                {
                    "candidate_id": candidate_id,
                    "source_sha256": source_hash,
                    "parent_source_sha256": parent_hash,
                    "status": summary["status"],
                    "static_report": report,
                    "public": public_score,
                    "executed": True,
                }
            )
            continue
        public_winners.append((metadata, source, public_runs))
        summaries[candidate_id] = {
            "status": "public_gate_passed",
            "executed": True,
            "public": public_score,
            "sealed": None,
            "source_sha256": source_hash,
            "reasons": [],
        }

    public_path = output_dir / "public-results.json"
    _atomic_json(public_path, public_payload)
    events.append(
        {
            "event_type": "sealed_opened",
            "public_results_persisted": public_path.is_file(),
            "public_winners": [row[0]["candidate_id"] for row in public_winners],
        }
    )
    parent_sealed_runs = _run_partition(
        candidate_id=parent_meta["candidate_id"],
        source=parent_source,
        cases=sealed_cases,
        partition="sealed",
        seeds=seeds,
        limits=limits,
        sandbox_parent=sandbox_parent,
        events=events,
    )
    parent_sealed = _score_pair(parent_sealed_runs)
    sealed_payload: dict[str, Any] = {parent_meta["candidate_id"]: parent_sealed_runs}
    verified: list[tuple[str, str]] = []
    for metadata, source, _ in public_winners:
        candidate_id = str(metadata["candidate_id"])
        sealed_runs = _run_partition(
            candidate_id=candidate_id,
            source=source,
            cases=sealed_cases,
            partition="sealed",
            seeds=seeds,
            limits=limits,
            sandbox_parent=sandbox_parent,
            events=events,
        )
        sealed_payload[candidate_id] = sealed_runs
        sealed_score = _score_pair(sealed_runs)
        noninferior_seeds = sum(
            row["passed_cases"] >= parent_sealed[0] for row in sealed_runs
        )
        status = (
            "verified"
            if sealed_score[0] >= parent_sealed[0] and noninferior_seeds >= 2
            else "rejected_sealed"
        )
        summaries[candidate_id]["status"] = status
        summaries[candidate_id]["sealed"] = sealed_score
        if status == "verified":
            verified.append((candidate_id, str(metadata["source_sha256"])))
        archive.append_record(
            {
                "candidate_id": candidate_id,
                "source_sha256": metadata["source_sha256"],
                "parent_source_sha256": parent_hash,
                "status": status,
                "static_report": capability_reports[candidate_id],
                "public": summaries[candidate_id]["public"],
                "sealed": sealed_score,
                "sealed_noninferior_seeds": noninferior_seeds,
                "executed": True,
            }
        )
    sealed_path = output_dir / "sealed-results.json"
    _atomic_json(sealed_path, sealed_payload)

    rejected_active_unchanged = archive.read_active() == active_before_candidates
    rollback = {"performed": False, "final_active_sha256": parent_hash}
    verified_candidate_id = None
    if len(verified) == 1:
        verified_candidate_id, verified_hash = verified[0]
        archive.activate(
            candidate_id=verified_candidate_id, source_sha256=verified_hash
        )
        rolled_back = archive.rollback(reason="predeclared rollback drill")
        rollback = {
            "performed": True,
            "verified_active_sha256": verified_hash,
            "final_active_sha256": rolled_back["source_sha256"],
        }

    hashes_after = {
        "config": _sha256_file(config_path),
        "evaluator": _sha256_file(evaluator_path),
        "candidate_manifest": _sha256_file(candidate_manifest_path),
    }
    unsafe_ids = {
        candidate_id
        for candidate_id, summary in summaries.items()
        if summary["status"] == "rejected_static"
    }
    sandbox_events = [
        event for event in events if event["event_type"] == "sandbox_execution"
    ]
    resource_minimum_applied = all(
        all(
            event["sandbox"]["resource_limits"].get(name, {}).get("applied") is True
            for name in ("cpu_seconds", "file_bytes", "open_files")
        )
        for event in sandbox_events
    )
    contract_checks = {
        "evaluator_unchanged": hashes_before["evaluator"] == hashes_after["evaluator"],
        "config_unchanged": hashes_before["config"] == hashes_after["config"],
        "candidate_manifest_unchanged": hashes_before["candidate_manifest"]
        == hashes_after["candidate_manifest"],
        "unsafe_candidates_never_executed": not any(
            event["candidate_id"] in unsafe_ids for event in sandbox_events
        ),
        "public_rejections_never_saw_sealed": all(
            summary["sealed"] is None
            for summary in summaries.values()
            if summary["status"] in {"rejected_static", "rejected_public"}
        ),
        "rejected_candidates_never_changed_active": rejected_active_unchanged,
        "exactly_one_verified": len(verified) == 1,
        "rollback_restored_parent": rollback["performed"]
        and rollback["final_active_sha256"] == parent_hash,
        "sandbox_cwd_remained_empty": all(
            event["sandbox"]["files_after"] == [] for event in sandbox_events
        ),
        "minimum_resource_limits_applied": resource_minimum_applied,
        "expected_candidate_verdicts_match": all(
            summaries[row["candidate_id"]]["status"] == row["expected_gate"]
            for row in candidates
        ),
    }
    accepted = all(contract_checks.values())
    stable = {
        "hashes": hashes_before,
        "parent": {
            "candidate_id": parent_meta["candidate_id"],
            "source_sha256": parent_hash,
            "public": parent_public,
            "sealed": parent_sealed,
        },
        "candidates": summaries,
        "verified_candidate_id": verified_candidate_id,
        "rollback": rollback,
        "contract_checks": contract_checks,
    }
    result = {
        "schema_version": 1,
        "stage": "v0.5.0-agent-code-mutation",
        **stable,
        "decision": "accepted" if accepted else "rejected",
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
        "claims": {
            "arbitrary_python_safe": False,
            "allowlist_only": True,
            "model_calls": 0,
            "network_calls": 0,
            "observer_used_for_admission": False,
            "production_ready": False,
        },
    }
    for sequence, event in enumerate(events, 1):
        event["sequence"] = sequence
    _atomic_text(
        output_dir / "events.jsonl",
        "".join(_canonical_json(event) + "\n" for event in events),
    )
    _atomic_json(output_dir / "capability-reports.json", capability_reports)
    _atomic_json(
        output_dir / "evidence.json",
        {
            "contract_checks": contract_checks,
            "active_before_candidates": active_before_candidates,
            "final_active": archive.read_active(),
            "archive_records": archive.read_records(),
            "rollback": rollback,
            "public_results_sha256": _sha256_file(public_path),
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
        "--evaluator",
        type=Path,
        default=DEFAULT_STAGE / "configs/harness-evaluator.json",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_STAGE / "configs/candidates.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_agent_code_experiment(
        config_path=args.config,
        evaluator_path=args.evaluator,
        candidate_manifest_path=args.candidates,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "verified_candidate_id": result["verified_candidate_id"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "rollback": result["rollback"],
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
