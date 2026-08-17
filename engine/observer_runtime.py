"""Asynchronous, failure-isolated observer matrix over the v0.2 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_program_runtime import run_agent_program_experiment
from observation_artifact import ObservationArtifact, collect_observation

ROOT = Path(__file__).resolve().parent
DEFAULT_OBSERVER_CONFIG = (
    ROOT / "artifacts/v1.0.0/v0.3.0-jlens-observer/configs/experiment.json"
)
DEFAULT_AGENT_CONFIGS = ROOT / "artifacts/v1.0.0/v0.2.0-agent-program/configs"
DEFAULT_RUNS = ROOT / "artifacts/v1.0.0/v0.3.0-jlens-observer/runs"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


def _config_hashes(
    observer_config_path: Path, agent_config_dir: Path
) -> dict[str, str]:
    paths = {
        "observer_experiment": observer_config_path,
        "agent_baseline": agent_config_dir / "baseline_agent_program.json",
        "agent_registry": agent_config_dir / "component_registry.json",
        "agent_proposals": agent_config_dir / "replay_proposals.json",
        "agent_experiment": agent_config_dir / "experiment.json",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _failed_artifact(
    *,
    mode: str,
    runtime_result: dict[str, Any],
    config_hashes: dict[str, str],
    status: str,
    error: str,
) -> ObservationArtifact:
    artifact = ObservationArtifact(
        schema_version=1,
        artifact_id=(
            f"observation-{mode}-{runtime_result['outcome_fingerprint'][:12]}-{status}"
        ),
        observer_mode=mode,
        status=status,
        runtime_outcome_fingerprint=runtime_result["outcome_fingerprint"],
        active_program_hash=runtime_result["final"]["program_hash"],
        config_hashes=config_hashes,
        source_refs=(),
        features={},
        causal_boundary="observational_not_causal",
        used_for_admission=False,
        error=error,
    )
    artifact.validate()
    return artifact


def _raise_injected_failure(mode: str) -> None:
    raise RuntimeError(f"injected observer failure: {mode}")


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "statuses": [row["artifact"]["status"] for row in rows],
        "runtime_outcome_fingerprints": [
            row["runtime"]["outcome_fingerprint"] for row in rows
        ],
        "unique_runtime_outcome_fingerprints": len(
            {row["runtime"]["outcome_fingerprint"] for row in rows}
        ),
        "active_program_hashes": [
            row["runtime"]["final"]["program_hash"] for row in rows
        ],
        "runtime_decisions": [row["runtime"]["decision"] for row in rows],
        "public_final_means": [
            row["runtime"]["final"]["public_passed_mean"] for row in rows
        ],
        "sealed_final_means": [
            row["runtime"]["sealed_audit"]["final_passed_mean"] for row in rows
        ],
        "artifact_fingerprints": [
            row["artifact"]["artifact_fingerprint"] for row in rows
        ],
        "unique_artifact_fingerprints": len(
            {row["artifact"]["artifact_fingerprint"] for row in rows}
        ),
    }


def run_observer_matrix(
    *,
    observer_config_path: Path,
    agent_config_dir: Path,
    output_dir: Path,
    replays_per_mode: int | None = None,
    inject_failure_mode: str | None = None,
) -> dict[str, Any]:
    config = json.loads(observer_config_path.read_text(encoding="utf-8"))
    if config.get("system_version") != "0.3.0":
        raise ValueError("observer experiment must use system_version 0.3.0")
    if config.get("admission_uses_observer") is not False:
        raise ValueError("observer cannot be used for admission")
    if (
        int(config.get("network_calls", -1)) != 0
        or int(config.get("model_downloads", -1)) != 0
    ):
        raise ValueError("observer replay must not use network or model downloads")
    modes = tuple(config["observer_modes"])
    if modes != ("off", "trace", "logit_lens", "jlens"):
        raise ValueError("observer mode matrix does not match the frozen protocol")
    repeats = int(replays_per_mode or config["replays_per_mode"])
    if repeats < 1:
        raise ValueError("replays_per_mode must be positive")
    if inject_failure_mode is not None and inject_failure_mode not in set(modes) - {
        "off"
    }:
        raise ValueError("failure injection requires trace, logit_lens, or jlens")
    timeout = float(config["observer_timeout_seconds"])
    lens_source = Path(config["historical_lens_source"])
    config_hashes = _config_hashes(observer_config_path, agent_config_dir)
    rows_by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    events: list[dict[str, Any]] = []

    for mode in modes:
        for replay in range(1, repeats + 1):
            run_dir = output_dir / "conditions" / mode / f"replay-{replay}" / "runtime"
            runtime_result = run_agent_program_experiment(
                baseline_path=agent_config_dir / "baseline_agent_program.json",
                registry_path=agent_config_dir / "component_registry.json",
                proposals_path=agent_config_dir / "replay_proposals.json",
                experiment_path=agent_config_dir / "experiment.json",
                output_dir=run_dir,
            )
            result_path = run_dir / "result.json"
            if not result_path.is_file():
                raise RuntimeError(
                    "runtime result was not persisted before observer submit"
                )
            events.append(
                {
                    "event_type": "runtime_complete",
                    "observer_mode": mode,
                    "replay": replay,
                    "result_path": str(result_path.resolve()),
                    "runtime_outcome_fingerprint": runtime_result[
                        "outcome_fingerprint"
                    ],
                }
            )
            executor = ThreadPoolExecutor(max_workers=1)
            if inject_failure_mode == mode:
                future = executor.submit(_raise_injected_failure, mode)
            else:
                future = executor.submit(
                    collect_observation,
                    mode=mode,
                    runtime_run_dir=run_dir,
                    lens_source=lens_source,
                    config_hashes=config_hashes,
                )
            events.append(
                {
                    "event_type": "observer_submitted",
                    "observer_mode": mode,
                    "replay": replay,
                    "runtime_already_persisted": result_path.is_file(),
                }
            )
            try:
                artifact = future.result(timeout=timeout)
            except FutureTimeout:
                future.cancel()
                artifact = _failed_artifact(
                    mode=mode,
                    runtime_result=runtime_result,
                    config_hashes=config_hashes,
                    status="timeout",
                    error=f"observer timeout after {timeout} seconds",
                )
            except Exception as exc:  # noqa: BLE001 - isolation evidence.
                artifact = _failed_artifact(
                    mode=mode,
                    runtime_result=runtime_result,
                    config_hashes=config_hashes,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            artifact_path = output_dir / "observations" / f"{mode}-replay-{replay}.json"
            _atomic_text(
                artifact_path,
                json.dumps(
                    artifact.to_dict(), indent=2, sort_keys=True, ensure_ascii=False
                )
                + "\n",
            )
            events.append(
                {
                    "event_type": "observer_complete",
                    "observer_mode": mode,
                    "replay": replay,
                    "status": artifact.status,
                    "artifact_path": str(artifact_path.resolve()),
                    "artifact_fingerprint": artifact.artifact_fingerprint,
                }
            )
            rows_by_mode[mode].append(
                {"runtime": runtime_result, "artifact": artifact.to_dict()}
            )

    conditions = {mode: _condition_summary(rows) for mode, rows in rows_by_mode.items()}
    all_rows = [row for rows in rows_by_mode.values() for row in rows]
    runtime_fingerprints = {row["runtime"]["outcome_fingerprint"] for row in all_rows}
    active_hashes = {row["runtime"]["final"]["program_hash"] for row in all_rows}
    runtime_views = {
        _canonical_json(
            {
                "baseline": row["runtime"]["baseline"],
                "steps": row["runtime"]["steps"],
                "final": row["runtime"]["final"],
                "sealed": row["runtime"]["sealed_audit"],
                "decision": row["runtime"]["decision"],
            }
        )
        for row in all_rows
    }
    mechanism_checks = {
        "one_runtime_outcome_fingerprint": len(runtime_fingerprints) == 1,
        "one_active_program_hash": len(active_hashes) == 1,
        "all_runtime_results_equal": len(runtime_views) == 1,
        "runtime_persisted_before_observer_submit": all(
            event.get("runtime_already_persisted", True)
            for event in events
            if event["event_type"] == "observer_submitted"
        ),
        "observer_never_used_for_admission": all(
            row["artifact"]["used_for_admission"] is False for row in all_rows
        ),
        "artifact_replay_stable_per_mode": all(
            condition["unique_artifact_fingerprints"] == 1
            for condition in conditions.values()
        ),
    }
    logit_completed = [
        row["artifact"]
        for row in rows_by_mode["logit_lens"]
        if row["artifact"]["status"] == "completed"
    ]
    jlens_completed = [
        row["artifact"]
        for row in rows_by_mode["jlens"]
        if row["artifact"]["status"] == "completed"
    ]
    if logit_completed and jlens_completed:
        logit_metric = float(logit_completed[0]["features"]["score_eta_squared"])
        jlens_metric = float(jlens_completed[0]["features"]["score_eta_squared"])
        source_incremental = bool(
            jlens_completed[0]["features"]["jlens_incremental_supported"]
        )
        advantage = jlens_metric - logit_metric
        supported = bool(
            advantage >= float(config["jlens_required_margin"]) and source_incremental
        )
        incremental = {
            "metric": config["incremental_metric"],
            "logit_lens": logit_metric,
            "jlens": jlens_metric,
            "advantage": advantage,
            "required_margin": float(config["jlens_required_margin"]),
            "source_incremental_supported": source_incremental,
            "conclusion": "supported" if supported else "not_supported",
        }
    else:
        incremental = {
            "metric": config["incremental_metric"],
            "conclusion": "unavailable_due_to_observer_failure",
        }
    failure_injection = None
    if inject_failure_mode:
        isolated = bool(
            conditions[inject_failure_mode]["statuses"] == ["failed"] * repeats
            and len(runtime_fingerprints) == 1
            and len(active_hashes) == 1
            and set(conditions[inject_failure_mode]["runtime_decisions"])
            == {"accepted"}
        )
        failure_injection = {"mode": inject_failure_mode, "isolated": isolated}

    expected_status = {
        "off": "disabled",
        "trace": "completed",
        "logit_lens": "completed",
        "jlens": "completed",
    }
    status_gate = all(
        condition["statuses"] == [expected_status[mode]] * repeats
        for mode, condition in conditions.items()
    )
    mechanism_accepted = all(mechanism_checks.values()) and (
        failure_injection["isolated"] if failure_injection else status_gate
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "config_hashes": config_hashes,
        "replays_per_mode": repeats,
        "conditions": conditions,
        "mechanism_checks": mechanism_checks,
        "failure_injection": failure_injection,
        "jlens_incremental": incremental,
        "decision": "accepted" if mechanism_accepted else "rejected",
        "production_ready": False,
        "claims": {
            "observer_isolated": mechanism_accepted,
            "observer_used_for_admission": False,
            "jlens_incremental_gain_proven": (
                incremental.get("conclusion") == "supported"
            ),
            "model_downloads": 0,
            "network_calls": 0,
        },
    }
    stable = {
        key: result[key]
        for key in (
            "config_hashes",
            "replays_per_mode",
            "conditions",
            "mechanism_checks",
            "failure_injection",
            "jlens_incremental",
            "decision",
            "claims",
        )
    }
    result["matrix_fingerprint"] = hashlib.sha256(
        _canonical_json(stable).encode("utf-8")
    ).hexdigest()
    for sequence, event in enumerate(events, start=1):
        event["sequence"] = sequence
    _atomic_text(
        output_dir / "matrix-events.jsonl",
        "".join(_canonical_json(event) + "\n" for event in events),
    )
    _atomic_text(
        output_dir / "evidence.json",
        json.dumps(
            {
                "mechanism_checks": mechanism_checks,
                "failure_injection": failure_injection,
                "jlens_incremental": incremental,
                "causal_boundary": "observational_not_causal",
                "observer_used_for_admission": False,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )
    _atomic_text(
        output_dir / "matrix-result.json",
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_OBSERVER_CONFIG)
    parser.add_argument("--agent-configs", type=Path, default=DEFAULT_AGENT_CONFIGS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int)
    parser.add_argument("--inject-failure-mode")
    args = parser.parse_args()
    result = run_observer_matrix(
        observer_config_path=args.config,
        agent_config_dir=args.agent_configs,
        output_dir=args.output,
        replays_per_mode=args.replays,
        inject_failure_mode=args.inject_failure_mode,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "matrix_fingerprint": result["matrix_fingerprint"],
                "runtime_fingerprint_count": len(
                    {
                        fingerprint
                        for condition in result["conditions"].values()
                        for fingerprint in condition["runtime_outcome_fingerprints"]
                    }
                ),
                "jlens_incremental": result["jlens_incremental"],
                "failure_injection": result["failure_injection"],
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
