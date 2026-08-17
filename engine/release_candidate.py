"""Build and audit the v0.9 three-operation release candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from durable_service import DurableOperationService

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/configs/experiment.json"
)
_POST_RC_FILES = frozenset(
    {
        "artifact_verifier.py",
        "evolve_service.py",
        "release_verification.py",
        "tests/test_artifact_verifier.py",
        "tests/test_evolve_service.py",
        "tests/test_release_verification.py",
        # v2 sources are deliberately excluded from the frozen v1 clean-room RC.
        "codex_changeset.py",
        "codex_evolution_runtime.py",
        "codex_target_runtime.py",
        "changeset_adapter.py",
        "live_codex_ab.py",
        "meta_evolution_runtime.py",
        "multi_model_eval.py",
        "swe_bench_adapter.py",
        "swe_cloud_runner.py",
        "v2_artifacts.py",
        "tests/test_codex_changeset.py",
        "tests/test_codex_evolution_runtime.py",
        "tests/test_codex_target_runtime.py",
        "tests/test_changeset_adapter.py",
        "tests/test_live_codex_ab.py",
        "tests/test_meta_evolution_runtime.py",
        "tests/test_multi_model_eval.py",
        "tests/test_swe_bench_adapter.py",
        "tests/test_swe_cloud_runner.py",
        "tests/test_v2_artifacts.py",
        # v2.1 sources are likewise outside the frozen v1 clean-room RC.
        "benchmark_adapters.py",
        "benchmark_catalog.py",
        "continuous_ab.py",
        "continuous_ab_service.py",
        "continuous_ab_smoke.py",
        "freeze_benchmark_pool.py",
        "verify_continuous_ab.py",
        "v21_artifacts.py",
        "evolve_jlens_harbor.py",
        "pilot_admission.py",
        "benchmark_execution.py",
        "agent_arm_runner.py",
        "native_result_adapter.py",
        "tests/test_benchmark_adapters.py",
        "tests/test_benchmark_catalog.py",
        "tests/test_continuous_ab.py",
        "tests/test_continuous_ab_service.py",
        "tests/test_continuous_ab_smoke.py",
        "tests/test_freeze_benchmark_pool.py",
        "tests/test_verify_continuous_ab.py",
        "tests/test_v21_artifacts.py",
        "tests/test_evolve_jlens_harbor.py",
        "tests/test_pilot_admission.py",
        "tests/test_benchmark_execution.py",
        "tests/test_agent_arm_runner.py",
        "tests/test_native_result_adapter.py",
        # v2.1.1 mutation/population/selection engine is outside frozen v1 RC.
        "pattern_miner.py",
        "mutation_proposer.py",
        "evolution_archive.py",
        "candidate_tournament.py",
        "evolution_controller.py",
        "evolution_runtime.py",
        "evolution_report.py",
        "evolution_fixture.py",
        "tests/test_pattern_miner.py",
        "tests/test_mutation_proposer.py",
        "tests/test_evolution_archive.py",
        "tests/test_candidate_tournament.py",
        "tests/test_evolution_controller.py",
        "tests/test_evolution_integration.py",
        "real_evolution_bridge.py",
        "real_mutation_proposer.py",
        "codex_mutation_caller.py",
        "trace_observer.py",
        "real_workspace_factory.py",
        "official_patch_evaluator.py",
        "loopback_connect_proxy.py",
        "real_evolution_run.py",
        "tests/test_real_evolution_bridge.py",
        "tests/test_real_mutation_proposer.py",
        "tests/test_codex_mutation_caller.py",
        "tests/test_trace_observer.py",
        "tests/test_real_workspace_factory.py",
        "tests/test_official_patch_evaluator.py",
        "tests/test_loopback_connect_proxy.py",
        "tests/test_real_evolution_run.py",
    }
)


class ReleaseCandidateError(ValueError):
    """Raised when the frozen RC contract or evidence tree is invalid."""


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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _partition_order(events: list[dict[str, Any]]) -> dict[str, Any]:
    public_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("partition") == "public"
    ]
    sealed_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("partition") == "sealed"
    ]
    opened_indexes = [
        index
        for index, event in enumerate(events)
        if event.get("event_type") == "sealed_opened"
    ]
    persisted = all(
        event.get("public_results_persisted") is True
        for event in events
        if event.get("event_type") == "sealed_opened"
    )
    valid = (
        bool(public_indexes)
        and bool(sealed_indexes)
        and len(opened_indexes) == 1
        and max(public_indexes) < opened_indexes[0] < min(sealed_indexes)
        and persisted
    )
    return {
        "first_sealed_index": min(sealed_indexes) if sealed_indexes else None,
        "last_public_index": max(public_indexes) if public_indexes else None,
        "public_results_persisted": persisted,
        "sealed_opened_index": opened_indexes[0] if len(opened_indexes) == 1 else None,
        "valid": valid,
    }


def _audit_operation(
    *, record: dict[str, Any], idempotent: dict[str, Any]
) -> dict[str, Any]:
    attempt_dir = Path(record["result_path"]).parent
    components = attempt_dir / "components"
    integration_result = json.loads(
        (attempt_dir / "result.json").read_text(encoding="utf-8")
    )
    agent_events = _jsonl(components / "agent-program/events.jsonl")
    psi_events = _jsonl(components / "psi/events.jsonl")
    agent_order = _partition_order(agent_events)
    psi_order = _partition_order(psi_events)
    agent_sealed_phases = {
        event.get("phase")
        for event in agent_events
        if event.get("partition") == "sealed"
    }
    agent_seeds = sorted(
        {
            int(event["seed"])
            for event in agent_events
            if event.get("event_type") == "evaluation" and "seed" in event
        }
    )
    psi_seeds = sorted(
        {
            int(event["seed"])
            for event in psi_events
            if event.get("event_type") == "evaluation" and "seed" in event
        }
    )
    psi_tasks = sorted(
        {
            str(event["task_id"])
            for event in psi_events
            if event.get("event_type") == "evaluation" and "task_id" in event
        }
    )
    observer = json.loads(
        (components / "observer-formal/matrix-result.json").read_text(encoding="utf-8")
    )
    psi = json.loads((components / "psi/result.json").read_text(encoding="utf-8"))
    code = json.loads(
        (components / "agent-code/result.json").read_text(encoding="utf-8")
    )
    shadow = json.loads(
        (components / "evaluator-shadow/result.json").read_text(encoding="utf-8")
    )
    observer_envelopes = [
        envelope
        for envelope in integration_result["envelopes"]
        if envelope["authority"] == "observe"
    ]
    return {
        "operation_id": record["operation_id"],
        "attempt": record["attempt"],
        "status": record["status"],
        "idempotent_replay": idempotent["idempotent_replay"],
        "integration_fingerprint": integration_result["experiment_fingerprint"],
        "integration_contract_checks": integration_result["contract_checks"],
        "agent_program_seeds": agent_seeds,
        "psi_seeds": psi_seeds,
        "psi_tasks": psi_tasks,
        "agent_public_before_sealed": agent_order["valid"],
        "agent_partition_order": agent_order,
        "agent_sealed_phases": sorted(agent_sealed_phases),
        "psi_public_before_sealed": psi_order["valid"],
        "psi_partition_order": psi_order,
        "observer_used_for_admission": any(
            envelope["used_for_admission"] for envelope in observer_envelopes
        ),
        "jlens_incremental": observer["jlens_incremental"]["conclusion"],
        "skill_candidate_status": psi["candidate_status"],
        "skill_active": False,
        "global_skill_installs": psi["claims"]["global_skill_installs"],
        "code_rollback_to_parent": code["rollback"]["performed"]
        and code["rollback"]["final_active_sha256"] == code["parent"]["source_sha256"],
        "evaluator_active_unchanged": shadow["active_evaluator_before"]
        == shadow["active_evaluator_after"],
        "evaluator_auto_promoted": shadow["review_proposal"]["auto_promoted"],
        "model_calls": integration_result["claims"]["model_calls"],
        "network_calls": integration_result["claims"]["network_calls"],
    }


_BUNDLE_MANIFEST = ROOT / "artifacts/v1.0.0/v1.0.0-release/bundle-manifest.json"


def _bundle_files() -> list[Path]:
    """Return the frozen bundle membership (issue #6 root fix).

    Membership is read from an explicit version-pinned manifest instead of
    globbing the tree, so adding new root/test/task modules no longer changes
    the release bundle fingerprint.  Content changes to listed files still
    require a deliberate re-pin (that is the intended signal).
    """
    if _BUNDLE_MANIFEST.is_file():
        manifest = json.loads(_BUNDLE_MANIFEST.read_text(encoding="utf-8"))
        missing = [
            rel
            for rel in manifest.get("files", ())
            if not (ROOT / rel).is_file()
        ]
        if missing:
            raise ReleaseCandidateError(
                f"bundle manifest lists missing files: {missing[:10]}"
            )
        return sorted(
            {(ROOT / rel).resolve() for rel in manifest.get("files", ())}
        )
    files = list(ROOT.glob("*.py"))
    files.extend(path for path in (ROOT / "tests").rglob("*.py") if path.is_file())
    files.extend(path for path in (ROOT / "tasks").rglob("*.py") if path.is_file())
    for stage in (
        "v0.2.0-agent-program",
        "v0.3.0-jlens-observer",
        "v0.4.0-psi-skill-library",
        "v0.5.0-agent-code-mutation",
        "v0.6.0-evaluator-shadow",
        "v0.7.0-integration",
        "v0.8.0-hardening",
    ):
        configs = ROOT / "artifacts/v1.0.0" / stage / "configs"
        files.extend(path for path in configs.rglob("*") if path.is_file())
    return sorted(
        {
            path.resolve()
            for path in files
            if path.resolve().relative_to(ROOT).as_posix() not in _POST_RC_FILES
        }
    )


def _build_bundle(path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    with tarfile.open(path, "w") as archive:
        for source in _bundle_files():
            relative = source.relative_to(ROOT)
            archive.add(source, arcname=str(relative), recursive=False)
            members.append(str(relative))
    return members


def _safe_tar_members(archive: tarfile.TarFile) -> bool:
    for member in archive.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            return False
    return True


def _run_checked(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONHASHSEED": "0",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _clean_room_replay(
    *, bundle: Path, output_dir: Path, core_tests: list[str]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="evolve-rc-clean-room-") as directory:
        clean_root = Path(directory) / "source"
        clean_root.mkdir()
        with tarfile.open(bundle, "r") as archive:
            safe_members = _safe_tar_members(archive)
            if not safe_members:
                raise ReleaseCandidateError("unsafe source bundle member")
            archive.extractall(clean_root, filter="data")
        import_check = _run_checked(
            [
                sys.executable,
                "-c",
                "import durable_service, integration_runtime, plugin_envelope",
            ],
            cwd=clean_root,
            timeout=30,
        )
        tests = _run_checked(
            [sys.executable, "-m", "pytest", "-q", *core_tests],
            cwd=clean_root,
            timeout=60,
        )
        replay_output = clean_root / "clean-replay"
        replay = _run_checked(
            [
                sys.executable,
                "integration_runtime.py",
                "--config",
                "artifacts/v1.0.0/v0.7.0-integration/configs/experiment.json",
                "--output",
                str(replay_output),
            ],
            cwd=clean_root,
            timeout=60,
        )
        replay_result = json.loads(
            (replay_output / "result.json").read_text(encoding="utf-8")
        )
        (output_dir / "import.stdout.txt").write_text(
            import_check.stdout, encoding="utf-8"
        )
        (output_dir / "tests.stdout.txt").write_text(tests.stdout, encoding="utf-8")
        (output_dir / "replay.stdout.txt").write_text(replay.stdout, encoding="utf-8")
        _atomic_json(output_dir / "replay-result.json", replay_result)
    test_summary = tests.stdout.strip().splitlines()[-1]
    try:
        tests_passed = int(test_summary.split()[0])
    except (IndexError, ValueError) as error:
        raise ReleaseCandidateError(
            f"cannot parse clean-room pytest summary: {test_summary}"
        ) from error
    return {
        "bundle_sha256": _sha256_file(bundle),
        "cli_replay_decision": replay_result["decision"],
        "cli_replay_fingerprint": replay_result["experiment_fingerprint"],
        "cli_replay_checks_passed": all(replay_result["contract_checks"].values()),
        "core_tests_passed": tests.returncode == 0,
        "import_passed": import_check.returncode == 0,
        "safe_members": safe_members,
        "test_summary": test_summary,
        "tests_passed": tests_passed,
    }


def _semantic_clean_room(clean_room: dict[str, Any]) -> dict[str, Any]:
    """Drop content hashes and wall-clock text from RC semantic identity."""
    return {
        key: value
        for key, value in clean_room.items()
        if key not in {"bundle_sha256", "test_summary"}
    }


def _docs_audit(stage_root: Path, names: list[str]) -> dict[str, Any]:
    required_tokens = {
        "OPERATIONS.zh-CN.md": ["release_candidate.py", "sqlite3"],
        "ARCHITECTURE.zh-CN.md": ["Observer", "Admission Gate"],
        "EVIDENCE.zh-CN.md": ["观察证据", "Sealed"],
        "ROLLBACK.zh-CN.md": ["回滚", "Durable"],
        "LIMITATIONS.zh-CN.md": ["限制", "production"],
    }
    rows = {}
    for name in names:
        path = stage_root / name
        content = path.read_text(encoding="utf-8") if path.is_file() else ""
        rows[name] = {
            "exists": path.is_file(),
            "nonempty": len(content) > 100,
            "required_tokens": all(
                token in content for token in required_tokens.get(name, [])
            ),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
    return {"documents": rows, "valid": all(all(row.values()) for row in rows.values())}


def run_release_candidate(*, config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("system_version") != "0.9.0" or config.get("schema_version") != 1:
        raise ReleaseCandidateError("unsupported RC config")
    if config.get("stage") != "v0.9.0-release-candidate":
        raise ReleaseCandidateError("invalid RC stage")
    if any(
        config.get(field) != 0
        for field in ("model_calls", "network_calls", "global_skill_installs")
    ):
        raise ReleaseCandidateError("RC external and model calls must remain zero")
    operation_ids = list(config["operation_ids"])
    if len(operation_ids) != 3 or len(set(operation_ids)) != 3:
        raise ReleaseCandidateError("RC requires three unique operation ids")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = (ROOT / config["base_durable_config"]).resolve()
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    database = output_dir / "rc-service.sqlite3"
    operations = output_dir / "operations"
    derived_configs = output_dir / "operation-configs"
    records: list[dict[str, Any]] = []
    replays: list[dict[str, Any]] = []
    for index, operation_id in enumerate(operation_ids, start=1):
        operation_config = derived_configs / f"{operation_id}.json"
        _atomic_json(operation_config, {**base, "operation_id": operation_id})
        service = DurableOperationService(
            config_path=operation_config,
            database_path=database,
            output_root=operations,
        )
        records.append(
            service.run(worker_id=f"rc-worker-{index}", now_ns=index * 1_000)
        )
        replays.append(
            service.run(worker_id=f"rc-replay-{index}", now_ns=index * 1_000 + 1)
        )
    audits = [
        _audit_operation(record=record, idempotent=replay)
        for record, replay in zip(records, replays, strict=True)
    ]
    bundle = output_dir / "clean-room/source-bundle.tar"
    bundle_members = _build_bundle(bundle)
    clean_room = _clean_room_replay(
        bundle=bundle,
        output_dir=output_dir / "clean-room",
        core_tests=list(config["clean_room_core_tests"]),
    )
    clean_room["member_count"] = len(bundle_members)
    docs = _docs_audit(config_path.parent.parent, list(config["required_docs"]))
    with sqlite3.connect(database) as connection:
        database_rows = [
            {"operation_id": row[0], "status": row[1], "attempt": row[2]}
            for row in connection.execute(
                "SELECT operation_id, status, attempt FROM operations ORDER BY operation_id"
            ).fetchall()
        ]
        database_integrity = str(
            connection.execute("PRAGMA integrity_check").fetchone()[0]
        )
    expected_seeds = list(config["expected_seeds"])
    expected_tasks = sorted(config["expected_tasks"])
    expected_fingerprint = str(config["expected_integration_fingerprint"])
    contract_checks = {
        "clean_room_replay_passed": clean_room["safe_members"]
        and clean_room["import_passed"]
        and clean_room["core_tests_passed"]
        and clean_room["cli_replay_decision"] == "accepted"
        and clean_room["cli_replay_checks_passed"]
        and clean_room["cli_replay_fingerprint"] == expected_fingerprint,
        "documentation_complete": docs["valid"],
        "durable_rows_complete": database_integrity == "ok"
        and len(database_rows) == 3
        and all(
            row["status"] == "completed" and row["attempt"] == 1
            for row in database_rows
        ),
        "evaluator_governance_preserved": all(
            audit["evaluator_active_unchanged"] and not audit["evaluator_auto_promoted"]
            for audit in audits
        ),
        "integration_fingerprint_stable": {
            audit["integration_fingerprint"] for audit in audits
        }
        == {expected_fingerprint},
        "model_and_network_frozen": all(
            audit["model_calls"] == 0 and audit["network_calls"] == 0
            for audit in audits
        ),
        "observer_is_diagnostic_only": all(
            not audit["observer_used_for_admission"]
            and audit["jlens_incremental"] == "not_supported"
            for audit in audits
        ),
        "operation_ids_unique_and_idempotent": {
            audit["operation_id"] for audit in audits
        }
        == set(operation_ids)
        and all(audit["idempotent_replay"] for audit in audits),
        "public_precedes_sealed": all(
            audit["agent_public_before_sealed"]
            and audit["psi_public_before_sealed"]
            and audit["agent_sealed_phases"] == ["post_search_audit"]
            for audit in audits
        ),
        "seed_and_task_coverage": all(
            audit["agent_program_seeds"] == expected_seeds
            and audit["psi_seeds"] == expected_seeds
            and audit["psi_tasks"] == expected_tasks
            for audit in audits
        ),
        "skill_and_code_governance_preserved": all(
            audit["skill_candidate_status"] == "transfer_verified"
            and not audit["skill_active"]
            and audit["global_skill_installs"] == 0
            and audit["code_rollback_to_parent"]
            for audit in audits
        ),
        "v0_7_contract_checks_preserved": all(
            all(audit["integration_contract_checks"].values()) for audit in audits
        ),
    }
    claims = {
        "clean_room_is_cross_machine": False,
        "global_skill_installs": 0,
        "model_calls": 0,
        "model_weights_frozen": True,
        "network_calls": 0,
        "production_ready": False,
        "three_operations_are_new_distributions": False,
    }
    stable = {
        "contract_checks": contract_checks,
        "operation_audits": audits,
        "clean_room": _semantic_clean_room(clean_room),
        "claims": claims,
    }
    result = {
        "schema_version": 1,
        "stage": "v0.9.0-release-candidate",
        "decision": "accepted" if all(contract_checks.values()) else "rejected",
        "contract_checks": contract_checks,
        "operation_audits": audits,
        "database_rows": database_rows,
        "database_integrity": database_integrity,
        "clean_room": clean_room,
        "documentation": docs,
        "claims": claims,
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(
        output_dir / "evidence.json",
        {"contract_checks": contract_checks, "database_rows": database_rows},
    )
    _atomic_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_release_candidate(config_path=args.config, output_dir=args.output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "experiment_fingerprint": result["experiment_fingerprint"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0 if result["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
