"""Run the predeclared v0.8 durability, migration, and failure matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from durable_service import (
    ContractConflict,
    DurableOperationService,
    HardeningContractError,
    InjectedCrash,
    OperationBusy,
    OperationExecutionError,
    Runner,
    migrate_database,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "artifacts/v1.0.0/v0.8.0-hardening/configs/experiment.json"


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


def _derived_config(
    base: dict[str, Any], path: Path, suffix: str, **changes: Any
) -> Path:
    data = {**base, "operation_id": f"{base['operation_id']}-{suffix}", **changes}
    _atomic_json(path, data)
    return path


def _events(service: DurableOperationService) -> list[str]:
    return [row["event_type"] for row in service.list_events()]


def _legacy_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE operations (
                operation_id TEXT PRIMARY KEY,
                contract_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE TABLE events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );
            INSERT INTO operations VALUES
                ('legacy-operation', 'legacy-contract', 'prepared', 1, 1);
            INSERT INTO events(operation_id, event_type, payload_json, created_ns)
                VALUES ('legacy-operation', 'prepared', '{}', 1);
            """
        )


def _database_integrity(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return str(connection.execute("PRAGMA integrity_check").fetchone()[0])


def _service(
    *,
    config: Path,
    database: Path,
    operations: Path,
    runner: Runner | None,
) -> DurableOperationService:
    return DurableOperationService(
        config_path=config,
        database_path=database,
        output_root=operations,
        runner=runner,
    )


def run_hardening_experiment(
    *,
    config_path: Path,
    output_dir: Path,
    integration_runner: Runner | None = None,
) -> dict[str, Any]:
    base = json.loads(config_path.read_text(encoding="utf-8"))
    if base.get("stage") != "v0.8.0-hardening":
        raise HardeningContractError("hardening experiment requires v0.8 stage")
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = output_dir / "scenario-configs"
    scenarios_root = output_dir / "scenarios"

    normal_service = _service(
        config=_derived_config(base, configs / "normal.json", "normal"),
        database=scenarios_root / "normal/service.sqlite3",
        operations=scenarios_root / "normal/operations",
        runner=integration_runner,
    )
    normal_first = normal_service.run(worker_id="normal-a", now_ns=1_000)
    normal_replay = normal_service.run(worker_id="normal-b", now_ns=2_000)

    prepare_service = _service(
        config=_derived_config(base, configs / "after-prepare.json", "prepare-crash"),
        database=scenarios_root / "after-prepare/service.sqlite3",
        operations=scenarios_root / "after-prepare/operations",
        runner=integration_runner,
    )
    prepare_crash_observed = False
    try:
        prepare_service.run(
            worker_id="prepare-a", now_ns=1_000, crash_point="after_prepare"
        )
    except InjectedCrash:
        prepare_crash_observed = True
    prepared_status = prepare_service.inspect_operation()["status"]
    prepare_recovered = prepare_service.run(worker_id="prepare-b", now_ns=2_000)

    integration_service = _service(
        config=_derived_config(
            base, configs / "after-integration.json", "integration-crash"
        ),
        database=scenarios_root / "after-integration/service.sqlite3",
        operations=scenarios_root / "after-integration/operations",
        runner=integration_runner,
    )
    integration_crash_observed = False
    try:
        integration_service.run(
            worker_id="integration-a",
            now_ns=1_000,
            crash_point="after_integration_before_commit",
        )
    except InjectedCrash:
        integration_crash_observed = True
    running_status = integration_service.inspect_operation()["status"]
    busy_before_expiry = False
    try:
        integration_service.run(worker_id="integration-b", now_ns=1_001)
    except OperationBusy:
        busy_before_expiry = True
    after_expiry_ns = int(base["lease_seconds"]) * 1_000_000_000 + 1_001
    integration_recovered = integration_service.run(
        worker_id="integration-b", now_ns=after_expiry_ns
    )

    corrupt_service = _service(
        config=_derived_config(base, configs / "corrupt.json", "corrupt-result"),
        database=scenarios_root / "corrupt-result/service.sqlite3",
        operations=scenarios_root / "corrupt-result/operations",
        runner=integration_runner,
    )
    corrupt_first = corrupt_service.run(worker_id="corrupt-a", now_ns=1_000)
    corrupt_target = Path(corrupt_first["result_path"])
    original_copy = corrupt_target.parent / "pre-injection-result.json"
    shutil.copyfile(corrupt_target, original_copy)
    original_sha256 = _sha256_file(original_copy)
    corrupt_target.write_bytes(corrupt_target.read_bytes()[:8])
    truncated_sha256 = _sha256_file(corrupt_target)
    corrupt_recovered = corrupt_service.run(worker_id="corrupt-b", now_ns=2_000)
    preserved_corrupt = sorted(corrupt_target.parent.glob("result.json.corrupt-*"))

    legacy_database = scenarios_root / "migration/legacy.sqlite3"
    _legacy_database(legacy_database)
    migration_report = migrate_database(legacy_database)
    with sqlite3.connect(legacy_database) as connection:
        preserved_operations = int(
            connection.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
        )
        preserved_events = int(
            connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )
        legacy_attempt = int(
            connection.execute(
                "SELECT attempt FROM operations WHERE operation_id='legacy-operation'"
            ).fetchone()[0]
        )

    changed_contract_rejected = False
    changed_path = _derived_config(
        base,
        configs / "changed-contract.json",
        "normal",
        max_runtime_seconds=int(base["max_runtime_seconds"]) + 1,
    )
    try:
        _service(
            config=changed_path,
            database=normal_service.database_path,
            operations=normal_service.output_root,
            runner=integration_runner,
        ).run(worker_id="conflict", now_ns=3_000)
    except ContractConflict:
        changed_contract_rejected = True

    validation_results: dict[str, str] = {}
    invalid_specs = {
        "outside_root": {"integration_config": "/private/tmp/outside.json"},
        "input_budget": {"max_input_bytes": 1},
        "unknown_field": {"unexpected": True},
    }
    for name, changes in invalid_specs.items():
        invalid_path = _derived_config(
            base, configs / f"invalid-{name}.json", f"invalid-{name}", **changes
        )
        try:
            _service(
                config=invalid_path,
                database=scenarios_root / f"invalid-{name}/service.sqlite3",
                operations=scenarios_root / f"invalid-{name}/operations",
                runner=integration_runner,
            )
        except HardeningContractError as error:
            validation_results[name] = type(error).__name__

    def timeout_runner(**_: Any) -> dict[str, Any]:
        raise OperationExecutionError("trusted integration timed out after 30s")

    timeout_service = _service(
        config=_derived_config(base, configs / "timeout.json", "timeout"),
        database=scenarios_root / "timeout/service.sqlite3",
        operations=scenarios_root / "timeout/operations",
        runner=timeout_runner,
    )
    timeout_observed = False
    try:
        timeout_service.run(worker_id="timeout-a", now_ns=1_000)
    except OperationExecutionError:
        timeout_observed = True
    timeout_row = timeout_service.inspect_operation()

    artifact_service = _service(
        config=_derived_config(
            base,
            configs / "artifact-budget.json",
            "artifact-budget",
            max_artifact_bytes=16,
        ),
        database=scenarios_root / "artifact-budget/service.sqlite3",
        operations=scenarios_root / "artifact-budget/operations",
        runner=integration_runner,
    )
    artifact_budget_observed = False
    try:
        artifact_service.run(worker_id="artifact-a", now_ns=1_000)
    except OperationExecutionError:
        artifact_budget_observed = True
    artifact_row = artifact_service.inspect_operation()

    services = {
        "normal": normal_service,
        "after_prepare": prepare_service,
        "after_integration": integration_service,
        "corrupt_result": corrupt_service,
        "timeout": timeout_service,
        "artifact_budget": artifact_service,
    }
    sqlite_integrity = {
        name: _database_integrity(service.database_path)
        for name, service in services.items()
    }
    sqlite_integrity["migration"] = _database_integrity(legacy_database)
    completed_results = [
        json.loads(Path(record["result_path"]).read_text(encoding="utf-8"))
        for record in (
            normal_first,
            prepare_recovered,
            integration_recovered,
            corrupt_recovered,
        )
    ]
    integration_authority_preserved = all(
        result.get("decision") == "accepted"
        and result.get("contract_checks")
        and all(result["contract_checks"].values())
        for result in completed_results
    )
    integration_fingerprints = sorted(
        {result["experiment_fingerprint"] for result in completed_results}
    )
    scenarios = {
        "normal": {
            "attempt": normal_first["attempt"],
            "events": _events(normal_service),
            "idempotent_replay": normal_replay["idempotent_replay"],
        },
        "after_prepare": {
            "crash_observed": prepare_crash_observed,
            "events": _events(prepare_service),
            "final_status": prepare_recovered["status"],
            "prepared_status_before_recovery": prepared_status,
        },
        "after_integration": {
            "attempt": integration_recovered["attempt"],
            "busy_before_expiry": busy_before_expiry,
            "crash_observed": integration_crash_observed,
            "events": _events(integration_service),
            "final_status": integration_recovered["status"],
            "running_status_before_recovery": running_status,
        },
        "corrupt_result": {
            "attempt": corrupt_recovered["attempt"],
            "events": _events(corrupt_service),
            "original_sha256": original_sha256,
            "preserved_corrupt_files": len(preserved_corrupt),
            "truncated_sha256": truncated_sha256,
        },
        "migration": {
            **migration_report,
            "legacy_attempt": legacy_attempt,
            "preserved_events": preserved_events,
            "preserved_operations": preserved_operations,
            "preserved_rows": preserved_operations == 1 and preserved_events == 1,
        },
        "timeout": {
            "events": _events(timeout_service),
            "final_status": timeout_row["status"],
            "observed": timeout_observed,
        },
        "artifact_budget": {
            "events": _events(artifact_service),
            "final_status": artifact_row["status"],
            "observed": artifact_budget_observed,
            "result_published": artifact_row["result_path"] is not None,
        },
        "validation": validation_results,
    }
    contract_checks = {
        "artifact_budget_fails_closed": artifact_budget_observed
        and artifact_row["status"] == "failed"
        and artifact_row["result_path"] is None,
        "changed_contract_rejected": changed_contract_rejected,
        "completed_replay_is_idempotent": normal_replay["idempotent_replay"]
        and normal_first["attempt"] == normal_replay["attempt"] == 1,
        "corrupt_result_preserved_and_rebuilt": len(preserved_corrupt) == 1
        and corrupt_recovered["attempt"] == 2
        and preserved_corrupt[0].read_bytes() == original_copy.read_bytes()[:8],
        "crash_after_integration_recovers_same_attempt": integration_crash_observed
        and busy_before_expiry
        and integration_recovered["attempt"] == 1
        and integration_recovered["status"] == "completed",
        "crash_after_prepare_recovers": prepare_crash_observed
        and prepared_status == "prepared"
        and prepare_recovered["status"] == "completed",
        "integration_authority_preserved": integration_authority_preserved,
        "legacy_migration_preserves_rows": migration_report["to_version"] == 2
        and preserved_operations == 1
        and preserved_events == 1
        and legacy_attempt == 0,
        "path_schema_and_budget_preflight": validation_results
        == {
            "input_budget": "HardeningContractError",
            "outside_root": "HardeningContractError",
            "unknown_field": "HardeningContractError",
        },
        "sqlite_integrity_ok": set(sqlite_integrity.values()) == {"ok"},
        "timeout_fails_closed": timeout_observed
        and timeout_row["status"] == "failed"
        and timeout_row["result_path"] is None,
    }
    claims = {
        "distributed_exactly_once": False,
        "global_skill_installs": 0,
        "model_calls": 0,
        "model_weights_frozen": True,
        "network_calls": 0,
        "production_ready": False,
    }
    stable = {
        "contract_checks": contract_checks,
        "integration_fingerprints": integration_fingerprints,
        "scenario_semantics": {
            name: {
                key: value
                for key, value in scenario.items()
                if key not in {"original_sha256", "truncated_sha256"}
            }
            for name, scenario in scenarios.items()
        },
        "claims": claims,
    }
    result = {
        "schema_version": 1,
        "stage": "v0.8.0-hardening",
        "decision": "accepted" if all(contract_checks.values()) else "rejected",
        "contract_checks": contract_checks,
        "scenarios": scenarios,
        "sqlite_integrity": sqlite_integrity,
        "integration_fingerprints": integration_fingerprints,
        "claims": claims,
        "experiment_fingerprint": hashlib.sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(output_dir / "scenario-summary.json", scenarios)
    _atomic_json(
        output_dir / "evidence.json",
        {
            "contract_checks": contract_checks,
            "integration_fingerprints": integration_fingerprints,
            "sqlite_integrity": sqlite_integrity,
        },
    )
    _atomic_json(output_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_hardening_experiment(config_path=args.config, output_dir=args.output)
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
