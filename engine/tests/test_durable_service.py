import json
import sqlite3
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from durable_service import (
    ContractConflict,
    DurableOperationService,
    HardeningContractError,
    InjectedCrash,
    OperationBusy,
    OperationExecutionError,
    migrate_database,
)

ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "artifacts/v1.0.0/v0.8.0-hardening/configs/experiment.json"


def _config(tmp_path: Path, operation_id: str = "test-operation") -> Path:
    data = json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))
    data["operation_id"] = operation_id
    path = tmp_path / f"{operation_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(
        self, *, integration_config: Path, output_dir: Path, timeout_seconds: int
    ) -> dict:
        self.calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "decision": "accepted",
            "experiment_fingerprint": "a" * 64,
            "contract_checks": {"authority_preserved": True},
            "integration_config": str(integration_config),
            "timeout_seconds": timeout_seconds,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result


def _service(tmp_path: Path, runner: RecordingRunner) -> DurableOperationService:
    return DurableOperationService(
        config_path=_config(tmp_path),
        database_path=tmp_path / "service.sqlite3",
        output_root=tmp_path / "operations",
        runner=runner,
    )


def test_normal_completion_is_idempotent_without_second_execution(tmp_path: Path):
    runner = RecordingRunner()
    service = _service(tmp_path, runner)

    first = service.run(worker_id="worker-a", now_ns=1_000)
    second = service.run(worker_id="worker-b", now_ns=2_000)

    assert first["status"] == "completed"
    assert first["attempt"] == 1
    assert second["idempotent_replay"] is True
    assert second["result_sha256"] == first["result_sha256"]
    assert len(runner.calls) == 1
    assert [event["event_type"] for event in service.list_events()] == [
        "prepared",
        "claimed",
        "completed",
        "idempotent_replay",
    ]


def test_after_prepare_crash_resumes_from_durable_prepared_state(tmp_path: Path):
    runner = RecordingRunner()
    service = _service(tmp_path, runner)

    with pytest.raises(InjectedCrash, match="after_prepare"):
        service.run(worker_id="worker-a", now_ns=1_000, crash_point="after_prepare")

    assert service.inspect_operation()["status"] == "prepared"
    recovered = service.run(worker_id="worker-b", now_ns=2_000)
    assert recovered["status"] == "completed"
    assert recovered["attempt"] == 1
    assert len(runner.calls) == 1


def test_crash_after_integration_requires_lease_expiry_then_reuses_attempt(
    tmp_path: Path,
):
    runner = RecordingRunner()
    service = _service(tmp_path, runner)

    with pytest.raises(InjectedCrash, match="after_integration_before_commit"):
        service.run(
            worker_id="worker-a",
            now_ns=1_000,
            crash_point="after_integration_before_commit",
        )

    running = service.inspect_operation()
    assert running["status"] == "running"
    assert running["attempt"] == 1
    with pytest.raises(OperationBusy):
        service.run(worker_id="worker-b", now_ns=1_001)

    recovered = service.run(worker_id="worker-b", now_ns=31_000_000_002)
    assert recovered["status"] == "completed"
    assert recovered["attempt"] == 1
    assert runner.calls == [runner.calls[0]]
    assert "lease_recovered" in {event["event_type"] for event in service.list_events()}


def test_concurrent_worker_observes_busy_lease_without_double_execution(
    tmp_path: Path,
):
    started = threading.Event()
    release = threading.Event()
    calls: list[Path] = []

    def blocking_runner(
        *, integration_config: Path, output_dir: Path, timeout_seconds: int
    ) -> dict:
        calls.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started.set()
        assert release.wait(timeout=2)
        result = {
            "decision": "accepted",
            "experiment_fingerprint": "b" * 64,
            "contract_checks": {"authority_preserved": True},
        }
        (output_dir / "result.json").write_text(
            json.dumps(result) + "\n", encoding="utf-8"
        )
        return result

    config_path = _config(tmp_path)
    service_a = DurableOperationService(
        config_path=config_path,
        database_path=tmp_path / "service.sqlite3",
        output_root=tmp_path / "operations",
        runner=blocking_runner,
    )
    service_b = DurableOperationService(
        config_path=config_path,
        database_path=tmp_path / "service.sqlite3",
        output_root=tmp_path / "operations",
        runner=blocking_runner,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(service_a.run, worker_id="worker-a", now_ns=1_000)
        assert started.wait(timeout=2)
        with pytest.raises(OperationBusy):
            service_b.run(worker_id="worker-b", now_ns=1_001)
        release.set()
        assert first.result(timeout=2)["status"] == "completed"

    assert len(calls) == 1
    assert service_a.inspect_operation()["attempt"] == 1


def test_changed_contract_for_same_operation_is_rejected(tmp_path: Path):
    runner = RecordingRunner()
    service = _service(tmp_path, runner)
    service.run(worker_id="worker-a", now_ns=1_000)

    changed = json.loads(service.config_path.read_text(encoding="utf-8"))
    changed["max_runtime_seconds"] += 1
    service.config_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ContractConflict):
        DurableOperationService(
            config_path=service.config_path,
            database_path=service.database_path,
            output_root=service.output_root,
            runner=runner,
        ).run(worker_id="worker-b", now_ns=2_000)
    assert len(runner.calls) == 1


def test_truncated_completed_result_is_preserved_and_rebuilt(tmp_path: Path):
    runner = RecordingRunner()
    service = _service(tmp_path, runner)
    first = service.run(worker_id="worker-a", now_ns=1_000)
    result_path = Path(first["result_path"])
    original_bytes = result_path.read_bytes()
    result_path.write_bytes(original_bytes[:8])

    recovered = service.run(worker_id="worker-b", now_ns=2_000)

    assert recovered["status"] == "completed"
    assert recovered["attempt"] == 2
    corrupt = list(result_path.parent.glob("result.json.corrupt-*"))
    assert len(corrupt) == 1
    assert corrupt[0].read_bytes() == original_bytes[:8]
    assert len(runner.calls) == 2
    assert "integrity_failure" in {
        event["event_type"] for event in service.list_events()
    }


def test_v1_database_migrates_to_v2_without_losing_operation(tmp_path: Path):
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
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
            INSERT INTO operations VALUES ('legacy-op', 'abc', 'prepared', 1, 1);
            INSERT INTO events(operation_id, event_type, payload_json, created_ns)
            VALUES ('legacy-op', 'prepared', '{}', 1);
            """
        )

    report = migrate_database(database)

    assert report == {"from_version": 1, "to_version": 2, "migrated": True}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM operations WHERE operation_id='legacy-op'"
        ).fetchone()
        assert row["status"] == "prepared"
        assert row["attempt"] == 0
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_input_bytes", 1, "input budget exceeded"),
        ("max_runtime_seconds", 0, "positive integer"),
        ("unknown", 1, "unknown config fields"),
    ],
)
def test_invalid_or_over_budget_config_is_rejected_before_execution(
    tmp_path: Path, field: str, value: int, message: str
):
    runner = RecordingRunner()
    config_path = _config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data[field] = value
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(HardeningContractError, match=message):
        DurableOperationService(
            config_path=config_path,
            database_path=tmp_path / "service.sqlite3",
            output_root=tmp_path / "operations",
            runner=runner,
        )
    assert runner.calls == []


def test_input_path_outside_project_is_rejected_before_execution(tmp_path: Path):
    runner = RecordingRunner()
    config_path = _config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["integration_config"] = str(tmp_path / "outside.json")
    (tmp_path / "outside.json").write_text("{}", encoding="utf-8")
    config_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(HardeningContractError, match="outside allowed input root"):
        DurableOperationService(
            config_path=config_path,
            database_path=tmp_path / "service.sqlite3",
            output_root=tmp_path / "operations",
            runner=runner,
        )
    assert runner.calls == []


def test_timeout_is_recorded_as_failed_not_completed(tmp_path: Path, monkeypatch):
    config_path = _config(tmp_path)
    service = DurableOperationService(
        config_path=config_path,
        database_path=tmp_path / "service.sqlite3",
        output_root=tmp_path / "operations",
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr("durable_service.subprocess.run", timeout)

    with pytest.raises(OperationExecutionError, match="timed out"):
        service.run(worker_id="worker-a", now_ns=1_000)
    assert service.inspect_operation()["status"] == "failed"
    assert service.list_events()[-1]["event_type"] == "failed"


def test_artifact_budget_overflow_is_failed_and_not_admitted(tmp_path: Path):
    runner = RecordingRunner()
    config_path = _config(tmp_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["max_artifact_bytes"] = 16
    config_path.write_text(json.dumps(data), encoding="utf-8")
    service = DurableOperationService(
        config_path=config_path,
        database_path=tmp_path / "service.sqlite3",
        output_root=tmp_path / "operations",
        runner=runner,
    )

    with pytest.raises(OperationExecutionError, match="artifact budget exceeded"):
        service.run(worker_id="worker-a", now_ns=1_000)

    assert service.inspect_operation()["status"] == "failed"
    assert service.inspect_operation()["result_path"] is None
