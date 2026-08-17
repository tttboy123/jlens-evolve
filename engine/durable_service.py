"""SQLite-backed crash recovery wrapper for the trusted v0.7 integration CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
INTEGRATION_RUNTIME = ROOT / "integration_runtime.py"
DEFAULT_CONFIG = ROOT / "artifacts/v1.0.0/v0.8.0-hardening/configs/experiment.json"
_SCHEMA_VERSION = 2
_CONFIG_FIELDS = frozenset(
    {
        "allowed_input_root",
        "database_schema_version",
        "global_skill_installs",
        "integration_config",
        "lease_seconds",
        "max_artifact_bytes",
        "max_input_bytes",
        "max_runtime_seconds",
        "model_calls",
        "network_calls",
        "operation_id",
        "schema_version",
        "stage",
        "system_version",
    }
)


class HardeningContractError(ValueError):
    """Raised before execution when the durable service contract is invalid."""


class ContractConflict(HardeningContractError):
    """Raised when an existing operation id is reused with another contract."""


class OperationBusy(RuntimeError):
    """Raised when another worker owns an unexpired operation lease."""


class OperationExecutionError(RuntimeError):
    """Raised when the trusted integration execution cannot complete."""


class InjectedCrash(RuntimeError):
    """Deterministic test-only crash at a predeclared durable checkpoint."""


Runner = Callable[..., dict[str, Any]]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def migrate_database(path: Path) -> dict[str, Any]:
    """Create schema v2 or transactionally migrate the frozen v1 fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, 1, _SCHEMA_VERSION}:
            raise HardeningContractError(f"unsupported database schema: {version}")
        if version == 0:
            connection.executescript(
                """
                CREATE TABLE operations (
                    operation_id TEXT PRIMARY KEY,
                    contract_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('prepared', 'running', 'completed', 'failed')),
                    attempt INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_ns INTEGER,
                    result_path TEXT,
                    result_sha256 TEXT,
                    semantic_fingerprint TEXT,
                    error TEXT,
                    created_ns INTEGER NOT NULL,
                    updated_ns INTEGER NOT NULL
                );
                CREATE TABLE events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_ns INTEGER NOT NULL,
                    FOREIGN KEY(operation_id) REFERENCES operations(operation_id)
                );
                CREATE INDEX events_operation_sequence
                    ON events(operation_id, sequence);
                PRAGMA user_version = 2;
                """
            )
            return {"from_version": 0, "to_version": 2, "migrated": True}
        if version == 1:
            connection.execute("BEGIN IMMEDIATE")
            for statement in (
                "ALTER TABLE operations ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE operations ADD COLUMN lease_owner TEXT",
                "ALTER TABLE operations ADD COLUMN lease_expires_ns INTEGER",
                "ALTER TABLE operations ADD COLUMN result_path TEXT",
                "ALTER TABLE operations ADD COLUMN result_sha256 TEXT",
                "ALTER TABLE operations ADD COLUMN semantic_fingerprint TEXT",
                "ALTER TABLE operations ADD COLUMN error TEXT",
            ):
                connection.execute(statement)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS events_operation_sequence "
                "ON events(operation_id, sequence)"
            )
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
            return {"from_version": 1, "to_version": 2, "migrated": True}
        return {"from_version": 2, "to_version": 2, "migrated": False}


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class DurableOperationService:
    """Own durable execution state while leaving admission inside v0.7."""

    def __init__(
        self,
        *,
        config_path: Path,
        database_path: Path,
        output_root: Path,
        runner: Runner | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.database_path = database_path.resolve()
        self.output_root = output_root.resolve()
        self.runner = runner or self._run_trusted_integration
        self.config = self._load_config()
        self.integration_config = self._resolve_integration_config()
        self.contract_hash = _sha256_bytes(
            _canonical_json(
                {
                    "service_config": self.config,
                    "integration_config_sha256": _sha256_file(self.integration_config),
                    "service_source_sha256": _sha256_file(Path(__file__)),
                    "integration_source_sha256": _sha256_file(INTEGRATION_RUNTIME),
                }
            ).encode("utf-8")
        )
        migrate_database(self.database_path)

    @property
    def operation_id(self) -> str:
        return str(self.config["operation_id"])

    def _load_config(self) -> dict[str, Any]:
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        unknown = sorted(data.keys() - _CONFIG_FIELDS)
        missing = sorted(_CONFIG_FIELDS - data.keys())
        if unknown:
            raise HardeningContractError(f"unknown config fields: {unknown}")
        if missing:
            raise HardeningContractError(f"missing config fields: {missing}")
        if data["schema_version"] != 1 or data["system_version"] != "0.8.0":
            raise HardeningContractError("unsupported service config version")
        if data["stage"] != "v0.8.0-hardening":
            raise HardeningContractError("invalid hardening stage")
        if data["database_schema_version"] != _SCHEMA_VERSION:
            raise HardeningContractError("database schema target must be 2")
        for field in (
            "lease_seconds",
            "max_artifact_bytes",
            "max_input_bytes",
            "max_runtime_seconds",
        ):
            value = data[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HardeningContractError(f"{field} must be a positive integer")
        for field in ("global_skill_installs", "model_calls", "network_calls"):
            if data[field] != 0:
                raise HardeningContractError(f"{field} must remain zero")
        operation_id = data["operation_id"]
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-."
                for character in operation_id
            )
        ):
            raise HardeningContractError("invalid operation_id")
        return data

    def _resolve_integration_config(self) -> Path:
        allowed_root = (ROOT / self.config["allowed_input_root"]).resolve()
        integration_config = (ROOT / self.config["integration_config"]).resolve()
        if not integration_config.is_relative_to(allowed_root):
            raise HardeningContractError(
                "integration config outside allowed input root"
            )
        if not integration_config.is_file():
            raise HardeningContractError("integration config does not exist")
        input_bytes = (
            self.config_path.stat().st_size + integration_config.stat().st_size
        )
        if input_bytes > self.config["max_input_bytes"]:
            raise HardeningContractError(
                f"input budget exceeded: {input_bytes} > {self.config['max_input_bytes']}"
            )
        return integration_config

    def _event(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        payload: dict[str, Any],
        now_ns: int,
    ) -> None:
        connection.execute(
            "INSERT INTO events(operation_id, event_type, payload_json, created_ns) "
            "VALUES (?, ?, ?, ?)",
            (self.operation_id, event_type, _canonical_json(payload), now_ns),
        )

    def _operation_row(self, connection: sqlite3.Connection) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM operations WHERE operation_id = ?", (self.operation_id,)
        ).fetchone()

    @staticmethod
    def _result_is_valid(row: sqlite3.Row) -> bool:
        if not row["result_path"] or not row["result_sha256"]:
            return False
        path = Path(row["result_path"])
        return path.is_file() and _sha256_file(path) == row["result_sha256"]

    @staticmethod
    def _preserve_corrupt_result(row: sqlite3.Row) -> str | None:
        if not row["result_path"]:
            return None
        path = Path(row["result_path"])
        if not path.is_file():
            return None
        actual = _sha256_file(path)
        preserved = path.with_name(f"{path.name}.corrupt-{actual[:16]}")
        path.replace(preserved)
        return str(preserved)

    def _claim(
        self, *, worker_id: str, now_ns: int, crash_point: str | None
    ) -> tuple[int, Path, bool, bool]:
        lease_expires_ns = now_ns + int(self.config["lease_seconds"]) * 1_000_000_000
        with _connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._operation_row(connection)
            if row is None:
                connection.execute(
                    "INSERT INTO operations(operation_id, contract_hash, status, "
                    "attempt, created_ns, updated_ns) VALUES (?, ?, 'prepared', 0, ?, ?)",
                    (self.operation_id, self.contract_hash, now_ns, now_ns),
                )
                self._event(
                    connection,
                    "prepared",
                    {"contract_hash": self.contract_hash},
                    now_ns,
                )
                connection.commit()
                if crash_point == "after_prepare":
                    raise InjectedCrash("injected crash: after_prepare")
                connection.execute("BEGIN IMMEDIATE")
                row = self._operation_row(connection)
                assert row is not None
            if row["contract_hash"] != self.contract_hash:
                connection.rollback()
                raise ContractConflict("operation id already has another contract hash")
            if row["status"] == "completed" and self._result_is_valid(row):
                self._event(
                    connection,
                    "idempotent_replay",
                    {"worker_id": worker_id, "attempt": row["attempt"]},
                    now_ns,
                )
                connection.commit()
                return (
                    int(row["attempt"]),
                    Path(row["result_path"]).parent,
                    True,
                    False,
                )
            if row["status"] == "completed":
                preserved = self._preserve_corrupt_result(row)
                self._event(
                    connection,
                    "integrity_failure",
                    {
                        "expected_sha256": row["result_sha256"],
                        "preserved_path": preserved,
                    },
                    now_ns,
                )
                connection.execute(
                    "UPDATE operations SET status='prepared', result_path=NULL, "
                    "result_sha256=NULL, semantic_fingerprint=NULL, error=NULL, "
                    "updated_ns=? WHERE operation_id=?",
                    (now_ns, self.operation_id),
                )
                row = self._operation_row(connection)
                assert row is not None
            recovering_running = row["status"] == "running"
            if recovering_running and int(row["lease_expires_ns"] or 0) > now_ns:
                connection.rollback()
                raise OperationBusy(
                    f"operation lease owned by {row['lease_owner']} until "
                    f"{row['lease_expires_ns']}"
                )
            if recovering_running:
                attempt = int(row["attempt"])
                event_type = "lease_recovered"
                staged_result = (
                    self.output_root
                    / self.operation_id
                    / f"attempt-{attempt}"
                    / "result.json"
                )
                reuse_staged_result = staged_result.is_file()
            else:
                attempt = int(row["attempt"]) + 1
                event_type = "claimed"
                reuse_staged_result = False
            attempt_dir = self.output_root / self.operation_id / f"attempt-{attempt}"
            connection.execute(
                "UPDATE operations SET status='running', attempt=?, lease_owner=?, "
                "lease_expires_ns=?, result_path=NULL, result_sha256=NULL, "
                "semantic_fingerprint=NULL, error=NULL, updated_ns=? "
                "WHERE operation_id=?",
                (
                    attempt,
                    worker_id,
                    lease_expires_ns,
                    now_ns,
                    self.operation_id,
                ),
            )
            self._event(
                connection,
                event_type,
                {
                    "attempt": attempt,
                    "lease_expires_ns": lease_expires_ns,
                    "worker_id": worker_id,
                },
                now_ns,
            )
            connection.commit()
            return attempt, attempt_dir, False, reuse_staged_result

    def _run_trusted_integration(
        self, *, integration_config: Path, output_dir: Path, timeout_seconds: int
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            str(INTEGRATION_RUNTIME),
            "--config",
            str(integration_config),
            "--output",
            str(output_dir),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={
                    "LANG": "C",
                    "LC_ALL": "C",
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "PYTHONHASHSEED": "0",
                },
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise OperationExecutionError(
                f"trusted integration timed out after {timeout_seconds}s"
            ) from error
        except subprocess.CalledProcessError as error:
            raise OperationExecutionError(
                f"trusted integration failed with exit {error.returncode}"
            ) from error
        (output_dir / "service-stdout.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        (output_dir / "service-stderr.txt").write_text(
            completed.stderr, encoding="utf-8"
        )
        result_path = output_dir / "result.json"
        if not result_path.is_file():
            raise OperationExecutionError("trusted integration produced no result.json")
        return json.loads(result_path.read_text(encoding="utf-8"))

    def _record_failure(self, *, error: Exception, now_ns: int) -> None:
        with _connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE operations SET status='failed', error=?, lease_owner=NULL, "
                "lease_expires_ns=NULL, updated_ns=? WHERE operation_id=?",
                (str(error), now_ns, self.operation_id),
            )
            self._event(
                connection,
                "failed",
                {"error": str(error), "error_type": type(error).__name__},
                now_ns,
            )
            connection.commit()

    def _record_completed(
        self,
        *,
        attempt: int,
        result_path: Path,
        result: dict[str, Any],
        worker_id: str,
        now_ns: int,
    ) -> dict[str, Any]:
        result_sha256 = _sha256_file(result_path)
        fingerprint = result.get("experiment_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise OperationExecutionError(
                "integration result lacks semantic fingerprint"
            )
        with _connect(self.database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._operation_row(connection)
            if row is None or row["status"] != "running":
                connection.rollback()
                raise OperationExecutionError(
                    "operation lost running state before commit"
                )
            if row["lease_owner"] != worker_id or int(row["attempt"]) != attempt:
                connection.rollback()
                raise OperationExecutionError("operation lease changed before commit")
            connection.execute(
                "UPDATE operations SET status='completed', result_path=?, "
                "result_sha256=?, semantic_fingerprint=?, error=NULL, lease_owner=NULL, "
                "lease_expires_ns=NULL, updated_ns=? WHERE operation_id=?",
                (
                    str(result_path.resolve()),
                    result_sha256,
                    fingerprint,
                    now_ns,
                    self.operation_id,
                ),
            )
            self._event(
                connection,
                "completed",
                {
                    "attempt": attempt,
                    "result_sha256": result_sha256,
                    "semantic_fingerprint": fingerprint,
                },
                now_ns,
            )
            connection.commit()
        return {
            "attempt": attempt,
            "contract_hash": self.contract_hash,
            "idempotent_replay": False,
            "operation_id": self.operation_id,
            "result_path": str(result_path.resolve()),
            "result_sha256": result_sha256,
            "semantic_fingerprint": fingerprint,
            "status": "completed",
        }

    def run(
        self,
        *,
        worker_id: str,
        now_ns: int | None = None,
        crash_point: str | None = None,
    ) -> dict[str, Any]:
        if not worker_id:
            raise HardeningContractError("worker_id must not be empty")
        if crash_point not in {
            None,
            "after_prepare",
            "after_integration_before_commit",
        }:
            raise HardeningContractError("unknown crash point")
        now_ns = time.time_ns() if now_ns is None else now_ns
        attempt, attempt_dir, idempotent, reuse_staged_result = self._claim(
            worker_id=worker_id, now_ns=now_ns, crash_point=crash_point
        )
        if idempotent:
            row = self.inspect_operation()
            return {**row, "idempotent_replay": True}
        result_path = attempt_dir / "result.json"
        try:
            if reuse_staged_result:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                result = self.runner(
                    integration_config=self.integration_config,
                    output_dir=attempt_dir,
                    timeout_seconds=int(self.config["max_runtime_seconds"]),
                )
            if _tree_bytes(attempt_dir) > int(self.config["max_artifact_bytes"]):
                raise OperationExecutionError("artifact budget exceeded")
            if not result_path.is_file():
                _atomic_json(result_path, result)
            if crash_point == "after_integration_before_commit":
                raise InjectedCrash("injected crash: after_integration_before_commit")
            return self._record_completed(
                attempt=attempt,
                result_path=result_path,
                result=result,
                worker_id=worker_id,
                now_ns=now_ns,
            )
        except InjectedCrash:
            raise
        except OperationExecutionError as error:
            self._record_failure(error=error, now_ns=now_ns)
            raise
        except Exception as error:
            wrapped = OperationExecutionError(f"trusted runner error: {error}")
            self._record_failure(error=wrapped, now_ns=now_ns)
            raise wrapped from error

    def inspect_operation(self) -> dict[str, Any]:
        with _connect(self.database_path) as connection:
            row = self._operation_row(connection)
            if row is None:
                raise KeyError(self.operation_id)
            return dict(row)

    def list_events(self) -> list[dict[str, Any]]:
        with _connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE operation_id=? ORDER BY sequence",
                (self.operation_id,),
            ).fetchall()
        return [
            {**dict(row), "payload": json.loads(row["payload_json"])} for row in rows
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument(
        "--crash-point",
        choices=["after_prepare", "after_integration_before_commit"],
    )
    args = parser.parse_args()
    service = DurableOperationService(
        config_path=args.config,
        database_path=args.database,
        output_root=args.output_root,
    )
    result = service.run(worker_id=args.worker_id, crash_point=args.crash_point)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
