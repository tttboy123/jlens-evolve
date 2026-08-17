"""Just-in-time benchmark input materialization for already-claimed rounds."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchmark_adapters import TaskPool
from continuous_ab import ABContractError, MatchedRoundLedger


class ExecutionContractError(ValueError):
    """Raised when a task input or pinned checkout crosses the execution contract."""


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise ExecutionContractError(f"frozen benchmark input is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ExecutionContractError("benchmark row must be an object")
                yield row


def _find_row(path: Path, *, instance_id: str) -> dict[str, Any]:
    matches = [
        row for row in _jsonl_rows(path) if row.get("instance_id") == instance_id
    ]
    if len(matches) != 1:
        raise ExecutionContractError(
            f"expected one frozen row for {instance_id}; found={len(matches)}"
        )
    return matches[0]


def _materialize_patch_task(
    contract: dict[str, Any], pool_root: Path
) -> tuple[str, str]:
    benchmark_id = contract["benchmark_id"]
    if benchmark_id in {"swe-bench-verified", "swe-bench-multilingual"}:
        source_path = pool_root / f"harness-inputs/{benchmark_id}.jsonl"
        row = _find_row(source_path, instance_id=contract["instance_id"])
        instruction = row.get("problem_statement")
    elif benchmark_id == "multi-swe-bench-flash":
        source_path = (
            pool_root / "inputs/multi-swe-bench-flash/multi_swe_bench_flash.jsonl"
        )
        row = _find_row(source_path, instance_id=contract["instance_id"])
        instruction = "\n\n".join(
            part.strip()
            for part in (str(row.get("title", "")), str(row.get("body", "")))
            if part.strip()
        )
    else:
        raise ExecutionContractError(f"unsupported patch benchmark: {benchmark_id}")
    actual_hash = _sha256_json(row)
    if actual_hash != contract["content_sha256"]:
        raise ExecutionContractError("frozen benchmark content hash mismatch")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ExecutionContractError("benchmark instruction is empty")
    return instruction.strip(), actual_hash


def _materialize_terminal_task(
    contract: dict[str, Any], terminal_dataset_root: Path | None
) -> tuple[str, str]:
    if terminal_dataset_root is None:
        raise ExecutionContractError("Terminal-Bench checkout is required")
    terminal_dataset_root = terminal_dataset_root.resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=terminal_dataset_root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or revision != contract["benchmark_revision"]:
        raise ExecutionContractError("Terminal-Bench checkout revision mismatch")
    instruction_path = (
        terminal_dataset_root / contract["instance_id"] / "instruction.md"
    ).resolve()
    if not instruction_path.is_relative_to(terminal_dataset_root):
        raise ExecutionContractError("Terminal-Bench instruction path escapes checkout")
    if not instruction_path.is_file():
        raise ExecutionContractError("Terminal-Bench instruction is missing")
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    if not instruction:
        raise ExecutionContractError("Terminal-Bench instruction is empty")
    return instruction, revision


def materialize_task_contract(
    *,
    task_contract: dict[str, Any],
    round_id: str,
    pool_root: Path,
    terminal_dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize one already-authorized task contract without admitting gold."""

    if not isinstance(round_id, str) or not round_id.strip():
        raise ExecutionContractError("materialized task round_id is empty")
    required = {
        "task_uid",
        "benchmark_id",
        "benchmark_revision",
        "instance_id",
        "repo",
        "base_commit",
        "language",
        "content_sha256",
    }
    if not isinstance(task_contract, dict) or not required.issubset(task_contract):
        raise ExecutionContractError("task contract is incomplete")
    if task_contract["benchmark_id"] == "terminal-bench-2":
        instruction, source_hash = _materialize_terminal_task(
            task_contract, terminal_dataset_root
        )
    else:
        instruction, source_hash = _materialize_patch_task(
            task_contract, pool_root.resolve()
        )
    return {
        "schema_version": "1.0",
        "round_id": round_id,
        "task_uid": task_contract["task_uid"],
        "benchmark_id": task_contract["benchmark_id"],
        "benchmark_revision": task_contract["benchmark_revision"],
        "instance_id": task_contract["instance_id"],
        "repo": task_contract["repo"],
        "base_commit": task_contract["base_commit"],
        "language": task_contract["language"],
        "instruction": instruction,
        "instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "source_content_sha256": source_hash,
        "gold_fields_included": False,
    }


def materialize_claimed_task(
    *,
    runtime_root: Path,
    round_id: str,
    pool_root: Path,
    terminal_dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Open only the task already claimed by a planned, tamper-evident round."""

    runtime_root = runtime_root.resolve()
    ledger = MatchedRoundLedger.load(runtime_root / "rounds" / f"{round_id}.json")
    if ledger.phase != "planned":
        raise ExecutionContractError("task materialization requires a planned round")
    pool = TaskPool.load(runtime_root / "TASK_POOL.json")
    task_uid = str(ledger.payload["task_uid"])
    records = [record for record in pool.records if record.task_uid == task_uid]
    if len(records) != 1:
        raise ExecutionContractError("round task is missing from the runtime pool")
    record = records[0]
    if record.state != record.assigned_partition:
        raise ExecutionContractError("task must be claimed before content is opened")
    payload = materialize_task_contract(
        task_contract=record.task_contract,
        round_id=round_id,
        pool_root=pool_root,
        terminal_dataset_root=terminal_dataset_root,
    )
    output_path = runtime_root / "evidence" / round_id / "task-input.json"
    _atomic_json(output_path, payload)
    try:
        ledger.record_task_materialization(output_path)
    except ABContractError as error:
        raise ExecutionContractError(str(error)) from error
    return payload
