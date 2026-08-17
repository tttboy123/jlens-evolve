from __future__ import annotations

from pathlib import Path

import pytest

from agent_code_mutation import MutationArchive, MutationContractError, validate_source
from sandbox_runner import run_candidate

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v1.0.0/v0.5.0-agent-code-mutation"
LIMITS = {
    "address_space_mb": 128,
    "cpu_seconds": 1,
    "file_bytes": 0,
    "max_ast_nodes": 64,
    "max_source_bytes": 2048,
    "open_files": 16,
    "timeout_seconds": 2.0,
}


def _source(name: str) -> str:
    return (STAGE / f"configs/candidates/{name}.py").read_text(encoding="utf-8")


def test_allowlist_accepts_only_the_bounded_route_function():
    safe = validate_source(_source("safe-retry-routing-v2"), limits=LIMITS)
    unsafe = {
        name: validate_source(_source(name), limits=LIMITS)
        for name in ("unsafe-import-v1", "unsafe-open-v1", "unsafe-loop-v1")
    }

    assert safe["allowed"] is True
    assert safe["function_name"] == "select_route"
    assert safe["function_args"] == ["attempt", "last_error"]
    assert all(report["allowed"] is False for report in unsafe.values())
    assert any("Import" in reason for reason in unsafe["unsafe-import-v1"]["reasons"])
    assert any("Call" in reason for reason in unsafe["unsafe-open-v1"]["reasons"])
    assert any("While" in reason for reason in unsafe["unsafe-loop-v1"]["reasons"])


def test_sandbox_runs_validated_source_with_empty_working_directory(tmp_path: Path):
    result = run_candidate(
        source=_source("safe-retry-routing-v2"),
        cases=[
            {"id": "timeout", "args": [1, "timeout"], "expected": "fallback"},
            {"id": "initial", "args": [0, "timeout"], "expected": "primary"},
        ],
        limits=LIMITS,
        sandbox_parent=tmp_path,
    )

    assert result["status"] == "completed"
    assert result["passed_cases"] == 2
    assert result["sandbox"]["isolated_python"] is True
    assert result["sandbox"]["empty_environment"] is True
    assert result["sandbox"]["files_after"] == []


def test_sandbox_refuses_source_that_failed_static_gate(tmp_path: Path):
    with pytest.raises(MutationContractError, match="static gate"):
        run_candidate(
            source=_source("unsafe-open-v1"),
            cases=[],
            limits=LIMITS,
            sandbox_parent=tmp_path,
        )


def test_archive_only_activates_verified_and_can_rollback(tmp_path: Path):
    archive = MutationArchive(tmp_path / "archive")
    parent_hash = "a" * 64
    child_hash = "b" * 64
    archive.initialize_active(candidate_id="parent", source_sha256=parent_hash)
    archive.append_record(
        {
            "candidate_id": "child",
            "source_sha256": child_hash,
            "parent_source_sha256": parent_hash,
            "status": "rejected_public",
        }
    )
    with pytest.raises(MutationContractError, match="verified"):
        archive.activate(candidate_id="child", source_sha256=child_hash)

    archive.append_record(
        {
            "candidate_id": "verified-child",
            "source_sha256": child_hash,
            "parent_source_sha256": parent_hash,
            "status": "verified",
        }
    )
    archive.activate(candidate_id="verified-child", source_sha256=child_hash)
    rolled_back = archive.rollback(reason="drill")

    assert rolled_back["source_sha256"] == parent_hash
    assert archive.read_active()["source_sha256"] == parent_hash
