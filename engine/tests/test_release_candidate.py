import sqlite3
from pathlib import Path

import pytest

from release_candidate import (
    _bundle_files,
    _semantic_clean_room,
    run_release_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "artifacts/v1.0.0/v0.9.0-release-candidate/configs/experiment.json"


def test_bundle_membership_is_explicit_frozen_list():
    """Issue #6 root fix: adding new root/test modules must not change the bundle."""
    before = {p.relative_to(ROOT).as_posix() for p in _bundle_files()}
    scratch = ROOT / "scratch_issue6_freeze_probe.py"
    scratch.write_text("x = 1\n", encoding="utf-8")
    try:
        after = {p.relative_to(ROOT).as_posix() for p in _bundle_files()}
    finally:
        scratch.unlink()
    assert before == after
    assert not any("scratch_issue6_freeze_probe" in name for name in after)


def test_clean_room_semantics_exclude_pytest_duration_noise():
    first = {
        "safe_members": True,
        "core_tests_passed": True,
        "tests_passed": 16,
        "test_summary": "16 passed in 0.06s",
        "bundle_sha256": "a" * 64,
    }
    second = {
        **first,
        "test_summary": "16 passed in 0.91s",
        "bundle_sha256": "b" * 64,
    }

    assert _semantic_clean_room(first) == _semantic_clean_room(second)


def test_release_candidate_audits_three_operations_and_clean_room(tmp_path: Path):
    result = run_release_candidate(
        config_path=CONFIG, output_dir=tmp_path / "release-candidate"
    )

    assert result["decision"] == "accepted"
    assert all(result["contract_checks"].values())
    assert len(result["operation_audits"]) == 3
    assert {audit["operation_id"] for audit in result["operation_audits"]} == {
        "rc-operation-001",
        "rc-operation-002",
        "rc-operation-003",
    }
    for audit in result["operation_audits"]:
        assert audit["agent_program_seeds"] == [11, 22, 33]
        assert audit["psi_seeds"] == [11, 22, 33]
        assert audit["psi_tasks"] == [
            "payout-record-cleaning-v1",
            "refund-record-cleaning-v1",
        ]
        assert audit["agent_public_before_sealed"]
        assert audit["psi_public_before_sealed"]
        assert audit["idempotent_replay"]
    assert result["clean_room"]["safe_members"]
    assert (
        result["clean_room"]["member_count"] == 124
    )  # v2.5 + pattern_card/t4_integration (amendment 029 re-pin)
    assert result["clean_room"]["core_tests_passed"]
    assert result["clean_room"]["cli_replay_decision"] == "accepted"
    assert result["clean_room"]["cli_replay_fingerprint"] == (
        "ace17a3054d0c52df08e0023d577da50cf46d7b64c6f65137d27c21ac7556c5e"
    )
    with sqlite3.connect(tmp_path / "release-candidate/rc-service.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM operations").fetchone()[0] == 3
        assert {
            row[0] for row in db.execute("SELECT status FROM operations").fetchall()
        } == {"completed"}
