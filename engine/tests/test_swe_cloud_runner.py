from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from swe_cloud_runner import (
    CloudResources,
    RunnerError,
    RunnerLedger,
    freeze_file,
    select_instances,
    verify_frozen_file,
    write_portable_checksums,
)


def test_selection_is_deterministic_and_excludes_gold_and_previous_tasks():
    universe = ["task-a", "task-b", "task-c", "task-d", "task-e"]
    excluded = {"task-b", "task-d"}

    first = select_instances(
        universe,
        excluded=excluded,
        seed_material="batch-002",
        count=3,
    )
    second = select_instances(
        reversed(universe),
        excluded=excluded,
        seed_material="batch-002",
        count=3,
    )

    assert first == second
    assert {row["instance_id"] for row in first} == {"task-a", "task-c", "task-e"}
    assert all(len(row["rank_sha256"]) == 64 for row in first)


def test_selection_fails_closed_when_unique_universe_is_too_small():
    with pytest.raises(RunnerError, match="not enough eligible instances"):
        select_instances(
            ["task-a", "task-a", "task-b"],
            excluded={"task-b"},
            seed_material="batch-002",
            count=2,
        )


def test_freeze_detects_prediction_mutation(tmp_path: Path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"instance_id":"task-a"}\n', encoding="utf-8")
    manifest = tmp_path / "prediction-freeze.json"

    frozen = freeze_file(predictions, manifest, kind="predictions")

    assert verify_frozen_file(manifest) == frozen["sha256"]
    predictions.write_text('{"instance_id":"task-b"}\n', encoding="utf-8")
    with pytest.raises(RunnerError, match="frozen file hash mismatch"):
        verify_frozen_file(manifest)


def test_checksum_manifest_uses_relative_paths_and_is_portable(tmp_path: Path):
    evidence = tmp_path / "cloud-root" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "result.json").write_text("{}\n", encoding="utf-8")
    nested = evidence / "logs"
    nested.mkdir()
    (nested / "run.log").write_text("ok\n", encoding="utf-8")
    output = evidence / "SHA256SUMS"

    rows = write_portable_checksums(evidence, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert all(str(tmp_path) not in line for line in lines)
    assert any(line.endswith("  logs/run.log") for line in lines)
    assert any(line.endswith("  result.json") for line in lines)


def test_ledger_rejects_out_of_order_progress_and_failed_commands(tmp_path: Path):
    ledger = RunnerLedger.create(tmp_path / "ledger.json", batch_id="batch-002")

    with pytest.raises(RunnerError, match="invalid phase transition"):
        ledger.advance("provisioned")

    ledger.advance("selected")
    failure = ledger.run_checked(
        [sys.executable, "-c", "raise SystemExit(7)"],
        evidence_path=tmp_path / "failed-command.json",
    )

    assert failure.returncode == 7
    assert ledger.phase == "failed"
    evidence = json.loads((tmp_path / "failed-command.json").read_text())
    assert evidence["returncode"] == 7
    with pytest.raises(RunnerError, match="runner is failed"):
        ledger.advance("selected")


def test_cloud_release_gate_requires_every_temporary_resource(tmp_path: Path):
    ledger = RunnerLedger.create(tmp_path / "ledger.json", batch_id="batch-002")
    for phase in (
        "selected",
        "provisioned",
        "predictions_frozen",
        "evaluated",
        "evidence_verified",
    ):
        ledger.advance(phase)

    incomplete = CloudResources(
        instance_id="ins-1",
        disk_ids=("disk-1",),
        security_group_ids=("sg-1",),
        ssh_key_id="skey-1",
        released_ids=frozenset({"ins-1", "disk-1", "sg-1"}),
    )
    with pytest.raises(RunnerError, match="unreleased cloud resources"):
        ledger.release_cloud(incomplete)

    complete = CloudResources(
        instance_id="ins-1",
        disk_ids=("disk-1",),
        security_group_ids=("sg-1",),
        ssh_key_id="skey-1",
        released_ids=frozenset({"ins-1", "disk-1", "sg-1", "skey-1"}),
    )
    ledger.release_cloud(complete)
    assert ledger.phase == "cloud_released"


def test_ledger_loads_for_cross_process_resume_and_rejects_tampering(tmp_path: Path):
    path = tmp_path / "ledger.json"
    ledger = RunnerLedger.create(path, batch_id="batch-002")
    ledger.advance("selected")
    ledger.advance("provisioned", instance_id="ins-1")

    resumed = RunnerLedger.load(path)

    assert resumed.phase == "provisioned"
    assert resumed.payload["events"][-1]["instance_id"] == "ins-1"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["phase"] = "invented"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunnerError, match="invalid ledger phase"):
        RunnerLedger.load(path)
