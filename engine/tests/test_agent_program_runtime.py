from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agent_program_runtime import run_agent_program_experiment

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v1.0.0/v0.2.0-agent-program"
CONFIGS = STAGE / "configs"


def run_experiment(output_dir: Path) -> dict:
    return run_agent_program_experiment(
        baseline_path=CONFIGS / "baseline_agent_program.json",
        registry_path=CONFIGS / "component_registry.json",
        proposals_path=CONFIGS / "replay_proposals.json",
        experiment_path=CONFIGS / "experiment.json",
        output_dir=output_dir,
    )


def test_replay_agent_program_improves_only_through_application_refs(tmp_path):
    result = run_experiment(tmp_path / "run")

    assert result["baseline"]["public_passed_by_seed"] == {
        "11": 3,
        "22": 3,
        "33": 3,
    }
    assert [row["candidate"]["public_passed_mean"] for row in result["steps"]] == [
        4.0,
        10.0,
        13.0,
    ]
    assert all(row["decision"] == "accepted" for row in result["steps"])
    assert result["final"]["public_passed_by_seed"] == {
        "11": 13,
        "22": 13,
        "33": 13,
    }
    assert (
        result["baseline"]["program"]["harness_code_ref"]
        == result["final"]["program"]["harness_code_ref"]
    )
    assert result["claims"]["task_program_mutated"] is False
    assert result["claims"]["model_calls"] == 0
    assert result["claims"]["network_calls"] == 0


def test_sealed_audit_occurs_only_after_search_and_passes_gate(tmp_path):
    result = run_experiment(tmp_path / "run")
    events = [
        json.loads(line)
        for line in (tmp_path / "run/events.jsonl").read_text().splitlines()
    ]
    partitions = [
        event["partition"] for event in events if event["event_type"] == "evaluation"
    ]

    first_holdout = partitions.index("sealed")
    assert set(partitions[:first_holdout]) == {"public"}
    assert set(partitions[first_holdout:]) == {"sealed"}
    checkpoint = tmp_path / "run/public-checkpoint.json"
    assert checkpoint.is_file()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    opened = next(event for event in events if event["event_type"] == "sealed_opened")
    opened_index = events.index(opened)
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
    assert max(public_indexes) < opened_index < min(sealed_indexes)
    assert opened["public_results_persisted"] is True
    assert opened["public_checkpoint_sha256"] == checkpoint_sha256
    assert result["search"]["sealed_used_for_search"] is False
    assert result["sealed_audit"]["baseline_passed_by_seed"] == {
        "11": 0,
        "22": 0,
        "33": 0,
    }
    assert result["sealed_audit"]["final_passed_by_seed"] == {
        "11": 6,
        "22": 6,
        "33": 6,
    }
    assert result["sealed_audit"]["noninferior_seeds"] == 3
    assert result["decision"] == "accepted"
    assert result["production_ready"] is False


def test_three_independent_replays_have_one_outcome_fingerprint(tmp_path):
    fingerprints = {
        run_experiment(tmp_path / f"run-{index}")["outcome_fingerprint"]
        for index in range(3)
    }

    assert len(fingerprints) == 1


def test_run_persists_lineage_archive_raw_evaluations_and_active_program(tmp_path):
    output = tmp_path / "run"
    result = run_experiment(output)

    assert (output / "result.json").is_file()
    assert (output / "evidence.json").is_file()
    assert (output / "events.jsonl").is_file()
    assert (output / "candidate_archive.jsonl").is_file()
    assert (output / "active_agent_program.json").is_file()
    active = json.loads((output / "active_agent_program.json").read_text())
    archive = [
        json.loads(line)
        for line in (output / "candidate_archive.jsonl").read_text().splitlines()
    ]
    assert active["program_id"] == "agent-program-retry-v1"
    assert active["harness_code_ref"] == "harness/record-cleaning-v1"
    assert len(archive) == 3
    assert [row["parent_program_hash"] for row in archive] == [
        result["baseline"]["program_hash"],
        result["steps"][0]["candidate"]["program_hash"],
        result["steps"][1]["candidate"]["program_hash"],
    ]
