from __future__ import annotations

import json
from pathlib import Path

from agent_code_runtime import run_agent_code_experiment

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v1.0.0/v0.5.0-agent-code-mutation"


def test_code_mutation_rejects_unsafe_executes_safe_and_rolls_back(tmp_path: Path):
    result = run_agent_code_experiment(
        config_path=STAGE / "configs/experiment.json",
        evaluator_path=STAGE / "configs/harness-evaluator.json",
        candidate_manifest_path=STAGE / "configs/candidates.json",
        output_dir=tmp_path / "run",
    )

    assert result["decision"] == "accepted"
    assert result["verified_candidate_id"] == "safe-retry-routing-v2"
    assert result["parent"]["public"] == [3, 6]
    assert result["parent"]["sealed"] == [2, 4]
    safe = result["candidates"]["safe-retry-routing-v2"]
    assert safe["status"] == "verified"
    assert safe["public"] == [6, 6]
    assert safe["sealed"] == [4, 4]
    regression = result["candidates"]["behavior-regression-v1"]
    assert regression["status"] == "rejected_public"
    assert regression["sealed"] is None
    for candidate_id in ("unsafe-import-v1", "unsafe-open-v1", "unsafe-loop-v1"):
        candidate = result["candidates"][candidate_id]
        assert candidate["status"] == "rejected_static"
        assert candidate["executed"] is False
    assert result["rollback"]["performed"] is True
    assert (
        result["rollback"]["final_active_sha256"] == result["parent"]["source_sha256"]
    )
    assert result["contract_checks"]["evaluator_unchanged"] is True
    assert result["contract_checks"]["config_unchanged"] is True


def test_sealed_is_only_opened_for_public_gate_winner(tmp_path: Path):
    output = tmp_path / "run"
    run_agent_code_experiment(
        config_path=STAGE / "configs/experiment.json",
        evaluator_path=STAGE / "configs/harness-evaluator.json",
        candidate_manifest_path=STAGE / "configs/candidates.json",
        output_dir=output,
    )
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    sealed_candidates = {
        row["candidate_id"]
        for row in events
        if row["event_type"] == "evaluation" and row["partition"] == "sealed"
    }

    assert sealed_candidates == {"route-parent-v1", "safe-retry-routing-v2"}
    assert not any(
        row["event_type"] == "sandbox_execution"
        and row["candidate_id"].startswith("unsafe-")
        for row in events
    )


def test_replay_fingerprint_is_stable(tmp_path: Path):
    fingerprints = []
    for replay in range(2):
        result = run_agent_code_experiment(
            config_path=STAGE / "configs/experiment.json",
            evaluator_path=STAGE / "configs/harness-evaluator.json",
            candidate_manifest_path=STAGE / "configs/candidates.json",
            output_dir=tmp_path / f"run-{replay}",
        )
        fingerprints.append(result["experiment_fingerprint"])

    assert len(set(fingerprints)) == 1
