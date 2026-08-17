from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator_shadow import (
    ShadowContractError,
    run_evaluator_shadow,
    score_program,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v1.0.0/v0.6.0-evaluator-shadow"


def test_cross_play_detects_lax_strict_and_reviewable_candidate(tmp_path: Path):
    result = run_evaluator_shadow(
        config_path=STAGE / "configs/experiment.json",
        evaluator_path=STAGE / "configs/evaluators.json",
        corpus_path=STAGE / "configs/program-corpus.json",
        output_dir=tmp_path / "shadow",
    )

    assert result["decision"] == "accepted"
    assert result["active_evaluator_before"] == result["active_evaluator_after"]
    assert result["anchor_shadow_off"] == result["anchor_shadow_on"]
    assert result["candidates"]["lax-v1"]["false_accepts"] == 2
    assert result["candidates"]["lax-v1"]["status"] == "rejected_false_accept"
    assert result["candidates"]["strict-v1"]["false_rejects"] == 2
    assert result["candidates"]["strict-v1"]["status"] == "rejected_false_reject"
    robust = result["candidates"]["robust-v2"]
    assert robust["false_accepts"] == 0
    assert robust["false_rejects"] == 0
    assert robust["champion_stable"] is True
    assert robust["rank_correlation"] >= 0.8
    assert robust["status"] == "review_proposed"
    assert result["review_proposal"] == {
        "activation_allowed": False,
        "auto_promoted": False,
        "candidate_id": "robust-v2",
        "epoch_boundary_only": True,
        "requires_human_review": True,
    }


def test_shadow_scores_never_enter_admission_events(tmp_path: Path):
    output = tmp_path / "shadow"
    run_evaluator_shadow(
        config_path=STAGE / "configs/experiment.json",
        evaluator_path=STAGE / "configs/evaluators.json",
        corpus_path=STAGE / "configs/program-corpus.json",
        output_dir=output,
    )
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    admissions = [row for row in events if row["event_type"] == "admission"]

    assert admissions
    assert all(row["authority"] == "record-cleaning-anchor-v1" for row in admissions)
    assert all("shadow_score" not in row for row in admissions)
    assert not any(row.get("event_type") == "evaluator_activated" for row in events)


def test_anchor_truth_mismatch_is_rejected_before_shadow_run(tmp_path: Path):
    evaluators = json.loads(
        (STAGE / "configs/evaluators.json").read_text(encoding="utf-8")
    )
    corpus = json.loads(
        (STAGE / "configs/program-corpus.json").read_text(encoding="utf-8")
    )
    corpus["truth"]["expected_admission"]["broken-v1"] = True
    tampered = tmp_path / "corpus.json"
    tampered.write_text(json.dumps(corpus), encoding="utf-8")

    with pytest.raises(ShadowContractError, match="anchor truth"):
        run_evaluator_shadow(
            config_path=STAGE / "configs/experiment.json",
            evaluator_path=STAGE / "configs/evaluators.json",
            corpus_path=tampered,
            output_dir=tmp_path / "shadow",
        )
    assert score_program(evaluators["anchor"], corpus["programs"][0])["score"] == 1.0


def test_shadow_replay_fingerprint_is_stable(tmp_path: Path):
    fingerprints = []
    for replay in range(2):
        result = run_evaluator_shadow(
            config_path=STAGE / "configs/experiment.json",
            evaluator_path=STAGE / "configs/evaluators.json",
            corpus_path=STAGE / "configs/program-corpus.json",
            output_dir=tmp_path / f"shadow-{replay}",
        )
        fingerprints.append(result["experiment_fingerprint"])

    assert len(set(fingerprints)) == 1
