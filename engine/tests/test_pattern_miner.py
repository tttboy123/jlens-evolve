from __future__ import annotations

import json

import pytest

from pattern_miner import FrozenObservationEvidence, PatternAdvantageMiner

SHA = {
    name: char * 64
    for name, char in {
        "original": "a",
        "parent": "b",
        "candidate": "c",
        "trajectory": "d",
        "tools": "e",
        "evaluator": "f",
        "cost": "1",
        "safety": "2",
    }.items()
}


def _evidence(
    evidence_id: str,
    *,
    delta: float,
    features: list[str],
    conditions: list[str],
    surfaces: list[str],
    safety_passed: bool = True,
) -> dict:
    return {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "task_uid": f"task-{evidence_id}",
        "benchmark_family": "swe-bench-verified",
        "agent_program_sha256": SHA["candidate"],
        "parent_agent_program_sha256": SHA["parent"],
        "native_evaluator_epoch": "native-adapters-v2.1.0-frozen",
        "native_score_delta": delta,
        "safety_passed": safety_passed,
        "observed_features": features,
        "conditions": conditions,
        "expected_surfaces": surfaces,
        "evidence": {
            "trajectory": {
                "path": f"{evidence_id}/trajectory.jsonl",
                "sha256": SHA["trajectory"],
            },
            "tool_events": {
                "path": f"{evidence_id}/tools.jsonl",
                "sha256": SHA["tools"],
            },
            "native_evaluator": {
                "path": f"{evidence_id}/eval.json",
                "sha256": SHA["evaluator"],
            },
            "cost": {"path": f"{evidence_id}/cost.json", "sha256": SHA["cost"]},
            "safety": {"path": f"{evidence_id}/safety.json", "sha256": SHA["safety"]},
        },
        "causal_boundary": "observational_not_causal",
        "admission_gate_allowed": False,
    }


def test_frozen_evidence_rejects_unknown_fields_and_admission_authority():
    payload = _evidence(
        "001",
        delta=1.0,
        features=["tests_before_edit"],
        conditions=["python"],
        surfaces=["prompt"],
    )
    payload["reward"] = 1.0
    with pytest.raises(ValueError, match="unknown evidence fields"):
        FrozenObservationEvidence.from_dict(payload)

    payload.pop("reward")
    payload["admission_gate_allowed"] = True
    with pytest.raises(ValueError, match="admission"):
        FrozenObservationEvidence.from_dict(payload)


def test_miner_aggregates_advantages_failures_and_counterexamples():
    rows = [
        _evidence(
            "001",
            delta=1.0,
            features=["tests_before_edit", "narrow_patch"],
            conditions=["python", "regression_task"],
            surfaces=["prompt", "policy"],
        ),
        _evidence(
            "002",
            delta=0.5,
            features=["tests_before_edit"],
            conditions=["python"],
            surfaces=["prompt"],
        ),
        _evidence(
            "003",
            delta=-0.5,
            features=["tests_before_edit", "broad_rewrite"],
            conditions=["python"],
            surfaces=["prompt", "constrained_harness_code"],
        ),
        _evidence(
            "004",
            delta=-1.0,
            features=["broad_rewrite"],
            conditions=["python", "high_coupling"],
            surfaces=["policy", "constrained_harness_code"],
        ),
    ]

    cards = PatternAdvantageMiner().mine(
        FrozenObservationEvidence.from_dict(row) for row in rows
    )

    by_key = {(card.pattern_kind, card.observed_feature): card for card in cards}
    advantage = by_key[("advantage", "tests_before_edit")]
    failure = by_key[("failure", "broad_rewrite")]
    assert advantage.support_count == 2
    assert advantage.counterexample_count == 1
    assert advantage.conditions == ("python",)
    assert advantage.counterexample_evidence_ids == ("003",)
    assert advantage.expected_surfaces == ("prompt", "policy")
    assert 0 < advantage.confidence < 1
    assert failure.support_count == 2
    assert failure.counterexample_count == 0
    assert "constrained_harness_code" in failure.expected_surfaces
    assert all(card.causal_boundary == "observational_not_causal" for card in cards)
    assert all(card.admission_gate_allowed is False for card in cards)


def test_pattern_card_ids_and_serialization_are_deterministic():
    rows = [
        FrozenObservationEvidence.from_dict(
            _evidence(
                "002",
                delta=0.5,
                features=["tests_before_edit"],
                conditions=["python"],
                surfaces=["prompt"],
            )
        ),
        FrozenObservationEvidence.from_dict(
            _evidence(
                "001",
                delta=1.0,
                features=["tests_before_edit"],
                conditions=["python"],
                surfaces=["prompt"],
            )
        ),
    ]
    miner = PatternAdvantageMiner()

    forward = [card.to_dict() for card in miner.mine(rows)]
    reverse = [card.to_dict() for card in miner.mine(reversed(rows))]

    assert forward == reverse
    assert "reward" not in json.dumps(forward)
    assert "rank" not in json.dumps(forward)
