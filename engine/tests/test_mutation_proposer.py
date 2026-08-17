from __future__ import annotations

import json

import pytest

from mutation_proposer import MutationContractError, MutationProposer
from pattern_miner import PatternCard

PARENT = "a" * 64
CANDIDATE = "b" * 64
EVIDENCE = "c" * 64


def _card(kind: str, feature: str, surfaces: tuple[str, ...]) -> PatternCard:
    return PatternCard(
        schema_version=1,
        pattern_id=f"pattern-{kind}-{feature}",
        pattern_kind=kind,
        observed_feature=feature,
        evidence_ids=("ev-1", "ev-2"),
        evidence_sha256s=(EVIDENCE,),
        counterexample_evidence_ids=("ev-3",),
        support_count=2,
        counterexample_count=1,
        conditions=("python",),
        expected_surfaces=surfaces,
        confidence=0.6,
        causal_boundary="observational_not_causal",
        admission_gate_allowed=False,
    )


def _valid_response(surface: str = "prompt") -> dict:
    path = {
        "prompt": "AGENTS.md",
        "skills": ".agents/skills/regression/SKILL.md",
        "policy": ".codex/evolution-policy.json",
        "router": ".codex/evolution-policy.json",
        "memory_policy": ".codex/evolution-policy.json",
        "constrained_harness_code": ".codex/harness/runner.py",
    }[surface]
    return {
        "schema_version": 1,
        "changeset_id": f"cs-{surface}",
        "status": "inactive",
        "parent_agent_program_sha256": PARENT,
        "candidate_agent_program_sha256": CANDIDATE,
        "hypothesis_ids": ["pattern-advantage-tests_before_edit"],
        "surface": surface,
        "operations": [{"op": "replace", "path": path, "after": "new"}],
        "rollback_operations": [{"op": "replace", "path": path, "after": "old"}],
        "proposer": {"platform": "codex", "model": "gpt"},
        "native_evaluator_epoch": "native-adapters-v2.1.0-frozen",
        "native_evaluator_authority": "external_fixed",
        "auto_apply": False,
        "production_promotion_allowed": False,
    }


def test_proposer_maps_cards_to_distinct_inactive_surface_requests():
    cards = (
        _card("advantage", "tests_before_edit", ("prompt", "policy")),
        _card("failure", "broad_rewrite", ("constrained_harness_code", "router")),
    )

    requests = MutationProposer().build_requests(
        cards,
        parent_agent_program_sha256=PARENT,
        native_evaluator_epoch="native-adapters-v2.1.0-frozen",
        maximum_candidates=4,
    )

    assert [request.surface for request in requests] == [
        "prompt",
        "policy",
        "constrained_harness_code",
        "router",
    ]
    assert all(request.status == "inactive" for request in requests)
    assert all(request.admission_gate_allowed is False for request in requests)


@pytest.mark.parametrize(
    "surface",
    [
        "prompt",
        "skills",
        "policy",
        "router",
        "memory_policy",
        "constrained_harness_code",
    ],
)
def test_validator_accepts_only_allowed_application_layer_surfaces(surface):
    proposal = MutationProposer().validate_response(
        json.dumps(_valid_response(surface)),
        expected_parent_sha256=PARENT,
        allowed_hypothesis_ids={"pattern-advantage-tests_before_edit"},
        frozen_native_evaluator_epoch="native-adapters-v2.1.0-frozen",
    )

    assert proposal.status == "inactive"
    assert proposal.surface == surface
    assert proposal.auto_apply is False
    assert proposal.production_promotion_allowed is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(surface="model_weights"), "surface"),
        (lambda p: p["operations"][0].update(path="../../global/SKILL.md"), "path"),
        (lambda p: p.update(status="promoted"), "inactive"),
        (lambda p: p.update(auto_apply=True), "auto_apply"),
        (lambda p: p.update(native_evaluator_epoch="changed"), "evaluator"),
    ],
)
def test_validator_rejects_weight_global_apply_and_evaluator_mutations(mutate, message):
    payload = _valid_response()
    mutate(payload)

    with pytest.raises(MutationContractError, match=message):
        MutationProposer().validate_response(
            json.dumps(payload),
            expected_parent_sha256=PARENT,
            allowed_hypothesis_ids={"pattern-advantage-tests_before_edit"},
            frozen_native_evaluator_epoch="native-adapters-v2.1.0-frozen",
        )


def test_propose_allows_exactly_one_bounded_schema_repair():
    calls = []

    def propose(_prompt: str) -> str:
        calls.append("first")
        return "not-json"

    def repair(_prompt: str) -> str:
        calls.append("repair")
        return json.dumps(_valid_response())

    result = MutationProposer().propose(
        request=MutationProposer().build_requests(
            (_card("advantage", "tests_before_edit", ("prompt",)),),
            parent_agent_program_sha256=PARENT,
            native_evaluator_epoch="native-adapters-v2.1.0-frozen",
            maximum_candidates=1,
        )[0],
        provider={"platform": "codex", "model": "gpt"},
        propose=propose,
        repair=repair,
    )

    assert result.repairs_used == 1
    assert result.changeset.status == "inactive"
    assert calls == ["first", "repair"]
