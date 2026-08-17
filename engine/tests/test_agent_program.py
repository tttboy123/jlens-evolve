from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_program import (
    AgentProgram,
    ComponentRegistry,
    ContractError,
    MutationProposal,
    ReplaySupervisor,
    apply_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v1.0.0/v0.2.0-agent-program"


def registry() -> ComponentRegistry:
    return ComponentRegistry.from_path(STAGE / "configs/component_registry.json")


def baseline() -> AgentProgram:
    return AgentProgram.from_path(STAGE / "configs/baseline_agent_program.json")


def test_agent_program_has_stable_canonical_hash_and_round_trip():
    program = baseline()

    assert program.sha256 == (
        "b47211fce748b6a69f4c3d90ecc6f2f6aa0e4060f08d283bfcef3d246456d3a7"
    )
    assert AgentProgram.from_dict(json.loads(program.canonical_json)) == program
    program.validate(registry())


def test_agent_program_rejects_unknown_fields_refs_and_duplicate_skills():
    payload = baseline().to_dict()
    payload["secret_override"] = True
    with pytest.raises(ContractError, match="unknown AgentProgram fields"):
        AgentProgram.from_dict(payload)

    with pytest.raises(ContractError, match="unknown system_prompt_ref"):
        replace(baseline(), system_prompt_ref="prompt/not-registered").validate(
            registry()
        )

    with pytest.raises(ContractError, match="duplicate skill_refs"):
        replace(
            baseline(),
            skill_refs=(
                "skill/identity-aggregation-output-v1",
                "skill/identity-aggregation-output-v1",
            ),
        ).validate(registry())


def test_replay_supervisor_reproduces_the_frozen_proposal_sequence():
    first = ReplaySupervisor.from_path(STAGE / "configs/replay_proposals.json")
    second = ReplaySupervisor.from_path(STAGE / "configs/replay_proposals.json")

    assert [proposal.canonical_json for proposal in first.proposals] == [
        proposal.canonical_json for proposal in second.proposals
    ]
    assert [proposal.mutation_type for proposal in first.proposals] == [
        "prompt_instruction",
        "skill_composition",
        "retry_policy",
    ]


def test_proposals_change_one_allowed_axis_and_preserve_lineage():
    current = baseline()
    supervisor = ReplaySupervisor.from_path(STAGE / "configs/replay_proposals.json")
    expected_hashes = [
        "335ec1694a8486c14d74e0c809337472c289552eed3de494b77f5cc42559a448",
        "cf7166c9b77f4f8cdae4f2fdab5ede2028c24fec21bf2d8efa6684ea54f8320d",
        "dcc7e43151f03fc561da9881b2210980851a0b18883823bac264611fecee6a77",
    ]

    for proposal, expected_hash in zip(
        supervisor.proposals, expected_hashes, strict=True
    ):
        child = apply_proposal(current, proposal, registry())
        assert child.parent_program_hash == current.sha256
        assert child.sha256 == expected_hash
        current = child


def test_proposal_rejects_parent_mismatch_and_frozen_field_change():
    proposal = MutationProposal.from_dict(
        {
            "schema_version": 1,
            "proposal_id": "bad-parent",
            "parent_program_hash": "0" * 64,
            "program_id": "bad-child",
            "mutation_type": "prompt_instruction",
            "changes": {"system_prompt_ref": "prompt/normalize-status-v1"},
            "reason": "test",
        }
    )
    with pytest.raises(ContractError, match="parent hash mismatch"):
        apply_proposal(baseline(), proposal, registry())

    frozen = {
        "schema_version": 1,
        "proposal_id": "frozen-change",
        "parent_program_hash": baseline().sha256,
        "program_id": "bad-child",
        "mutation_type": "prompt_instruction",
        "changes": {"harness_code_ref": "harness/other"},
        "reason": "test",
    }
    with pytest.raises(ContractError, match="does not match mutation_type"):
        MutationProposal.from_dict(frozen)
