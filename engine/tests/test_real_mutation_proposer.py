from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mutation_proposer import MutationProposer
from pattern_miner import PatternCard
from real_mutation_proposer import RealMutationProposerAdapter

EPOCH = "native-adapters-v2.1.0-frozen"


def _tree_hash(root: Path) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _profile(root: Path) -> str:
    root.mkdir(parents=True)
    (root / "AGENTS.md").write_text("Original policy.\n", encoding="utf-8")
    policy = root / ".codex/evolution-policy.json"
    policy.parent.mkdir(parents=True)
    policy.write_text('{"mode":"original"}\n', encoding="utf-8")
    return _tree_hash(root)


def _card(surface: str = "prompt") -> PatternCard:
    return PatternCard(
        schema_version=1,
        pattern_id=f"pattern-advantage-{surface}",
        pattern_kind="advantage",
        observed_feature="tests_before_edit",
        evidence_ids=("ev-1",),
        evidence_sha256s=("e" * 64,),
        counterexample_evidence_ids=("ev-2",),
        support_count=1,
        counterexample_count=1,
        conditions=("python",),
        expected_surfaces=(surface,),
        confidence=0.5,
        causal_boundary="observational_not_causal",
        admission_gate_allowed=False,
    )


def _request(parent: str, surface: str = "prompt"):
    return MutationProposer().build_requests(
        (_card(surface),),
        parent_agent_program_sha256=parent,
        native_evaluator_epoch=EPOCH,
        maximum_candidates=1,
    )[0]


def _response(parent: str, *, surface: str = "prompt") -> dict:
    if surface == "skills":
        path = ".agents/skills/test-first/SKILL.md"
        operation = {"op": "create", "path": path, "after": "# Test first\n"}
        rollback = {"op": "delete", "path": path, "after": None}
    else:
        path = "AGENTS.md"
        operation = {
            "op": "replace",
            "path": path,
            "after": "Run tests before editing.\n",
        }
        rollback = {"op": "replace", "path": path, "after": "Original policy.\n"}
    return {
        "schema_version": 1,
        "changeset_id": f"cs-{surface}",
        "status": "inactive",
        "parent_agent_program_sha256": parent,
        "candidate_agent_program_sha256": "0" * 64,
        "hypothesis_ids": [f"pattern-advantage-{surface}"],
        "surface": surface,
        "operations": [operation],
        "rollback_operations": [rollback],
        "proposer": {"platform": "codex", "model": "gpt-test"},
        "native_evaluator_epoch": EPOCH,
        "native_evaluator_authority": "external_fixed",
        "auto_apply": False,
        "production_promotion_allowed": False,
    }


def test_real_proposer_materializes_authoritative_hash_and_verified_rollback(
    tmp_path: Path,
):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    raw = json.dumps(_response(parent_hash))
    adapter = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=tmp_path / "proposals",
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=lambda _prompt: raw,
    )

    result = adapter.propose(_request(parent_hash), generation=0)

    candidate = result.changeset
    candidate_root = adapter.profile_root(candidate.candidate_agent_program_sha256)
    assert candidate.status == "inactive"
    assert candidate.candidate_agent_program_sha256 == _tree_hash(candidate_root)
    assert candidate.candidate_agent_program_sha256 != "0" * 64
    assert candidate.candidate_agent_program_sha256 != parent_hash
    assert (candidate_root / "AGENTS.md").read_text() == "Run tests before editing.\n"
    assert result.raw_responses == (raw,)
    rollback = adapter.rollback(candidate)
    assert rollback["verified"] is True
    assert len(rollback["forward_patch_sha256"]) == 64
    assert len(rollback["rollback_patch_sha256"]) == 64
    proposal_root = tmp_path / "proposals/generation-0/mutation-01-prompt"
    assert (proposal_root / "raw-1.txt").read_text() == raw
    normalized = json.loads((proposal_root / "ADMITTED-CHANGESET.json").read_text())
    assert normalized["candidate_agent_program_sha256"] == _tree_hash(candidate_root)


def test_real_proposer_prompt_contains_exact_parent_profile_and_call_receipts(
    tmp_path: Path,
):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    raw = json.dumps(_response(parent_hash))
    prompts = []
    reserved = []
    completed = []

    def model_call(prompt: str) -> str:
        prompts.append(json.loads(prompt))
        return raw

    adapter = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=tmp_path / "proposals",
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=model_call,
        reserve_call=lambda reservation_id: reserved.append(reservation_id) or True,
        complete_call=lambda reservation_id, response_sha256: completed.append(
            (reservation_id, response_sha256)
        ),
    )

    adapter.propose(_request(parent_hash), generation=0)

    assert len(prompts) == 1
    files = prompts[0]["parent_profile"]["files"]
    assert {item["path"] for item in files} == {
        "AGENTS.md",
        ".codex/evolution-policy.json",
    }
    assert (
        next(item for item in files if item["path"] == "AGENTS.md")["content"]
        == "Original policy.\n"
    )
    assert prompts[0]["parent_profile"]["tree_sha256"] == parent_hash
    assert reserved == ["proposer|g0|mutation-01-prompt|attempt-1"]
    assert completed == [(reserved[0], hashlib.sha256(raw.encode()).hexdigest())]


def test_real_proposer_prompt_explains_rollback_after_rule(tmp_path: Path):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    raw = json.dumps(_response(parent_hash))
    prompts = []

    def model_call(prompt: str) -> str:
        prompts.append(json.loads(prompt))
        return raw

    adapter = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=tmp_path / "proposals",
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=model_call,
    )
    adapter.propose(_request(parent_hash), generation=0)

    assert len(prompts) == 1
    path_rule = prompts[0]["response_schema"]["path_rule"]
    assert "every operation object must contain exactly op, path and after" in path_rule
    assert "when op is delete, after must be null" in path_rule


def test_real_proposer_resume_uses_persisted_raw_response_without_redispatch(
    tmp_path: Path,
):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    raw = json.dumps(_response(parent_hash))
    output_root = tmp_path / "proposals"
    first = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=output_root,
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=lambda _prompt: raw,
    )
    first.propose(_request(parent_hash), generation=0)

    calls = []
    resumed = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=output_root,
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=lambda prompt: calls.append(prompt) or raw,
    )

    result = resumed.propose(_request(parent_hash), generation=0)

    assert calls == []
    assert result.raw_responses == (raw,)


def test_created_skill_is_removed_by_rollback_and_parent_tree_is_restored(
    tmp_path: Path,
):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    raw = json.dumps(_response(parent_hash, surface="skills"))
    adapter = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=tmp_path / "proposals",
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=lambda _prompt: raw,
    )

    result = adapter.propose(_request(parent_hash, "skills"), generation=1)

    candidate_root = adapter.profile_root(
        result.changeset.candidate_agent_program_sha256
    )
    assert (candidate_root / ".agents/skills/test-first/SKILL.md").is_file()
    assert adapter.rollback(result.changeset)["verified"] is True
    assert _tree_hash(parent_root) == parent_hash


def test_materialization_error_allows_exactly_one_bounded_repair(tmp_path: Path):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    invalid = _response(parent_hash)
    invalid["rollback_operations"][0]["after"] = "not the parent\n"
    repaired = _response(parent_hash)
    calls = []

    def model_call(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(invalid if len(calls) == 1 else repaired)

    adapter = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=tmp_path / "proposals",
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=model_call,
    )

    result = adapter.propose(_request(parent_hash), generation=2)

    assert result.repairs_used == 1
    assert len(result.raw_responses) == 2
    assert len(calls) == 2
    proposal_root = tmp_path / "proposals/generation-2/mutation-01-prompt"
    assert (proposal_root / "raw-1.txt").is_file()
    assert (proposal_root / "raw-2.txt").is_file()


def test_forbidden_profile_path_is_rejected_without_candidate(tmp_path: Path):
    parent_root = tmp_path / "parent"
    parent_hash = _profile(parent_root)
    response = _response(parent_hash)
    response["operations"][0]["path"] = "../../global/SKILL.md"
    response["rollback_operations"][0]["path"] = "../../global/SKILL.md"
    adapter = RealMutationProposerAdapter(
        profile_roots={parent_hash: parent_root},
        output_root=tmp_path / "proposals",
        provider={"platform": "codex", "model": "gpt-test"},
        model_call=lambda _prompt: json.dumps(response),
        repair_call=None,
    )

    with pytest.raises(ValueError, match="path"):
        adapter.propose(_request(parent_hash), generation=0)
    assert not list((tmp_path / "proposals").rglob("profile"))
