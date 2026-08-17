from __future__ import annotations

import json
from pathlib import Path

from codex_changeset import materialize_changeset, propose_changeset
from codex_target_runtime import (
    CodexHistoryContract,
    CodexTargetAgentAdapter,
    evaluate_profile,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v2.0.0/v1.1.0-codex-target"
CONTRACT = STAGE / "configs/history-contract.json"
BASELINE = STAGE / "configs/baseline-profile"
SYNTHETIC_FIXTURE_DIR = ROOT / "tests/fixtures/v1.1.0-codex-target"
SYNTHETIC_CONTRACT = SYNTHETIC_FIXTURE_DIR / "synthetic-history-contract.json"


def test_public_history_proposes_one_three_surface_codex_changeset():
    history = CodexHistoryContract.from_path(
        SYNTHETIC_CONTRACT, sessions_root=SYNTHETIC_FIXTURE_DIR
    )
    public = history.load_partition("public")

    changeset = propose_changeset(BASELINE, public)

    assert changeset.status == "candidate"
    assert (
        changeset.parent_tree_hash
        == CodexTargetAgentAdapter.from_profile(BASELINE).profile.tree_hash
    )
    assert [change.path for change in changeset.changes] == [
        "AGENTS.md",
        ".agents/skills/evidence-to-agent-change/SKILL.md",
        ".codex/evolution-policy.json",
    ]
    assert {change.surface for change in changeset.changes} == {
        "prompt",
        "skill",
        "policy",
    }
    assert {ref["ordinal"] for ref in changeset.evidence_refs} == {
        task.ordinal for task in public
    }
    assert all(ref["partition"] == "public" for ref in changeset.evidence_refs)
    assert changeset.model_calls == 0


def test_changeset_materializes_reviewable_round_trip_patches(tmp_path: Path):
    history = CodexHistoryContract.from_path(
        SYNTHETIC_CONTRACT, sessions_root=SYNTHETIC_FIXTURE_DIR
    )
    public = history.load_partition("public")
    changeset = propose_changeset(BASELINE, public)

    result = materialize_changeset(
        changeset=changeset,
        baseline_root=BASELINE,
        output_dir=tmp_path / "materialized",
    )

    assert result["apply_patch_check"] is True
    assert result["rollback_patch_check"] is True
    assert result["rollback_tree_hash_equal"] is True
    assert result["auto_applied_to_live_profile"] is False
    assert (tmp_path / "materialized/AgentChangeSet.json").is_file()
    assert (tmp_path / "materialized/apply.patch").is_file()
    assert (tmp_path / "materialized/rollback.patch").is_file()
    apply_patch = (tmp_path / "materialized/apply.patch").read_text(encoding="utf-8")
    assert "a/AGENTS.md" in apply_patch
    assert "b/.agents/skills/evidence-to-agent-change/SKILL.md" in apply_patch
    assert "b/.codex/evolution-policy.json" in apply_patch
    stored = json.loads(
        (tmp_path / "materialized/AgentChangeSet.json").read_text(encoding="utf-8")
    )
    assert stored["changeset_hash"] == changeset.sha256

    candidate = CodexTargetAgentAdapter.from_profile(
        tmp_path / "materialized/candidate-profile"
    )
    candidate_result = evaluate_profile(candidate, public)
    assert candidate.profile.capabilities == frozenset(
        {
            "operation_contract",
            "outcome_contract",
            "causal_change",
            "complexity_control",
            "plugin_governance",
            "rollback_safety",
        }
    )
    assert candidate_result["mean_score"] == 1
