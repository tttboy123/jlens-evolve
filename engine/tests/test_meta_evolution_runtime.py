from __future__ import annotations

import json
from pathlib import Path

from evolve_service import run_cli
from meta_evolution_runtime import load_meta_programs, run_meta_evolution

ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "artifacts/v2.0.0/v1.1.0-codex-target"
V20 = ROOT / "artifacts/v2.0.0/v2.0.0-meta-evolution"
CONTRACT = V11 / "configs/history-contract.json"
BASELINE = V11 / "configs/baseline-profile"
PROGRAMS = V20 / "configs/meta-programs.json"
SYNTHETIC_FIXTURE_DIR = ROOT / "tests/fixtures/v1.1.0-codex-target"
SYNTHETIC_CONTRACT = SYNTHETIC_FIXTURE_DIR / "synthetic-history-contract.json"


def test_meta_programs_have_hash_lineage_and_shrinking_candidate_budget():
    programs = load_meta_programs(PROGRAMS)

    assert [program.generation for program in programs] == [0, 1, 2]
    assert programs[0].parent_program_hash is None
    assert programs[1].parent_program_hash == programs[0].sha256
    assert programs[2].parent_program_hash == programs[1].sha256
    assert [program.proposed_changesets for program in programs] == [3, 2, 1]


def test_synthetic_meta_evolution_accepts_v2_mechanism_but_rejects_rsi_claim(
    tmp_path: Path,
):
    output = tmp_path / "meta-evolution"

    result = run_meta_evolution(
        contract_path=SYNTHETIC_CONTRACT,
        baseline_root=BASELINE,
        programs_path=PROGRAMS,
        output_dir=output,
        sessions_root=SYNTHETIC_FIXTURE_DIR,
    )

    assert result["release_decision"] == "accepted"
    assert result["agentic_rsi_decision"] == "rejected"
    assert all(result["mechanism_checks"].values())
    assert result["rsi_gates"]["multi_generation_public_improvement"] is True
    assert result["rsi_gates"]["unit_cost_efficiency_improvement"] is True
    assert result["rsi_gates"]["protocol_holdout_improvement"] is True
    assert result["rsi_gates"]["original_task_safety_non_degradation"] is True
    assert result["rsi_gates"]["fresh_meta_tasks"] is False
    assert result["rsi_gates"]["live_target_execution"] is False
    assert result["selected_program_id"] == "meta-v2-compositional"
    evaluations = result["meta_public_evaluations"]
    assert evaluations[1]["mean_score"] > evaluations[0]["mean_score"]
    assert (
        evaluations[2]["score_per_proposed_changeset"]
        > evaluations[1]["score_per_proposed_changeset"]
    )
    assert result["meta_sealed_ab"]["delta"] > 0
    assert result["meta_sealed_ab"]["all_tasks_noninferior"] is True
    assert result["claims"] == {
        "bounded_metaprogram_mechanism": True,
        "optimizer_components_evolved": ["proposer", "search", "routing"],
        "model_calls": 0,
        "live_codex_runs": 0,
        "developer_blind_meta_sealed": False,
        "open_world_agentic_rsi_proven": False,
    }
    assert (output / "REPORT.zh-CN.md").is_file()
    assert (output / "RESULT.json").is_file()
    assert (output / "meta-program-transition.patch").is_file()
    assert (output / "meta-program-rollback.patch").is_file()
    assert (output / "archive/meta-v0-exact.json").is_file()
    assert (output / "archive/meta-v1-routed.json").is_file()
    assert (output / "archive/meta-v2-compositional.json").is_file()
    report = (output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    for heading in (
        "v2.0 结论",
        "MetaProgram 怎么演化",
        "外层 public 搜索",
        "meta-sealed 协议留出",
        "RSI 为什么仍未通过",
        "下一步",
    ):
        assert heading in report
    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    types = [event["event_type"] for event in events]
    assert types.index("meta_program_frozen") < types.index("meta_sealed_opened")


def test_stable_cli_runs_synthetic_v2_meta_evolution(tmp_path: Path, capsys):
    output = tmp_path / "cli-meta"

    code = run_cli(
        [
            "meta-run",
            "--contract",
            str(SYNTHETIC_CONTRACT),
            "--profile",
            str(BASELINE),
            "--sessions-root",
            str(SYNTHETIC_FIXTURE_DIR),
            "--programs",
            str(PROGRAMS),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["release_decision"] == "accepted"
    assert payload["agentic_rsi_decision"] == "rejected"
    assert payload["report"].endswith("REPORT.zh-CN.md")
