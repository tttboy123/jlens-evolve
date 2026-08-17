from __future__ import annotations

import json
from pathlib import Path

from codex_evolution_runtime import run_codex_evolution
from evolve_service import run_cli

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v2.0.0/v1.1.0-codex-target"
CONTRACT = STAGE / "configs/history-contract.json"
BASELINE = STAGE / "configs/baseline-profile"
SYNTHETIC_FIXTURE_DIR = ROOT / "tests/fixtures/v1.1.0-codex-target"
SYNTHETIC_CONTRACT = SYNTHETIC_FIXTURE_DIR / "synthetic-history-contract.json"


def test_synthetic_codex_history_evolution_freezes_before_sealed_and_reports(
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "codex-evolution"
    monkeypatch.setattr("codex_target_runtime.shutil.which", lambda _name: None)

    result = run_codex_evolution(
        contract_path=SYNTHETIC_CONTRACT,
        baseline_root=BASELINE,
        output_dir=output,
        sessions_root=SYNTHETIC_FIXTURE_DIR,
    )

    assert result["decision"] == "accepted"
    assert all(result["contract_checks"].values())
    assert result["public_ab"]["candidate_mean"] == 1
    assert result["public_ab"]["delta"] > 0
    assert result["sealed_ab"]["candidate_mean"] == 1
    assert result["sealed_ab"]["delta"] > 0
    assert result["sealed_ab"]["all_tasks_noninferior"] is True
    assert result["claims"] == {
        "real_codex_cli_bound": False,
        "real_user_history_tasks": 0,
        "synthetic_history_tasks": 14,
        "history_source_kind": "synthetic_fixture",
        "model_calls": 0,
        "global_codex_writes": 0,
        "live_model_improvement_proven": False,
        "offline_application_contract_improvement_proven": True,
        "jlens_causal_claims": 0,
        "runtime_sealed_order_enforced": True,
        "developer_blind_sealed": False,
    }
    assert (output / "REPORT.zh-CN.md").is_file()
    assert (output / "RESULT.json").is_file()
    assert (output / "changeset/AgentChangeSet.json").is_file()
    assert (output / "changeset/apply.patch").is_file()
    assert (output / "changeset/rollback.patch").is_file()
    report = (output / "REPORT.zh-CN.md").read_text(encoding="utf-8")
    assert "合成 Codex 历史" in report
    assert "真实 Codex 用户反馈" not in report
    for heading in (
        "样本怎么产生",
        "观察证据",
        "确定性干预",
        "sealed 泛化审计",
        "怎么反馈优化 Codex Agent",
        "没有证明什么",
        "如何回滚",
    ):
        assert heading in report

    events = [
        json.loads(line)
        for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = [row["event_type"] for row in events]
    assert event_types.index("candidate_frozen") < event_types.index("sealed_opened")
    assert all(
        row.get("partition") != "sealed"
        for row in events[: event_types.index("sealed_opened")]
    )
    source_index = json.loads(
        (output / "source-index.json").read_text(encoding="utf-8")
    )
    assert source_index["selected_messages"] == 14
    assert source_index["excluded_roles"] == [
        "assistant",
        "developer",
        "system",
        "tool",
    ]


def test_stable_cli_runs_synthetic_codex_history_evolution(tmp_path: Path, capsys):
    output = tmp_path / "cli-codex"

    code = run_cli(
        [
            "codex-run",
            "--contract",
            str(SYNTHETIC_CONTRACT),
            "--profile",
            str(BASELINE),
            "--sessions-root",
            str(SYNTHETIC_FIXTURE_DIR),
            "--output",
            str(output),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["decision"] == "accepted"
    assert payload["report"].endswith("REPORT.zh-CN.md")
    assert payload["rollback_patch"].endswith("rollback.patch")
