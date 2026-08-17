from __future__ import annotations

import json
from pathlib import Path

import pytest

from codex_target_runtime import (
    CodexHistoryContract,
    CodexHistoryError,
    CodexTargetAgentAdapter,
    evaluate_profile,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "artifacts/v2.0.0/v1.1.0-codex-target"
CONTRACT = STAGE / "configs/history-contract.json"
BASELINE = STAGE / "configs/baseline-profile"
SYNTHETIC_FIXTURE_DIR = ROOT / "tests/fixtures/v1.1.0-codex-target"
SYNTHETIC_CONTRACT = SYNTHETIC_FIXTURE_DIR / "synthetic-history-contract.json"


def test_synthetic_history_contract_loads_only_selected_user_tasks():
    history = CodexHistoryContract.from_path(
        SYNTHETIC_CONTRACT, sessions_root=SYNTHETIC_FIXTURE_DIR
    )

    public = history.load_partition("public")

    assert history.thread_id == "019f7285-085b-7d92-bf32-5f887f2a9bfd"
    assert len(public) == 8
    assert public[0].text == "最终得到了什么结果？"
    assert {task.task_family for task in public} == {
        "result_legibility",
        "evidence_to_change",
        "operation_contract",
        "complexity_control",
        "plugin_boundary",
    }
    assert all(task.source_role == "user" for task in public)
    assert all("<environment_context>" not in task.text for task in public)


def test_synthetic_history_contract_fails_closed_on_message_hash_drift(tmp_path: Path):
    payload = json.loads(SYNTHETIC_CONTRACT.read_text(encoding="utf-8"))
    payload["partitions"]["public"][0]["sha256"] = "0" * 64
    changed = tmp_path / "history-contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CodexHistoryError, match="message hash mismatch"):
        CodexHistoryContract.from_path(
            changed, sessions_root=SYNTHETIC_FIXTURE_DIR
        ).load_partition("public")


def test_history_contract_rejects_sources_outside_codex_sessions(tmp_path: Path):
    payload = json.loads(SYNTHETIC_CONTRACT.read_text(encoding="utf-8"))
    payload["source_path"] = str(tmp_path / "unrelated.jsonl")
    changed = tmp_path / "history-contract.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CodexHistoryError, match="outside Codex sessions"):
        CodexHistoryContract.from_path(changed, sessions_root=SYNTHETIC_FIXTURE_DIR)


def test_synthetic_codex_target_adapter_uses_local_codex_and_project_native_profile():
    adapter = CodexTargetAgentAdapter.from_profile(BASELINE)
    public = CodexHistoryContract.from_path(
        SYNTHETIC_CONTRACT, sessions_root=SYNTHETIC_FIXTURE_DIR
    ).load_partition("public")

    result = evaluate_profile(adapter, public)

    assert adapter.identity.binary_path.endswith("codex")
    assert adapter.identity.cli_version.startswith("codex-cli ")
    assert adapter.identity.execution_mode == "offline_history_replay"
    assert adapter.profile.capabilities == frozenset({"operation_contract"})
    assert 0 < result["mean_score"] < 1
    assert result["model_calls"] == 0
    assert result["global_writes"] == 0
    assert result["tasks_total"] == 8
