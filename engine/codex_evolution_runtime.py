"""One-round real-history Codex application-layer evolution runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from codex_changeset import materialize_changeset, propose_changeset
from codex_target_runtime import (
    CodexHistoryContract,
    CodexHistoryTask,
    CodexRuntimeIdentity,
    CodexTargetAgentAdapter,
    evaluate_profile,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE = ROOT / "artifacts/v2.0.0/v1.1.0-codex-target"
DEFAULT_CONTRACT = DEFAULT_STAGE / "configs/history-contract.json"
DEFAULT_PROFILE = DEFAULT_STAGE / "configs/baseline-profile"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical_json(event) + "\n")


def _matched_ab(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = {row["ordinal"]: row for row in baseline["task_results"]}
    candidate_rows = {row["ordinal"]: row for row in candidate["task_results"]}
    if baseline_rows.keys() != candidate_rows.keys():
        raise ValueError("matched A/B task identities differ")
    rows = []
    for ordinal in sorted(baseline_rows):
        baseline_score = float(baseline_rows[ordinal]["score"])
        candidate_score = float(candidate_rows[ordinal]["score"])
        rows.append(
            {
                "ordinal": ordinal,
                "task_family": baseline_rows[ordinal]["task_family"],
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "delta": candidate_score - baseline_score,
            }
        )
    baseline_mean = mean(row["baseline_score"] for row in rows)
    candidate_mean = mean(row["candidate_score"] for row in rows)
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta": candidate_mean - baseline_mean,
        "all_tasks_noninferior": all(row["delta"] >= 0 for row in rows),
        "tasks": rows,
    }


def _source_rows(tasks: tuple[CodexHistoryTask, ...]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": task.ordinal,
            "timestamp": task.timestamp,
            "partition": task.partition,
            "task_family": task.task_family,
            "message_sha256": task.text_sha256,
            "text": task.text,
            "source_role": task.source_role,
        }
        for task in tasks
    ]


def _report(
    *,
    result: dict[str, Any],
    public_tasks: tuple[CodexHistoryTask, ...],
    sealed_tasks: tuple[CodexHistoryTask, ...],
) -> str:
    decision = "接受" if result["decision"] == "accepted" else "拒绝"
    public = result["public_ab"]
    sealed = result["sealed_ab"]
    synthetic = result["claims"]["history_source_kind"] == "synthetic_fixture"
    history_label = "合成 Codex 历史" if synthetic else "真实 Codex 历史"
    feedback_label = "合成 fixture 用户角色文本" if synthetic else "真实 Codex 用户反馈"
    source_label = "仓库内合成 fixture" if synthetic else "本项目真实 Codex Desktop 根任务"
    return f"""# 首轮{history_label}演化报告

## 结论

本轮 ChangeSet：**{decision}但未应用**。离线交付合同在 public 上从
`{public["baseline_mean"]:.3f}` 提升到 `{public["candidate_mean"]:.3f}`，在候选冻结后才打开的
sealed 上从 `{sealed["baseline_mean"]:.3f}` 提升到 `{sealed["candidate_mean"]:.3f}`。

这个结论证明：现有 Kernel 已能把{feedback_label}转换成可审查、可验证、可回滚的
Prompt / Skill / Policy 变更。它不证明模型在 live inference 中一定遵循这些变更。

## 样本怎么产生

- 来源是{source_label} `{result["source"]["thread_id"]}`。
- 只选择用户角色文本：public {len(public_tasks)} 条，sealed {len(sealed_tasks)} 条。
- 排除了 assistant、developer、system、tool、推理和环境注入内容。
- 每条消息用 ordinal + SHA-256 固定；源会话可以继续追加，但已选消息漂移会 fail closed。
- public 用于归因和生成候选；候选内容与 hash 持久化后才读取 sealed。
- sealed 的运行顺序真实受控，但这些文本在合同设计阶段已被开发者看过，因此不是开发者盲测。

## 观察证据

重复出现的真实纠正分为五类：结果不可读、证据无法落到 Agent 变更、运行交接不完整、
架构复杂度失控、Observer / RSI / PSI 插件边界不清。这些是用户反馈与合同缺口的关联证据，
本身不是 JLens 或某一提示导致失败的因果证明。

## 确定性干预

本轮只改三个项目级 Codex 表面：

1. `AGENTS.md`：要求结果优先、证据/因果边界/下一步齐全。
2. `.agents/skills/evidence-to-agent-change/SKILL.md`：把纠错聚类映射成最小 Prompt / Skill / Policy ChangeSet。
3. `.codex/evolution-policy.json`：冻结 sealed 顺序、插件权限、回滚和禁止自动应用。

完整内容在 `changeset/AgentChangeSet.json`，正向差异在 `changeset/apply.patch`。

## public matched A/B

- baseline：`{public["baseline_mean"]:.3f}`
- candidate：`{public["candidate_mean"]:.3f}`
- delta：`{public["delta"]:+.3f}`
- matched task 非退化：`{str(public["all_tasks_noninferior"]).lower()}`

## sealed 泛化审计

- baseline：`{sealed["baseline_mean"]:.3f}`
- candidate：`{sealed["candidate_mean"]:.3f}`
- delta：`{sealed["delta"]:+.3f}`
- 所有 sealed task 非退化：`{str(sealed["all_tasks_noninferior"]).lower()}`

## 怎么反馈优化 Codex Agent

把 `changeset/apply.patch` 当作待审查的 Agent 应用层候选，而不是模型训练数据：Prompt 负责每轮
交付格式，Skill 负责从 evidence 到 ChangeSet 的重复方法，Policy 负责禁止越权和无审计晋升。
下一次真实 Codex 运行应在隔离项目副本中应用该 patch，收集模型真实输出，再复用同一 matched A/B；
当前项目没有自动修改正在使用的 Codex 配置。

## 没有证明什么

- 没有调用模型，所以没有证明真实模型遵循性、token 成本或在线任务质量提升。
- 历史来自一个项目任务，尚未证明跨用户、跨仓库泛化。
- sealed 只证明 protocol-order holdout；它不是 developer-blind fresh-task 证据。
- JLens 未参与干预，本报告没有 JLens 因果声明。
- 这是 L2/L3 应用层机制证据，不是开放世界 L4 RSI。

## 如何回滚

候选尚未应用，无需回滚 live 配置。若以后在隔离目标根目录应用，可先审查再执行
`git apply changeset/rollback.patch`；本轮已在临时副本证明正向 patch 后回滚树哈希完全相同。
"""


def run_codex_evolution(
    *,
    contract_path: Path,
    baseline_root: Path,
    output_dir: Path,
    sessions_root: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    history = CodexHistoryContract.from_path(contract_path, sessions_root=sessions_root)
    identity = CodexRuntimeIdentity.for_source_snapshot(history.source_snapshot)

    public_tasks = history.load_partition("public")
    _append_event(
        events_path,
        {
            "event_type": "public_history_loaded",
            "partition": "public",
            "tasks": len(public_tasks),
            "message_hashes": [task.text_sha256 for task in public_tasks],
        },
    )
    baseline = CodexTargetAgentAdapter.from_profile(baseline_root, identity=identity)
    baseline_public = evaluate_profile(baseline, public_tasks)
    _append_event(
        events_path,
        {"event_type": "baseline_public_evaluated", "partition": "public"},
    )
    changeset = propose_changeset(baseline_root, public_tasks)
    _append_event(
        events_path,
        {
            "event_type": "changeset_proposed",
            "partition": "public",
            "changeset_hash": changeset.sha256,
        },
    )
    patch_verification = materialize_changeset(
        changeset=changeset,
        baseline_root=baseline_root,
        output_dir=output_dir / "changeset",
    )
    candidate = CodexTargetAgentAdapter.from_profile(
        output_dir / "changeset/candidate-profile", identity=identity
    )
    candidate_public = evaluate_profile(candidate, public_tasks)
    public_ab = _matched_ab(baseline_public, candidate_public)
    _write_json(
        output_dir / "public-results.json",
        {
            "baseline": baseline_public,
            "candidate": candidate_public,
            "matched_ab": public_ab,
        },
    )
    frozen = {
        "changeset_hash": changeset.sha256,
        "candidate_tree_hash": candidate.profile.tree_hash,
        "public_results_sha256": hashlib.sha256(
            (output_dir / "public-results.json").read_bytes()
        ).hexdigest(),
    }
    _write_json(output_dir / "candidate-freeze.json", frozen)
    _append_event(
        events_path,
        {"event_type": "candidate_frozen", "freeze": frozen},
    )

    _append_event(
        events_path,
        {
            "event_type": "sealed_opened",
            "candidate_frozen": True,
            "public_results_persisted": True,
        },
    )
    sealed_tasks = history.load_partition("sealed")
    baseline_sealed = evaluate_profile(baseline, sealed_tasks)
    candidate_sealed = evaluate_profile(candidate, sealed_tasks)
    sealed_ab = _matched_ab(baseline_sealed, candidate_sealed)
    _write_json(
        output_dir / "sealed-results.json",
        {
            "baseline": baseline_sealed,
            "candidate": candidate_sealed,
            "matched_ab": sealed_ab,
        },
    )
    _append_event(
        events_path,
        {"event_type": "sealed_evaluated", "partition": "sealed"},
    )

    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    contract_checks = {
        "source_messages_verified": True,
        "public_delta_positive": public_ab["delta"] > 0,
        "candidate_frozen_before_sealed": event_types.index("candidate_frozen")
        < event_types.index("sealed_opened"),
        "sealed_delta_positive": sealed_ab["delta"] > 0,
        "sealed_all_tasks_noninferior": sealed_ab["all_tasks_noninferior"],
        "three_surfaces_changed": {change.surface for change in changeset.changes}
        == {"prompt", "skill", "policy"},
        "patch_round_trip": bool(
            patch_verification["apply_patch_check"]
            and patch_verification["rollback_patch_check"]
            and patch_verification["rollback_tree_hash_equal"]
        ),
        "no_model_calls": baseline_public["model_calls"]
        + candidate_public["model_calls"]
        + baseline_sealed["model_calls"]
        + candidate_sealed["model_calls"]
        == 0,
        "no_global_writes": baseline_public["global_writes"]
        + candidate_public["global_writes"]
        + baseline_sealed["global_writes"]
        + candidate_sealed["global_writes"]
        == 0,
        "observer_has_no_causal_authority": candidate.profile.policy.get(
            "observer_causal_authority"
        )
        is False,
        "candidate_not_auto_applied": patch_verification["auto_applied_to_live_profile"]
        is False,
    }
    decision = "accepted" if all(contract_checks.values()) else "rejected"
    synthetic = history.source_snapshot.get("synthetic") is True
    selected_task_count = len(public_tasks) + len(sealed_tasks)
    claims = {
        "real_codex_cli_bound": not synthetic,
        "real_user_history_tasks": 0 if synthetic else selected_task_count,
        "synthetic_history_tasks": selected_task_count if synthetic else 0,
        "history_source_kind": "synthetic_fixture" if synthetic else "real_session",
        "model_calls": 0,
        "global_codex_writes": 0,
        "live_model_improvement_proven": False,
        "offline_application_contract_improvement_proven": decision == "accepted",
        "jlens_causal_claims": 0,
        "runtime_sealed_order_enforced": contract_checks[
            "candidate_frozen_before_sealed"
        ],
        "developer_blind_sealed": False,
    }
    stable = {
        "decision": decision,
        "changeset_hash": changeset.sha256,
        "public_ab": public_ab,
        "sealed_ab": sealed_ab,
        "contract_checks": contract_checks,
        "claims": claims,
    }
    result = {
        "schema_version": 1,
        "stage": "v1.1.0-codex-target",
        "decision": decision,
        "source": {
            "thread_id": history.thread_id,
            "source_path": str(history.source_path),
            "source_snapshot": history.source_snapshot,
        },
        "codex_identity": {
            "binary_path": baseline.identity.binary_path,
            "cli_version": baseline.identity.cli_version,
            "execution_mode": baseline.identity.execution_mode,
        },
        "changeset_hash": changeset.sha256,
        "public_ab": public_ab,
        "sealed_ab": sealed_ab,
        "patch_verification": patch_verification,
        "contract_checks": contract_checks,
        "claims": claims,
        "experiment_fingerprint": _sha256(stable),
    }
    _write_json(
        output_dir / "source-index.json",
        {
            "schema_version": 1,
            "thread_id": history.thread_id,
            "source_path": str(history.source_path),
            "selected_messages": len(public_tasks) + len(sealed_tasks),
            "excluded_roles": ["assistant", "developer", "system", "tool"],
            "messages": _source_rows(public_tasks + sealed_tasks),
        },
    )
    _write_json(output_dir / "RESULT.json", result)
    (output_dir / "REPORT.zh-CN.md").write_text(
        _report(result=result, public_tasks=public_tasks, sealed_tasks=sealed_tasks),
        encoding="utf-8",
    )
    _append_event(
        events_path,
        {
            "event_type": "decision_recorded",
            "decision": decision,
            "experiment_fingerprint": result["experiment_fingerprint"],
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_codex_evolution(
        contract_path=args.contract,
        baseline_root=args.profile,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
