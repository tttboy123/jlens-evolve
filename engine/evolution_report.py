"""Chinese evidence reports for the multi-generation evolution runtime."""

from __future__ import annotations

from typing import Any


def render_generation_report(summary: dict[str, Any]) -> str:
    patterns = summary["patterns"]
    decision = summary.get("parent_decision")
    failures = summary.get("proposal_failures") or []
    candidate_note = ""
    if failures:
        candidate_note = (
            "\n\n候选数少于 4 的原因（proposer 失败，已 fail-closed 记录）：\n"
            + "\n".join(
                f"- {item.get('request_id')} / {item.get('surface')}: "
                f"{item.get('error_type')} - {item.get('reason', '')[:200]}"
                for item in failures
            )
        )
        candidate_note += "\n\n原始失败证据见 PROPOSAL-FAILURES.jsonl"
    lines = [
        f"# 第 {summary['generation']} 代结果",
        "",
        "## 每代发现了什么",
        "",
        (
            f"本代在 {summary['tasks_retired']} 个全新 search task 上完成冻结证据。"
            f"PatternCard 共 {patterns['total']} 张，其中优势 {patterns['advantage']} 张、"
            f"失败 {patterns['failure']} 张。它们是 observational hypothesis，不是 reward 或 admission。"
        ),
        "",
        "## inactive ChangeSet",
        "",
        f"生成 {len(summary['proposed_candidates'])} 个 inactive ChangeSet："
        + (", ".join(summary["proposed_candidates"]) or "无")
        + candidate_note,
        "",
        "## 淘汰候选",
        "",
        (", ".join(summary["rejected_candidates"]) or "本代没有已评测淘汰候选。"),
        "",
        "## search parent",
        "",
    ]
    if decision is None:
        lines.append("G0 只观察和提案，search parent 保持 seed parent。")
    elif decision["advance"]:
        lines.append(
            f"实验 search parent 从 `{decision['previous_parent_sha256']}` 前进到 "
            f"`{decision['search_parent_sha256']}`；这不是 production promotion。"
        )
    else:
        lines.append(
            f"search parent 保持 `{decision['previous_parent_sha256']}`；原因："
            + ", ".join(decision["reasons"])
        )
    convergence = summary.get("convergence_metrics")
    if convergence is not None:
        lines.extend(
            [
                "",
                "## 收敛指标（candidate-vs-original / candidate-vs-parent）",
                "",
                (
                    f"本代 mean |native_score delta| = {convergence.get('mean_abs_delta')}，"
                    f"safety regression = {convergence.get('safety_regression')}"
                    f"（epsilon={convergence.get('epsilon')}, K={convergence.get('k_consecutive')}）。"
                ),
                "",
            ]
        )
        for candidate, row in convergence.get("per_candidate", {}).items():
            lines.append(
                f"- 候选 `{candidate[:16]}…`：vs original "
                f"{row.get('vs_original')}；vs parent {row.get('vs_parent')}。"
            )
    if patterns.get("generalized", 0) > 0:
        lines.append(
            f"\n本代含 {patterns['generalized']} 张 generalized PatternCard"
            "（“A 修复 B”合并，仍 observational_not_causal）。"
        )
    lines.extend(
        [
            "",
            "## 证据链",
            "",
            (
                "本代原始 arm evidence、Observer evidence、PatternCard、ChangeSet、Tournament 和 "
                "Parent Decision 均保存在同代目录并进入最终语义指纹。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_final_report(result: dict[str, Any]) -> str:
    lines = [
        "# v2.1.1 JLens 多代 Agent 应用层进化报告",
        "",
        "## 结论",
        "",
        (
            f"完成 {result['completed_generations']} 代、退役 "
            f"{result['unique_search_tasks_retired']} 个全新 search task。"
            f"真实调用记账 {result['usage']['real_codex_calls']} 次，其中 Agent task calls "
            f"{result['usage']['agent_task_calls']} 次。"
            f"终态：{result.get('terminal_state', result.get('status'))}。"
        ),
        "",
        (
            "当前完成的是可恢复 mutation/population/selection 进化发动机及搜索谱系证据；"
            "search parent 的前进仅限实验谱系。final sealed 未打开，因此 `agent_optimized=false`，"
            "不能声明生产晋升或 Agentic RSI。"
        ),
        "",
        "## 每代发现了什么",
        "",
    ]
    for generation in result["generations"]:
        convergence = generation.get("convergence_metrics")
        convergence_note = (
            f"，mean |Δ|={convergence.get('mean_abs_delta')}"
            f"/safety回归={convergence.get('safety_regression')}"
            if convergence is not None
            else ""
        )
        lines.append(
            f"- G{generation['generation']}：PatternCard {generation['patterns']['total']} 张"
            f"（优势 {generation['patterns']['advantage']} / 失败 {generation['patterns']['failure']}"
            f" / generalized {generation['patterns'].get('generalized', 0)}），"
            f"inactive ChangeSet {len(generation['proposed_candidates'])} 个，"
            f"淘汰候选 {len(generation['rejected_candidates'])} 个{convergence_note}。"
        )
    lines.extend(
        [
            "",
            "## 收敛判定",
            "",
        ]
    )
    convergence = result.get("convergence")
    if convergence is not None:
        lines.append(
            f"- 连续 K={convergence.get('k_consecutive')} 代 mean |Δ|<"
            f"{convergence.get('epsilon')} 且无 safety regression → converged 停止；"
            f"停止时 mean |Δ|={convergence.get('mean_abs_delta')}。"
        )
    lines.extend(
        [
            "",
            "## PatternCard 如何反馈 Agent",
            "",
            (
                "PatternCard 把冻结 trajectory/tool/native evaluator/cost/safety 中反复出现的优势和失败行为"
                "转成带适用条件、反例、置信度和作用面的假设；MutationProposer 只把它们编译为 inactive "
                "ChangeSet。候选是否保留完全由下一批 fresh task 的 native evaluator/safety/cost 决定。"
            ),
            "",
            "## inactive ChangeSet 与淘汰候选",
            "",
            (
                f"共提出 {result['candidates']['proposed']} 个候选；选择 "
                f"{result['candidates']['selected']}、淘汰 {result['candidates']['rejected']}、"
                f"仍未评测 inactive {result['candidates']['inactive']}。所有负候选、正反操作和原因均保留。"
            ),
            "",
            "## search parent 变化",
            "",
        ]
    )
    for item in result["search_parent_history"]:
        lines.append(
            f"- G{item['generation']}: `{item['previous_parent_sha256']}` → "
            f"`{item['search_parent_sha256']}`（experimental only）"
        )
    lines.extend(
        [
            "",
            "## 证据链",
            "",
            (
                "每代目录保存 EXECUTION-LEDGER、OBSERVER-EVIDENCE、PATTERN-CARDS、候选 ChangeSet、"
                "TOURNAMENT、PARENT-DECISION 和中文结果；archive/events.jsonl 保存 hash-linked lineage，"
                "controller/STATE.json 保存任务、调用和预算状态。"
            ),
            "",
            "## 声明边界",
            "",
            "- JLens 仍是 Observer，不参与 reward、rank 或 admission；",
            "- final sealed 未打开；",
            "- production/global active ref 未写入；",
            "- 模型权重未改变，Skill 未自动安装；",
            "- 本报告中的 parent advance 不等于生产 Agent 已优化。",
            "",
        ]
    )
    return "\n".join(lines)
