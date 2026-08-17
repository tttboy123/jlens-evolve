#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical technical report artifact."
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=EXPERIMENT_DIR / "analysis"
    )
    parser.add_argument(
        "--output", type=Path, default=EXPERIMENT_DIR / "analysis/artifact.json"
    )
    return parser.parse_args()


def clean_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def p_text(value: float) -> str:
    return f"p={value:.3f}" if value >= 0.001 else "p<0.001"


def main() -> None:
    args = parse_args()
    summary = json.loads((args.analysis_dir / "analysis_summary.json").read_text())
    quality = summary["data_quality"]
    jlens = summary["lenses"]["jlens"]
    logit = summary["lenses"]["logit_lens"]
    edges = pd.read_csv(args.analysis_dir / "edges.csv")
    trajectory = pd.read_csv(args.analysis_dir / "score_trajectory.csv")
    clusters = pd.read_csv(args.analysis_dir / "cluster_summary.csv")
    components = pd.read_csv(args.analysis_dir / "component_deltas.csv")
    heatmap = pd.read_csv(args.analysis_dir / "cluster_heatmap.csv")
    representatives = pd.read_csv(args.analysis_dir / "representatives.csv")
    profiles = pd.read_csv(args.analysis_dir / "cluster_profiles.csv")
    trace_audit = json.loads(
        (args.analysis_dir / "trace_repair_audit.json").read_text()
    )

    best_score = float(trajectory["best_score"].max())
    baseline_score = float(trajectory.iloc[0]["score"])
    score_changed = quality["improved_edges"] + quality["regressed_edges"] > 0
    if score_changed:
        main_result = (
            f"50 轮中有 {quality['improved_edges']} 条改善、{quality['regressed_edges']} 条退化、"
            f"{quality['neutral_edges']} 条持平；最佳分从 {baseline_score:.3f} 到 {best_score:.3f}。"
            f"J‑Lens 聚类为 k={jlens['k']}、silhouette={jlens['silhouette']:.3f}，"
            f"但它与 score delta 的关联强度 η²={jlens['score_association']['eta_squared']:.3f} "
            f"({p_text(jlens['score_association']['p_value'])})，只能解释为观察性关联。"
        )
    else:
        main_result = (
            f"50 轮全部被 OpenEvolve 接受，但 combined score 始终为 {baseline_score:.3f}；"
            "因此本次实验没有识别出可称为“提升分数”的语义簇。"
            f"J‑Lens 特征仍形成 k={jlens['k']} 个形态簇（silhouette={jlens['silhouette']:.3f}），"
            f"但 score η²={jlens['score_association']['eta_squared']:.3f}，不能把簇当成训练收益因果证据。"
        )
    diversity_result = (
        f"更关键的质量门槛是：{quality['unique_programs']} 个 program ID 只有 "
        f"{quality['unique_code_hashes']} 个唯一源码，形成 "
        f"{quality['unique_transition_hash_pairs']} 种唯一源码迁移。"
        "因此 50 条 lineage edge 是搜索频次，不是 50 个独立实验样本；"
        "silhouette、ARI 和 permutation p-value 只能作为本次搜索内部的描述性诊断。"
    )

    component_means = (
        components.groupby(["lens", "component"], as_index=False)["delta"]
        .mean()
        .query("lens == 'jlens'")
        .sort_values("delta")
    )
    largest_loss = component_means.iloc[0]
    largest_gain = component_means.iloc[-1]
    tradeoff_text = (
        f"组件层面存在抵消：平均最大增益是 `{largest_gain.component}` "
        f"({largest_gain.delta:+.3f})，平均最大损失是 `{largest_loss.component}` "
        f"({largest_loss.delta:+.3f})。这说明 combined score 的平台期不是“什么都没变”，"
        "而是行为变化没有形成净收益。"
    )

    lens_metrics = pd.DataFrame(
        [
            {
                "lens": "J-Lens",
                "silhouette": jlens["silhouette"],
                "shuffled_q95": jlens["shuffled_null"]["q95"],
                "stability_ari": jlens["stability_ari"],
                "score_eta_squared": jlens["score_association"]["eta_squared"],
                "null_p_value": jlens["shuffled_null"]["p_value"],
                "k": jlens["k"],
            },
            {
                "lens": "Logit lens",
                "silhouette": logit["silhouette"],
                "shuffled_q95": logit["shuffled_null"]["q95"],
                "stability_ari": logit["stability_ari"],
                "score_eta_squared": logit["score_association"]["eta_squared"],
                "null_p_value": logit["shuffled_null"]["p_value"],
                "k": logit["k"],
            },
        ]
    )
    component_summary = (
        components.groupby(["lens", "cluster", "component"], as_index=False)["delta"]
        .mean()
        .assign(
            cluster=lambda frame: frame["cluster"].map(lambda value: f"C{int(value)}")
        )
    )
    depth_heatmap = heatmap.assign(
        depth_band=pd.cut(
            heatmap["layer"],
            bins=[-1, 9, 20, 100],
            labels=["early L0–9", "middle L10–20", "late L21+"],
        )
    )
    depth_heatmap = (
        depth_heatmap.groupby(
            ["lens", "cluster", "concept", "depth_band"], observed=True, as_index=False
        )["mean_delta"]
        .mean()
        .assign(
            cluster_depth=lambda frame: frame.apply(
                lambda row: f"C{int(row['cluster'])} · {row['depth_band']}", axis=1
            )
        )
    )
    depth_order = {
        value: index
        for index, value in enumerate(depth_heatmap["cluster_depth"].unique())
    }
    depth_heatmap["cluster_depth_index"] = depth_heatmap["cluster_depth"].map(
        depth_order
    )
    representative_table = representatives[
        [
            "lens",
            "cluster",
            "rank",
            "iteration",
            "score_delta",
            "edge_id",
            "distance_to_centroid",
        ]
    ].copy()
    representative_table["cluster"] = representative_table["cluster"].map(
        lambda value: f"C{int(value)}"
    )
    cluster_table = clusters.drop(columns=["top_features"]).copy()
    cluster_table["cluster"] = cluster_table["cluster"].map(
        lambda value: f"C{int(value)}"
    )
    profile_detail = "\n\n".join(
        (
            f"### {row.cluster} · {row.label}（{int(row.n_edges)}/50，{row.share:.0%}）\n\n"
            f"- **迁移：** `{row.transition}`；{int(row.parent_lines)}→{int(row.child_lines)} 行，"
            f"AST 等价={bool(row.ast_equivalent)}，精确复制={bool(row.exact_copy)}。\n"
            f"- **行为：** combined Δ={row.score_delta:+.3f}；{row.component_delta}。\n"
            f"- **J‑Lens：** {row.jlens_top_features}。\n"
            f"- **Logit-lens：** {row.matched_logit_cluster}；{row.logit_top_features}。\n"
            f"- **判断：** {row.interpretation}\n"
            f"- **训练用途：** {row.training_use}"
        )
        for row in profiles.itertuples(index=False)
    )

    headline = pd.DataFrame(
        [
            {
                "edges": quality["unique_edges"],
                "programs": quality["unique_programs"],
                "code_variants": quality["unique_code_hashes"],
                "best_score": best_score,
                "jlens_silhouette": jlens["silhouette"],
                "cross_lens_ari": summary["cross_lens_cluster_ari"],
            }
        ]
    )
    now = datetime.now(UTC).isoformat()
    source = {
        "id": "analysis_pipeline",
        "label": "OpenEvolve lineage + J-Lens analysis snapshot",
        "path": "analysis/analysis_summary.json",
        "query": {
            "engine": "local-python",
            "language": "python",
            "description": (
                "Parses the 50-edge OpenEvolve trace, joins score-blind program signatures, "
                "clusters child-minus-parent semantic deltas, and runs permutation/null checks."
            ),
            "executed_at": now,
            "filters": [
                "50 accepted lineage edges",
                "Qwen3.5-4B J-Lens and logit-lens at the final prompt token",
                "scores excluded until after clustering",
            ],
            "metric_definitions": [
                "score_delta = child combined_score - parent combined_score",
                "silhouette is measured on standardized, PCA-95% mutation features",
                "eta_squared is the variance in outcome associated with fixed cluster labels",
            ],
            "tables_used": [
                "analysis/evolution_trace_complete.jsonl",
                "analysis/trace_repair_audit.json",
                "analysis/lens_signatures.jsonl",
                "analysis/edges.csv",
            ],
        },
    }
    sources = [
        source,
        {
            "id": "openevolve_official",
            "label": "Official OpenEvolve repository",
            "href": "https://github.com/algorithmicsuperintelligence/openevolve",
        },
        {
            "id": "jlens_official",
            "label": "Official Jacobian Lens repository",
            "href": "https://github.com/anthropics/jacobian-lens",
        },
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "layout": "full",
            "body": "# J‑Lens × OpenEvolve 聚类归因实验",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": f"## 技术结论：搜索发生模式坍缩，语义簇不是训练收益证据\n\n{main_result}\n\n{diversity_result}\n\n{tradeoff_text}",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["run_scale", "best_score", "jlens_quality", "cross_lens"],
        },
        {
            "id": "trajectory_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## 搜索轨迹显示的是平台期，而不是稳定收敛\n\n"
                "折线同时显示每轮 child score 与历史 best。连续持平意味着当前 proposer、提示和评分函数的组合没有产生净改善；"
                "源码多样性审计进一步显示 proposer 反复生成少数等价模式，属于搜索坍缩；"
                "它不能证明模型已经学会任务，也不能证明 J‑Lens 方向无效。"
            ),
        },
        {
            "id": "trajectory_chart_block",
            "type": "chart",
            "chartId": "trajectory_chart",
        },
        {
            "id": "cluster_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## J‑Lens 簇的结构必须通过 shuffled null 才算可用\n\n"
                f"J‑Lens silhouette={jlens['silhouette']:.3f}，其 shuffled-feature 95% 分位为 "
                f"{jlens['shuffled_null']['q95']:.3f}（{p_text(jlens['shuffled_null']['p_value'])}）；"
                f"多随机种子稳定性 ARI={jlens['stability_ari']:.3f}。散点图展示 PCA 空间，而聚类实际使用保留 95% 方差的全部主成分。"
            ),
        },
        {"id": "jlens_scatter_block", "type": "chart", "chartId": "jlens_scatter"},
        {
            "id": "lens_comparison_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## J‑Lens 与 logit lens 的差异是增量证据，不是自动胜出\n\n"
                f"两种 lens 的 cluster agreement ARI={summary['cross_lens_cluster_ari']:.3f}。"
                f"本轮两者都选择 k={jlens['k']}，silhouette 分别为 "
                f"{jlens['silhouette']:.3f} 与 {logit['silhouette']:.3f}；标签完全一致，"
                "说明 J‑Lens 没有提供超出普通 logit lens 的额外分簇信息。"
                "比较 silhouette、null 阈值与稳定性可以判断 Jacobian 修正是否带来更清晰的变异结构；"
                "若只在 J‑Lens 出现而无法连接到 held-out outcome，应视为待验证假设。"
            ),
        },
        {"id": "lens_quality_block", "type": "chart", "chartId": "lens_quality"},
        {
            "id": "component_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## 组件归因揭示了“得一项、丢一项”的抵消\n\n"
                f"{tradeoff_text} 分组柱图按 cluster 展示 evaluator 组件的平均 child-parent delta；"
                "这比单一 combined score 更适合构造 Pareto preference pair。"
            ),
        },
        {"id": "component_chart_block", "type": "chart", "chartId": "component_chart"},
        {
            "id": "depth_interpretation",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## layer×concept 热图描述模型如何表征变异\n\n"
                "颜色是同一 cluster 内 child-parent raw-logit delta 的均值，并按 early/middle/late 深度汇总。"
                "它适合定位“语义方向在何处出现”，但不能下钻到 attention head/MLP，也不能证明该方向导致分数变化。"
            ),
        },
        {"id": "depth_heatmap_block", "type": "chart", "chartId": "depth_heatmap"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## 范围、数据与指标定义\n\n"
                f"- **样本单位：** {quality['unique_edges']} 条 parent→child mutation edge，"
                f"覆盖 {quality['unique_programs']} 个唯一程序。\n"
                f"- **有效源码多样性：** {quality['unique_code_hashes']} 个唯一代码哈希、"
                f"{quality['unique_transition_hash_pairs']} 种唯一哈希迁移；edge 非独立。\n"
                f"- **Trace 审计：** OpenEvolve 原始 trace 为 {trace_audit['raw_rows']} 行；"
                f"从 checkpoint 可验证地恢复 iteration {trace_audit['recovered_iterations']}，"
                f"得到 {trace_audit['complete_rows']} 行连续 lineage；原始文件保持不变。\n"
                "- **任务：** 纯 Python 交易记录过滤、归一化、校验、聚合、舍入与排序。\n"
                "- **观察模型：** 本地 BF16 Qwen3.5‑4B；proposer 使用同模型的 MLX 4-bit 量化副本。\n"
                "- **J‑Lens 特征：** 固定、无评分泄露的 code-review prompt，在末 token 位置读取所有 source layer 的 10 个概念组。\n"
                "- **结果指标：** combined score 与 8 个 evaluator component delta；这些指标仅在聚类完成后连接。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": (
                "## 方法：先聚类，后看分数\n\n"
                "1. 对每个唯一程序收集 J‑Lens 与 logit-lens 的 layer×concept raw-logit signature。\n"
                "2. 对每条 lineage edge 计算 child−parent delta；cluster input 不含 score、component score 或 evaluator feedback。\n"
                "3. 特征标准化后以 PCA 保留 95% 方差；在 k=2…6 中按 silhouette 选 k，并用 12 个独立种子报告 ARI 稳定性。\n"
                "4. 用逐列独立 shuffle 的 200 次 null 检查聚类是否只是边际分布造成；用 5,000 次 outcome permutation 检查 cluster/outcome η²。\n"
                "5. 同步分析 AST diff、组件分数 delta、代表性 centroid-nearest mutation，并以 BH-FDR 校正 feature/outcome Spearman 检验。"
                "由于 lineage 中存在大量重复源码，这些检验仅作描述，不作总体显著性推断。"
            ),
        },
        {
            "id": "cluster_table_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## Cluster 精确统计\n\n表格给出样本量、score delta 分布与改善/退化/持平计数；cluster 编号没有跨运行语义。",
        },
        {"id": "cluster_table_block", "type": "table", "tableId": "cluster_table"},
        {
            "id": "cluster_profile_detail",
            "type": "markdown",
            "layout": "full",
            "sourceId": "analysis_pipeline",
            "body": "## 逐簇归因：4 个簇实际是 4 种源码迁移\n\n" + profile_detail,
        },
        {
            "id": "cluster_profile_table_block",
            "type": "table",
            "tableId": "cluster_profile_table",
        },
        {
            "id": "representative_table_intro",
            "type": "markdown",
            "layout": "full",
            "body": "## 代表性 mutation\n\n每簇列出离 centroid 最近的 3 条 edge；完整 parent/child 源码保存在配套 CSV，而报告只呈现索引和距离。",
        },
        {
            "id": "representative_table_block",
            "type": "table",
            "tableId": "representative_table",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 限制与不确定性\n\n"
                "- 这是单任务、单 seed、单 proposer 的 50-edge 观察性实验，不能外推到通用代码模型。\n"
                f"- 50 edge 只含 {quality['unique_code_hashes']} 个唯一源码，存在严重伪重复；"
                "搜索频次会放大重复 mutation 的权重，统计 p-value 不是独立样本推断。\n"
                "- J‑Lens 权重来自 WikiText 校准，概念词表由研究者预定义；二者都可能造成 domain/measurement bias。\n"
                "- 4-bit proposer 与 BF16 observer 不完全同分布；量化会影响 mutation，但不进入 lens readout。\n"
                "- J‑Lens 提供 residual→vocabulary 的 layer×position 读出，不提供 attention-head/MLP 级 pathway attribution。\n"
                "- 即使 cluster/outcome permutation 显著，也只说明关联；需要干预、复现实验和 held-out eval 才能接近因果结论。"
            ),
        },
        {
            "id": "training_actions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 如何把结果用于优化训练\n\n"
                "1. **先扩充分辨率，不把 tie 当偏好。** combined-score 持平 pair 不应直接进入 DPO；增加 hidden cases 或 Pareto 规则，区分组件得失。\n"
                "2. **先修复搜索坍缩。** 在进入训练前提高 proposer 的结构化输出可靠性，加入重复代码惩罚、archive novelty 和 syntax-preserving mutation；当前 50 轮只有少数唯一代码，不能当作 50 条训练样本。\n"
                "3. **构造 Pareto preference pairs。** child 至少改善一个组件且不损害其他组件时才标 positive；有明确 trade-off 的 pair 作为 hard negative 或 critic 训练样本。\n"
                "4. **按语义簇做 cluster-balanced replay。** 对样本稀少但可靠的簇过采样，对大量 neutral、近重复簇降权，防止训练集被一种 mutation 模式淹没。\n"
                "5. **把 J‑Lens 当 router/诊断元数据。** 用 early→late 一致的概念变化选择课程、critic prompt 或 data slice；不要直接把 lens logit 当优化目标。\n"
                "6. **设置上线门槛。** 在新 seed、新任务和 held-out test 上复现 cluster stability、shuffled-null 超越和真实 score uplift，才进入 SFT/DPO 主训练。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## 仍需回答的问题\n\n"
                "- 更高分辨率 evaluator 是否会把当前 neutral edges 分成真实改善与退化？\n"
                "- 更换 proposer seed、温度或模型后，cluster 的语义和稳定性是否复现？\n"
                "- 用随机概念词表、随机位置或随机 lens 权重作对照时，J‑Lens 优势是否仍存在？\n"
                "- 哪些 cluster-balanced SFT/DPO 配方能在 held-out 代码任务上带来净提升，而不是只提高当前 evaluator？"
            ),
        },
    ]

    cards = [
        {
            "id": "run_scale",
            "dataset": "headline",
            "sourceId": "analysis_pipeline",
            "description": "Accepted lineage scale.",
            "metrics": [
                {"label": "Mutation edges", "field": "edges", "format": "number"},
                {
                    "label": "Unique code variants",
                    "field": "code_variants",
                    "format": "number",
                },
            ],
        },
        {
            "id": "best_score",
            "dataset": "headline",
            "sourceId": "analysis_pipeline",
            "description": "Best deterministic evaluator score.",
            "metrics": [
                {
                    "label": "Best combined score",
                    "field": "best_score",
                    "format": "number",
                }
            ],
        },
        {
            "id": "jlens_quality",
            "dataset": "headline",
            "sourceId": "analysis_pipeline",
            "description": "J-Lens cluster separation.",
            "metrics": [
                {
                    "label": "J-Lens silhouette",
                    "field": "jlens_silhouette",
                    "format": "number",
                }
            ],
        },
        {
            "id": "cross_lens",
            "dataset": "headline",
            "sourceId": "analysis_pipeline",
            "description": "Agreement between J-Lens and logit-lens labels.",
            "metrics": [
                {
                    "label": "Cross-lens ARI",
                    "field": "cross_lens_ari",
                    "format": "number",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "trajectory_chart",
            "title": "OpenEvolve score trajectory",
            "subtitle": "51 observations: baseline plus 50 accepted mutations",
            "type": "line",
            "intent": "trend",
            "question": "Did accepted mutations improve the deterministic evaluator over iterations?",
            "rationale": "An ordered line view exposes improvement, regression, and plateau shape.",
            "dataset": "trajectory",
            "sourceId": "analysis_pipeline",
            "encodings": {
                "x": {
                    "field": "iteration",
                    "type": "ordinal",
                    "label": "Iteration",
                },
                "y": {
                    "fields": ["score", "best_score"],
                    "type": "quantitative",
                    "label": "Combined score",
                },
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "jlens_scatter",
            "title": "J-Lens mutation embedding",
            "subtitle": f"{len(edges)} edges; color denotes unsupervised cluster, score excluded from fitting",
            "type": "scatter",
            "intent": "relationship",
            "question": "Do child-parent semantic deltas form separable groups?",
            "rationale": "A PCA scatter shows cluster shape and overlap at the mutation-edge grain.",
            "dataset": "edges",
            "sourceId": "analysis_pipeline",
            "encodings": {
                "x": {"field": "pc1_jlens", "type": "quantitative", "label": "PC1"},
                "y": {"field": "pc2_jlens", "type": "quantitative", "label": "PC2"},
                "color": {
                    "field": "cluster_jlens",
                    "type": "nominal",
                    "label": "Cluster",
                },
                "label": {"field": "iteration", "type": "text", "label": "Iteration"},
                "tooltip": [
                    {"field": "edge_id", "type": "text", "label": "Edge"},
                    {
                        "field": "score_delta",
                        "type": "quantitative",
                        "label": "Score delta",
                    },
                    {"field": "outcome", "type": "nominal", "label": "Outcome"},
                ],
            },
            "layout": "full",
        },
        {
            "id": "lens_quality",
            "title": "Cluster quality by lens",
            "subtitle": "Observed silhouette versus shuffled-feature 95% threshold; ARI is seed stability",
            "type": "bar",
            "intent": "comparison",
            "question": "Does Jacobian correction produce stronger and more stable cluster structure?",
            "rationale": "Grouped bars compare same-scale quality diagnostics across two lenses.",
            "dataset": "lens_metrics",
            "sourceId": "analysis_pipeline",
            "encodings": {
                "x": {"field": "lens", "type": "nominal", "label": "Readout"},
                "y": {
                    "fields": ["silhouette", "shuffled_q95", "stability_ari"],
                    "type": "quantitative",
                    "label": "Metric value",
                },
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "component_chart",
            "title": "Evaluator component delta by cluster",
            "subtitle": "Mean child-parent component score; grouped by unsupervised J-Lens cluster",
            "type": "bar",
            "intent": "comparison",
            "question": "Which evaluator capabilities improve or regress within each semantic cluster?",
            "rationale": "Grouped signed bars reveal component trade-offs hidden by the combined score.",
            "dataset": "component_summary_jlens",
            "sourceId": "analysis_pipeline",
            "encodings": {
                "x": {"field": "component", "type": "nominal", "label": "Component"},
                "y": {
                    "field": "delta",
                    "type": "quantitative",
                    "label": "Mean score delta",
                },
                "color": {"field": "cluster", "type": "nominal", "label": "Cluster"},
            },
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "depth_heatmap",
            "title": "J-Lens concept delta by cluster and depth",
            "subtitle": "Mean raw-logit child-parent delta across early, middle, and late layer bands",
            "type": "heatmap",
            "intent": "relationship",
            "question": "Where in depth does each semantic mutation direction become visible?",
            "rationale": "A heatmap is the most compact view of dense cluster-by-depth-by-concept structure.",
            "dataset": "depth_heatmap_jlens",
            "sourceId": "analysis_pipeline",
            "encodings": {
                "x": {"field": "concept", "type": "nominal", "label": "Concept"},
                "y": {
                    "field": "cluster_depth_index",
                    "type": "quantitative",
                    "label": "Cluster · depth index",
                },
                "color": {
                    "field": "mean_delta",
                    "type": "quantitative",
                    "label": "Mean logit delta",
                },
                "tooltip": [
                    {
                        "field": "cluster_depth",
                        "type": "nominal",
                        "label": "Cluster · depth",
                    },
                    {"field": "cluster", "type": "nominal", "label": "Cluster"},
                    {"field": "depth_band", "type": "nominal", "label": "Depth band"},
                    {
                        "field": "mean_delta",
                        "type": "quantitative",
                        "label": "Mean delta",
                    },
                ],
            },
            "layout": "full",
        },
    ]
    tables = [
        {
            "id": "cluster_table",
            "title": "Cluster outcome summary",
            "subtitle": "All J-Lens and logit-lens clusters; exact distribution statistics",
            "dataset": "clusters",
            "sourceId": "analysis_pipeline",
            "defaultSort": {"field": "lens", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "lens", "label": "Lens", "type": "text"},
                {"field": "cluster", "label": "Cluster", "type": "text"},
                {"field": "n", "label": "N", "format": "number"},
                {
                    "field": "score_delta_mean",
                    "label": "Mean Δ",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "score_delta_median",
                    "label": "Median Δ",
                    "format": "number",
                    "movement": True,
                },
                {"field": "score_delta_q25", "label": "Q25", "format": "number"},
                {"field": "score_delta_q75", "label": "Q75", "format": "number"},
                {"field": "improved_n", "label": "Improved", "format": "number"},
                {"field": "regressed_n", "label": "Regressed", "format": "number"},
                {"field": "neutral_n", "label": "Neutral", "format": "number"},
            ],
        },
        {
            "id": "representative_table",
            "title": "Centroid-nearest mutation edges",
            "subtitle": "Three representative edges per cluster; source code is in the companion CSV",
            "dataset": "representatives",
            "sourceId": "analysis_pipeline",
            "defaultSort": {"field": "lens", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "lens", "label": "Lens", "type": "text"},
                {"field": "cluster", "label": "Cluster", "type": "text"},
                {"field": "rank", "label": "Rank", "format": "number"},
                {"field": "iteration", "label": "Iteration", "format": "number"},
                {
                    "field": "score_delta",
                    "label": "Score Δ",
                    "format": "number",
                    "movement": True,
                },
                {
                    "field": "distance_to_centroid",
                    "label": "Centroid distance",
                    "format": "number",
                },
                {"field": "edge_id", "label": "Edge ID", "type": "text"},
            ],
        },
        {
            "id": "cluster_profile_table",
            "title": "Cluster interpretation and training use",
            "subtitle": "Human review of exact source transitions, behavior, and lens controls",
            "dataset": "cluster_profiles",
            "sourceId": "analysis_pipeline",
            "defaultSort": {"field": "cluster", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "cluster", "label": "Cluster", "type": "text"},
                {"field": "label", "label": "Profile", "type": "text"},
                {"field": "n_edges", "label": "N", "format": "number"},
                {"field": "share", "label": "Share", "format": "percent"},
                {"field": "transition", "label": "Code transition", "type": "text"},
                {
                    "field": "component_delta",
                    "label": "Component delta",
                    "type": "text",
                },
                {
                    "field": "training_use",
                    "label": "Training use",
                    "type": "text",
                },
            ],
        },
    ]

    headline_sql = (
        "SELECT "
        f"{quality['unique_edges']}::INTEGER AS edges, "
        f"{quality['unique_programs']}::INTEGER AS programs, "
        f"{quality['unique_code_hashes']}::INTEGER AS code_variants, "
        f"{best_score}::DOUBLE AS best_score, "
        f"{jlens['silhouette']}::DOUBLE AS jlens_silhouette, "
        f"{summary['cross_lens_cluster_ari']}::DOUBLE AS cross_lens_ari"
    )
    for card in cards:
        card["source"] = {
            "label": "Exact headline values from analysis_summary.json",
            "query": {"engine": "duckdb", "sql": headline_sql},
        }
    chart_sql = {
        "trajectory_chart": (
            "SELECT iteration, score, best_score, program_id "
            "FROM read_csv_auto('analysis/score_trajectory.csv') ORDER BY iteration"
        ),
        "jlens_scatter": (
            "SELECT edge_id, iteration, score_delta, outcome, cluster_jlens, "
            "pc1_jlens, pc2_jlens FROM read_csv_auto('analysis/edges.csv')"
        ),
        "lens_quality": (
            "SELECT * FROM (VALUES "
            f"('J-Lens', {jlens['silhouette']}, {jlens['shuffled_null']['q95']}, {jlens['stability_ari']}), "
            f"('Logit lens', {logit['silhouette']}, {logit['shuffled_null']['q95']}, {logit['stability_ari']})) "
            "AS t(lens, silhouette, shuffled_q95, stability_ari)"
        ),
        "component_chart": (
            "SELECT component, 'C' || CAST(cluster AS VARCHAR) AS cluster, AVG(delta) AS delta "
            "FROM read_csv_auto('analysis/component_deltas.csv') WHERE lens = 'jlens' "
            "GROUP BY component, cluster"
        ),
        "depth_heatmap": (
            "WITH base AS (SELECT cluster, concept, "
            "CASE WHEN layer <= 9 THEN 'early L0-9' WHEN layer <= 20 THEN 'middle L10-20' "
            "ELSE 'late L21+' END AS depth_band, mean_delta "
            "FROM read_csv_auto('analysis/cluster_heatmap.csv') WHERE lens = 'jlens'), "
            "agg AS (SELECT cluster, concept, depth_band, AVG(mean_delta) AS mean_delta FROM base "
            "GROUP BY cluster, concept, depth_band) SELECT *, "
            "DENSE_RANK() OVER (ORDER BY cluster, depth_band) - 1 AS cluster_depth_index, "
            "'C' || CAST(cluster AS VARCHAR) || ' · ' || depth_band AS cluster_depth FROM agg"
        ),
    }
    for chart in charts:
        chart["source"] = {
            "label": f"Reproducible local view for {chart['id']}",
            "query": {"engine": "duckdb", "sql": chart_sql[chart["id"]]},
        }
    table_sql = {
        "cluster_table": (
            "SELECT * EXCLUDE (top_features) FROM read_csv_auto('analysis/cluster_summary.csv')"
        ),
        "representative_table": (
            "SELECT lens, cluster, rank, iteration, score_delta, edge_id, distance_to_centroid "
            "FROM read_csv_auto('analysis/representatives.csv')"
        ),
        "cluster_profile_table": (
            "SELECT cluster, label, n_edges, share, transition, component_delta, training_use "
            "FROM read_csv_auto('analysis/cluster_profiles.csv')"
        ),
    }
    for table in tables:
        table["source"] = {
            "label": f"Reproducible local view for {table['id']}",
            "query": {"engine": "duckdb", "sql": table_sql[table["id"]]},
        }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "J‑Lens × OpenEvolve 聚类归因实验",
            "description": "50-edge technical experiment on score-blind semantic mutation clustering.",
            "generatedAt": now,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": now,
            "status": "ready",
            "datasets": {
                "headline": clean_records(headline),
                "trajectory": clean_records(trajectory),
                "edges": clean_records(edges),
                "lens_metrics": clean_records(lens_metrics),
                "component_summary_jlens": clean_records(
                    component_summary.query("lens == 'jlens'")
                ),
                "depth_heatmap_jlens": clean_records(
                    depth_heatmap.query("lens == 'jlens'")
                ),
                "clusters": clean_records(cluster_table),
                "representatives": clean_records(representative_table),
                "cluster_profiles": clean_records(profiles),
            },
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {"originUrl": "artifact://evolve-jlens-clustering"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
