#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the reproducible clustering notebook."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir-name", default="evolve-jlens-clustering-data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    quality = summary["data_quality"]
    jlens = summary["lenses"]["jlens"]
    score_changed = quality["improved_edges"] + quality["regressed_edges"] > 0
    if score_changed:
        answer = (
            f"50 条 mutation 中，改善 {quality['improved_edges']}、退化 {quality['regressed_edges']}、"
            f"持平 {quality['neutral_edges']}。J-Lens 聚类 k={jlens['k']}，"
            f"silhouette={jlens['silhouette']:.3f}，score η²="
            f"{jlens['score_association']['eta_squared']:.3f}。"
        )
    else:
        answer = (
            f"50 条 mutation 的 combined score 全部持平；J-Lens 虽形成 k={jlens['k']} 个簇"
            f"（silhouette={jlens['silhouette']:.3f}），但没有 score variance 可归因。"
            "分析重点因此转为组件抵消、语义探索模式与 evaluator 分辨率。"
        )
    diversity = (
        f"{quality['unique_programs']} 个 program ID 只有 {quality['unique_code_hashes']} 个唯一源码、"
        f"{quality['unique_transition_hash_pairs']} 种源码迁移；50 edge 不是独立样本。"
    )

    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# J-Lens × OpenEvolve 聚类归因实验\n\n"
            f"**结论先行：** {answer} {diversity}\n\n"
            "本 notebook 从配套 CSV/JSON 重新加载已核验结果，展示数据质量、聚类、null 对照、"
            "组件归因和代表 mutation。所有 score 与 evaluator component 都在无监督聚类完成后才连接。"
        ),
        nbf.v4.new_markdown_cell(
            "## 分析口径\n\n"
            "- 样本：50 条 OpenEvolve parent→child lineage edge。\n"
            f"- 多样性：{quality['unique_code_hashes']} 个唯一源码、"
            f"{quality['unique_transition_hash_pairs']} 种唯一源码迁移；edge 存在伪重复。\n"
            "- 特征：固定 score-blind prompt 下，J-Lens/logit-lens 的 layer×concept raw-logit child−parent delta。\n"
            "- 聚类：standardize → PCA 95% → KMeans，k=2…6 以 silhouette 选择。\n"
            "- 稳健性：多 seed ARI、200 次逐列 shuffled-feature null、5,000 次 outcome permutation；"
            "因 lineage 重复，p-value 只作搜索内描述。\n"
            "- 解释边界：描述性关联，不是 attention-head/MLP 级因果 pathway attribution。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from IPython.display import Markdown, display\n\n"
            f"DATA_DIR = Path({args.data_dir_name!r})\n"
            "assert DATA_DIR.is_dir(), f'Missing data directory: {DATA_DIR.resolve()}'\n"
            "summary = json.loads((DATA_DIR / 'analysis_summary.json').read_text())\n"
            "quality = json.loads((DATA_DIR / 'data_quality.json').read_text())\n"
            "edges = pd.read_csv(DATA_DIR / 'edges.csv')\n"
            "trajectory = pd.read_csv(DATA_DIR / 'score_trajectory.csv')\n"
            "clusters = pd.read_csv(DATA_DIR / 'cluster_summary.csv')\n"
            "components = pd.read_csv(DATA_DIR / 'component_deltas.csv')\n"
            "heatmap = pd.read_csv(DATA_DIR / 'cluster_heatmap.csv')\n"
            "correlations = pd.read_csv(DATA_DIR / 'feature_correlations.csv')\n"
            "representatives = pd.read_csv(DATA_DIR / 'representatives.csv')\n\n"
            "sns.set_theme(style='whitegrid', context='notebook')\n"
            "BLUE, ORANGE, GREY = '#1473E6', '#D97706', '#6B7280'\n"
            "quality"
        ),
        nbf.v4.new_markdown_cell(
            "## 数据质量：lineage 与 signature 完整配对\n\n"
            "先检查 edge 去重、program signature 覆盖、code hash、一致 layer set 与 token 截断。"
        ),
        nbf.v4.new_code_cell(
            "quality_table = pd.DataFrame({\n"
            "    'check': ['unique edges', 'program IDs', 'unique code hashes', 'unique source transitions',\n"
            "              'signature rows', 'missing signatures',\n"
            "              'hash mismatches', 'truncated prompts', 'token range'],\n"
            "    'value': [quality['unique_edges'], quality['unique_programs'], quality['unique_code_hashes'],\n"
            "              quality['unique_transition_hash_pairs'], quality['signature_rows'],\n"
            "              len(quality['missing_signatures']), len(quality['code_hash_mismatches']),\n"
            "              quality['truncated_prompts'],\n"
            "              f\"{quality['token_count_min']}–{quality['token_count_max']}\"],\n"
            "})\n"
            "display(quality_table)\n"
            "assert quality['unique_edges'] == 50\n"
            "assert not quality['missing_signatures']\n"
            "assert not quality['code_hash_mismatches']\n"
            "assert quality['truncated_prompts'] == 0"
        ),
        nbf.v4.new_markdown_cell(
            "## 搜索轨迹：判断改善、退化还是平台期\n\n"
            "蓝线是每轮 child score，橙色虚线是历史 best。51 个观测点足以展示搜索形态。"
        ),
        nbf.v4.new_code_cell(
            "fig, ax = plt.subplots(figsize=(10, 4.2))\n"
            "ax.plot(trajectory['iteration'], trajectory['score'], color=BLUE, marker='o', ms=3, label='Child score')\n"
            "ax.plot(trajectory['iteration'], trajectory['best_score'], color=ORANGE, ls='--', lw=2, label='Best score')\n"
            "ax.set(title='OpenEvolve score trajectory', xlabel='Iteration', ylabel='Combined score')\n"
            "ax.legend(frameon=False, ncol=2)\n"
            "sns.despine(ax=ax)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## 聚类质量：J-Lens 必须超过 shuffled null\n\n"
            "silhouette 衡量簇分离；shuffled q95 是逐列独立打乱后最佳 silhouette 的 95% 分位；"
            "stability ARI 衡量不同 KMeans seed 的一致性。"
        ),
        nbf.v4.new_code_cell(
            "lens_rows = []\n"
            "for name in ['jlens', 'logit_lens']:\n"
            "    item = summary['lenses'][name]\n"
            "    lens_rows.append({\n"
            "        'lens': name, 'k': item['k'], 'silhouette': item['silhouette'],\n"
            "        'shuffled_q95': item['shuffled_null']['q95'],\n"
            "        'null_p': item['shuffled_null']['p_value'],\n"
            "        'stability_ari': item['stability_ari'],\n"
            "        'score_eta_squared': item['score_association']['eta_squared'],\n"
            "        'score_p': item['score_association']['p_value'],\n"
            "    })\n"
            "lens_quality = pd.DataFrame(lens_rows)\n"
            "display(lens_quality.round(4))\n\n"
            "plot_df = lens_quality.melt('lens', value_vars=['silhouette', 'shuffled_q95', 'stability_ari'],\n"
            "                            var_name='metric', value_name='value')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "sns.barplot(data=plot_df, x='lens', y='value', hue='metric',\n"
            "            palette=[BLUE, GREY, ORANGE], ax=ax)\n"
            "ax.set(title='Cluster quality by lens', xlabel='', ylabel='Metric value')\n"
            "ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.16))\n"
            "sns.despine(ax=ax)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## PCA 投影：查看簇形态与重叠\n\n"
            "每个点是一条 mutation edge；颜色只表示无监督 cluster。PCA 图只用于展示，拟合使用保留 95% 方差的全部主成分。"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)\n"
            "for ax, lens, ccol, xcol, ycol in [\n"
            "    (axes[0], 'J-Lens', 'cluster_jlens', 'pc1_jlens', 'pc2_jlens'),\n"
            "    (axes[1], 'Logit lens', 'cluster_logit_lens', 'pc1_logit_lens', 'pc2_logit_lens'),\n"
            "]:\n"
            "    sns.scatterplot(data=edges, x=xcol, y=ycol, hue=ccol, style='outcome',\n"
            "                    palette='tab10', s=75, edgecolor='white', linewidth=.5, ax=ax)\n"
            "    ax.set_title(f'{lens} mutation embedding')\n"
            "    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')\n"
            "    ax.legend(frameon=False, fontsize=8)\n"
            "    sns.despine(ax=ax)\n"
            "plt.show()\n"
            "display(Markdown(f\"**Cross-lens cluster ARI:** {summary['cross_lens_cluster_ari']:.3f}\"))"
        ),
        nbf.v4.new_markdown_cell(
            "## 组件归因：combined score 会隐藏抵消\n\n"
            "下面按 J-Lens cluster 汇总 8 个 evaluator component 的 child−parent delta。"
        ),
        nbf.v4.new_code_cell(
            "component_mean = (components.query(\"lens == 'jlens'\")\n"
            "                  .groupby(['cluster', 'component'], as_index=False)['delta'].mean())\n"
            "pivot = component_mean.pivot(index='cluster', columns='component', values='delta')\n"
            "fig, ax = plt.subplots(figsize=(11, max(3.5, 1.1 * len(pivot))))\n"
            "sns.heatmap(pivot, center=0, cmap=sns.diverging_palette(28, 220, as_cmap=True),\n"
            "            annot=True, fmt='.2f', linewidths=.5, cbar_kws={'label': 'Mean component delta'}, ax=ax)\n"
            "ax.set(title='Evaluator component delta by J-Lens cluster', xlabel='Component', ylabel='Cluster')\n"
            "plt.show()\n"
            "display(pivot.round(3))"
        ),
        nbf.v4.new_markdown_cell(
            "## Layer×concept 归因：语义方向在哪里出现\n\n"
            "raw-logit delta 按 early/middle/late 深度汇总。它描述 residual→vocabulary readout，不等同于 head/MLP attribution。"
        ),
        nbf.v4.new_code_cell(
            "jheat = heatmap.query(\"lens == 'jlens'\").copy()\n"
            "jheat['depth'] = pd.cut(jheat['layer'], [-1, 9, 20, 100], labels=['early L0–9', 'middle L10–20', 'late L21+'])\n"
            "depth = (jheat.groupby(['cluster', 'depth', 'concept'], observed=True)['mean_delta']\n"
            "         .mean().reset_index())\n"
            "depth['row'] = depth.apply(lambda r: f\"C{int(r['cluster'])} · {r['depth']}\", axis=1)\n"
            "matrix = depth.pivot(index='row', columns='concept', values='mean_delta')\n"
            "fig, ax = plt.subplots(figsize=(12, max(4.5, 0.42 * len(matrix))))\n"
            "sns.heatmap(matrix, center=0, cmap=sns.diverging_palette(28, 220, as_cmap=True),\n"
            "            linewidths=.25, cbar_kws={'label': 'Mean raw-logit delta'}, ax=ax)\n"
            "ax.set(title='J-Lens concept delta by cluster and depth', xlabel='Concept', ylabel='Cluster · depth')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Feature/outcome 检验与 FDR\n\n"
            "Spearman 相关在全部 layer×concept feature 上计算，并用 Benjamini–Hochberg 校正。"
        ),
        nbf.v4.new_code_cell(
            "top_corr = (correlations.sort_values(['lens', 'q_value', 'spearman_rho'], ascending=[True, True, False])\n"
            "            .groupby('lens', as_index=False).head(12))\n"
            "display(top_corr[['lens', 'feature', 'spearman_rho', 'p_value', 'q_value']].round(5))"
        ),
        nbf.v4.new_markdown_cell(
            "## 代表 mutation\n\n"
            "每簇选择离 centroid 最近的 3 条 edge；完整 parent/child 源码保留在 CSV，便于人工语义审查。"
        ),
        nbf.v4.new_code_cell(
            "display(representatives[['lens', 'cluster', 'rank', 'iteration', 'score_delta',\n"
            "                         'distance_to_centroid', 'edge_id']].sort_values(['lens', 'cluster', 'rank']))"
        ),
        nbf.v4.new_markdown_cell(
            "## 对模型训练的可执行建议\n\n"
            "1. **不要把 combined-score tie 直接当 DPO preference。** 先提高 hidden evaluator 的分辨率，或用组件 Pareto dominance 重标。\n"
            "2. **先修复搜索坍缩。** 加入 code-hash novelty、archive 去重和 syntax-preserving mutation；不要把重复源码复制成训练样本。\n"
            "3. **用无退化的组件改善构造 positive。** 明确 trade-off 的 child 作为 hard negative/critic 样本。\n"
            "4. **做 cluster-balanced replay。** 稀有且稳定的语义簇过采样，近重复 neutral 簇降权。\n"
            "5. **把 J-Lens 用作 router 与观测元数据。** 不直接对 lens logit 反向传播；先在 held-out task 证明 uplift。\n"
            "6. **复现门槛。** 新 seed、新任务需同时通过 stability、shuffled null 和真实 outcome uplift。"
        ),
        nbf.v4.new_markdown_cell(
            "## 限制\n\n"
            f"单任务/单 seed；50 edge 只有 {quality['unique_code_hashes']} 个唯一源码，存在严重伪重复；"
            "4-bit proposer 与 BF16 observer 存在量化差异；J-Lens 权重来自 WikiText；"
            "概念词表是人为测量选择；所有归因均为观察性。"
        ),
        nbf.v4.new_code_cell(
            "import platform, sklearn, scipy, matplotlib\n"
            "pd.DataFrame([{'python': platform.python_version(), 'pandas': pd.__version__,\n"
            "               'numpy': np.__version__, 'scikit_learn': sklearn.__version__,\n"
            "               'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__}])"
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
