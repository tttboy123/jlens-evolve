#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build human-readable cluster profiles."
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=EXPERIMENT_DIR / "analysis"
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=EXPERIMENT_DIR / "analysis/evolution_trace_complete.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ast_equivalent(left: str, right: str) -> bool:
    try:
        return ast.dump(ast.parse(left), include_attributes=False) == ast.dump(
            ast.parse(right), include_attributes=False
        )
    except SyntaxError:
        return False


def short_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:8]


def feature_text(raw: str, limit: int = 4) -> str:
    features = json.loads(raw)[:limit]
    return "; ".join(
        f"{item['feature']} {float(item['mean_delta']):+.3f}" for item in features
    )


def classify_profile(
    parent_code: str,
    child_code: str,
    component_changes: dict[str, float],
) -> tuple[str, str, str]:
    nonzero = {
        key: value for key, value in component_changes.items() if abs(value) > 1e-12
    }
    if parent_code == child_code:
        return (
            "精确自复制",
            "源码、AST、J-Lens delta 与行为全部为零，是搜索坍缩而不是有效探索。",
            "丢弃重复 pair；按 code hash 去重，并对 archive 中已存在源码施加 novelty penalty。",
        )
    if nonzero:
        return (
            "结构性重写与能力交换",
            "唯一发生 evaluator 行为变化的模式；综合分持平来自组件收益与损失相互抵消。",
            "作为 critic/Pareto hard negative，不能标成 DPO positive；补齐丢失组件的针对性样本。",
        )
    if ast_equivalent(parent_code, child_code):
        return (
            "注释级等价改写",
            "AST 与 evaluator 行为不变，但 lens signature 明显变化，主要反映源码措辞与注释的可读语义。",
            "作为行为保持的 invariance pair；训练或筛选时压低对注释措辞的敏感度。",
        )
    return (
        "行为持平的结构改写",
        "语法结构变化但当前 evaluator 未分辨出行为差异，需要更强 hidden cases。",
        "先扩充 evaluator，再决定是否进入偏好数据。",
    )


def main() -> None:
    args = parse_args()
    edges = pd.read_csv(args.analysis_dir / "edges.csv")
    clusters = pd.read_csv(args.analysis_dir / "cluster_summary.csv")
    components = pd.read_csv(args.analysis_dir / "component_deltas.csv")
    trace = read_jsonl(args.trace)
    trace_by_edge = {f"{row['parent_id']}->{row['child_id']}": row for row in trace}
    jlens_clusters = clusters.query("lens == 'jlens'").set_index("cluster")
    logit_clusters = clusters.query("lens == 'logit_lens'").set_index("cluster")
    rows: list[dict[str, Any]] = []
    detail_sections: list[str] = []
    for cluster in sorted(edges["cluster_jlens"].unique()):
        members = edges.query("cluster_jlens == @cluster")
        representative = trace_by_edge[str(members.iloc[0]["edge_id"])]
        parent_code = str(representative["parent_code"])
        child_code = str(representative["child_code"])
        transitions = {
            (
                short_hash(trace_by_edge[edge_id]["parent_code"]),
                short_hash(trace_by_edge[edge_id]["child_code"]),
            )
            for edge_id in members["edge_id"]
        }
        if len(transitions) != 1:
            raise ValueError(
                f"J-Lens cluster {cluster} spans multiple code transitions"
            )
        transition = next(iter(transitions))
        logit_cluster = int(members["cluster_logit_lens"].mode().iloc[0])
        component_changes = (
            components.query("lens == 'jlens' and cluster == @cluster")
            .groupby("component")["delta"]
            .mean()
            .to_dict()
        )
        label, interpretation, training_use = classify_profile(
            parent_code, child_code, component_changes
        )
        nonzero_components = (
            "; ".join(
                f"{name} {value:+.3f}"
                for name, value in component_changes.items()
                if abs(value) > 1e-12
            )
            or "全部 0"
        )
        jlens_top = feature_text(jlens_clusters.loc[cluster, "top_features"])
        logit_top = feature_text(logit_clusters.loc[logit_cluster, "top_features"])
        row = {
            "cluster": f"C{int(cluster)}",
            "label": label,
            "n_edges": len(members),
            "share": float(len(members) / len(edges)),
            "transition": f"{transition[0]}→{transition[1]}",
            "parent_lines": len(parent_code.splitlines()),
            "child_lines": len(child_code.splitlines()),
            "ast_equivalent": ast_equivalent(parent_code, child_code),
            "exact_copy": parent_code == child_code,
            "score_delta": float(members["score_delta"].mean()),
            "component_delta": nonzero_components,
            "jlens_top_features": jlens_top,
            "matched_logit_cluster": f"C{logit_cluster}",
            "logit_top_features": logit_top,
            "interpretation": interpretation,
            "training_use": training_use,
        }
        rows.append(row)
        detail_sections.append(
            f"### {row['cluster']} · {label}（{row['n_edges']}/50，{row['share']:.0%}）\n\n"
            f"- **源码迁移：** `{row['transition']}`，{row['parent_lines']}→{row['child_lines']} 行；"
            f"AST 等价={row['ast_equivalent']}，精确复制={row['exact_copy']}。\n"
            f"- **行为结果：** combined Δ={row['score_delta']:+.3f}；组件 Δ：{nonzero_components}。\n"
            f"- **J‑Lens 主信号：** {jlens_top}。\n"
            f"- **Logit-lens 对照：** 映射到 {row['matched_logit_cluster']}；{logit_top}。\n"
            f"- **归因判断：** {interpretation}\n"
            f"- **训练用途：** {training_use}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.analysis_dir / "cluster_profiles.csv", index=False)
    (args.analysis_dir / "cluster_profiles.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = (
        "# Cluster 逐簇归因\n\n"
        "聚类输入不含 score，但 50 条 edge 只对应 4 种唯一源码迁移；以下簇是搜索模式描述，"
        "不是 50 个独立样本上的因果机制。\n\n" + "\n\n".join(detail_sections) + "\n"
    )
    (args.analysis_dir / "cluster_profiles.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
