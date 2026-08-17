#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

from attribution import (
    ast_diff_features,
    benjamini_hochberg,
    choose_kmeans,
    parse_evolution_trace,
    permutation_eta_squared,
    shuffled_feature_silhouettes,
)
from evaluator_core import CASE_GROUPS
from lens_features import CONCEPT_GROUPS, mutation_feature_rows

EXPERIMENT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster score-blind J-lens mutation deltas."
    )
    parser.add_argument(
        "--trace", type=Path, default=EXPERIMENT_DIR / "runs/main/evolution_trace.jsonl"
    )
    parser.add_argument(
        "--signatures",
        type=Path,
        default=EXPERIMENT_DIR / "analysis/lens_signatures.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "analysis")
    parser.add_argument("--null-repeats", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def cluster_one(
    lens_name: str,
    *,
    edges: list[dict[str, Any]],
    signatures: dict[str, dict[str, Any]],
    null_repeats: int,
    permutations: int,
    seed: int,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    feature_rows = mutation_feature_rows(
        edges, signatures, lens_name=lens_name, metric="score"
    )
    feature_frame = pd.DataFrame(feature_rows).set_index("edge_id")
    feature_names = feature_frame.columns.tolist()
    matrix = feature_frame.to_numpy(dtype=float)
    clustering = choose_kmeans(matrix, random_seed=seed)
    labels = np.asarray(clustering["labels"], dtype=int)
    embedding = np.asarray(clustering["embedding"], dtype=float)
    outcomes = np.asarray([edge["score_delta"] for edge in edges], dtype=float)
    association = permutation_eta_squared(
        outcomes, labels, permutations=permutations, random_seed=seed + 101
    )
    shuffled = shuffled_feature_silhouettes(
        matrix, repeats=null_repeats, random_seed=seed + 202
    )
    silhouette = float(clustering["silhouette"])
    null_p = float(
        (1 + np.count_nonzero(np.asarray(shuffled) >= silhouette)) / (len(shuffled) + 1)
    )

    edge_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for index, edge in enumerate(edges):
        ast_features = ast_diff_features(edge["parent_code"], edge["child_code"])
        row = {
            "edge_id": edge["edge_id"],
            "iteration": int(edge["iteration"]),
            "parent_id": edge["parent_id"],
            "child_id": edge["child_id"],
            "parent_score": edge["parent_score"],
            "child_score": edge["child_score"],
            "score_delta": edge["score_delta"],
            "outcome": (
                "improved"
                if edge["score_delta"] > 1e-12
                else "regressed"
                if edge["score_delta"] < -1e-12
                else "neutral"
            ),
            "cluster": int(labels[index]),
            "pc1": float(embedding[index, 0]),
            "pc2": float(embedding[index, 1]) if embedding.shape[1] > 1 else 0.0,
            **ast_features,
        }
        edge_rows.append(row)
        for component in CASE_GROUPS:
            parent_value = float(edge["parent_metrics"].get(component, 0.0))
            child_value = float(edge["child_metrics"].get(component, 0.0))
            component_rows.append(
                {
                    "lens": lens_name,
                    "edge_id": edge["edge_id"],
                    "cluster": int(labels[index]),
                    "component": component,
                    "delta": child_value - parent_value,
                }
            )
    edge_frame = pd.DataFrame(edge_rows)
    component_frame = pd.DataFrame(component_rows)
    component_association = {}
    for component_index, component in enumerate(CASE_GROUPS):
        component_deltas = np.asarray(
            [
                float(edge["child_metrics"].get(component, 0.0))
                - float(edge["parent_metrics"].get(component, 0.0))
                for edge in edges
            ],
            dtype=float,
        )
        component_association[component] = permutation_eta_squared(
            component_deltas,
            labels,
            permutations=permutations,
            random_seed=seed + 300 + component_index,
        )

    cluster_rows: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    for cluster in sorted(set(labels)):
        mask = labels == cluster
        cluster_outcomes = outcomes[mask]
        cluster_features = matrix[mask]
        mean_features = cluster_features.mean(axis=0)
        top_indices = np.argsort(np.abs(mean_features))[::-1][:8]
        top_features = [
            {"feature": feature_names[idx], "mean_delta": float(mean_features[idx])}
            for idx in top_indices
        ]
        cluster_rows.append(
            {
                "lens": lens_name,
                "cluster": int(cluster),
                "n": int(mask.sum()),
                "score_delta_mean": float(cluster_outcomes.mean()),
                "score_delta_median": float(np.median(cluster_outcomes)),
                "score_delta_q25": float(np.quantile(cluster_outcomes, 0.25)),
                "score_delta_q75": float(np.quantile(cluster_outcomes, 0.75)),
                "improved_n": int(np.count_nonzero(cluster_outcomes > 1e-12)),
                "regressed_n": int(np.count_nonzero(cluster_outcomes < -1e-12)),
                "neutral_n": int(np.count_nonzero(np.abs(cluster_outcomes) <= 1e-12)),
                "top_features": json.dumps(top_features, ensure_ascii=False),
            }
        )
        for concept in CONCEPT_GROUPS:
            for layer in signatures[next(iter(signatures))]["layers"]:
                feature = f"layer_{int(layer):02d}__{concept}__score"
                if feature in feature_names:
                    feature_index = feature_names.index(feature)
                    heatmap_rows.append(
                        {
                            "lens": lens_name,
                            "cluster": int(cluster),
                            "concept": concept,
                            "layer": int(layer),
                            "mean_delta": float(
                                cluster_features[:, feature_index].mean()
                            ),
                        }
                    )
        member_indices = np.flatnonzero(mask)
        center = np.asarray(clustering["centers"])[cluster]
        distances = np.linalg.norm(embedding[member_indices] - center, axis=1)
        for rank, local_index in enumerate(np.argsort(distances)[:3], start=1):
            edge_index = int(member_indices[local_index])
            edge = edges[edge_index]
            representative_rows.append(
                {
                    "lens": lens_name,
                    "cluster": int(cluster),
                    "rank": rank,
                    "edge_id": edge["edge_id"],
                    "iteration": int(edge["iteration"]),
                    "score_delta": float(edge["score_delta"]),
                    "distance_to_centroid": float(distances[local_index]),
                    "parent_code": edge["parent_code"],
                    "child_code": edge["child_code"],
                }
            )

    correlation_rows = []
    raw_p_values = []
    for feature in feature_names:
        values = feature_frame[feature].to_numpy(dtype=float)
        if np.ptp(values) == 0 or np.ptp(outcomes) == 0:
            rho, p_value = 0.0, 1.0
        else:
            result = spearmanr(values, outcomes)
            rho, p_value = float(result.statistic), float(result.pvalue)
            if not np.isfinite(rho) or not np.isfinite(p_value):
                rho, p_value = 0.0, 1.0
        correlation_rows.append(
            {
                "lens": lens_name,
                "feature": feature,
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )
        raw_p_values.append(p_value)
    adjusted = benjamini_hochberg(raw_p_values)
    for row, q_value in zip(correlation_rows, adjusted, strict=True):
        row["q_value"] = q_value
    correlation_frame = pd.DataFrame(correlation_rows).sort_values(
        ["q_value", "spearman_rho"], ascending=[True, False]
    )

    constant_features = int((feature_frame.nunique(dropna=False) <= 1).sum())
    summary = {
        "lens": lens_name,
        "n_edges": len(edges),
        "n_features": len(feature_names),
        "constant_features": constant_features,
        "k": int(clustering["k"]),
        "silhouette": silhouette,
        "stability_ari": float(clustering["stability_ari"]),
        "pca_components": int(clustering["pca_components"]),
        "pca_variance": float(clustering["pca_variance"]),
        "candidates": clustering["candidates"],
        "score_association": association,
        "component_association": component_association,
        "shuffled_null": {
            "repeats": len(shuffled),
            "mean": float(np.mean(shuffled)),
            "q95": float(np.quantile(shuffled, 0.95)),
            "p_value": null_p,
        },
    }
    feature_export = feature_frame.reset_index()
    feature_export.insert(1, "lens", lens_name)
    return (
        summary,
        edge_frame,
        feature_export,
        pd.DataFrame(cluster_rows),
        pd.DataFrame(heatmap_rows),
        component_frame,
        correlation_frame,
        pd.DataFrame(representative_rows),
    )


def main() -> None:
    args = parse_args()
    edges = parse_evolution_trace(args.trace)
    signature_rows = read_jsonl(args.signatures)
    signatures = {str(row["program_id"]): row for row in signature_rows}
    if len(signatures) != len(signature_rows):
        raise ValueError("duplicate program ids in lens signatures")
    program_ids = {edge[role] for edge in edges for role in ("parent_id", "child_id")}
    missing = sorted(program_ids - signatures.keys())
    extra = sorted(signatures.keys() - program_ids)
    if missing:
        raise ValueError(f"missing lens signatures: {missing}")
    code_by_id = {
        edge[f"{role}_id"]: edge[f"{role}_code"]
        for edge in edges
        for role in ("parent", "child")
    }
    hash_mismatches = [
        program_id
        for program_id, code in code_by_id.items()
        if hashlib.sha256(code.encode()).hexdigest()
        != signatures[program_id]["code_sha256"]
    ]
    if hash_mismatches:
        raise ValueError(f"signature/code hash mismatches: {hash_mismatches}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lens_summaries = []
    edge_frames = []
    feature_frames = []
    cluster_frames = []
    heatmap_frames = []
    component_frames = []
    correlation_frames = []
    representative_frames = []
    for offset, lens_name in enumerate(("jlens", "logit_lens")):
        result = cluster_one(
            lens_name,
            edges=edges,
            signatures=signatures,
            null_repeats=args.null_repeats,
            permutations=args.permutations,
            seed=args.seed + offset * 1000,
        )
        (
            summary,
            edge_frame,
            feature_frame,
            cluster_frame,
            heatmap_frame,
            component_frame,
            correlation_frame,
            representative_frame,
        ) = result
        lens_summaries.append(summary)
        edge_frames.append(
            edge_frame.rename(
                columns={
                    "cluster": f"cluster_{lens_name}",
                    "pc1": f"pc1_{lens_name}",
                    "pc2": f"pc2_{lens_name}",
                }
            )
        )
        feature_frames.append(feature_frame)
        cluster_frames.append(cluster_frame)
        heatmap_frames.append(heatmap_frame)
        component_frames.append(component_frame)
        correlation_frames.append(correlation_frame)
        representative_frames.append(representative_frame)

    edges_export = edge_frames[0].merge(
        edge_frames[1][
            ["edge_id", "cluster_logit_lens", "pc1_logit_lens", "pc2_logit_lens"]
        ],
        on="edge_id",
        validate="one_to_one",
    )
    agreement = float(
        adjusted_rand_score(
            edges_export["cluster_jlens"], edges_export["cluster_logit_lens"]
        )
    )
    running_best = float(edges[0]["parent_score"])
    trajectory_rows = [
        {
            "iteration": 0,
            "score": running_best,
            "best_score": running_best,
            "program_id": edges[0]["parent_id"],
        }
    ]
    for edge in edges:
        child_score = float(edge["child_score"])
        running_best = max(running_best, child_score)
        trajectory_rows.append(
            {
                "iteration": int(edge["iteration"]),
                "score": child_score,
                "best_score": running_best,
                "program_id": edge["child_id"],
            }
        )
    trajectory = pd.DataFrame(trajectory_rows)
    token_counts = np.asarray([row["token_count"] for row in signature_rows], dtype=int)
    token_limits = np.asarray(
        [row.get("observation_max_seq_len", 512) for row in signature_rows],
        dtype=int,
    )
    score_deltas = np.asarray([edge["score_delta"] for edge in edges], dtype=float)
    code_hash_by_id = {
        program_id: hashlib.sha256(code.encode()).hexdigest()
        for program_id, code in code_by_id.items()
    }
    transition_hash_pairs = {
        (code_hash_by_id[edge["parent_id"]], code_hash_by_id[edge["child_id"]])
        for edge in edges
    }
    unique_feature_vectors = {
        lens_name: int(
            feature_frame.drop(columns=["edge_id", "lens"]).drop_duplicates().shape[0]
        )
        for lens_name, feature_frame in zip(
            ("jlens", "logit_lens"), feature_frames, strict=True
        )
    }
    data_quality = {
        "trace_rows": len(read_jsonl(args.trace)),
        "unique_edges": len(edges),
        "unique_programs": len(program_ids),
        "signature_rows": len(signature_rows),
        "unique_code_hashes": len({row["code_sha256"] for row in signature_rows}),
        "unique_transition_hash_pairs": len(transition_hash_pairs),
        "unique_feature_vectors": unique_feature_vectors,
        "lineage_edges_are_independent": False,
        "cached_signature_rows": int(
            sum("reused_from_program_id" in row for row in signature_rows)
        ),
        "missing_signatures": missing,
        "extra_signatures": extra,
        "code_hash_mismatches": hash_mismatches,
        "token_count_min": int(token_counts.min()),
        "token_count_max": int(token_counts.max()),
        "token_limit": int(token_limits.max()),
        "token_limits": sorted({int(value) for value in token_limits}),
        "truncated_prompts": int(np.count_nonzero(token_counts > token_limits)),
        "layer_sets": sorted({tuple(row["layers"]) for row in signature_rows}),
        "score_delta_unique": sorted({float(value) for value in score_deltas}),
        "improved_edges": int(np.count_nonzero(score_deltas > 1e-12)),
        "regressed_edges": int(np.count_nonzero(score_deltas < -1e-12)),
        "neutral_edges": int(np.count_nonzero(np.abs(score_deltas) <= 1e-12)),
    }
    summary = {
        "schema_version": 1,
        "seed": args.seed,
        "cluster_input_excludes_scores": True,
        "observation_claim": "descriptive association; not causal pathway attribution",
        "cross_lens_cluster_ari": agreement,
        "data_quality": data_quality,
        "lenses": {item["lens"]: item for item in lens_summaries},
    }

    pd.concat(feature_frames, ignore_index=True).to_csv(
        args.output_dir / "mutation_features.csv", index=False
    )
    edges_export.to_csv(args.output_dir / "edges.csv", index=False)
    pd.concat(cluster_frames, ignore_index=True).to_csv(
        args.output_dir / "cluster_summary.csv", index=False
    )
    pd.concat(heatmap_frames, ignore_index=True).to_csv(
        args.output_dir / "cluster_heatmap.csv", index=False
    )
    pd.concat(component_frames, ignore_index=True).to_csv(
        args.output_dir / "component_deltas.csv", index=False
    )
    pd.concat(correlation_frames, ignore_index=True).to_csv(
        args.output_dir / "feature_correlations.csv", index=False
    )
    pd.concat(representative_frames, ignore_index=True).to_csv(
        args.output_dir / "representatives.csv", index=False
    )
    trajectory.to_csv(args.output_dir / "score_trajectory.csv", index=False)
    (args.output_dir / "data_quality.json").write_text(
        json.dumps(json_safe(data_quality), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
