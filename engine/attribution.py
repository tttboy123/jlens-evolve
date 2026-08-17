"""Pure analysis helpers for mutation clustering and score attribution."""

from __future__ import annotations

import ast
import difflib
import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def parse_evolution_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load trace JSONL, deduplicate identical edges, and enforce lineage fields."""
    rows: list[dict[str, Any]] = []
    by_edge: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not {"parent_id", "child_id"} <= raw.keys():
            continue
        required = {
            "iteration",
            "parent_metrics",
            "child_metrics",
            "parent_code",
            "child_code",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"line {line_number} missing fields: {sorted(missing)}")
        parent_score = float(raw["parent_metrics"].get("combined_score", 0.0))
        child_score = float(raw["child_metrics"].get("combined_score", 0.0))
        row = {
            **raw,
            "edge_id": f"{raw['parent_id']}->{raw['child_id']}",
            "parent_score": parent_score,
            "child_score": child_score,
            "score_delta": child_score - parent_score,
        }
        existing = by_edge.get(row["edge_id"])
        if existing is not None:
            comparable = ("iteration", "parent_code", "child_code", "child_score")
            if any(existing[key] != row[key] for key in comparable):
                raise ValueError(f"conflicting duplicate edge: {row['edge_id']}")
            continue
        by_edge[row["edge_id"]] = row
        rows.append(row)
    return sorted(rows, key=lambda row: (int(row["iteration"]), row["edge_id"]))


def _ast_counts(source: str) -> dict[str, float]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "nodes": 0,
            "for": 0,
            "while": 0,
            "if": 0,
            "calls": 0,
            "comprehensions": 0,
            "returns": 0,
            "try": 0,
        }
    nodes = list(ast.walk(tree))
    return {
        "nodes": float(len(nodes)),
        "for": float(sum(isinstance(node, (ast.For, ast.AsyncFor)) for node in nodes)),
        "while": float(sum(isinstance(node, ast.While) for node in nodes)),
        "if": float(sum(isinstance(node, (ast.If, ast.IfExp)) for node in nodes)),
        "calls": float(sum(isinstance(node, ast.Call) for node in nodes)),
        "comprehensions": float(
            sum(
                isinstance(
                    node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
                )
                for node in nodes
            )
        ),
        "returns": float(sum(isinstance(node, ast.Return) for node in nodes)),
        "try": float(sum(isinstance(node, ast.Try) for node in nodes)),
    }


def ast_diff_features(parent_code: str, child_code: str) -> dict[str, float]:
    parent = _ast_counts(parent_code)
    child = _ast_counts(child_code)
    features = {f"delta_{name}": child[name] - parent[name] for name in parent}
    features["parent_lines"] = float(len(parent_code.splitlines()))
    features["child_lines"] = float(len(child_code.splitlines()))
    features["delta_lines"] = features["child_lines"] - features["parent_lines"]
    features["line_edit_ratio"] = (
        1.0
        - difflib.SequenceMatcher(
            None, parent_code.splitlines(), child_code.splitlines()
        ).ratio()
    )
    return features


def _prepare_matrix(
    matrix: np.ndarray,
) -> tuple[np.ndarray, StandardScaler, PCA | None]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("matrix must have at least 3 rows and 1 feature")
    if not np.isfinite(values).all():
        raise ValueError("matrix contains non-finite values")
    scaled = StandardScaler().fit_transform(values)
    max_components = min(scaled.shape[0] - 1, scaled.shape[1])
    if max_components <= 1:
        return scaled, StandardScaler().fit(values), None
    pca = PCA(n_components=0.95, svd_solver="full")
    embedded = pca.fit_transform(scaled)
    return embedded, StandardScaler().fit(values), pca


def choose_kmeans(
    matrix: np.ndarray,
    *,
    k_values: Iterable[int] = range(2, 7),
    random_seed: int = 42,
    stability_runs: int = 12,
) -> dict[str, Any]:
    """Choose k by silhouette; report label stability across independent seeds."""
    values = np.asarray(matrix, dtype=float)
    embedded, _, pca = _prepare_matrix(values)
    unique_rows = np.unique(values, axis=0).shape[0]
    candidates: list[dict[str, Any]] = []
    for k in k_values:
        if k < 2 or k >= len(embedded) or k > unique_rows:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            canonical = KMeans(n_clusters=k, random_state=random_seed, n_init=30).fit(
                embedded
            )
        if len(set(canonical.labels_)) < 2:
            continue
        silhouette = float(silhouette_score(embedded, canonical.labels_))
        stability_scores = []
        for offset in range(1, stability_runs + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                alternate = KMeans(
                    n_clusters=k,
                    random_state=random_seed + offset,
                    n_init=10,
                ).fit_predict(embedded)
            stability_scores.append(adjusted_rand_score(canonical.labels_, alternate))
        candidates.append(
            {
                "k": int(k),
                "labels": canonical.labels_.astype(int),
                "centers": canonical.cluster_centers_,
                "silhouette": silhouette,
                "stability_ari": float(np.median(stability_scores)),
            }
        )
    if not candidates:
        raise ValueError("no valid k values for matrix")
    selected = max(
        candidates, key=lambda item: (item["silhouette"], item["stability_ari"])
    )
    return {
        **selected,
        "embedding": embedded,
        "pca_components": int(embedded.shape[1]),
        "pca_variance": float(pca.explained_variance_ratio_.sum()) if pca else 1.0,
        "candidates": [
            {
                key: value
                for key, value in item.items()
                if key not in {"labels", "centers"}
            }
            for item in candidates
        ],
    }


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if len(values) != len(labels) or len(values) == 0:
        raise ValueError("values and labels must have equal non-zero length")
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between += len(group) * float((group.mean() - values.mean()) ** 2)
    return min(1.0, max(0.0, between / total))


def permutation_eta_squared(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    permutations: int = 5000,
    random_seed: int = 42,
) -> dict[str, float]:
    """Test cluster/outcome association by permuting outcomes across fixed labels."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    observed = eta_squared(values, labels)
    rng = np.random.default_rng(random_seed)
    null = np.array(
        [eta_squared(rng.permutation(values), labels) for _ in range(permutations)],
        dtype=float,
    )
    p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    return {
        "eta_squared": observed,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_q95": float(np.quantile(null, 0.95)),
    }


def shuffled_feature_silhouettes(
    matrix: np.ndarray,
    *,
    repeats: int = 200,
    random_seed: int = 42,
    k_values: Iterable[int] = range(2, 7),
) -> list[float]:
    """Destroy cross-feature structure while preserving each feature's marginal values."""
    values = np.asarray(matrix, dtype=float)
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    rng = np.random.default_rng(random_seed)
    silhouettes: list[float] = []
    for repeat in range(repeats):
        shuffled = np.column_stack(
            [rng.permutation(values[:, column]) for column in range(values.shape[1])]
        )
        embedded, _, _ = _prepare_matrix(shuffled)
        unique_rows = np.unique(shuffled, axis=0).shape[0]
        candidates: list[float] = []
        for k in k_values:
            if k < 2 or k >= len(embedded) or k > unique_rows:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                labels = KMeans(
                    n_clusters=k,
                    random_state=random_seed + repeat + 1,
                    n_init=5,
                ).fit_predict(embedded)
            if len(set(labels)) == k:
                candidates.append(float(silhouette_score(embedded, labels)))
        if not candidates:
            raise ValueError("no valid shuffled-null k values for matrix")
        silhouettes.append(max(candidates))
    return silhouettes


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=float)
    if values.size == 0:
        return []
    if ((values < 0) | (values > 1) | ~np.isfinite(values)).any():
        raise ValueError("p-values must be finite and within [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted.tolist()
