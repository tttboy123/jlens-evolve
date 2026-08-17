from __future__ import annotations

import numpy as np
import pytest

from attribution import (
    ast_diff_features,
    benjamini_hochberg,
    choose_kmeans,
    eta_squared,
    parse_evolution_trace,
    permutation_eta_squared,
    shuffled_feature_silhouettes,
)


def test_parse_trace_deduplicates_edges_and_computes_score_delta(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        """{"iteration": 2, "parent_id": "p0", "child_id": "p1", "parent_metrics": {"combined_score": 0.25}, "child_metrics": {"combined_score": 0.75}, "parent_code": "def solve(x): return []", "child_code": "def solve(x): return x"}\n
{"iteration": 2, "parent_id": "p0", "child_id": "p1", "parent_metrics": {"combined_score": 0.25}, "child_metrics": {"combined_score": 0.75}, "parent_code": "def solve(x): return []", "child_code": "def solve(x): return x"}\n
""",
        encoding="utf-8",
    )

    rows = parse_evolution_trace(trace)

    assert len(rows) == 1
    assert rows[0]["score_delta"] == pytest.approx(0.5)
    assert rows[0]["edge_id"] == "p0->p1"


def test_parse_trace_rejects_conflicting_duplicate_edges(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        """{"iteration": 2, "parent_id": "p0", "child_id": "p1", "parent_metrics": {"combined_score": 0.25}, "child_metrics": {"combined_score": 0.75}, "parent_code": "a", "child_code": "b"}\n
{"iteration": 3, "parent_id": "p0", "child_id": "p1", "parent_metrics": {"combined_score": 0.25}, "child_metrics": {"combined_score": 0.5}, "parent_code": "a", "child_code": "c"}\n
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting duplicate edge"):
        parse_evolution_trace(trace)


def test_ast_diff_features_detects_new_control_flow_and_calls():
    parent = "def solve(records):\n    return records\n"
    child = (
        "def solve(records):\n"
        "    out = {}\n"
        "    for row in records:\n"
        "        if row:\n"
        "            out[row] = out.get(row, 0) + 1\n"
        "    return sorted(out.items())\n"
    )

    features = ast_diff_features(parent, child)

    assert features["delta_for"] == 1
    assert features["delta_if"] == 1
    assert features["delta_calls"] >= 3
    assert features["line_edit_ratio"] > 0


def test_choose_kmeans_recovers_three_well_separated_clusters():
    rng = np.random.default_rng(7)
    matrix = np.vstack(
        [
            rng.normal(loc=-5, scale=0.15, size=(12, 4)),
            rng.normal(loc=0, scale=0.15, size=(12, 4)),
            rng.normal(loc=5, scale=0.15, size=(12, 4)),
        ]
    )

    result = choose_kmeans(matrix, k_values=range(2, 6), random_seed=11)

    assert result["k"] == 3
    assert result["silhouette"] > 0.8
    assert result["stability_ari"] > 0.95
    assert len(result["labels"]) == len(matrix)


def test_eta_squared_and_bh_adjustment_have_expected_bounds():
    values = np.array([0.0, 0.1, 0.0, 1.0, 1.1, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])

    assert eta_squared(values, labels) > 0.9
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.9])
    assert all(0 <= value <= 1 for value in adjusted)
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[-1] == pytest.approx(0.9)


def test_permutation_eta_squared_detects_strong_group_separation():
    values = np.array([0.0, 0.05, 0.1, 0.9, 0.95, 1.0] * 3)
    labels = np.array([0, 0, 0, 1, 1, 1] * 3)

    result = permutation_eta_squared(values, labels, permutations=999, random_seed=17)

    assert result["eta_squared"] > 0.95
    assert 0 < result["p_value"] < 0.01
    assert result["null_q95"] < result["eta_squared"]


def test_shuffled_feature_silhouettes_are_deterministic_and_bounded():
    rng = np.random.default_rng(9)
    matrix = np.vstack(
        [rng.normal(-3, 0.1, size=(12, 5)), rng.normal(3, 0.1, size=(12, 5))]
    )

    first = shuffled_feature_silhouettes(matrix, repeats=8, random_seed=23)
    second = shuffled_feature_silhouettes(matrix, repeats=8, random_seed=23)

    assert first == second
    assert len(first) == 8
    assert all(-1 <= value <= 1 for value in first)
