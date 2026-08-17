from __future__ import annotations

import pytest

from lens_features import (
    build_observation_prompt,
    mutation_feature_rows,
    signature_feature_vector,
)


def _record(program_id: str, validation: float, aggregation: float):
    return {
        "program_id": program_id,
        "layer_records": [
            {
                "layer": 0,
                "jlens": {
                    "validation": {"score": validation, "rank": 5, "percentile": 0.9},
                    "aggregation": {"score": aggregation, "rank": 8, "percentile": 0.8},
                },
                "logit_lens": {
                    "validation": {
                        "score": validation / 2,
                        "rank": 15,
                        "percentile": 0.7,
                    },
                    "aggregation": {
                        "score": aggregation / 2,
                        "rank": 18,
                        "percentile": 0.6,
                    },
                },
            },
            {
                "layer": 1,
                "jlens": {
                    "validation": {
                        "score": validation + 1,
                        "rank": 2,
                        "percentile": 0.95,
                    },
                    "aggregation": {
                        "score": aggregation + 1,
                        "rank": 4,
                        "percentile": 0.92,
                    },
                },
                "logit_lens": {
                    "validation": {
                        "score": validation / 2 + 1,
                        "rank": 9,
                        "percentile": 0.85,
                    },
                    "aggregation": {
                        "score": aggregation / 2 + 1,
                        "rank": 11,
                        "percentile": 0.81,
                    },
                },
            },
        ],
    }


def test_observation_prompt_has_fixed_semantic_query_without_score_leakage():
    prompt = build_observation_prompt("def solve(records):\n    return []")

    assert "def solve(records)" in prompt
    assert prompt.endswith("Strategy:")
    assert "score" not in prompt.lower()
    assert "fitness" not in prompt.lower()
    assert "evaluator" not in prompt.lower()


def test_signature_feature_vector_is_deterministic_and_lens_specific():
    record = _record("p0", validation=2.0, aggregation=3.0)

    jlens = signature_feature_vector(record, lens_name="jlens", metric="score")
    logit = signature_feature_vector(record, lens_name="logit_lens", metric="score")

    assert list(jlens) == sorted(jlens)
    assert jlens["layer_00__aggregation__score"] == pytest.approx(3.0)
    assert jlens["layer_01__validation__score"] == pytest.approx(3.0)
    assert logit["layer_00__validation__score"] == pytest.approx(1.0)


def test_mutation_feature_rows_compute_child_minus_parent_without_scores():
    records = {"p0": _record("p0", 1.0, 2.0), "p1": _record("p1", 2.5, 1.5)}
    edges = [
        {"edge_id": "p0->p1", "parent_id": "p0", "child_id": "p1", "score_delta": 0.75}
    ]

    rows = mutation_feature_rows(edges, records, lens_name="jlens", metric="score")

    assert len(rows) == 1
    assert rows[0]["edge_id"] == "p0->p1"
    assert rows[0]["layer_00__validation__score"] == pytest.approx(1.5)
    assert rows[0]["layer_00__aggregation__score"] == pytest.approx(-0.5)
    assert "score_delta" not in rows[0]


def test_mutation_feature_rows_reject_missing_program_signatures():
    edges = [{"edge_id": "p0->missing", "parent_id": "p0", "child_id": "missing"}]

    with pytest.raises(KeyError, match="missing"):
        mutation_feature_rows(edges, {"p0": _record("p0", 1.0, 2.0)}, lens_name="jlens")
