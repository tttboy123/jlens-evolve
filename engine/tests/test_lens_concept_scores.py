from __future__ import annotations

from lens_concept_scores import concept_scores, to_signature_records
from lens_features import signature_feature_vector


def _records() -> list[dict]:
    return [
        {"layer": 0, "last_token_full": [1.0, 0.0, 0.0]},
        {"layer": 1, "last_token_full": [0.0, 1.0, 0.0]},
    ]


def _concepts() -> dict[str, list[float]]:
    return {
        "validation": [1.0, 0.0, 0.0],
        "filtering": [0.0, 1.0, 0.0],
        "sorting": [0.0, 0.0, 1.0],
    }


def test_concept_scores_are_cosine_per_layer() -> None:
    rows = concept_scores(_records(), _concepts())
    assert len(rows) == 2
    assert rows[0]["layer"] == 0
    assert abs(rows[0]["concepts"]["validation"] - 1.0) < 1e-6
    assert abs(rows[0]["concepts"]["filtering"] - 0.0) < 1e-6
    assert abs(rows[1]["concepts"]["filtering"] - 1.0) < 1e-6


def test_signature_feature_vector_accepts_converted_records() -> None:
    rows = concept_scores(_records(), _concepts())
    sig = signature_feature_vector(
        {"layer_records": to_signature_records(rows)}, lens_name="jlens"
    )
    assert "layer_00__validation__score" in sig
    assert "layer_01__filtering__score" in sig
    assert abs(sig["layer_00__validation__score"] - 1.0) < 1e-6


def test_missing_last_token_full_is_skipped() -> None:
    rows = concept_scores([{"layer": 3, "last_token_full": []}], _concepts())
    assert rows == []
