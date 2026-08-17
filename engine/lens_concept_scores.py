"""Per-layer concept scores from raw hidden states (T4 integration).

v2.5: convert the local JLens track's raw per-layer hidden vectors into
layer-by-concept scores aligned with ``lens_features.signature_feature_vector``
so the same PatternAdvantageMiner pipeline can consume them.

Score = cosine similarity between a layer's last-token hidden vector and the
embedding direction of representative concept tokens (CONCEPT_GROUPS).
Forward-only, weights frozen; observational_not_causal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lens_features import CONCEPT_GROUPS


def _norm(v: list[float]) -> float:
    import math
    return math.sqrt(sum(x * x for x in v)) or 1.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (_norm(a) * _norm(b))


def build_concept_vectors(
    embed_tokens: Any,
    tokenizer: Any,
    concepts: Mapping[str, list[str]] | None = None,
) -> dict[str, list[float]]:
    """Embed one representative token per concept group -> direction vector."""
    import mlx.core as mx

    groups = concepts or CONCEPT_GROUPS
    vectors: dict[str, list[float]] = {}
    for group, words in groups.items():
        word = words[0]
        ids = tokenizer.encode(word)
        if not ids:
            continue
        vec = embed_tokens(mx.array([ids[0]]))
        vectors[group] = [float(v) for v in vec[0, 0, :].tolist()]
    return vectors


def concept_scores(
    layer_records: list[Mapping[str, Any]],
    concept_vectors: Mapping[str, list[float]],
) -> list[dict[str, Any]]:
    """Per-layer {concept: score} from raw records with 'last_token_full'."""
    out: list[dict[str, Any]] = []
    for rec in sorted(layer_records, key=lambda r: int(r["layer"])):
        hidden = rec.get("last_token_full")
        if not hidden:
            continue
        scores = {name: cosine(hidden, vec) for name, vec in concept_vectors.items()}
        out.append({"layer": int(rec["layer"]), "concepts": scores})
    return out


def to_signature_records(concept_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape records for lens_features.signature_feature_vector (jlens)."""
    return [
        {"layer": r["layer"],
         "jlens": {name: {"score": value}
                   for name, value in r["concepts"].items()}}
        for r in concept_rows
    ]
