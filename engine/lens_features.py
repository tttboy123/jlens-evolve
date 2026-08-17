from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

CONCEPT_GROUPS: dict[str, list[str]] = {
    "validation": ["validation", "validate", "invalid", "safe"],
    "filtering": ["filter", "paid", "exclude", "status"],
    "normalization": ["normalize", "strip", "lowercase", "canonical"],
    "aggregation": ["aggregate", "total", "sum", "accumulate"],
    "sorting": ["sort", "order", "descending", "rank"],
    "rounding": ["round", "decimal", "precision"],
    "robustness": ["robust", "malformed", "error", "edge"],
    "iteration": ["loop", "iterate", "iteration"],
    "mapping": ["dictionary", "map", "lookup"],
    "simplicity": ["simple", "concise", "refactor"],
}


def build_observation_prompt(code: str) -> str:
    """Build the fixed score-blind prompt used for every program signature."""
    return (
        "Review this Python implementation of deterministic transaction aggregation.\n"
        "```python\n"
        f"{code.rstrip()}\n"
        "```\n"
        "Name the implementation's main strategy in one word. Strategy:"
    )


def signature_feature_vector(
    record: Mapping[str, Any],
    *,
    lens_name: str,
    metric: str = "score",
) -> dict[str, float]:
    """Flatten a program's layer-by-concept signature in stable key order."""
    if lens_name not in {"jlens", "logit_lens"}:
        raise ValueError(f"unsupported lens: {lens_name}")
    features: dict[str, float] = {}
    for layer_record in sorted(
        record["layer_records"], key=lambda item: int(item["layer"])
    ):
        layer = int(layer_record["layer"])
        concepts = layer_record[lens_name]
        for concept in sorted(concepts):
            value = concepts[concept]
            if value is None or metric not in value:
                continue
            key = f"layer_{layer:02d}__{concept}__{metric}"
            features[key] = float(value[metric])
    return dict(sorted(features.items()))


def mutation_feature_rows(
    edges: Iterable[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    *,
    lens_name: str,
    metric: str = "score",
) -> list[dict[str, Any]]:
    """Return score-blind child-minus-parent feature rows for lineage edges."""
    rows: list[dict[str, Any]] = []
    for edge in edges:
        parent_id = str(edge["parent_id"])
        child_id = str(edge["child_id"])
        if parent_id not in records:
            raise KeyError(f"missing program signature: {parent_id}")
        if child_id not in records:
            raise KeyError(f"missing program signature: {child_id}")
        parent = signature_feature_vector(
            records[parent_id], lens_name=lens_name, metric=metric
        )
        child = signature_feature_vector(
            records[child_id], lens_name=lens_name, metric=metric
        )
        if parent.keys() != child.keys():
            raise ValueError(f"signature feature mismatch for edge {edge['edge_id']}")
        row: dict[str, Any] = {"edge_id": str(edge["edge_id"])}
        row.update({key: child[key] - parent[key] for key in parent})
        rows.append(row)
    return rows
