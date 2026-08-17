"""PatternCard aggregation for the v2.5 local JLens rail (T4/T5).

Aggregates per-layer hidden-state records plus the tool trace into an
observational PatternCard.  Pure functions: no model calls, no network, no
external writes.  The card is *observational* evidence (never admission);
causality claims require the transfer gate + human review downstream.
"""

from __future__ import annotations

import re
from typing import Any


def _trigger_keywords(instruction: str, limit: int = 6) -> list[str]:
    """Extract stable identifier-ish keywords from the task instruction."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", instruction or "")
    seen: list[str] = []
    for word in words:
        low = word.lower()
        if low not in seen:
            seen.append(low)
        if len(seen) >= limit:
            break
    return seen


def _layer_profile(layer_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-layer mean/l2 across all steps; keeps only scalar stats."""
    by_layer: dict[int, dict[str, list[float]]] = {}
    for record in layer_records or ():
        if not isinstance(record, dict):
            continue
        try:
            layer = int(record.get("layer"))
        except (TypeError, ValueError):
            continue
        bucket = by_layer.setdefault(layer, {"mean": [], "l2": []})
        if isinstance(record.get("mean"), (int, float)):
            bucket["mean"].append(float(record["mean"]))
        if isinstance(record.get("l2_norm"), (int, float)):
            bucket["l2"].append(float(record["l2_norm"]))
    profile: dict[str, Any] = {}
    for layer in sorted(by_layer):
        bucket = by_layer[layer]
        profile[str(layer)] = {
            "mean": sum(bucket["mean"]) / len(bucket["mean"])
            if bucket["mean"]
            else None,
            "l2_norm": sum(bucket["l2"]) / len(bucket["l2"]) if bucket["l2"] else None,
            "records": max(len(bucket["mean"]), len(bucket["l2"])),
        }
    return profile


def _tool_trace_summary(tool_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in tool_events or ():
        kind = event.get("kind", "unknown") if isinstance(event, dict) else "unknown"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def build_pattern_card(
    *,
    layer_records: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    instruction: str,
    pattern_id: str = "local-lens",
    evidence_refs: list[str] | None = None,
) -> dict[str, Any]:
    """Build an observational PatternCard from a local-lens run artifact.

    Output fields: pattern_id, trigger_keywords, layer_profile,
    tool_trace_summary, observational_boundary, evidence_refs, layer_record_count.
    """
    return {
        "pattern_id": pattern_id,
        "trigger_keywords": _trigger_keywords(instruction),
        "layer_profile": _layer_profile(layer_records),
        "tool_trace_summary": _tool_trace_summary(tool_events),
        "observational_boundary": "observational_not_causal",
        "evidence_refs": list(evidence_refs or ()),
        "layer_record_count": len(layer_records or ()),
    }
