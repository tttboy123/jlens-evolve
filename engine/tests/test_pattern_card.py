"""Tests for pattern_card.py (v2.5 T4/T5 PatternCard aggregation)."""

from __future__ import annotations

from pattern_card import build_pattern_card


def _fake_records(n_layers: int = 32) -> list[dict]:
    return [
        {
            "step_label": "plan",
            "layer": i,
            "shape": [1, 8, 16],
            "mean": float(i) * 0.01,
            "l2_norm": float(i) + 1.0,
        }
        for i in range(n_layers)
    ]


def test_build_pattern_card_structure() -> None:
    events = [
        {"kind": "read", "detail": {"path": "pkg/mod.py"}},
        {"kind": "generate", "detail": {"stage": "plan"}},
        {"kind": "grep", "detail": {"pattern": "bug"}},
        {"kind": "generate", "detail": {"stage": "edit"}},
    ]
    card = build_pattern_card(
        layer_records=_fake_records(),
        tool_events=events,
        instruction="Fix callable storage deconstruct bug",
    )
    assert card["observational_boundary"] == "observational_not_causal"
    assert card["layer_record_count"] == 32
    assert len(card["layer_profile"]) == 32
    assert card["layer_profile"]["0"]["l2_norm"] == 1.0
    assert card["tool_trace_summary"] == {"read": 1, "generate": 2, "grep": 1}
    assert "deconstruct" in card["trigger_keywords"]


def test_empty_inputs() -> None:
    card = build_pattern_card(layer_records=[], tool_events=[], instruction="")
    assert card["layer_record_count"] == 0
    assert card["layer_profile"] == {}
    assert card["tool_trace_summary"] == {}
    assert card["trigger_keywords"] == []


def test_bad_records_ignored() -> None:
    card = build_pattern_card(
        layer_records=[{"layer": "x"}, None, {"layer": 1, "mean": 2.0, "l2_norm": 3.0}],
        tool_events=[None, {"kind": "edit"}],
        instruction="hello world test",
    )
    assert card["layer_record_count"] == 3
    assert "1" in card["layer_profile"]
    assert card["tool_trace_summary"] == {"unknown": 1, "edit": 1}
