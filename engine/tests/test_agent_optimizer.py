from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from agent_optimizer import (
    compile_agent_strategy,
    load_agent_strategy,
    load_policy_agent_strategy,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_analysis(tmp_path: Path) -> tuple[Path, Path]:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_json(
        analysis / "analysis_summary.json",
        {
            "cluster_input_excludes_scores": True,
            "observation_claim": "descriptive association; not causal pathway attribution",
            "cross_lens_cluster_ari": 0.8,
            "lenses": {
                "jlens": {"score_association": {"eta_squared": 0.7}},
                "logit_lens": {"score_association": {"eta_squared": 0.75}},
            },
        },
    )
    edges = [
        {
            "edge_id": "p0->p1",
            "score_delta": "0.5",
            "outcome": "improved",
            "cluster_jlens": "1",
            "cluster_logit_lens": "2",
        },
        {
            "edge_id": "p0b->p1b",
            "score_delta": "0.5",
            "outcome": "improved",
            "cluster_jlens": "1",
            "cluster_logit_lens": "2",
        },
        {
            "edge_id": "p0->p2",
            "score_delta": "-0.2",
            "outcome": "regressed",
            "cluster_jlens": "3",
            "cluster_logit_lens": "3",
        },
    ]
    with (analysis / "edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(edges[0]))
        writer.writeheader()
        writer.writerows(edges)
    feature_names = [
        "edge_id",
        "lens",
        "layer_26__validation__score",
        "layer_26__aggregation__score",
        "layer_26__filtering__score",
    ]
    feature_rows = []
    for lens in ("jlens", "logit_lens"):
        feature_rows.extend(
            [
                {
                    "edge_id": "p0->p1",
                    "lens": lens,
                    "layer_26__validation__score": 2,
                    "layer_26__aggregation__score": 3,
                    "layer_26__filtering__score": 0,
                },
                {
                    "edge_id": "p0b->p1b",
                    "lens": lens,
                    "layer_26__validation__score": 2,
                    "layer_26__aggregation__score": 3,
                    "layer_26__filtering__score": 0,
                },
                {
                    "edge_id": "p0->p2",
                    "lens": lens,
                    "layer_26__validation__score": 7,
                    "layer_26__aggregation__score": 8,
                    "layer_26__filtering__score": 9,
                },
            ]
        )
    with (analysis / "mutation_features.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_names)
        writer.writeheader()
        writer.writerows(feature_rows)
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "parent_id": "p0",
                    "child_id": "p1",
                    "parent_code": "def solve(x): return []",
                    "child_code": "def solve(x): return [1]",
                },
                {
                    "parent_id": "p0b",
                    "child_id": "p1b",
                    "parent_code": "def solve(x): return []",
                    "child_code": "def solve(x): return [1]",
                },
                {
                    "parent_id": "p0",
                    "child_id": "p2",
                    "parent_code": "def solve(x): return []",
                    "child_code": "def solve(x): return [2]",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return analysis, trace


def test_strategy_compiler_collapses_repeated_transitions_and_stays_observational(
    tmp_path,
):
    analysis, trace = _fixture_analysis(tmp_path)

    result = compile_agent_strategy(analysis, trace)

    assert result["status"] == "candidate"
    assert result["causal_boundary"] == "observational_not_causal"
    assert result["admission_gate_allowed"] is False
    assert result["evidence"]["trace_edges"] == 3
    assert result["evidence"]["unique_transitions"] == 2
    assert result["evidence"]["repeated_transition_fraction"] == pytest.approx(1 / 3)
    assert result["evidence"]["jlens_incremental_supported"] is False
    assert "structurally new" in result["prompt_guidance"]
    assert result["recommended_search_overrides"]["stochastic_llm"] is True


def test_loader_rejects_strategy_that_claims_admission_authority(tmp_path):
    path = tmp_path / "strategy.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "strategy_id": "unsafe",
            "status": "candidate",
            "causal_boundary": "observational_not_causal",
            "admission_gate_allowed": True,
            "prompt_guidance": "unsafe",
            "recommended_search_overrides": {},
        },
    )

    with pytest.raises(ValueError, match="admission gate"):
        load_agent_strategy(path)


def test_policy_strategy_requires_declared_overrides_to_match(tmp_path):
    path = tmp_path / "strategy.json"
    strategy = {
        "schema_version": 1,
        "strategy_id": "guided-v1",
        "status": "candidate",
        "causal_boundary": "observational_not_causal",
        "admission_gate_allowed": False,
        "prompt_guidance": "Use observer evidence only as bounded prompt guidance.",
        "recommended_search_overrides": {
            "temperature": 0.95,
            "stochastic_llm": True,
        },
    }
    _write_json(path, strategy)
    policy = {
        "id": "jlens-guided-v1",
        "agent_strategy_file": "strategy.json",
        "temperature": 0.95,
        "stochastic_llm": True,
    }

    loaded, loaded_path, digest = load_policy_agent_strategy(tmp_path, policy)

    assert loaded["strategy_id"] == "guided-v1"
    assert loaded_path == path
    assert len(digest) == 64

    policy["temperature"] = 0.85
    with pytest.raises(ValueError, match="does not match"):
        load_policy_agent_strategy(tmp_path, policy)
