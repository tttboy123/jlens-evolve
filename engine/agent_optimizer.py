"""Compile score-blind JLens observations into an auditable Agent strategy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_FEATURE = re.compile(r"^layer_(\d+)__(.+)__score$")
_ALLOWED_OVERRIDES = {
    "temperature",
    "num_top_programs",
    "num_diverse_programs",
    "stochastic_llm",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _semantic_profile(
    edge_ids: list[str], features: dict[str, dict[str, float]]
) -> dict[str, float]:
    by_concept: dict[str, list[float]] = defaultdict(list)
    for edge_id in edge_ids:
        concept_values: dict[str, list[float]] = defaultdict(list)
        for feature, value in features[edge_id].items():
            match = _FEATURE.fullmatch(feature)
            if match and 26 <= int(match.group(1)) <= 30:
                concept_values[match.group(2)].append(float(value))
        for concept, values in concept_values.items():
            by_concept[concept].append(_mean(values))
    return {concept: _mean(values) for concept, values in sorted(by_concept.items())}


def compile_agent_strategy(analysis_dir: Path, trace_path: Path) -> dict[str, Any]:
    """Build a candidate prompt/search intervention from unique transitions."""
    analysis_dir = analysis_dir.resolve()
    trace_path = trace_path.resolve()
    summary_path = analysis_dir / "analysis_summary.json"
    edges_path = analysis_dir / "edges.csv"
    features_path = analysis_dir / "mutation_features.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("cluster_input_excludes_scores"):
        raise ValueError("JLens strategy requires score-blind clustering inputs")
    if "not causal" not in str(summary.get("observation_claim", "")):
        raise ValueError("analysis must preserve the non-causal observation boundary")

    edges = _read_csv(edges_path)
    trace = {
        f"{row['parent_id']}->{row['child_id']}": row for row in _read_jsonl(trace_path)
    }
    if {row["edge_id"] for row in edges} != trace.keys():
        raise ValueError("edge/trace inventories do not match")
    jlens_features = {
        row["edge_id"]: {
            key: float(value)
            for key, value in row.items()
            if key not in {"edge_id", "lens"} and value not in {None, ""}
        }
        for row in _read_csv(features_path)
        if row["lens"] == "jlens"
    }
    if set(jlens_features) != trace.keys():
        raise ValueError("JLens feature/trace inventories do not match")

    transition_edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    edge_rows = {row["edge_id"]: row for row in edges}
    for edge_id, row in trace.items():
        transition_edges[
            (_code_hash(str(row["parent_code"])), _code_hash(str(row["child_code"])))
        ].append(edge_id)

    transitions: list[dict[str, Any]] = []
    for (parent_hash, child_hash), edge_ids in sorted(transition_edges.items()):
        score_delta = _mean(
            [float(edge_rows[edge_id]["score_delta"]) for edge_id in edge_ids]
        )
        outcome = (
            "improved"
            if score_delta > 1e-12
            else "regressed"
            if score_delta < -1e-12
            else "neutral"
        )
        transitions.append(
            {
                "transition_id": f"{parent_hash[:12]}->{child_hash[:12]}",
                "parent_sha256": parent_hash,
                "child_sha256": child_hash,
                "occurrences": len(edge_ids),
                "score_delta": score_delta,
                "outcome": outcome,
                "late_layer_semantic_delta": _semantic_profile(
                    edge_ids, jlens_features
                ),
            }
        )

    outcome_profiles: dict[str, dict[str, float]] = {}
    for outcome in ("improved", "neutral", "regressed"):
        members = [row for row in transitions if row["outcome"] == outcome]
        concepts = sorted(
            {concept for row in members for concept in row["late_layer_semantic_delta"]}
        )
        outcome_profiles[outcome] = {
            concept: _mean(
                [
                    float(row["late_layer_semantic_delta"].get(concept, 0.0))
                    for row in members
                ]
            )
            for concept in concepts
        }

    improved = outcome_profiles["improved"]
    neutral = outcome_profiles["neutral"]
    regressed = outcome_profiles["regressed"]
    balanced_concepts = [
        name
        for name, _ in sorted(
            improved.items(),
            key=lambda item: item[1] - neutral.get(item[0], 0.0),
            reverse=True,
        )[:3]
    ]
    overshoot_concepts = [
        name
        for name, _ in sorted(
            regressed.items(),
            key=lambda item: item[1] - improved.get(item[0], 0.0),
            reverse=True,
        )[:3]
    ]
    edge_count = len(edges)
    unique_count = len(transitions)
    repeated_fraction = 1.0 - unique_count / max(1, edge_count)
    jlens_eta = float(summary["lenses"]["jlens"]["score_association"]["eta_squared"])
    logit_eta = float(
        summary["lenses"]["logit_lens"]["score_association"]["eta_squared"]
    )
    incremental_supported = jlens_eta > logit_eta + 0.01
    guidance = (
        "JLens observer evidence, not a correctness or admission signal: "
        f"{repeated_fraction:.0%} of sampled transitions repeated an existing "
        "source transition. Produce a structurally new implementation; do not copy "
        "the current or reference program. Make one narrow structural change for the "
        "named target and preserve every passing case. Historically improved unique "
        f"transitions showed balanced movement in {', '.join(balanced_concepts)}; "
        f"the regressed transition showed broad overshoot in {', '.join(overshoot_concepts)}. "
        "Avoid a broad rewrite and let the deterministic evaluator decide correctness."
    )
    evidence_files = {
        "analysis_summary": {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
        },
        "edges": {"path": str(edges_path), "sha256": _sha256(edges_path)},
        "mutation_features": {
            "path": str(features_path),
            "sha256": _sha256(features_path),
        },
        "trace": {"path": str(trace_path), "sha256": _sha256(trace_path)},
    }
    strategy_seed = json.dumps(
        {
            "evidence_files": evidence_files,
            "transitions": transitions,
            "guidance": guidance,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "strategy_id": f"jlens-agent-{hashlib.sha256(strategy_seed.encode()).hexdigest()[:12]}",
        "status": "candidate",
        "causal_boundary": "observational_not_causal",
        "admission_gate_allowed": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "prompt_guidance": guidance,
        "recommended_search_overrides": {
            "temperature": 0.95,
            "num_top_programs": 1,
            "num_diverse_programs": 1,
            "stochastic_llm": True,
        },
        "evidence": {
            "trace_edges": edge_count,
            "unique_transitions": unique_count,
            "repeated_transition_fraction": repeated_fraction,
            "unique_outcomes": {
                outcome: sum(row["outcome"] == outcome for row in transitions)
                for outcome in ("improved", "neutral", "regressed")
            },
            "outcome_semantic_profiles": outcome_profiles,
            "jlens_score_eta_squared": jlens_eta,
            "logit_lens_score_eta_squared": logit_eta,
            "jlens_incremental_supported": incremental_supported,
            "cross_lens_cluster_ari": float(summary["cross_lens_cluster_ari"]),
            "lineage_edges_are_independent": False,
            "transitions": transitions,
            "files": evidence_files,
        },
    }


def load_agent_strategy(path: Path) -> dict[str, Any]:
    """Load a strategy only when it preserves the observer-only boundary."""
    data = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "schema_version",
        "strategy_id",
        "status",
        "causal_boundary",
        "admission_gate_allowed",
        "prompt_guidance",
        "recommended_search_overrides",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"agent strategy missing fields: {missing}")
    if data["schema_version"] != 1:
        raise ValueError("unsupported agent strategy schema")
    if data["status"] not in {"candidate", "approved"}:
        raise ValueError("agent strategy status must be candidate or approved")
    if data["causal_boundary"] != "observational_not_causal":
        raise ValueError("agent strategy lost the observation boundary")
    if data["admission_gate_allowed"] is not False:
        raise ValueError("JLens strategy cannot claim admission gate authority")
    guidance = str(data["prompt_guidance"]).strip()
    if not guidance or len(guidance) > 2000:
        raise ValueError("agent strategy prompt guidance is empty or too long")
    overrides = data["recommended_search_overrides"]
    if not isinstance(overrides, dict) or set(overrides) - _ALLOWED_OVERRIDES:
        raise ValueError("agent strategy contains unsupported search overrides")
    return data


def load_policy_agent_strategy(
    project_root: Path, selected_policy: dict[str, Any]
) -> tuple[dict[str, Any] | None, Path | None, str | None]:
    """Resolve and validate the optional strategy bound to one search policy."""
    configured = selected_policy.get("agent_strategy_file")
    if not configured:
        return None, None, None
    project_root = project_root.resolve()
    path = (project_root / str(configured)).resolve()
    if not path.is_relative_to(project_root):
        raise ValueError("agent strategy must stay inside the project root")
    strategy = load_agent_strategy(path)
    for name, expected in strategy["recommended_search_overrides"].items():
        if selected_policy.get(name) != expected:
            raise ValueError(
                f"policy override {name} does not match JLens strategy: "
                f"{selected_policy.get(name)!r} != {expected!r}"
            )
    return strategy, path, _sha256(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    strategy = compile_agent_strategy(args.analysis_dir, args.trace)
    _atomic_json(args.output, strategy)
    print(json.dumps(strategy, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
