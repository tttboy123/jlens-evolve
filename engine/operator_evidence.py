"""Compile public structured-operator evidence into RSI/PSI candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_OPERATOR_DESCRIPTIONS = {
    "canonicalize_before_predicate": (
        "Canonicalize a categorical string before evaluating its predicate."
    ),
    "finite_numeric_guard": (
        "Reject bool, non-numeric, non-finite, and non-positive values before use."
    ),
}


def build_operator_evidence(
    audits: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Join proxy selections to public candidate events by source hash."""
    remaining_candidates = [
        event
        for event in events
        if event.get("event_type") == "candidate"
        and (run_id is None or event.get("run_id") == run_id)
        and not (
            int(event.get("iteration", -1)) == 0 and event.get("parent_id") is None
        )
    ]

    operators: dict[str, dict[str, Any]] = {}
    matched = 0
    for audit in audits:
        operator_id = str(audit.get("operator_id", "free_form_rewrite"))
        row = operators.setdefault(
            operator_id,
            {
                "attempts": 0,
                "matched_public_candidates": 0,
                "deterministic_transforms": 0,
                "postcondition_valid": 0,
                "evaluator_valid": 0,
                "accepted": 0,
                "public_improvements": 0,
                "structural_duplicates": 0,
            },
        )
        row["attempts"] += 1
        row["deterministic_transforms"] += int(
            bool(audit.get("deterministic_transform_applied"))
        )
        row["postcondition_valid"] += int(
            bool(audit.get("repair_postcondition_valid"))
            or audit.get("selected_origin") == "deterministic_scaffold"
        )
        source_hash = audit.get("selected_source_sha256")
        ast_hash = audit.get("selected_ast_sha256")
        matching_index = next(
            (
                index
                for index, candidate in enumerate(remaining_candidates)
                if source_hash and candidate.get("source_hash") == source_hash
            ),
            None,
        )
        if matching_index is None:
            matching_index = next(
                (
                    index
                    for index, candidate in enumerate(remaining_candidates)
                    if ast_hash and candidate.get("ast_hash") == ast_hash
                ),
                None,
            )
        event = (
            remaining_candidates.pop(matching_index)
            if matching_index is not None
            else None
        )
        if event is None:
            continue
        matched += 1
        row["matched_public_candidates"] += 1
        metrics = event.get("metrics", {})
        row["evaluator_valid"] += int(float(metrics.get("evaluator_valid", 1.0)) >= 1.0)
        accepted = bool(event.get("accepted"))
        row["accepted"] += int(accepted)
        row["public_improvements"] += int(accepted and bool(event.get("gained_cases")))
        reasons = set(event.get("admission_reasons", []))
        row["structural_duplicates"] += int(
            bool({"exact_duplicate", "ast_duplicate"}.intersection(reasons))
        )

    return {
        "schema_version": 1,
        "definition": "public-only structured operator evidence",
        "run_id": run_id,
        "audit_rows": len(audits),
        "matched_public_candidates": matched,
        "operators": operators,
    }


def merge_operator_evidence(
    evidence_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pool public-only operator counts across independent completed windows."""
    operators: dict[str, dict[str, int]] = {}
    run_ids: list[str] = []
    for evidence in evidence_windows:
        run_id = evidence.get("run_id")
        if run_id is not None:
            run_ids.append(str(run_id))
        for operator_id, row in evidence.get("operators", {}).items():
            pooled = operators.setdefault(str(operator_id), {})
            for key, value in row.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                pooled[key] = pooled.get(key, 0) + int(value)
    return {
        "schema_version": 1,
        "definition": "pooled public-only structured operator evidence",
        "run_id": None,
        "run_ids": run_ids,
        "window_count": len(evidence_windows),
        "audit_rows": sum(int(row.get("audit_rows", 0)) for row in evidence_windows),
        "matched_public_candidates": sum(
            int(row.get("matched_public_candidates", 0)) for row in evidence_windows
        ),
        "operators": operators,
    }


def propose_operator_policy(
    evidence: dict[str, Any], *, parent_policy_id: str
) -> dict[str, Any]:
    """Create a versioned candidate; the next window must prove RSI."""
    scores: dict[str, float] = {}
    for operator_id, row in evidence.get("operators", {}).items():
        if operator_id not in _OPERATOR_DESCRIPTIONS:
            continue
        attempts = int(row.get("attempts", 0))
        improvements = int(row.get("public_improvements", 0))
        valid = int(row.get("postcondition_valid", 0))
        improvement_score = (improvements + 1.0) / (attempts + 2.0)
        validity_score = (valid + 1.0) / (attempts + 1.0)
        scores[operator_id] = improvement_score * validity_score
    if not scores:
        scores = {operator_id: 1.0 for operator_id in _OPERATOR_DESCRIPTIONS}
    total = sum(scores.values())
    weights = {key: value / total for key, value in sorted(scores.items())}
    digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "schema_version": 1,
        "policy_id": f"operator-policy-{digest}",
        "parent_policy_id": parent_policy_id,
        "status": "candidate",
        "operator_weights": weights,
        "evidence_sha256": hashlib.sha256(
            json.dumps(evidence, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "rsi_pass": False,
        "rsi_reason": "requires a separate fixed-budget next-window evaluation",
    }


def render_operator_skill_candidate(operator_id: str, evidence: dict[str, Any]) -> str:
    """Render generic PSI candidate metadata without target-task source code."""
    row = evidence.get("operators", {}).get(operator_id, {})
    description = _OPERATOR_DESCRIPTIONS.get(
        operator_id, "Apply one bounded, machine-verifiable AST mutation."
    )
    return f"""---
name: operator-{operator_id}
status: candidate
operator_id: {operator_id}
---

# Operator candidate: {operator_id}

{description}

## Public evidence

- attempts: {int(row.get("attempts", 0))}
- matched_public_candidates: {int(row.get("matched_public_candidates", 0))}
- postcondition_valid: {int(row.get("postcondition_valid", 0))}
- accepted: {int(row.get("accepted", 0))}
- public_improvements: {int(row.get("public_improvements", 0))}
- structural_duplicates: {int(row.get("structural_duplicates", 0))}

This artifact is a PSI candidate only. Cross-task transfer evaluation is required
before promotion, and the deterministic evaluator remains the correctness authority.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--evidence-input", type=Path, action="append")
    parser.add_argument("--evidence-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument("--parent-policy-id", default="structured-mutation-v4")
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.evidence_input:
        if args.audit or args.events or args.run_id:
            parser.error(
                "--evidence-input cannot be mixed with --audit/--events/--run-id"
            )
        evidence = merge_operator_evidence(
            [
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.evidence_input
            ]
        )
    else:
        if args.audit is None or args.events is None:
            parser.error("--audit and --events are required without --evidence-input")
        evidence = build_operator_evidence(
            _read_jsonl(args.audit), _read_jsonl(args.events), run_id=args.run_id
        )
    evidence["generated_at"] = datetime.now(UTC).isoformat()
    policy = propose_operator_policy(evidence, parent_policy_id=args.parent_policy_id)
    _atomic_text(
        args.evidence_output,
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(
        args.policy_output,
        json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    for operator_id in sorted(_OPERATOR_DESCRIPTIONS):
        _atomic_text(
            args.skills_dir / operator_id / "SKILL.md",
            render_operator_skill_candidate(operator_id, evidence),
        )
    print(json.dumps({"evidence": evidence, "policy": policy}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
