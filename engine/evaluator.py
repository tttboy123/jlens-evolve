"""OpenEvolve evaluator backed by fixed, deterministic component tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

from evaluator_core import CASE_GROUPS, CASES, score_program_path


def evaluate(program_path: str) -> EvaluationResult:
    result = score_program_path(program_path)
    metrics = {group: float(result[group]) for group in CASE_GROUPS}
    case_values = {f"case_{case['id']}": 0.0 for case in CASES}
    case_values.update(
        {
            f"case_{row['id']}": float(row["passed"])
            for row in result.get("case_results", [])
        }
    )
    metrics.update(case_values)
    metrics.update(
        {
            "combined_score": float(result["combined_score"]),
            "case_pass_rate": float(result["passed_cases"] / result["total_cases"]),
            "passed_cases": float(result["passed_cases"]),
            "weighted_score": float(result["weighted_score"]),
            "ast_complexity": float(result["ast_complexity"]),
            "evaluator_valid": float(result["evaluator_valid"]),
        }
    )
    behavior_bits = "".join(
        str(int(case_values[f"case_{case['id']}"])) for case in CASES
    )
    behavior_signature = hashlib.sha256(behavior_bits.encode("ascii")).hexdigest()
    failures = [
        {"id": row["id"], "group": row["group"], "error": row["error"]}
        for row in result.get("case_results", [])
        if not row["passed"]
    ]
    failed_groups = sorted({row["group"] for row in failures})
    passing_cases = [
        row["id"] for row in result.get("case_results", []) if row["passed"]
    ]
    target_failure = failures[0] if failures else None
    retrieved_lessons = []
    lessons_path = os.environ.get("EVOLVE_RETRIEVED_LESSONS_FILE")
    if lessons_path and Path(lessons_path).is_file():
        try:
            retrieved_lessons = json.loads(
                Path(lessons_path).read_text(encoding="utf-8")
            )[:3]
        except (OSError, json.JSONDecodeError, TypeError):
            retrieved_lessons = []
    artifacts = {
        "summary": (
            f"passed={result['passed_cases']}/{result['total_cases']} "
            f"combined_score={result['combined_score']:.4f}"
        ),
        "failed_groups": failed_groups,
        "passing_cases": passing_cases,
        "target_failure": target_failure,
        "remaining_failure_count": len(failures),
        "rejection_reasons": result.get("rejection_reasons", []),
        "behavior_signature": behavior_signature,
        "validated_experience": [
            {
                "lesson": lesson.get("lesson"),
                "evidence_count": lesson.get("evidence_count"),
            }
            for lesson in retrieved_lessons
        ],
        "guidance": (
            f"Preserve passing cases {passing_cases}. "
            f"Target only {target_failure['id'] if target_failure else 'no remaining failure'}. "
            "Improve only solve(records); never use file, network, process, environment, "
            "reflection, eval, or exec access."
        ),
    }
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
