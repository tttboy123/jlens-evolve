"""Evaluate one fixed-budget planner-control vs structured-mutation trial."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _audit_structure(audits: list[dict[str, Any]]) -> tuple[int, int, float]:
    seen: set[str] = set()
    duplicates = 0
    for row in audits:
        ast_hash = row.get("selected_ast_sha256")
        if not ast_hash or ast_hash in seen:
            duplicates += 1
        else:
            seen.add(str(ast_hash))
    return duplicates, len(seen), duplicates / max(1, len(audits))


def evaluate_structured_ab(
    control_report: dict[str, Any],
    treatment_report: dict[str, Any],
    control_stats: dict[str, Any],
    treatment_stats: dict[str, Any],
    control_audits: list[dict[str, Any]],
    treatment_audits: list[dict[str, Any]],
    *,
    completed_seed_count: int,
) -> dict[str, Any]:
    """Apply capability gates while keeping formal multi-seed promotion closed."""
    control = control_report["run"]
    treatment = treatment_report["run"]
    contract_keys = (
        "task_id",
        "config_hash",
        "evaluator_hash",
        "initial_hash",
        "model_id",
        "iterations_requested_total",
        "experience_mode",
        "experiment_seed",
    )
    mismatches = [
        key for key in contract_keys if control.get(key) != treatment.get(key)
    ]
    control_attempts = int(control_report["admission"].get("candidate_attempts", 0))
    treatment_attempts = int(treatment_report["admission"].get("candidate_attempts", 0))
    contract_matched = bool(
        not mismatches
        and control_attempts == treatment_attempts
        and completed_seed_count >= 1
    )
    controller_binding_valid = bool(
        control.get("proposal_controller_mode") == "planner-control"
        and treatment.get("proposal_controller_mode") == "structured-mutation"
        and control.get("proposal_controller_endpoint_verified") is True
        and treatment.get("proposal_controller_endpoint_verified") is True
        and control.get("proposal_controller_sha256")
        and treatment.get("proposal_controller_sha256")
        and control.get("proposal_controller_implementation_sha256")
        == treatment.get("proposal_controller_implementation_sha256")
        and int(control.get("proposal_controller_calls_per_request", 0)) == 2
        and int(treatment.get("proposal_controller_calls_per_request", 0)) == 2
    )
    control_requests = int(control_stats.get("requests", 0))
    treatment_requests = int(treatment_stats.get("requests", 0))
    control_calls = int(control_stats.get("upstream_calls", 0))
    treatment_calls = int(treatment_stats.get("upstream_calls", 0))
    fixed_call_budget = bool(
        control_stats.get("mode") == "planner-control"
        and treatment_stats.get("mode") == "structured-mutation"
        and control_stats.get("protocol_version") == "structured-mutation-v4"
        and treatment_stats.get("protocol_version") == "structured-mutation-v4"
        and control_requests == control_attempts
        and treatment_requests == treatment_attempts
        and control_calls == treatment_calls == 2 * control_attempts
    )
    operator_execution_valid = bool(
        int(control_stats.get("structured_plans", 0)) == 0
        and int(control_stats.get("deterministic_transforms", 0)) == 0
        and int(treatment_stats.get("structured_plans", 0)) > 0
        and int(treatment_stats.get("deterministic_transforms", 0)) > 0
        and int(treatment_stats.get("model_repairs_selected", 0))
        + int(treatment_stats.get("deterministic_fallbacks", 0))
        >= int(treatment_stats.get("deterministic_transforms", 0))
    )

    control_public = float(control.get("best_public_score", 0.0))
    treatment_public = float(treatment.get("best_public_score", 0.0))
    control_holdout = float(control.get("final_holdout_score", 0.0))
    treatment_holdout = float(treatment.get("final_holdout_score", 0.0))
    control_initial = float(control.get("initial_holdout_score", 0.0))
    treatment_initial = float(treatment.get("initial_holdout_score", 0.0))
    performance_noninferior = bool(
        treatment_public >= control_public - 1e-12
        and treatment_holdout >= control_holdout - 1e-12
        and treatment_holdout - treatment_initial
        >= control_holdout - control_initial - 1e-12
    )

    control_duplicates, control_unique_asts, control_duplicate_rate = _audit_structure(
        control_audits
    )
    treatment_duplicates, treatment_unique_asts, treatment_duplicate_rate = (
        _audit_structure(treatment_audits)
    )
    control_unique_behaviors = int(
        control_report["admission"].get("unique_behavior_signatures", 0)
    )
    treatment_unique_behaviors = int(
        treatment_report["admission"].get("unique_behavior_signatures", 0)
    )
    structural_novelty_improved = bool(
        treatment_duplicate_rate < control_duplicate_rate - 1e-12
        and treatment_unique_asts >= control_unique_asts
        and treatment_unique_behaviors >= control_unique_behaviors
    )
    no_accepted_regression = bool(
        int(control_report["admission"].get("accepted_parent_regressions", 0)) == 0
        and int(treatment_report["admission"].get("accepted_parent_regressions", 0))
        == 0
    )
    capability_pass = bool(
        contract_matched
        and controller_binding_valid
        and fixed_call_budget
        and operator_execution_valid
        and performance_noninferior
        and structural_novelty_improved
        and no_accepted_regression
    )
    formal_seed_requirement_met = completed_seed_count >= 3
    promotion_decision = (
        "approved"
        if capability_pass and formal_seed_requirement_met
        else "capability_pass_formal_pending"
        if capability_pass
        else "rejected"
    )
    return {
        "definition": "structured mutation proposer v4 matched A/B",
        "contract_mismatches": mismatches,
        "contract_matched": contract_matched,
        "controller_binding_valid": controller_binding_valid,
        "fixed_call_budget": fixed_call_budget,
        "operator_execution_valid": operator_execution_valid,
        "control_run_id": control.get("run_id"),
        "treatment_run_id": treatment.get("run_id"),
        "experiment_seed": control.get("experiment_seed"),
        "control_candidate_attempts": control_attempts,
        "treatment_candidate_attempts": treatment_attempts,
        "control_upstream_calls": control_calls,
        "treatment_upstream_calls": treatment_calls,
        "control_public_score": control_public,
        "treatment_public_score": treatment_public,
        "public_delta": treatment_public - control_public,
        "control_holdout_score": control_holdout,
        "treatment_holdout_score": treatment_holdout,
        "holdout_delta": treatment_holdout - control_holdout,
        "performance_noninferior": performance_noninferior,
        "control_structural_duplicates": control_duplicates,
        "treatment_structural_duplicates": treatment_duplicates,
        "control_structural_duplicate_rate": control_duplicate_rate,
        "treatment_structural_duplicate_rate": treatment_duplicate_rate,
        "control_unique_asts": control_unique_asts,
        "treatment_unique_asts": treatment_unique_asts,
        "control_unique_behaviors": control_unique_behaviors,
        "treatment_unique_behaviors": treatment_unique_behaviors,
        "treatment_structured_plans": int(treatment_stats.get("structured_plans", 0)),
        "treatment_deterministic_transforms": int(
            treatment_stats.get("deterministic_transforms", 0)
        ),
        "treatment_model_plans_valid": int(treatment_stats.get("model_plans_valid", 0)),
        "structural_novelty_improved": structural_novelty_improved,
        "no_accepted_regression": no_accepted_regression,
        "capability_trial_pass": capability_pass,
        "completed_seed_count": completed_seed_count,
        "formal_seed_requirement_met": formal_seed_requirement_met,
        "promotion_decision": promotion_decision,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_chinese(result: dict[str, Any]) -> str:
    ab = result["structured_ab"]
    return f"""# 结构化 AST Mutation v4 capability A/B

生成时间：{result["generated_at"]}

本报告是单 seed capability trial。只有完成至少 3 个独立 seed 的同协议评测后，才具备正式晋升资格。

| 指标 | planner-control | structured-mutation |
|---|---:|---:|
| 模型调用 | {ab["control_upstream_calls"]} | {ab["treatment_upstream_calls"]} |
| 公开最佳 | {_fmt(ab["control_public_score"])} | {_fmt(ab["treatment_public_score"])} |
| 隐藏最终 | {_fmt(ab["control_holdout_score"])} | {_fmt(ab["treatment_holdout_score"])} |
| audit 结构重复 | {ab["control_structural_duplicates"]} | {ab["treatment_structural_duplicates"]} |
| 唯一 AST | {ab["control_unique_asts"]} | {ab["treatment_unique_asts"]} |
| 唯一行为 | {ab["control_unique_behaviors"]} | {ab["treatment_unique_behaviors"]} |

- 契约匹配：{_fmt(ab["contract_matched"])}
- 固定调用预算：{_fmt(ab["fixed_call_budget"])}
- operator 执行可验证：{_fmt(ab["operator_execution_valid"])}
- 性能非劣：{_fmt(ab["performance_noninferior"])}
- 结构新颖性改善：{_fmt(ab["structural_novelty_improved"])}
- capability trial 通过：{_fmt(ab["capability_trial_pass"])}
- 已完成 seed：{ab["completed_seed_count"]}/3
- 决定：`{ab["promotion_decision"]}`
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-stats", type=Path, required=True)
    parser.add_argument("--treatment-stats", type=Path, required=True)
    parser.add_argument("--control-audit", type=Path, required=True)
    parser.add_argument("--treatment-audit", type=Path, required=True)
    parser.add_argument("--completed-seed-count", type=int, default=1)
    parser.add_argument("--experiment-seed", type=int)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    control_report = json.loads(args.control.read_text(encoding="utf-8"))
    treatment_report = json.loads(args.treatment.read_text(encoding="utf-8"))
    if args.experiment_seed is not None:
        control_report["run"]["experiment_seed"] = args.experiment_seed
        treatment_report["run"]["experiment_seed"] = args.experiment_seed
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "structured_ab": evaluate_structured_ab(
            control_report,
            treatment_report,
            json.loads(args.control_stats.read_text(encoding="utf-8")),
            json.loads(args.treatment_stats.read_text(encoding="utf-8")),
            _read_jsonl(args.control_audit),
            _read_jsonl(args.treatment_audit),
            completed_seed_count=args.completed_seed_count,
        ),
    }
    _atomic_text(
        args.json_output,
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(args.markdown_output, render_chinese(result))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
