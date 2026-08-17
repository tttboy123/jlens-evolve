"""Aggregate independent structured-mutation A/B seeds into a formal decision."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sum(rows: list[dict[str, Any]], key: str) -> int:
    return sum(int(row.get(key, 0)) for row in rows)


def evaluate_multiseed(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared three-seed promotion gate."""
    rows = [report["structured_ab"] for report in seed_reports]
    seeds = [row.get("experiment_seed") for row in rows]
    distinct_seeds = {seed for seed in seeds if seed is not None}
    seed_count = len(rows)
    required_nonhigher = math.ceil(2 * seed_count / 3) if seed_count else 0

    control_attempts = _sum(rows, "control_candidate_attempts")
    treatment_attempts = _sum(rows, "treatment_candidate_attempts")
    control_duplicates = _sum(rows, "control_structural_duplicates")
    treatment_duplicates = _sum(rows, "treatment_structural_duplicates")
    control_duplicate_rate = control_duplicates / max(1, control_attempts)
    treatment_duplicate_rate = treatment_duplicates / max(1, treatment_attempts)
    nonhigher_count = sum(
        float(row.get("treatment_structural_duplicate_rate", 1.0))
        <= float(row.get("control_structural_duplicate_rate", 0.0)) + 1e-12
        for row in rows
    )

    seed_requirement_met = bool(seed_count >= 3 and len(distinct_seeds) == seed_count)
    all_seed_capability_pass = bool(
        rows and all(row.get("capability_trial_pass") is True for row in rows)
    )
    all_contracts_valid = bool(
        rows
        and all(
            row.get("contract_matched") is True
            and row.get("controller_binding_valid") is True
            and row.get("fixed_call_budget") is True
            for row in rows
        )
    )
    all_operator_execution_valid = bool(
        rows and all(row.get("operator_execution_valid") is True for row in rows)
    )
    all_performance_noninferior = bool(
        rows and all(row.get("performance_noninferior") is True for row in rows)
    )
    no_accepted_regression = bool(
        rows and all(row.get("no_accepted_regression") is True for row in rows)
    )
    pooled_duplicate_rate_improved = bool(
        control_attempts == treatment_attempts
        and control_attempts > 0
        and treatment_duplicate_rate < control_duplicate_rate - 1e-12
    )
    duplicate_seed_gate = bool(seed_count > 0 and nonhigher_count >= required_nonhigher)
    control_unique_asts = _sum(rows, "control_unique_asts")
    treatment_unique_asts = _sum(rows, "treatment_unique_asts")
    control_unique_behaviors = _sum(rows, "control_unique_behaviors")
    treatment_unique_behaviors = _sum(rows, "treatment_unique_behaviors")
    diversity_noninferior = bool(
        treatment_unique_asts >= control_unique_asts
        and treatment_unique_behaviors >= control_unique_behaviors
    )

    formal_pass = bool(
        seed_requirement_met
        and all_contracts_valid
        and all_operator_execution_valid
        and all_performance_noninferior
        and no_accepted_regression
        and pooled_duplicate_rate_improved
        and duplicate_seed_gate
        and diversity_noninferior
    )
    return {
        "definition": "structured mutation proposer v4 formal multi-seed A/B",
        "seeds": seeds,
        "seed_count": seed_count,
        "distinct_seed_count": len(distinct_seeds),
        "seed_requirement_met": seed_requirement_met,
        "all_seed_capability_pass": all_seed_capability_pass,
        "all_contracts_valid": all_contracts_valid,
        "all_operator_execution_valid": all_operator_execution_valid,
        "all_performance_noninferior": all_performance_noninferior,
        "no_accepted_regression": no_accepted_regression,
        "control_candidate_attempts": control_attempts,
        "treatment_candidate_attempts": treatment_attempts,
        "control_structural_duplicates": control_duplicates,
        "treatment_structural_duplicates": treatment_duplicates,
        "control_structural_duplicate_rate": control_duplicate_rate,
        "treatment_structural_duplicate_rate": treatment_duplicate_rate,
        "pooled_duplicate_rate_improved": pooled_duplicate_rate_improved,
        "nonhigher_duplicate_seed_count": nonhigher_count,
        "required_nonhigher_duplicate_seed_count": required_nonhigher,
        "duplicate_seed_gate": duplicate_seed_gate,
        "control_unique_asts_sum": control_unique_asts,
        "treatment_unique_asts_sum": treatment_unique_asts,
        "control_unique_behaviors_sum": control_unique_behaviors,
        "treatment_unique_behaviors_sum": treatment_unique_behaviors,
        "diversity_noninferior": diversity_noninferior,
        "formal_promotion_pass": formal_pass,
        "promotion_decision": "approved" if formal_pass else "rejected",
        "rsi_status": "candidate_requires_next_window",
        "psi_status": "candidate_requires_cross_task_transfer_ab",
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_chinese(result: dict[str, Any]) -> str:
    formal = result["formal_multiseed"]
    return f"""# 结构化 AST Mutation v4 正式三 Seed 报告

生成时间：{result["generated_at"]}

本报告只判断 proposer 策略是否可晋升，不把策略晋升等同于 RSI 或 PSI。

| 指标 | control | treatment |
|---|---:|---:|
| 候选数 | {formal["control_candidate_attempts"]} | {formal["treatment_candidate_attempts"]} |
| 结构重复数 | {formal["control_structural_duplicates"]} | {formal["treatment_structural_duplicates"]} |
| pooled 结构重复率 | {_fmt(formal["control_structural_duplicate_rate"])} | {_fmt(formal["treatment_structural_duplicate_rate"])} |
| 唯一 AST（逐 seed 求和） | {formal["control_unique_asts_sum"]} | {formal["treatment_unique_asts_sum"]} |
| 唯一行为（逐 seed 求和） | {formal["control_unique_behaviors_sum"]} | {formal["treatment_unique_behaviors_sum"]} |

- 独立 seed：{formal["distinct_seed_count"]}/{formal["seed_count"]}
- 单 seed strict capability 全部通过（诊断项）：{_fmt(formal["all_seed_capability_pass"])}
- pooled 重复率严格下降：{_fmt(formal["pooled_duplicate_rate_improved"])}
- 单 seed 重复率非升：{formal["nonhigher_duplicate_seed_count"]}/{formal["seed_count"]}（要求 {formal["required_nonhigher_duplicate_seed_count"]}）
- 公开与隐藏性能逐 seed 非劣：{_fmt(formal["all_performance_noninferior"])}
- operator 后置条件逐 seed 可审计：{_fmt(formal["all_operator_execution_valid"])}
- 接受回归为零：{_fmt(formal["no_accepted_regression"])}
- 正式决定：`{formal["promotion_decision"]}`

RSI 仍为 `candidate_requires_next_window`：必须在下一固定预算窗口启用候选权重并证明 improvement yield 与任务性能非劣。

PSI 仍为 `candidate_requires_cross_task_transfer_ab`：必须在相邻任务做 transfer/off 匹配 A/B 并守住隐藏集。
"""


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
    parser.add_argument("--seed-report", type=Path, action="append", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.seed_report
    ]
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "formal_multiseed": evaluate_multiseed(reports),
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
