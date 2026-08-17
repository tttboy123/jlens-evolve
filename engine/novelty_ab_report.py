"""Evaluate a fixed-budget shadow-control vs duplicate-aware Agent A/B."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def evaluate_novelty_ab(
    control_report: dict[str, Any],
    treatment_report: dict[str, Any],
    control_stats: dict[str, Any],
    treatment_stats: dict[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared performance, novelty, and compute gates."""
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
    )

    def contract_value(run: dict[str, Any], key: str) -> Any:
        if key == "iterations_requested_total":
            return run.get(key, run.get("iterations_requested"))
        return run.get(key)

    mismatches = [
        key
        for key in contract_keys
        if contract_value(control, key) != contract_value(treatment, key)
    ]
    control_admission = control_report["admission"]
    treatment_admission = treatment_report["admission"]
    control_attempts = int(control_admission.get("candidate_attempts", 0))
    treatment_attempts = int(treatment_admission.get("candidate_attempts", 0))
    contract_matched = not mismatches and control_attempts == treatment_attempts

    controller_binding_valid = bool(
        control.get("proposal_controller_mode") == "shadow-control"
        and treatment.get("proposal_controller_mode") == "duplicate-aware"
        and control.get("proposal_controller_id")
        and treatment.get("proposal_controller_id")
        and control.get("proposal_controller_sha256")
        and treatment.get("proposal_controller_sha256")
        and control.get("proposal_controller_endpoint_verified") is True
        and treatment.get("proposal_controller_endpoint_verified") is True
        and treatment.get("proposal_controller_implementation_sha256")
        and int(control.get("proposal_controller_calls_per_request", 0)) == 2
        and int(treatment.get("proposal_controller_calls_per_request", 0)) == 2
    )
    control_requests = int(control_stats.get("requests", 0))
    treatment_requests = int(treatment_stats.get("requests", 0))
    control_calls = int(control_stats.get("upstream_calls", 0))
    treatment_calls = int(treatment_stats.get("upstream_calls", 0))
    fixed_call_budget = bool(
        control_stats.get("mode") == "shadow-control"
        and treatment_stats.get("mode") == "duplicate-aware"
        and control_requests == control_attempts
        and treatment_requests == treatment_attempts
        and control_calls == 2 * control_requests
        and treatment_calls == 2 * treatment_requests
        and control_calls == treatment_calls
    )
    # The shadow controller never selects its escape completion, so its
    # detector version is behaviorally inert. The treatment detector must be
    # the corrected global-best implementation.
    stagnation_eval_valid = bool(
        control_stats.get("mode") == "shadow-control"
        and treatment_stats.get("stagnation_detector_version") == "global-best-v2"
    )

    control_public = float(control.get("best_public_score", 0.0))
    treatment_public = float(treatment.get("best_public_score", 0.0))
    control_initial = float(control.get("initial_holdout_score", 0.0))
    treatment_initial = float(treatment.get("initial_holdout_score", 0.0))
    control_holdout = float(control.get("final_holdout_score", 0.0))
    treatment_holdout = float(treatment.get("final_holdout_score", 0.0))
    control_gain = control_holdout - control_initial
    treatment_gain = treatment_holdout - treatment_initial
    performance_noninferior = bool(
        treatment_public >= control_public - 1e-12
        and treatment_holdout >= control_holdout - 1e-12
        and treatment_gain >= control_gain - 1e-12
    )

    def structural_duplicates(admission: dict[str, Any]) -> int:
        reasons = admission.get("rejection_reasons", {})
        return int(reasons.get("exact_duplicate", 0)) + int(
            reasons.get("ast_duplicate", 0)
        )

    control_duplicates = structural_duplicates(control_admission)
    treatment_duplicates = structural_duplicates(treatment_admission)
    control_duplicate_rate = control_duplicates / max(1, control_attempts)
    treatment_duplicate_rate = treatment_duplicates / max(1, treatment_attempts)
    control_unique_ast = int(control_admission.get("unique_ast_hashes", 0))
    treatment_unique_ast = int(treatment_admission.get("unique_ast_hashes", 0))
    control_unique_behavior = int(
        control_admission.get("unique_behavior_signatures", 0)
    )
    treatment_unique_behavior = int(
        treatment_admission.get("unique_behavior_signatures", 0)
    )
    structural_novelty_improved = bool(
        treatment_duplicate_rate < control_duplicate_rate - 1e-12
        and treatment_unique_ast >= control_unique_ast
        and treatment_unique_behavior >= control_unique_behavior
    )
    no_accepted_regression = bool(
        int(control_admission.get("accepted_parent_regressions", 0)) == 0
        and int(treatment_admission.get("accepted_parent_regressions", 0)) == 0
    )
    passed = bool(
        contract_matched
        and controller_binding_valid
        and fixed_call_budget
        and stagnation_eval_valid
        and performance_noninferior
        and structural_novelty_improved
        and no_accepted_regression
    )
    return {
        "definition": "fixed-budget duplicate-aware proposal-controller A/B",
        "contract_keys": list(contract_keys),
        "contract_mismatches": mismatches,
        "contract_matched": contract_matched,
        "controller_binding_valid": controller_binding_valid,
        "fixed_call_budget": fixed_call_budget,
        "stagnation_eval_valid": stagnation_eval_valid,
        "control_run_id": control.get("run_id"),
        "treatment_run_id": treatment.get("run_id"),
        "control_controller_id": control.get("proposal_controller_id"),
        "treatment_controller_id": treatment.get("proposal_controller_id"),
        "control_requests": control_requests,
        "treatment_requests": treatment_requests,
        "control_upstream_calls": control_calls,
        "treatment_upstream_calls": treatment_calls,
        "control_public_score": control_public,
        "treatment_public_score": treatment_public,
        "public_delta": treatment_public - control_public,
        "control_holdout_score": control_holdout,
        "treatment_holdout_score": treatment_holdout,
        "holdout_delta": treatment_holdout - control_holdout,
        "control_holdout_gain": control_gain,
        "treatment_holdout_gain": treatment_gain,
        "performance_noninferior": performance_noninferior,
        "control_structural_duplicates": control_duplicates,
        "treatment_structural_duplicates": treatment_duplicates,
        "control_structural_duplicate_rate": control_duplicate_rate,
        "treatment_structural_duplicate_rate": treatment_duplicate_rate,
        "control_unique_asts": control_unique_ast,
        "treatment_unique_asts": treatment_unique_ast,
        "control_unique_behaviors": control_unique_behavior,
        "treatment_unique_behaviors": treatment_unique_behavior,
        "control_first_duplicates": int(control_stats.get("first_duplicates", 0)),
        "treatment_first_duplicates": int(treatment_stats.get("first_duplicates", 0)),
        "control_stagnation_detector_version": control_stats.get(
            "stagnation_detector_version"
        ),
        "treatment_stagnation_detector_version": treatment_stats.get(
            "stagnation_detector_version"
        ),
        "treatment_stagnation_triggers": int(
            treatment_stats.get("stagnation_triggers", 0)
        ),
        "treatment_retry_feedback_requests": int(
            treatment_stats.get("retry_feedback_requests", 0)
        ),
        "treatment_selected_second": int(treatment_stats.get("selected_second", 0)),
        "structural_novelty_improved": structural_novelty_improved,
        "no_accepted_regression": no_accepted_regression,
        "agent_optimization_pass": passed,
        "promotion_decision": "approved" if passed else "rejected",
    }


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_chinese(result: dict[str, Any]) -> str:
    ab = result["novelty_ab"]
    return f"""# 重复感知 Agent proposer 固定预算 A/B

生成时间：{result["generated_at"]}

## 实验设计

control 与 treatment 每个外层候选都调用本地模型两次。control 始终选择第一次，第二次作为 shadow；treatment 在第一次与 prompt/archive 的源码或 AST 重复，或公开搜索连续未超过历史全局最佳时，给第二次追加有限反馈并按预声明规则选择。JLens 只提供“重复搜索”这一 observer 假设，不参与正确性或准入。

## 运行绑定

- control run：`{ab["control_run_id"]}` / `{ab["control_controller_id"]}`
- treatment run：`{ab["treatment_run_id"]}` / `{ab["treatment_controller_id"]}`
- treatment 停滞检测器：`{ab["treatment_stagnation_detector_version"]}`
- treatment 停滞触发次数：{ab["treatment_stagnation_triggers"]}

## 结果

| 指标 | control | duplicate-aware | 差值/说明 |
|---|---:|---:|---:|
| 上游模型调用 | {ab["control_upstream_calls"]} | {ab["treatment_upstream_calls"]} | 固定预算：{_fmt(ab["fixed_call_budget"])} |
| 公开最佳分数 | {_fmt(ab["control_public_score"])} | {_fmt(ab["treatment_public_score"])} | {_fmt(ab["public_delta"])} |
| 隐藏最终分数 | {_fmt(ab["control_holdout_score"])} | {_fmt(ab["treatment_holdout_score"])} | {_fmt(ab["holdout_delta"])} |
| 结构重复率 | {_fmt(ab["control_structural_duplicate_rate"])} | {_fmt(ab["treatment_structural_duplicate_rate"])} | — |
| 唯一 AST | {ab["control_unique_asts"]} | {ab["treatment_unique_asts"]} | — |
| 唯一行为 | {ab["control_unique_behaviors"]} | {ab["treatment_unique_behaviors"]} | — |
| 首案重复 | {ab["control_first_duplicates"]} | {ab["treatment_first_duplicates"]} | — |
| 有反馈重提案 | 0 | {ab["treatment_retry_feedback_requests"]} | 最多一次/候选 |

- 契约匹配：{_fmt(ab["contract_matched"])}
- controller 绑定：{_fmt(ab["controller_binding_valid"])}
- 停滞状态机评测有效：{_fmt(ab["stagnation_eval_valid"])}
- 性能不劣：{_fmt(ab["performance_noninferior"])}
- 结构新颖性改善：{_fmt(ab["structural_novelty_improved"])}
- 接受回归为零：{_fmt(ab["no_accepted_regression"])}
- **Agent 优化通过：{_fmt(ab["agent_optimization_pass"])}**
- 晋升决定：`{ab["promotion_decision"]}`

若“停滞状态机评测有效”为否，说明该次运行使用了父代相对分数而非历史全局最佳来判断停滞；重复感知结果仍可分析，但不能把该运行作为状态机晋升证据。shadow-control 永远返回第一次 completion，因此其停滞检测器在行为上不生效；treatment 必须绑定 `global-best-v2` 和经过哈希校验的代理实现。
"""


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-stats", type=Path, required=True)
    parser.add_argument("--treatment-stats", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "novelty_ab": evaluate_novelty_ab(
            json.loads(args.control.read_text(encoding="utf-8")),
            json.loads(args.treatment.read_text(encoding="utf-8")),
            json.loads(args.control_stats.read_text(encoding="utf-8")),
            json.loads(args.treatment_stats.read_text(encoding="utf-8")),
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
