"""Evaluate and report a matched JLens-guided Agent policy A/B."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def evaluate_agent_ab(
    control_report: dict[str, Any], treatment_report: dict[str, Any]
) -> dict[str, Any]:
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
    control_schedule = list(
        control.get("operator_policy_schedule") or [control.get("operator_policy_id")]
    )
    treatment_schedule = list(
        treatment.get("operator_policy_schedule")
        or [treatment.get("operator_policy_id")]
    )
    strategy_binding_valid = bool(
        not control.get("agent_strategy_id")
        and treatment.get("agent_strategy_id")
        and treatment.get("agent_strategy_sha256")
        and control_schedule == ["focused-v1"]
        and treatment_schedule[-1:] == ["jlens-guided-v1"]
        and all(policy == "focused-v1" for policy in treatment_schedule[:-1])
    )
    control_admission = control_report["admission"]
    treatment_admission = treatment_report["admission"]
    control_attempts = int(control_admission.get("candidate_attempts", 0))
    treatment_attempts = int(treatment_admission.get("candidate_attempts", 0))
    control_exact = int(
        control_admission.get("rejection_reasons", {}).get("exact_duplicate", 0)
    )
    treatment_exact = int(
        treatment_admission.get("rejection_reasons", {}).get("exact_duplicate", 0)
    )
    control_ast_duplicates = int(
        control_admission.get("rejection_reasons", {}).get("ast_duplicate", 0)
    )
    treatment_ast_duplicates = int(
        treatment_admission.get("rejection_reasons", {}).get("ast_duplicate", 0)
    )
    control_exact_rate = control_exact / max(1, control_attempts)
    treatment_exact_rate = treatment_exact / max(1, treatment_attempts)
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
    control_unique = int(control_admission.get("unique_source_hashes", 0))
    treatment_unique = int(treatment_admission.get("unique_source_hashes", 0))
    diversity_improved = bool(
        treatment_exact_rate < control_exact_rate - 1e-12
        and treatment_unique >= control_unique
    )
    no_accepted_regression = bool(
        int(control_admission.get("accepted_parent_regressions", 0)) == 0
        and int(treatment_admission.get("accepted_parent_regressions", 0)) == 0
    )
    contract_matched = not mismatches and control_attempts == treatment_attempts
    passed = bool(
        contract_matched
        and strategy_binding_valid
        and performance_noninferior
        and diversity_improved
        and no_accepted_regression
    )
    return {
        "definition": "matched JLens-guided Agent policy A/B",
        "contract_keys": list(contract_keys),
        "contract_mismatches": mismatches,
        "contract_matched": contract_matched,
        "strategy_binding_valid": strategy_binding_valid,
        "control_run_id": control.get("run_id"),
        "treatment_run_id": treatment.get("run_id"),
        "control_policy": control.get("operator_policy_id"),
        "treatment_policy": treatment.get("operator_policy_id"),
        "control_policy_schedule": control_schedule,
        "treatment_policy_schedule": treatment_schedule,
        "strategy_id": treatment.get("agent_strategy_id"),
        "strategy_sha256": treatment.get("agent_strategy_sha256"),
        "control_public_score": control_public,
        "treatment_public_score": treatment_public,
        "public_delta": treatment_public - control_public,
        "control_holdout_score": control_holdout,
        "treatment_holdout_score": treatment_holdout,
        "holdout_delta": treatment_holdout - control_holdout,
        "control_holdout_gain": control_gain,
        "treatment_holdout_gain": treatment_gain,
        "holdout_gain_delta": treatment_gain - control_gain,
        "performance_noninferior": performance_noninferior,
        "control_candidate_attempts": control_attempts,
        "treatment_candidate_attempts": treatment_attempts,
        "control_exact_duplicates": control_exact,
        "treatment_exact_duplicates": treatment_exact,
        "control_ast_duplicate_rejections": control_ast_duplicates,
        "treatment_ast_duplicate_rejections": treatment_ast_duplicates,
        "control_exact_duplicate_rate": control_exact_rate,
        "treatment_exact_duplicate_rate": treatment_exact_rate,
        "control_unique_sources": control_unique,
        "treatment_unique_sources": treatment_unique,
        "control_unique_asts": int(control_admission.get("unique_ast_hashes", 0)),
        "treatment_unique_asts": int(treatment_admission.get("unique_ast_hashes", 0)),
        "control_unique_behaviors": int(
            control_admission.get("unique_behavior_signatures", 0)
        ),
        "treatment_unique_behaviors": int(
            treatment_admission.get("unique_behavior_signatures", 0)
        ),
        "diversity_improved": diversity_improved,
        "no_accepted_regression": no_accepted_regression,
        "agent_optimization_pass": passed,
        "promotion_decision": "approved" if passed else "rejected",
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (float, int)):
        return f"{value:.{digits}f}"
    return str(value)


def render_chinese(result: dict[str, Any], strategy: dict[str, Any]) -> str:
    ab = result["agent_ab"]
    evidence = strategy["evidence"]
    return f"""# JLens 驱动的 Agent 优化 A/B 报告

生成时间：{result["generated_at"]}

## 一、JLens 发现的规律

- 真实搜索边：{evidence["trace_edges"]}；唯一源码迁移：{evidence["unique_transitions"]}；重复迁移比例：{evidence["repeated_transition_fraction"]:.1%}。
- 唯一迁移结果：{json.dumps(evidence["unique_outcomes"], ensure_ascii=False, sort_keys=True)}。
- JLens score 关联 η²：{evidence["jlens_score_eta_squared"]:.4f}；logit-lens：{evidence["logit_lens_score_eta_squared"]:.4f}。
- JLens 是否提供增量证据：{_fmt(evidence["jlens_incremental_supported"])}。
- 因果边界：`{strategy["causal_boundary"]}`；准入权限：{_fmt(strategy["admission_gate_allowed"])}。

策略 `{strategy["strategy_id"]}` 只修改 proposer 提示、随机解码和多样父代数量，保持 evaluator、hidden cases 与 admission 不变。本次 treatment 的策略链为 `{json.dumps(ab["treatment_policy_schedule"], ensure_ascii=False)}`，因此冷启动与 JLens 介入阶段可分别审计。

## 二、匹配 A/B

| 指标 | control | JLens-guided | treatment 差值 |
|---|---:|---:|---:|
| 公开最佳分数 | {_fmt(ab["control_public_score"])} | {_fmt(ab["treatment_public_score"])} | {_fmt(ab["public_delta"])} |
| 隐藏最终分数 | {_fmt(ab["control_holdout_score"])} | {_fmt(ab["treatment_holdout_score"])} | {_fmt(ab["holdout_delta"])} |
| 隐藏增益 | {_fmt(ab["control_holdout_gain"])} | {_fmt(ab["treatment_holdout_gain"])} | {_fmt(ab["holdout_gain_delta"])} |
| 精确重复率 | {_fmt(ab["control_exact_duplicate_rate"])} | {_fmt(ab["treatment_exact_duplicate_rate"])} | — |
| 唯一源码 | {ab["control_unique_sources"]} | {ab["treatment_unique_sources"]} | — |
| 唯一 AST | {ab["control_unique_asts"]} | {ab["treatment_unique_asts"]} | — |
| 唯一行为 | {ab["control_unique_behaviors"]} | {ab["treatment_unique_behaviors"]} | — |
| AST 重复拒绝 | {ab["control_ast_duplicate_rejections"]} | {ab["treatment_ast_duplicate_rejections"]} | — |

- 契约匹配：{_fmt(ab["contract_matched"])}
- 策略绑定有效：{_fmt(ab["strategy_binding_valid"])}
- 性能不劣：{_fmt(ab["performance_noninferior"])}
- 多样性改善：{_fmt(ab["diversity_improved"])}
- 接受回归为零：{_fmt(ab["no_accepted_regression"])}
- **Agent 优化通过：{_fmt(ab["agent_optimization_pass"])}**
- 晋升决定：`{ab["promotion_decision"]}`

## 三、解释边界

该决定只说明这个由 JLens 观测启发的 Agent 策略是否在本次匹配 A/B 中有效。JLens 没有直接控制输出、拒绝候选或修改模型权重；若 treatment 失败，策略保持未晋升，规律仍仅作诊断证据。

## 四、当前诊断

- 本轮公开分与 control 相同，但隐藏集下降 {_fmt(ab["holdout_delta"])}，不满足性能不劣条件。
- treatment 的唯一源码为 {ab["treatment_unique_sources"]}，control 为 {ab["control_unique_sources"]}；但精确重复率没有严格下降，AST 重复拒绝为 {ab["treatment_ast_duplicate_rejections"]} 次，所以新增文本形态没有转化为稳定的结构或行为增益。
- 这表明当前瓶颈主要在 4B proposer 的结构模式坍缩和公开评估的同分辨识能力。继续加长 JLens 提示不是充分修复；下一版应在 proposer 层做有预算上限的 duplicate-aware re-proposal，并用多 seed、JLens 对 logit-lens 的增量 A/B 再验证。

更完整的实现、两轮 A/B 和根因说明见项目内 `AGENT_OPTIMIZATION.zh-CN.md`。
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
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    control = json.loads(args.control.read_text(encoding="utf-8"))
    treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
    strategy = json.loads(args.strategy.read_text(encoding="utf-8"))
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "agent_ab": evaluate_agent_ab(control, treatment),
        "strategy": strategy,
    }
    _atomic_text(
        args.json_output,
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(args.markdown_output, render_chinese(result, strategy))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
