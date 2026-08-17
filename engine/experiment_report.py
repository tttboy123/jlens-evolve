"""Build auditable model-comparison and PSI A/B reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from self_improvement_eval import evaluate_psi_ab


def _winner(control: float, treatment: float) -> str:
    if treatment > control + 1e-12:
        return "treatment"
    if control > treatment + 1e-12:
        return "control"
    return "tie"


def compare_model_runs(
    control_report: dict[str, Any], treatment_report: dict[str, Any]
) -> dict[str, Any]:
    """Compare two runs only when every non-model contract field matches."""
    control = control_report["run"]
    treatment = treatment_report["run"]
    contract_keys = (
        "task_id",
        "search_protocol_hash",
        "evaluator_hash",
        "initial_hash",
        "iterations_requested",
        "experience_mode",
        "operator_policy_id",
    )
    mismatches = [
        key for key in contract_keys if control.get(key) != treatment.get(key)
    ]
    models_differ = control.get("model_id") != treatment.get("model_id")
    control_public = float(control.get("best_public_score", 0.0))
    treatment_public = float(treatment.get("best_public_score", 0.0))
    control_holdout = float(control.get("final_holdout_score", 0.0))
    treatment_holdout = float(treatment.get("final_holdout_score", 0.0))
    control_admission = control_report.get("admission", {})
    treatment_admission = treatment_report.get("admission", {})
    return {
        "definition": "matched proposer model comparison",
        "contract_keys": list(contract_keys),
        "contract_mismatches": mismatches,
        "contract_matched": not mismatches,
        "models_differ": models_differ,
        "comparison_valid": not mismatches and models_differ,
        "control_run_id": control.get("run_id"),
        "treatment_run_id": treatment.get("run_id"),
        "task_id": control.get("task_id"),
        "search_protocol_hash": control.get("search_protocol_hash"),
        "iterations_requested": control.get("iterations_requested"),
        "control_model": control.get("model_id"),
        "treatment_model": treatment.get("model_id"),
        "control_public_score": control_public,
        "treatment_public_score": treatment_public,
        "public_score_delta": treatment_public - control_public,
        "public_winner": _winner(control_public, treatment_public),
        "control_holdout_score": control_holdout,
        "treatment_holdout_score": treatment_holdout,
        "holdout_score_delta": treatment_holdout - control_holdout,
        "holdout_winner": _winner(control_holdout, treatment_holdout),
        "control_best_passed_cases": control_admission.get("best_passed_cases"),
        "treatment_best_passed_cases": treatment_admission.get("best_passed_cases"),
        "control_accept_rate": control_admission.get("accept_rate"),
        "treatment_accept_rate": treatment_admission.get("accept_rate"),
        "control_unique_behaviors": control_admission.get("unique_behavior_signatures"),
        "treatment_unique_behaviors": treatment_admission.get(
            "unique_behavior_signatures"
        ),
        "control_accepted_regressions": control_admission.get(
            "accepted_parent_regressions"
        ),
        "treatment_accepted_regressions": treatment_admission.get(
            "accepted_parent_regressions"
        ),
        "control_rejection_reasons": control_admission.get("rejection_reasons", {}),
        "treatment_rejection_reasons": treatment_admission.get("rejection_reasons", {}),
        "control_duration_seconds": control.get("duration_seconds"),
        "treatment_duration_seconds": treatment.get("duration_seconds"),
    }


def summarize_psi(
    resume_report: dict[str, Any], psi_ab: dict[str, Any]
) -> dict[str, Any]:
    """Combine same-search resume evidence with the matched cross-task A/B."""
    resume = resume_report.get("psi", {})
    same_search_pass = bool(resume.get("same_search_resume_pass", False))
    cross_task_pass = bool(psi_ab.get("psi_ab_pass", False))
    return {
        "definition": "persistent self-improvement",
        "same_search_resume_pass": same_search_pass,
        "resume_trials": int(resume.get("resume_trials", 0)),
        "cross_task_ab_pass": cross_task_pass,
        "strict_transfer_benefit": bool(psi_ab.get("strict_transfer_benefit", False)),
        "psi_pass": same_search_pass and cross_task_pass,
    }


def verify_experience_snapshot(
    source_path: Path, control_path: Path, transfer_path: Path
) -> dict[str, Any]:
    """Verify that both arms began with the same append-only lesson snapshot."""
    source = source_path.read_bytes()
    control = control_path.read_bytes()
    transfer = transfer_path.read_bytes()
    control_matches = control.startswith(source)
    transfer_matches = transfer.startswith(source)
    return {
        "source_path": str(source_path.resolve()),
        "control_path": str(control_path.resolve()),
        "transfer_path": str(transfer_path.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_bytes": len(source),
        "source_lines": len(source.splitlines()),
        "control_prefix_matches": control_matches,
        "transfer_prefix_matches": transfer_matches,
        "snapshot_matched": control_matches and transfer_matches,
    }


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_chinese_report(result: dict[str, Any]) -> str:
    model = result["model_comparison"]
    psi = result["psi_ab"]
    overall = result["psi_overall"]
    snapshot = result["psi_experience_snapshot"]
    strict = "取得严格正收益" if psi["strict_transfer_benefit"] else "未取得严格正收益"
    transfer_interpretation = (
        "本次经验迁移不劣于 control。"
        if psi["noninferior_to_control"]
        else "本次经验迁移劣于 control，属于负迁移。"
    )
    return f"""# Coder Proposer 对照与跨任务 PSI A/B 报告

生成时间：{result["generated_at"]}

## 一、模型对照

本实验只替换 proposer 模型。任务：`{model["task_id"]}`；每组 {model["iterations_requested"]} 轮；搜索协议哈希：`{model["search_protocol_hash"]}`。协议匹配：{_format_number(model["contract_matched"])}；对照有效：{_format_number(model["comparison_valid"])}。

- control：`{model["control_model"]}`
- treatment：`{model["treatment_model"]}`

| 指标 | 4B control | 7B coder treatment | treatment 差值 |
|---|---:|---:|---:|
| 公开综合分数 | {_format_number(model["control_public_score"])} | {_format_number(model["treatment_public_score"])} | {_format_number(model["public_score_delta"])} |
| 隐藏集分数 | {_format_number(model["control_holdout_score"])} | {_format_number(model["treatment_holdout_score"])} | {_format_number(model["holdout_score_delta"])} |
| 最佳公开通过数 | {_format_number(model["control_best_passed_cases"], 0)} | {_format_number(model["treatment_best_passed_cases"], 0)} | — |
| 接受率 | {_format_number(model["control_accept_rate"])} | {_format_number(model["treatment_accept_rate"])} | — |
| 唯一行为签名 | {_format_number(model["control_unique_behaviors"], 0)} | {_format_number(model["treatment_unique_behaviors"], 0)} | — |
| 接受回归 | {_format_number(model["control_accepted_regressions"], 0)} | {_format_number(model["treatment_accepted_regressions"], 0)} | — |
| 运行秒数 | {_format_number(model["control_duration_seconds"], 1)} | {_format_number(model["treatment_duration_seconds"], 1)} | — |

公开指标胜者：`{model["public_winner"]}`；隐藏指标胜者：`{model["holdout_winner"]}`。

control 拒绝原因：`{json.dumps(model["control_rejection_reasons"], ensure_ascii=False, sort_keys=True)}`；treatment 拒绝原因：`{json.dumps(model["treatment_rejection_reasons"], ensure_ascii=False, sort_keys=True)}`。这只是当前固定协议下的模型×搜索交互结果，不是模型的通用能力排名。

## 二、跨任务 PSI A/B

实验 ID：`{psi["experiment_id"]}`。任务：`{psi["task_id"]}`；模型：`{psi["model_id"]}`；搜索协议哈希：`{psi["search_protocol_hash"]}`。control 禁用经验，treatment 只允许检索不同 task ID、同任务族且经过隐藏集验证的经验。

| 指标 | control | transfer | transfer 差值 |
|---|---:|---:|---:|
| 最佳公开分数 | {_format_number(psi["control_public_score"])} | {_format_number(psi["transfer_public_score"])} | {_format_number(psi["public_score_delta_vs_control"])} |
| 初始隐藏分数 | {_format_number(psi["control_initial_holdout"])} | {_format_number(psi["transfer_initial_holdout"])} | — |
| 最终隐藏分数 | {_format_number(psi["control_final_holdout"])} | {_format_number(psi["transfer_final_holdout"])} | {_format_number(psi["holdout_delta_vs_control"])} |
| 隐藏增益 | {_format_number(psi["control_holdout_gain"])} | {_format_number(psi["transfer_holdout_gain"])} | {_format_number(psi["gain_delta_vs_control"])} |

- 契约完全匹配：{_format_number(psi["contract_matched"])}
- 跨任务来源可追溯：{_format_number(psi["cross_task_provenance"])}
- transfer 不劣于 control：{_format_number(psi["noninferior_to_control"])}
- PSI A/B 通过：{_format_number(psi["psi_ab_pass"])}
- 严格收益判断：{strict}

来源经验：`{json.dumps(psi["foreign_lesson_sources"], ensure_ascii=False, sort_keys=True)}`。{transfer_interpretation}

## 三、PSI 总判定

- 同一搜索跨进程恢复：{_format_number(overall["same_search_resume_pass"])}（{overall["resume_trials"]} 次恢复证据）
- 跨任务匹配 A/B：{_format_number(overall["cross_task_ab_pass"])}
- PSI 总判定：{_format_number(overall["psi_pass"])}
- 两组初始经验快照一致：{_format_number(snapshot["snapshot_matched"])}（SHA-256 `{snapshot["source_sha256"]}`，{snapshot["source_lines"]} 行）

## 四、解释边界

`psi_ab_pass` 表示经过验证的跨任务经验在匹配实验中被真实检索，且目标隐藏集表现不劣于禁用经验的 control。严格正收益单独报告；若两组打平，不会描述为经验带来了提升。

本报告的模型对照只纳入 `{model["control_run_id"]}` 与 `{model["treatment_run_id"]}`；PSI A/B 只纳入 `{psi["control_run_id"]}` 与 `{psi["transfer_run_id"]}`。预检失败或非 v2 状态目录的试跑不参与任何结论。
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-control", required=True)
    parser.add_argument("--model-treatment", required=True)
    parser.add_argument("--psi-control", required=True)
    parser.add_argument("--psi-transfer", required=True)
    parser.add_argument("--resume-evidence", required=True)
    parser.add_argument("--psi-source-lessons", required=True)
    parser.add_argument("--psi-control-lessons", required=True)
    parser.add_argument("--psi-transfer-lessons", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    model_control = json.loads(Path(args.model_control).read_text(encoding="utf-8"))
    model_treatment = json.loads(Path(args.model_treatment).read_text(encoding="utf-8"))
    psi_control = json.loads(Path(args.psi_control).read_text(encoding="utf-8"))
    psi_transfer = json.loads(Path(args.psi_transfer).read_text(encoding="utf-8"))
    resume_evidence = json.loads(Path(args.resume_evidence).read_text(encoding="utf-8"))
    psi_ab = evaluate_psi_ab(
        [psi_control["run"], psi_transfer["run"]],
        experiment_id=args.experiment_id,
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_comparison": compare_model_runs(model_control, model_treatment),
        "psi_ab": psi_ab,
        "psi_overall": summarize_psi(resume_evidence, psi_ab),
        "psi_experience_snapshot": verify_experience_snapshot(
            Path(args.psi_source_lessons),
            Path(args.psi_control_lessons),
            Path(args.psi_transfer_lessons),
        ),
    }
    _atomic_text(
        Path(args.json_output),
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(Path(args.markdown_output), render_chinese_report(result))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
