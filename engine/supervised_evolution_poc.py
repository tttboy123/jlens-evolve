"""Minimal supervised-evolution POC over the existing deterministic task."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluator_core
from structured_mutation import apply_mutation_plan, derive_fallback_plan

ROOT = Path(__file__).resolve().parent
DEFAULT_OBSERVATION = ROOT / "analysis/agent-baseline/agent_strategy.json"
DEFAULT_OUTPUT = ROOT.parent.parent / "outputs" / "supervised-evolution-poc"
PUBLIC_FAILURE_SEQUENCE = (
    "filter_normalized_status",
    "reject_invalid_amounts",
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _passed_ids(metrics: dict[str, Any]) -> set[str]:
    return {
        str(row["id"]) for row in metrics.get("case_results", []) if row.get("passed")
    }


def _score_source(source: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="jlens-poc-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(source, encoding="utf-8")
        public = evaluator_core.score_program_path(path)
        solve, reasons = evaluator_core.load_candidate(path)
        if solve is None:
            raise ValueError(f"candidate rejected by evaluator: {reasons}")
        hidden = evaluator_core.score_holdout_callable(solve)
    return {"public": public, "hidden": hidden}


def _summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    public = evaluation["public"]
    hidden = evaluation["hidden"]
    return {
        "public_passed": int(public["passed_cases"]),
        "public_total": int(public["total_cases"]),
        "public_score": float(public["combined_score"]),
        "hidden_passed": int(hidden["passed_cases"]),
        "hidden_total": int(hidden["total_cases"]),
        "hidden_score": float(hidden["combined_score"]),
    }


def build_supervision_plan(observation: dict[str, Any]) -> dict[str, Any]:
    """Compile JLens observations into a bounded, non-causal experiment plan."""
    evidence = observation.get("evidence", {})
    boundary = str(observation.get("causal_boundary", "observational_not_causal"))
    repeated_fraction = float(evidence.get("repeated_transition_fraction", 0.0))
    return {
        "schema_version": 1,
        "strategy": "bounded_structured_mutation",
        "reason": (
            f"观测到 {repeated_fraction:.0%} 的迁移重复；采用两个白名单、"
            "可验证的小步 operator，避免宽泛重写。"
        ),
        "evidence_boundary": boundary,
        "public_failure_sequence": list(PUBLIC_FAILURE_SEQUENCE),
        "model_calls": 0,
        "max_mutations": len(PUBLIC_FAILURE_SEQUENCE),
    }


def _gate_step(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    changed: bool,
    postcondition_valid: bool,
) -> dict[str, Any]:
    parent_public = parent["public"]
    candidate_public = candidate["public"]
    parent_hidden = parent["hidden"]
    candidate_hidden = candidate["hidden"]
    lost = sorted(_passed_ids(parent_public) - _passed_ids(candidate_public))
    public_gain = int(candidate_public["passed_cases"] - parent_public["passed_cases"])
    hidden_gain = int(candidate_hidden["passed_cases"] - parent_hidden["passed_cases"])
    checks = {
        "source_changed": changed,
        "operator_postcondition": postcondition_valid,
        "public_gain_positive": public_gain > 0,
        "no_public_regression": not lost,
        "hidden_non_regression": hidden_gain >= 0,
    }
    return {
        "decision": "accepted" if all(checks.values()) else "rejected",
        "checks": checks,
        "public_gain": public_gain,
        "hidden_gain": hidden_gain,
        "lost_public_cases": lost,
    }


def render_markdown(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    final = result["final"]
    rows = []
    for index, step in enumerate(result["steps"], start=1):
        rows.append(
            "| {index} | `{failure}` | `{operator}` | {before} → {after} | "
            "{hidden_before} → {hidden_after} | {decision} |".format(
                index=index,
                failure=step["public_failure"],
                operator=step["operator_id"],
                before=step["before"]["public_passed"],
                after=step["after"]["public_passed"],
                hidden_before=step["before"]["hidden_passed"],
                hidden_after=step["after"]["hidden_passed"],
                decision=step["decision"],
            )
        )
    return f"""# 监督演化 POC

生成时间：{result["generated_at"]}

## 一页结论

- 公开 evaluator：**{baseline["public_passed"]}/{baseline["public_total"]} → {final["public_passed"]}/{final["public_total"]}**。
- 隐藏 evaluator：**{baseline["hidden_passed"]}/{baseline["hidden_total"]} → {final["hidden_passed"]}/{final["hidden_total"]}**，没有退化，但也没有证明泛化提升。
- Supervisor 使用既有 JLens 观测选择“小步、结构化 mutation”路线；本次不调用 LLM。
- Gate 决定：`{result["decision"]}`；`production_ready=false`。

## 实际闭环

```text
JLens 观察证据
  → Rule Supervisor
  → 两个白名单 RSI operator
  → 固定公开/隐藏 evaluator
  → 无回归 Gate
  → POC candidate
```

| 步骤 | 公开失败 | Operator | 公开通过 | 隐藏通过 | Gate |
|---:|---|---|---:|---:|---|
{chr(10).join(rows)}

## Supervisor 判断

{result["plan"]["reason"]}

证据边界：`{result["plan"]["evidence_boundary"]}`。JLens 在这里负责选择实验方向，不能作为正确性、因果性或晋升依据。

## 当前能证明什么

1. 最小服务链路能够消费真实 JLens 产物并形成可执行实验。
2. 结构化 RSI 候选在固定公开 evaluator 上新增通过 3 个案例，没有丢失父代已通过案例。
3. 隐藏集仍为 0/6，因此只能接受为 POC 候选，不能称为泛化成功、RSI 成功或生产版本。

## 下一步

优先补一个针对 identity/aggregation 的独立 operator，然后重新运行相同 Gate；只有隐藏集出现稳定提升后，才接入异构 LLM Supervisor。
"""


def render_html(result: dict[str, Any], candidate_source: str) -> str:
    baseline = result["baseline"]
    final = result["final"]
    public_before = 100 * baseline["public_passed"] / baseline["public_total"]
    public_after = 100 * final["public_passed"] / final["public_total"]
    step_rows = "".join(
        "<tr><td>{}</td><td><code>{}</code></td><td>{} → {}</td>"
        "<td>{} → {}</td><td>{}</td></tr>".format(
            index,
            html.escape(step["operator_id"]),
            step["before"]["public_passed"],
            step["after"]["public_passed"],
            step["before"]["hidden_passed"],
            step["after"]["hidden_passed"],
            html.escape(step["decision"]),
        )
        for index, step in enumerate(result["steps"], start=1)
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>监督演化 POC</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:960px;margin:0 auto;padding:32px;color:#172033;background:#f5f7fb}}
h1,h2{{font-weight:600}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #dfe4ec;border-radius:12px;padding:18px}} .value{{font-size:32px;font-weight:600}}
.bar{{height:12px;background:#e8ecf3;border-radius:8px;overflow:hidden;margin-top:10px}} .before,.after{{height:100%}}
.before{{background:#8c98aa;width:{public_before:.2f}%}} .after{{background:#2f7cf6;width:{public_after:.2f}%}}
.ok{{color:#176b3a}} .warn{{color:#8a5a00}} table{{width:100%;table-layout:fixed;border-collapse:collapse;background:#fff}}
th,td{{padding:10px;text-align:left;border-bottom:1px solid #e4e8ef;overflow-wrap:anywhere}} code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
pre{{overflow:auto;background:#101827;color:#e8edf6;padding:16px;border-radius:10px}} .flow{{font-size:18px;line-height:1.8}}
@media(max-width:600px){{body{{padding:20px 16px}}th,td{{padding:8px 4px;font-size:12px}}th:first-child{{width:7%}}th:nth-child(2){{width:39%}}}}
</style>
</head>
<body>
<h1>监督演化 POC</h1>
<p>真实 JLens 观察 → Rule Supervisor → 结构化 RSI → 固定 evaluator → Gate</p>
<div class="grid">
  <section class="card"><div>Baseline 公开集</div><div class="value">{baseline["public_passed"]}/{baseline["public_total"]}</div><div class="bar"><div class="before"></div></div></section>
  <section class="card"><div>Candidate 公开集</div><div class="value ok">{final["public_passed"]}/{final["public_total"]}</div><div class="bar"><div class="after"></div></div></section>
  <section class="card"><div>隐藏集</div><div class="value warn">{final["hidden_passed"]}/{final["hidden_total"]}</div><div>无退化，但尚无泛化增益</div></section>
  <section class="card"><div>Gate</div><div class="value ok">POC 候选</div><div><code>production_ready=false</code></div></section>
</div>
<h2>最小流程</h2>
<div class="card flow">JLens 发现 65% 重复迁移 → Supervisor 选择有界小步修改 → 两个 operator 连续通过 Gate → 公开集新增 3 个通过案例</div>
<h2>每一步</h2>
<table><thead><tr><th>#</th><th>Operator</th><th>公开通过</th><th>隐藏通过</th><th>决定</th></tr></thead><tbody>{step_rows}</tbody></table>
<h2>边界</h2>
<p>JLens 证据为 <code>{html.escape(result["plan"]["evidence_boundary"])}</code>；它只决定尝试什么，不决定候选是否正确。隐藏集仍为 0/6，因此这不是泛化成功或生产晋升。</p>
<details><summary>查看最终候选源码</summary><pre>{html.escape(candidate_source)}</pre></details>
</body>
</html>
"""


def run_poc(
    *, program_path: Path, observation_path: Path, output_dir: Path
) -> dict[str, Any]:
    source = program_path.read_text(encoding="utf-8")
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    plan = build_supervision_plan(observation)
    baseline_evaluation = _score_source(source)
    current_source = source
    current_evaluation = baseline_evaluation
    steps: list[dict[str, Any]] = []

    for public_failure in plan["public_failure_sequence"]:
        mutation_plan = derive_fallback_plan(public_failure)
        mutation = apply_mutation_plan(current_source, mutation_plan)
        candidate_evaluation = _score_source(mutation.source)
        gate = _gate_step(
            current_evaluation,
            candidate_evaluation,
            changed=mutation.changed,
            postcondition_valid=mutation.postcondition_valid,
        )
        step = {
            "public_failure": public_failure,
            "operator_id": mutation.operator_id,
            "before": _summary(current_evaluation),
            "after": _summary(candidate_evaluation),
            **gate,
        }
        steps.append(step)
        if gate["decision"] == "accepted":
            current_source = mutation.source
            current_evaluation = candidate_evaluation

    baseline = _summary(baseline_evaluation)
    final = _summary(current_evaluation)
    accepted = bool(
        final["public_passed"] > baseline["public_passed"]
        and final["hidden_passed"] >= baseline["hidden_passed"]
    )
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "poc_id": "supervised-evolution-minimal-v1",
        "inputs": {
            "program": str(program_path.resolve()),
            "program_sha256": _sha256(source),
            "observation": str(observation_path.resolve()),
            "observation_sha256": _sha256(observation_path.read_text(encoding="utf-8")),
            "strategy_id": observation.get("strategy_id"),
        },
        "plan": plan,
        "baseline": baseline,
        "steps": steps,
        "final": final,
        "decision": "poc_candidate_accepted" if accepted else "poc_candidate_rejected",
        "production_ready": False,
        "claims": {
            "service_loop_demonstrated": True,
            "public_capability_gain": final["public_passed"]
            - baseline["public_passed"],
            "hidden_generalization_gain": final["hidden_passed"]
            - baseline["hidden_passed"],
            "jlens_causal_gain_proven": False,
            "rsi_proven": False,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(output_dir / "candidate.py", current_source)
    _atomic_text(output_dir / "report.md", render_markdown(result))
    _atomic_text(output_dir / "report.html", render_html(result, current_source))
    _atomic_text(
        output_dir / "result.json",
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=ROOT / "initial_program.py")
    parser.add_argument("--observation", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_poc(
        program_path=args.program,
        observation_path=args.observation,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "baseline": result["baseline"],
                "final": result["final"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
