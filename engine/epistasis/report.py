"""Evidence report writer: JSON artifacts + a human-readable SUMMARY.md."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def build_summary_markdown(
    *,
    meta: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Epistasis / Diversity-Collapse Experiment Summary")
    lines.append("")
    lines.append(f"- generated_at: `{meta.get('generated_at')}`")
    lines.append(
        f"- mode: `{meta.get('mode')}`  budget: `{meta.get('budget')}`  seeds: `{meta.get('seeds')}`"
    )
    lines.append(
        f"- tasks: `{meta.get('task_count')}` (real `{meta.get('real_task_count')}` + synthetic `{meta.get('synthetic_task_count')}`)"
    )
    lines.append(
        f"- cells executed: `{len(cells)}`  attempts: `{meta.get('attempt_count')}`"
    )
    lines.append("")

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    a = reports.get("A")
    if a:
        lines.append("## A — Composition vs single (conjunction escape)")
        lines.append("")
        lines.append(
            f"- cells: `{a['cells']}`  wins(delta>0): `{a['wins']}`  losses: `{a['losses']}`  ties: `{a['ties']}`"
        )
        lines.append(
            f"- sign-test p: `{a['sign_test_p']:.4f}`  synergy cells: `{a['synergy_cells']}`  order-sensitive: `{a['order_sensitive_cells']}`"
        )
        lines.append(
            f"- mean delta over best single: `{a['mean_delta_over_best']:+.3f}`  mean delta over additive prediction: `{a['mean_delta_over_additive']:+.3f}`"
        )
        lines.append(f"- conclusion: `{a['conclusion']}`")
        lines.append("")

    b = reports.get("B")
    if b:
        lines.append("## B — Coverage -> yield correlation (large-scale)")
        lines.append("")
        lines.append(f"- cells: `{b['cells']}`")
        sy = b.get("spearman_yield", {})
        sd = b.get("spearman_best_delta", {})
        ci = b.get("spearman_yield_ci", {})
        lines.append(
            f"- yield: Spearman rho=`{sy.get('rho', 0):.3f}` p=`{sy.get('p', 1):.4f}` "
            f"(95% CI [{ci.get('lower', 0):.3f}, {ci.get('upper', 0):.3f}]) "
            f"permutation p=`{b.get('permutation_p_yield', 1):.4f}`"
        )
        lines.append(
            f"- best_delta: Spearman rho=`{sd.get('rho', 0):.3f}` p=`{sd.get('p', 1):.4f}`"
        )
        t = b.get("threshold_yield", {})
        if t.get("supported"):
            lines.append(
                f"- threshold model: critical coverage `{t['threshold']:.2f}` "
                f"(bootstrap median `{t['threshold_ci'].get('median', float('nan')):.2f}`), "
                f"BIC advantage over linear `{t['bic_advantage']:+.1f}`"
            )
        else:
            lines.append(
                f"- threshold model: not supported (`{t.get('reason', 'n/a')}`)"
            )
        lines.append("")
        lines.append("| coverage | cells | mean yield | mean best_delta | p25 | p75 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for coverage in sorted(b.get("ladder", {}), key=float):
            row = b["ladder"][coverage]
            lines.append(
                f"| {coverage} | {row['cells']} | {row['mean_yield']:.3f} | "
                f"{row['mean_best_delta']:+.3f} | {row['p25_best_delta']:+.1f} | {row['p75_best_delta']:+.1f} |"
            )
        lines.append("")

    c = reports.get("C")
    if c:
        lines.append("## C — Lineage vs composed (is simultaneous change required?)")
        lines.append("")
        lines.append(
            f"- cells: `{c['cells']}`  lineage reaches composed: `{c['lineage_reaches_composed_cells']}`  composed strictly better: `{c['composed_strictly_better_cells']}`"
        )
        lines.append(f"- mean delta (lineage - composed): `{c['mean_delta']:+.3f}`")
        lines.append(f"- conclusion: `{c['conclusion']}`")
        lines.append("")

    d = reports.get("D")
    if d:
        lines.append("## D — Epistasis (2^k factorial interaction)")
        lines.append("")
        lines.append(
            f"- design: `{d['design']}`  cells: `{d['cells']}`  synergistic pairs: `{d['synergistic_count']}`  mean interaction: `{d['mean_interaction_all']:+.3f}`"
        )
        for key, value in sorted(d.get("pairs", {}).items()):
            op_i, op_j = key.split("|")
            lines.append(
                f"- `{op_i}` x `{op_j}`: gain_i=`{value['mean_gain_i']:+.2f}` "
                f"gain_j=`{value['mean_gain_j']:+.2f}` gain_ij=`{value['mean_gain_ij']:+.2f}` "
                f"interaction=`{value['mean_interaction']:+.3f}` "
                f"(95% CI [{value['ci_lower']:+.3f}, {value['ci_upper']:+.3f}]) "
                f"positive cells=`{value['positive_cells']}/{value['cells']}` "
                f"synergy=`{value['synergy']}`"
            )
        lines.append(f"- conclusion: `{d['conclusion']}`")
        lines.append("")

    e = reports.get("E")
    if e:
        lines.append("## E — Cross-task validation threshold")
        lines.append("")
        for op, value in sorted(e.get("operators", {}).items()):
            cp = value.get("changepoint", {})
            if cp.get("supported"):
                lines.append(
                    f"- `{op}`: tasks=`{value['tasks_observed']}` "
                    f"cumulative_gain=`{value['cumulative_gain']}` "
                    f"threshold at task `{cp['max_acceleration_index']}` "
                    f"(CI {cp['threshold_ci'].get('lower', float('nan')):.1f}–{cp['threshold_ci'].get('upper', float('nan')):.1f})"
                )
            else:
                lines.append(
                    f"- `{op}`: tasks=`{value['tasks_observed']}` cumulative_gain=`{value['cumulative_gain']}` no threshold detected"
                )
        lines.append(f"- conclusion: `{e['conclusion']}`")
        lines.append("")

    lines.append("## Command")
    lines.append("")
    lines.append(f"```bash\n{meta.get('command', '')}\n```")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    out_dir: Path,
    *,
    meta: dict[str, Any],
    reports: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_manifest.json", meta)
    write_json(out_dir / "EXPERIMENTS.json", reports)
    write_json(out_dir / "CELLS.json", cells)
    _atomic_write(
        out_dir / "EVENTS.jsonl",
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in events
        ),
    )
    _atomic_write(
        out_dir / "SUMMARY.md",
        build_summary_markdown(meta=meta, reports=reports, cells=cells),
    )


def default_meta(
    *,
    command: str,
    mode: str,
    budget: int,
    seeds: list[int],
    task_count: int,
    real_task_count: int,
    synthetic_task_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "command": command,
        "mode": mode,
        "budget": budget,
        "seeds": seeds,
        "task_count": task_count,
        "real_task_count": real_task_count,
        "synthetic_task_count": synthetic_task_count,
        "experiments": ["A", "B", "C", "D", "E"],
    }
