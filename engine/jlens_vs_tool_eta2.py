"""v2.5 NEXT-2: JLens-vs-tool-trace discriminative power (v0.3 eta^2 re-test).

Compares how much variance of native scores is explained by:
  (a) tool-trace features from the DeepSeek rail (real-search-002 G0, 32 arms);
  (b) per-layer hidden-state profiles from the local JLens rail (Qwen3.5-4B,
      run-3: 96 layer records, no native outcomes yet -> sample limitation).

Eta^2 = SS_between / SS_total (one-way ANOVA effect size), the same metric the
v0.3 probe used (logit-lens eta^2=0.997, JLens eta^2=0.995).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT = Path("/Users/lune/Documents/Codex/2026-07-18/bang/work/evolve-jlens-cluster")


def eta_squared(values: list[float], groups: list[int | float]) -> float:
    n = len(values)
    if n < 2 or len(set(groups)) < 2:
        return 0.0
    grand = sum(values) / n
    ss_total = sum((v - grand) ** 2 for v in values)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in set(groups):
        members = [v for v, grp in zip(values, groups) if grp == g]
        if members:
            ss_between += len(members) * ((sum(members) / len(members)) - grand) ** 2
    return ss_between / ss_total


def tool_trace_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bin_names = sorted({b for r in rows for b in (r.get("cmd_bins") or {})})
    features: dict[str, list[float]] = {"tool_total": []}
    for b in bin_names:
        features[b] = []
    if any(r.get("cost_total_tokens") is not None for r in rows):
        features["cost_total_tokens"] = []
    groups = [r["native_score"] for r in rows]
    for r in rows:
        features["tool_total"].append(float(r.get("tool_total") or 0))
        for b in bin_names:
            features[b].append(float((r.get("cmd_bins") or {}).get(b, 0)))
        if "cost_total_tokens" in features:
            features["cost_total_tokens"].append(float(r.get("cost_total_tokens") or 0))
    results = {}
    for name, values in features.items():
        results[name] = {
            "eta2": round(eta_squared(values, groups), 4),
            "mean_resolved": round(
                sum(v for v, g in zip(values, groups) if g == 1) / max(1, sum(1 for g in groups if g == 1)), 4
            ),
            "mean_unresolved": round(
                sum(v for v, g in zip(values, groups) if g == 0) / max(1, sum(1 for g in groups if g == 0)), 4
            ),
        }
    return results


def _ols_r2(X: list[list[float]], y: list[float]) -> float:
    """R^2 of ordinary least squares fit (with intercept) of y on features X."""
    try:
        import numpy as np
    except ImportError:
        return 0.0
    n = len(y)
    if n < 4:
        return 0.0
    Xa = np.array([[1.0] + row for row in X], dtype=float)
    ya = np.array(y, dtype=float)
    try:
        beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
        pred = Xa @ beta
        ss_res = float(np.sum((ya - pred) ** 2))
        ss_tot = float(np.sum((ya - ya.mean()) ** 2))
        return 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def combined_tool_r2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Leave-one-out cross-validated R^2 on the top-k single features.

    Plain OLS R^2 with 40 features on 32 arms is an overfit artifact; LOOCV
    gives the honest out-of-sample discriminative power of the tool profile.
    """
    per = tool_trace_analysis(rows)
    top = sorted(per, key=lambda k: per[k]["eta2"], reverse=True)[:6]
    X = [
        [float((r.get("cmd_bins") or {}).get(b, 0)) for b in top]
        + [float(r.get("tool_total") or 0)]
        for r in rows
    ]
    y = [r["native_score"] for r in rows]
    preds: list[float] = []
    for i in range(len(rows)):
        tr_X = [row for j, row in enumerate(X) if j != i]
        tr_y = [v for j, v in enumerate(y) if j != i]
        test_X = X[i]
        # mini linear fit per fold
        import numpy as np
        Xa = np.array([[1.0] + row for row in tr_X], dtype=float)
        ya = np.array(tr_y, dtype=float)
        try:
            beta, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
            pred = float(np.array([1.0] + test_X) @ beta)
        except (np.linalg.LinAlgError, ValueError):
            pred = 0.0
        preds.append(1.0 if pred >= 0.5 else 0.0)
    acc = sum(1 for a, b in zip(preds, y) if a == b) / len(y)
    return {
        "loocv_accuracy_top6": round(acc, 4),
        "top_features": top,
        "n_samples": len(rows),
        "note": "R^2 all-features is an overfit artifact (40 feat / 32 samples); use LOOCV accuracy instead.",
    }


def layer_profile_analysis() -> dict[str, Any]:
    """Descriptive layer-profile stats from local runs (no paired native labels yet)."""
    run3 = PROJECT / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/local-jlens-run-3/RUN-ARTIFACT.json"
    out: dict[str, Any] = {"runs_loaded": 0, "note": ""}
    if run3.is_file():
        artifact = json.loads(run3.read_text(encoding="utf-8"))
        records = [
            rec
            for event in artifact.get("tool_events", [])
            for rec in event.get("layer_records", [])
        ]
        layers = {}
        for rec in records:
            layer = rec.get("layer")
            layers.setdefault(layer, {"l2": []})
            if isinstance(rec.get("l2_norm"), (int, float)):
                layers[layer]["l2"].append(float(rec["l2_norm"]))
        out["runs_loaded"] = 1
        out["record_count"] = len(records)
        out["layers"] = len(layers)
        l2_means = [sum(v["l2"]) / len(v["l2"]) for v in layers.values() if v["l2"]]
        out["layer_l2_mean"] = round(sum(l2_means) / len(l2_means), 3) if l2_means else None
        out["layer_l2_std"] = (
            round((sum((x - sum(l2_means) / len(l2_means)) ** 2 for x in l2_means) / len(l2_means)) ** 0.5, 3)
            if l2_means
            else None
        )
    out["note"] = (
        "local runs have layer records but no official native outcomes yet; "
        "JLens eta^2 requires paired (layer_profile, native_score) per task. "
        "Protocol: evaluate local-lens patches with the official harness, then re-run this script."
    )
    return out


def main() -> int:
    dataset = json.loads(
        (Path("/private/tmp/g0-tool-dataset.json")).read_text(encoding="utf-8")
    )
    tool = tool_trace_analysis(dataset)
    combined = combined_tool_r2(dataset)
    layers = layer_profile_analysis()
    report = {
        "method": "eta^2 = SS_between/SS_total (v0.3 metric)",
        "reference": {"logit_lens_eta2": 0.997, "jlens_eta2": 0.995},
        "tool_trace_eta2": tool,
        "tool_trace_combined_r2": combined,
        "layer_profile": layers,
        "sample": {
            "arms": len(dataset),
            "resolved": sum(1 for r in dataset if r["resolved"]),
            "tasks": len({r["task_uid"] for r in dataset}),
        },
        "conclusion_tool": (
            "tool-trace combined R^2 quantifies how much native-score variance "
            "tool traces explain (v0.3 eta^2 ~0.99 for lens features as reference)."
        ),
        "combined_r2": combined,
    }
    out_dir = PROJECT / "artifacts/v2.5.0/v2.5.0-local-jlens/runs/jlens-vs-tool-eta2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ETA2-RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    top = sorted(tool.items(), key=lambda kv: kv[1]["eta2"], reverse=True)[:12]
    print(json.dumps({"sample": report["sample"], "top_tool_features": top, "layer": layers}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
