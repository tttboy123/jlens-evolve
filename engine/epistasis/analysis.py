"""Small, auditable statistics used by the A-E experiment reports.

Everything here is a pure function over plain lists so it is easy to unit test
and to re-derive in an external auditor.  No hidden state.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy import stats


def pearson(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    if len(x) < 2 or len(x) != len(y):
        return (0.0, 1.0)
    r, p = stats.pearsonr(x, y)
    if isinstance(r, float):
        return (r, float(p))
    return (float(r[0]), float(p))


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    if len(x) < 2 or len(x) != len(y):
        return (0.0, 1.0)
    rho, p = stats.spearmanr(x, y)
    return (float(rho), float(p))


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float],
    *,
    n_resamples: int = 2000,
    seed: int = 20260802,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for ``stat`` over ``values``."""
    if not values:
        return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan")}
    rng = random.Random(seed)
    sample = list(values)
    estimates: list[float] = []
    for _ in range(n_resamples):
        resampled = [sample[rng.randrange(len(sample))] for _ in sample]
        estimates.append(stat(resampled))
    estimates.sort()
    lo = estimates[int(round((alpha / 2) * (n_resamples - 1)))]
    hi = estimates[int(round((1 - alpha / 2) * (n_resamples - 1)))]
    return {"estimate": float(stat(sample)), "lower": float(lo), "upper": float(hi)}


def bootstrap_correlation_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    kind: str = "spearman",
    n_resamples: int = 2000,
    seed: int = 20260802,
) -> dict[str, float]:
    pairs = list(zip(x, y, strict=True))
    if len(pairs) < 3:
        return {"estimate": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    corr_fn = stats.spearmanr if kind == "spearman" else stats.pearsonr
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        xs = [p[0] for p in sample]
        ys = [p[1] for p in sample]
        if len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        try:
            r, _ = corr_fn(xs, ys)
            estimates.append(float(r))
        except Exception:  # noqa: BLE001 - degenerate resample
            continue
    estimates.sort()
    if not estimates:
        return {"estimate": float(corr_fn(x, y)[0]), "lower": 0.0, "upper": 0.0}
    lo = estimates[int(round(0.025 * (n_resamples - 1)))]
    hi = estimates[int(round(0.975 * (n_resamples - 1)))]
    return {"estimate": float(corr_fn(x, y)[0]), "lower": float(lo), "upper": float(hi)}


def permutation_correlation_pvalue(
    x: Sequence[float],
    y: Sequence[float],
    *,
    kind: str = "spearman",
    n_permutations: int = 2000,
    seed: int = 20260802,
) -> float:
    """Permutation p-value for the observed correlation under label shuffling."""
    pairs = list(zip(x, y, strict=True))
    if len(pairs) < 3:
        return 1.0
    rng = random.Random(seed)
    corr_fn = stats.spearmanr if kind == "spearman" else stats.pearsonr
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    observed = abs(float(corr_fn(xs, ys)[0]))
    count = 0
    for _ in range(n_permutations):
        shuffled = list(ys)
        rng.shuffle(shuffled)
        try:
            r, _ = corr_fn(xs, shuffled)
        except Exception:  # noqa: BLE001 - degenerate permutation
            r = 0.0
        if abs(float(r)) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def exact_binomial_p(wins: int, losses: int) -> float:
    """Two-sided exact sign-test p-value for paired comparisons."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p_value = 2.0 * sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, p_value)


def threshold_vs_linear(
    x: Sequence[float],
    y: Sequence[float],
    *,
    n_bootstrap: int = 500,
    seed: int = 20260802,
) -> dict[str, Any]:
    """Compare a linear model to a step (coverage-threshold) model by BIC.

    Returns the best threshold candidate, its BIC advantage over the linear fit,
    and the fit summaries.  A large positive ``bic_advantage`` is evidence for a
    critical-coverage step rather than a smooth linear relationship.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    n = len(xs)
    if n < 4:
        return {"supported": False, "reason": "insufficient points"}

    def bic_linear() -> tuple[float, dict[str, float]]:
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        rss = float(np.sum((ys - pred) ** 2))
        bic = n * math.log(max(rss, 1e-12) / n) + 2 * math.log(n)
        return bic, {"slope": float(slope), "intercept": float(intercept), "rss": rss}

    lin_bic, lin_fit = bic_linear()
    best: dict[str, Any] | None = None
    candidates = sorted(set(xs))
    for c in candidates:
        if min(xs) < c < max(xs):
            step = np.where(xs < c, 0.0, 1.0)
            try:
                slope, intercept = np.polyfit(step, ys, 1)
            except Exception:  # noqa: BLE001 - degenerate threshold
                continue
            pred = slope * step + intercept
            rss = float(np.sum((ys - pred) ** 2))
            bic = n * math.log(max(rss, 1e-12) / n) + 2 * math.log(n)
            if best is None or bic < best["bic"]:
                best = {
                    "threshold": float(c),
                    "bic": float(bic),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "rss": rss,
                }
    if best is None:
        return {"supported": False, "reason": "no interior threshold"}

    # Bootstrap the critical-threshold estimate.
    rng = random.Random(seed)
    thresholds: list[float] = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = xs[idx]
        by = ys[idx]
        b_best = None
        for c in sorted(set(bx)):
            if min(bx) < c < max(bx):
                step = np.where(bx < c, 0.0, 1.0)
                try:
                    slope, intercept = np.polyfit(step, by, 1)
                except Exception:  # noqa: BLE001
                    continue
                pred = slope * step + intercept
                rss = float(np.sum((by - pred) ** 2))
                bic = len(bx) * math.log(max(rss, 1e-12) / len(bx)) + 2 * math.log(
                    len(bx)
                )
                if b_best is None or bic < b_best["bic"]:
                    b_best = {"threshold": float(c), "bic": float(bic)}
        if b_best is not None:
            thresholds.append(b_best["threshold"])
    ci: dict[str, float] = {"lower": 0.0, "upper": 0.0, "median": 0.0}
    if thresholds:
        thresholds.sort()
        ci = {
            "lower": float(thresholds[int(round(0.05 * (len(thresholds) - 1)))]),
            "upper": float(thresholds[int(round(0.95 * (len(thresholds) - 1)))]),
            "median": float(thresholds[len(thresholds) // 2]),
        }
    return {
        "supported": True,
        "linear_bic": float(lin_bic),
        "linear_fit": lin_fit,
        "threshold_bic": float(best["bic"]),
        "threshold": float(best["threshold"]),
        "threshold_fit": {
            "slope": float(best["slope"]),
            "intercept": float(best["intercept"]),
            "rss": float(best["rss"]),
        },
        "bic_advantage": float(lin_bic - best["bic"]),
        "threshold_ci": ci,
    }


def changepoint_cumulative(
    cumulative: Sequence[float],
    *,
    n_bootstrap: int = 500,
    seed: int = 20260802,
) -> dict[str, Any]:
    """Detect the largest jump in a cumulative curve (cross-task threshold).

    Returns the index with the maximum second-difference (acceleration) and a
    bootstrap CI for it.  ``k`` is the number of distinct tasks.
    """
    ys = list(cumulative)
    if len(ys) < 3:
        return {"supported": False, "reason": "insufficient points"}
    deltas = [ys[i] - ys[i - 1] for i in range(1, len(ys))]
    accel = [deltas[i] - deltas[i - 1] for i in range(1, len(deltas))]
    peak = int(np.argmax(accel)) + 2  # 1-based task count at the acceleration peak
    rng = random.Random(seed)
    peaks: list[int] = []
    for _ in range(n_bootstrap):
        idx = sorted(rng.randrange(len(ys)) for _ in range(len(ys)))
        bys = [ys[i] for i in idx]
        bdeltas = [bys[i] - bys[i - 1] for i in range(1, len(bys))]
        baccel = [bdeltas[i] - bdeltas[i - 1] for i in range(1, len(bdeltas))]
        if baccel:
            peaks.append(int(np.argmax(baccel)) + 2)
    ci: dict[str, float] = {}
    if peaks:
        peaks.sort()
        ci = {
            "lower": float(peaks[int(round(0.05 * (len(peaks) - 1)))]),
            "upper": float(peaks[int(round(0.95 * (len(peaks) - 1)))]),
            "median": float(peaks[len(peaks) // 2]),
        }
    return {
        "supported": True,
        "max_acceleration_index": peak,
        "acceleration": float(accel[peak - 2]),
        "cumulative": [float(v) for v in ys],
        "threshold_ci": ci,
    }


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0
