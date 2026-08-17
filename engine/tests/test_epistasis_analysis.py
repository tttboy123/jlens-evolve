from __future__ import annotations

from epistasis.analysis import (
    bootstrap_ci,
    changepoint_cumulative,
    exact_binomial_p,
    pearson,
    permutation_correlation_pvalue,
    spearman,
    threshold_vs_linear,
)


def test_spearman_perfect_monotone():
    x = [0.0, 0.25, 0.5, 1.0]
    y = [0.0, 1.0, 2.0, 4.0]
    rho, p = spearman(x, y)
    assert rho == 1.0
    assert p < 0.05


def test_pearson_sign_and_magnitude():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    r, _ = pearson(x, y)
    assert r < -0.9


def test_bootstrap_ci_contains_estimate():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    ci = bootstrap_ci(values, lambda v: sum(v) / len(v), n_resamples=200)
    assert ci["lower"] <= ci["estimate"] <= ci["upper"]


def test_permutation_p_small_for_real_correlation():
    x = list(range(20))
    y = [v * 2 + 1 for v in x]
    p = permutation_correlation_pvalue(x, y, n_permutations=200)
    assert p < 0.05


def test_sign_test():
    assert exact_binomial_p(9, 1) < 0.05
    assert exact_binomial_p(5, 5) >= 0.9


def test_threshold_model_detects_step():
    # y jumps sharply at coverage 0.5 -> threshold model should beat linear.
    x = [0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
    y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    result = threshold_vs_linear(x, y)
    assert result["supported"] is True
    assert result["threshold"] == 0.5
    assert result["bic_advantage"] > 0


def test_changepoint_finds_jump():
    cumulative = [0.0, 0.0, 0.0, 0.5, 0.8, 0.9, 1.0]
    result = changepoint_cumulative(cumulative)
    assert result["supported"] is True
    assert result["max_acceleration_index"] == 3
