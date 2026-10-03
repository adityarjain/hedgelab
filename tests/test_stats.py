import numpy as np
import pytest

from hedgelab.stats import mean_ci, newey_west_se, paired_diff, sharpe_ci, stationary_indices


def ar1(n, phi, seed=0):
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_bootstrap_indices_are_valid_blocks():
    idx = stationary_indices(50, n_boot=200, block=5)
    assert idx.shape == (200, 50) and idx.min() >= 0 and idx.max() < 50
    runs = (np.diff(idx, axis=1) == 1) | (np.diff(idx, axis=1) == -49)  # consecutive, incl. wraparound
    assert 0.7 < runs.mean() < 0.9  # ~1 - 1/block of steps continue the block


def test_mean_ci_on_iid_data_matches_textbook_width():
    x = np.random.default_rng(1).normal(0.3, 2.0, 600)
    est, lo, hi = mean_ci(x, block=1)
    assert est == pytest.approx(x.mean()) and lo < est < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 * x.std(ddof=1) / np.sqrt(600), rel=0.15)


def test_serial_correlation_widens_intervals():
    x = ar1(600, 0.6)
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert newey_west_se(x) > 1.5 * naive
    _, lo_b, hi_b = mean_ci(x, block=20)
    _, lo_i, hi_i = mean_ci(x, block=1)
    assert (hi_b - lo_b) > 1.3 * (hi_i - lo_i)


def test_newey_west_equals_naive_without_lags():
    x = np.random.default_rng(2).normal(size=300)
    assert newey_west_se(x, lags=0) == pytest.approx(x.std(ddof=0) / np.sqrt(300))


def test_paired_diff_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(3)
    b = rng.normal(0, 3, 400)
    shifted = paired_diff(b + 0.5 + rng.normal(0, 0.2, 400), b)
    assert shifted[1] > 0 and shifted[3] < 0.01
    noise = paired_diff(b + rng.normal(0, 0.2, 400), b)
    assert noise[1] < 0 < noise[2] and noise[3] > 0.05


def test_sharpe_ci_brackets_estimate():
    x = np.random.default_rng(4).normal(0.5, 1.0, 250)
    est, lo, hi = sharpe_ci(x)
    assert est == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(12)) and lo < est < hi
