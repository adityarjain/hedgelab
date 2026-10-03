import numpy as np
import pandas as pd
import pytest

from hedgelab.volatility import MODELS, dm_test, features, garch_fit, qlike, walk_forward


def sim_garch(n, omega=0.02, alpha=0.1, beta=0.85, seed=0):
    rng = np.random.default_rng(seed)
    h, r = np.empty(n), np.empty(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(n):
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
        if t + 1 < n:
            h[t + 1] = omega + alpha * r[t] ** 2 + beta * h[t]
    return r, h


def sim_frame(n=1600, seed=0):
    r, h = sim_garch(n, seed=seed)
    rv = h * np.exp(np.random.default_rng(seed + 1).normal(-0.125, 0.5, n))  # noisy, mean-unbiased proxy
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(r, idx), pd.Series(rv, idx)


def test_garch_fit_recovers_parameters():
    r, _ = sim_garch(8000, seed=1)
    _, _, a, b = garch_fit(r)
    assert a == pytest.approx(0.1, abs=0.04) and b == pytest.approx(0.85, abs=0.05)


def test_qlike_is_zero_at_truth_and_positive_elsewhere():
    y = np.array([1.0, 2.0, 0.5])
    assert np.allclose(qlike(y, y), 0)
    assert (qlike(y * 1.3, y) > 0).all() and (qlike(y * 0.7, y) > 0).all()


def test_dm_detects_a_biased_forecast():
    rng = np.random.default_rng(2)
    truth = np.exp(rng.normal(0, 0.5, 3000))
    y = truth * rng.exponential(1.0, 3000)
    stat, p = dm_test(qlike(truth, y), qlike(1.5 * truth, y))
    assert stat < 0 and p < 0.01


def test_no_lookahead():
    """Scramble raw data after row k: every model's forecasts on rows <= k must be unchanged."""
    r, rv = sim_frame()
    k = 1250
    r2, rv2 = r.copy(), rv.copy()
    noise = np.random.default_rng(9).uniform(0.3, 3.0, len(r) - k - 1)
    r2.iloc[k + 1 :] *= noise
    rv2.iloc[k + 1 :] *= noise
    kw = dict(first=1000, step=100, seeds=1, mlp_iter=50)
    f, f2 = features(r, rv), features(r2, rv2)
    a, b = walk_forward(f, **kw), walk_forward(f2, **kw)
    cut = f.index[f.index <= r.index[k]]
    pd.testing.assert_frame_equal(a.loc[cut, MODELS], b.loc[cut, MODELS])
