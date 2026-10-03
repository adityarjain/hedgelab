"""Inference for per-trade P&L, which is fat-tailed and serially correlated (crashes cluster).

Confidence intervals use the stationary bootstrap (Politis & Romano, 1994): resample blocks of consecutive
observations with geometric lengths, so clustering survives resampling. Newey-West gives a HAC standard
error of the mean as a cross-check. Paired comparisons (hedge A vs hedge B on the same trades) bootstrap the
per-trade difference, which is far tighter than comparing two separate means.
"""
import numpy as np


def stationary_indices(n, n_boot=5000, block=None, seed=0):
    """(n_boot, n) resampling indices: blocks of mean length `block` (default n^(1/3)), wrapping around."""
    block = block or max(1, round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, n), dtype=int)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jump = rng.random((n_boot, n)) < 1 / block
    starts = rng.integers(0, n, (n_boot, n))
    for j in range(1, n):
        idx[:, j] = np.where(jump[:, j], starts[:, j], (idx[:, j - 1] + 1) % n)
    return idx


def _ci(x, stat, alpha=0.05, **kw):
    x = np.asarray(x, float)
    draws = stat(x[stationary_indices(len(x), **kw)])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(stat(x)), float(lo), float(hi)


def mean_ci(x, **kw):
    """(mean, lo, hi), 95% stationary-bootstrap percentile interval."""
    return _ci(x, lambda a: a.mean(axis=-1), **kw)


def sharpe_ci(x, periods=12, **kw):
    """(annualised Sharpe, lo, hi) of a per-period P&L series (rf already netted out or zero)."""
    return _ci(x, lambda a: a.mean(axis=-1) / a.std(axis=-1, ddof=1) * np.sqrt(periods), **kw)


def paired_diff(a, b, **kw):
    """Per-trade a - b: (mean diff, lo, hi, two-sided bootstrap p-value for 'mean diff = 0')."""
    d = np.asarray(a, float) - np.asarray(b, float)
    est, lo, hi = mean_ci(d, **kw)
    draws = d[stationary_indices(len(d), **kw)].mean(axis=1)
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return est, lo, hi, float(p)


def newey_west_se(x, lags=None):
    """HAC (Bartlett kernel) standard error of the mean."""
    x = np.asarray(x, float)
    n = len(x)
    lags = int(4 * (n / 100) ** (2 / 9)) if lags is None else lags
    e = x - x.mean()
    var = e @ e / n + 2 * sum((1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, lags + 1))
    return float(np.sqrt(var / n))
