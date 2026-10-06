import numpy as np
import pandas as pd
import pytest

from hedgelab.trades import mark_to_market, run_all, run_trade, straddle


def test_daily_marks_add_up_to_trade_totals():
    df = gbm_frame(years=3)
    df["vix"] = 20 + 5 * np.sin(np.arange(len(df)) / 40)  # marks move with VIX, entry pricing too
    t = run_all(df, cost=3e-4, entry_cost=0.02, start="2000-01-01")
    marks = mark_to_market(df, cost=3e-4, entry_cost=0.02, start="2000-01-01")
    assert marks.sum() == pytest.approx(per_100(t).sum(), abs=1e-9)
    assert marks.index.is_monotonic_increasing and marks.index.isin(df.index).all()


def test_daily_marks_ignore_the_future():
    df = gbm_frame(years=3)
    k = df.index[400]
    df2 = df.copy()
    after = df2.index > k
    df2.loc[after, "close"] *= np.random.default_rng(7).uniform(0.7, 1.3, after.sum())
    df2.loc[after, "vix"] = 45.0
    a, b = mark_to_market(df, start="2000-01-01"), mark_to_market(df2, start="2000-01-01")
    pd.testing.assert_series_equal(a.loc[:k], b.loc[:k])


def gbm_frame(years=40, sigma=0.2, vix=20.0, seed=0):
    """Synthetic data.load()-shaped frame. Variance accrues in calendar time (weekends count), matching
    how the engine and VIX measure time, so 'realized vol = sigma' holds exactly in expectation."""
    idx = pd.bdate_range("2000-01-03", periods=int(years * 261))
    dt = np.diff(idx.values).astype("timedelta64[D]").astype(float) / 365
    z = np.random.default_rng(seed).standard_normal(len(dt))
    logs = np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z)])
    return pd.DataFrame({"close": np.exp(logs), "dividend": 0.0, "vix": vix, "rate": 0.0}, index=idx)


def per_100(t):
    return t.total / t.S0 * 100


def test_fair_world_breaks_even():
    """Realized vol = implied vol, no costs: a delta-hedged short straddle earns nothing on average."""
    t = run_all(gbm_frame(), cost=0.0, start="2000-01-01")
    p = per_100(t)
    assert abs(p.mean()) < 3 * p.std() / np.sqrt(len(p))


def test_vrp_world_profits_and_vrp_term_explains_it():
    """Implied 20% > realized 15%: positive P&L, almost all of it the gamma-weighted variance term."""
    t = run_all(gbm_frame(sigma=0.15), cost=0.0, start="2000-01-01")
    assert per_100(t).mean() > 0
    assert abs(t.residual.mean()) < 0.1 * t.total.mean()
    assert np.corrcoef(t.total, t.vrp)[0, 1] > 0.9


def test_decomposition_sums_exactly():
    t = run_all(gbm_frame(years=5), cost=3e-4, entry_cost=0.02, start="2000-01-01")
    np.testing.assert_allclose(t.total, t.vrp + t.residual - t.stock_cost - t.entry_cost, rtol=0, atol=1e-12)


def test_costs_are_accounted_exactly():
    days = np.array([0, 1, 4, 5, 6, 7, 8, 11])
    S = 100 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(1).normal(0, 0.01, 7)]))
    args = (days, S, np.zeros(8), 0.2, 0.0, 0.0)
    free = run_trade(*args)
    paid = run_trade(*args, cost=5e-4, entry_cost=0.03)
    assert free["total"] - paid["total"] == pytest.approx(paid["stock_cost"] + paid["entry_cost"], abs=1e-12)
    assert paid["stock_cost"] > 0 and paid["entry_cost"] == pytest.approx(0.03 * paid["premium"])


def test_dividend_is_paid_to_the_hedge():
    """Same path, one dividend on day 3: the long-delta hedge collects h * D."""
    days, S = np.arange(6.0), np.full(6, 100.0)
    div = np.zeros(6)
    base = run_trade(days, S, div, 0.2, 0.0, 0.0)
    div[3] = 1.0
    with_div = run_trade(days, S, div, 0.2, 0.0, 0.0)
    h2 = straddle(np.array([100.0]), 100.0, np.array([3 / 365]), 0.0, 0.0, 0.2)[1][0]  # held over day 2 -> 3
    assert with_div["total"] - base["total"] == pytest.approx(h2 * 1.0)


def test_straddle_greeks_match_finite_differences():
    S, h = np.array([97.0, 100.0, 104.0]), 1e-3
    v, d, g = straddle(S, 100.0, 0.08, 0.03, 0.01, 0.2)
    up, dn = straddle(S + h, 100.0, 0.08, 0.03, 0.01, 0.2)[0], straddle(S - h, 100.0, 0.08, 0.03, 0.01, 0.2)[0]
    np.testing.assert_allclose(d, (up - dn) / (2 * h), rtol=1e-5)
    np.testing.assert_allclose(g, (up - 2 * v + dn) / h**2, rtol=1e-3)


def test_hedge_uses_only_past_prices():
    """Two paths identical through session j: hedge holdings identical through session j."""
    days = np.arange(0.0, 31.0)
    S = 100 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(4).normal(0, 0.01, 30)]))
    j = 12
    S2 = S.copy()
    S2[j + 1 :] *= 1.15
    h1 = run_trade(days, S, np.zeros(31), 0.2, 0.0, 0.0)["holdings"]
    h2 = run_trade(days, S2, np.zeros(31), 0.2, 0.0, 0.0)["holdings"]
    np.testing.assert_array_equal(h1[: j + 1], h2[: j + 1])
    assert not np.array_equal(h1, h2)  # and the scramble did reach later decisions


def test_no_lookahead():
    """Scramble everything after date k: trades settled by k are unchanged, and every trade entered by k
    (including the one straddling k) was priced identically."""
    df = gbm_frame(years=4)
    k = df.index[600]
    df2 = df.copy()
    after = df2.index > k
    rng = np.random.default_rng(3)
    df2.loc[after, "close"] *= rng.uniform(0.7, 1.3, after.sum())
    df2.loc[after, "vix"] = rng.uniform(10, 60, after.sum())
    df2.loc[after, "dividend"] = rng.uniform(0, 2, after.sum())
    a, b = run_all(df, start="2000-01-01"), run_all(df2, start="2000-01-01")
    done = a.settle <= k
    assert done.sum() > 20
    pd.testing.assert_frame_equal(a[done], b[done])
    entered = a.entry <= k
    assert (entered & ~done).any()  # the straddling trade is in the check
    pd.testing.assert_frame_equal(a.loc[entered, ["sigma", "premium"]], b.loc[entered, ["sigma", "premium"]])
