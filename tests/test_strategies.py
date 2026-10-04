import numpy as np
import pandas as pd
import pytest
import torch
from test_trades import gbm_frame

from hedgelab import heston
from hedgelab import strategies as St
from hedgelab.trades import run_all, run_trade
from hedgelab.volatility import har_monthly

DAYS = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29], float)


def path(seed=0):
    return 100 * np.exp(np.cumsum(np.r_[0, np.random.default_rng(seed).normal(0, 0.012, len(DAYS) - 1)]))


def as_row(a):
    return torch.tensor(np.asarray(a)[None], dtype=torch.float32)


def test_policy_sees_exactly_the_history():
    seen = []

    def spy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
        seen.append((t, len(S), len(vix), T))
        return 0.0

    run_trade(DAYS, path(), np.zeros(len(DAYS)), 0.2, 0.0, 0.0, hedge=spy, vix=np.full(len(DAYS), 20.0))
    assert [(t, n, m) for t, n, m, _ in seen] == [(t, t + 1, t + 1) for t in range(len(DAYS) - 1)]
    assert all(T == pytest.approx(29 / 365) for *_, T in seen)


def test_every_k_only_rebalances_on_schedule():
    args = (DAYS, path(8), np.zeros(len(DAYS)), 0.2, 0.0, 0.0)
    daily = run_trade(*args, hedge=St.implied_delta)["holdings"]
    weekly = run_trade(*args, hedge=St.every(5, St.implied_delta))["holdings"]
    for t in range(len(weekly)):
        assert weekly[t] == pytest.approx(daily[t - t % 5])


def test_run_all_options_daily_entries_and_vol_column():
    df = gbm_frame(years=1).assign(vix9d=np.nan, vix3m=25.0)
    df.loc[df.index[100]:, "vix9d"] = 15.0
    monthly, daily = run_all(df, start="2000-01-01"), run_all(df, start="2000-01-01", entries="daily")
    assert len(daily) > 15 * len(monthly)
    nine = run_all(df, days=9, vol_col="vix9d", start="2000-01-01")
    assert (nine.entry >= df.index[100]).all() and np.allclose(nine.sigma, 0.15)
    assert np.allclose(run_all(df, days=93, vol_col="vix3m", start="2000-01-01").sigma, 0.25)


@pytest.mark.parametrize("make_policy", [
    lambda: St.implied_delta,
    lambda: St.forecast_delta,  # given forecast_vol = the implied vol below
    lambda: St.leland(0.0),
    lambda: St.whalley_wilmott(1e-18, 1.0),  # band ~ cost^(1/3): 1e-18 -> ~5e-6 shares
    lambda: St.skew_delta(0.0),  # no spot-vol correlation -> plain delta
])
def test_policies_reduce_to_implied_delta(make_policy):
    args = (DAYS, path(1), np.zeros(len(DAYS)), 0.2, 0.01, 0.005)
    ref = run_trade(*args)
    got = run_trade(*args, hedge=make_policy(), ctx={"forecast_vol": 0.2})
    np.testing.assert_allclose(got["holdings"], ref["holdings"], atol=1e-4)
    assert got["total"] == pytest.approx(ref["total"], abs=1e-3)


def test_skew_delta_holds_less_when_vol_rises_on_selloffs():
    args = (DAYS, path(7), np.zeros(len(DAYS)), 0.2, 0.0, 0.0)
    plain = run_trade(*args)["holdings"]
    skewed = run_trade(*args, hedge=St.skew_delta(-0.5))["holdings"]
    assert (skewed < plain).all()


def test_wider_band_trades_less():
    df = gbm_frame(years=3)
    tight = run_all(df, hedge=St.whalley_wilmott(2e-4, 100.0), start="2000-01-01")
    loose = run_all(df, hedge=St.whalley_wilmott(2e-4, 0.1), start="2000-01-01")
    assert loose.stock_cost.sum() < tight.stock_cost.sum()


def test_torch_training_pnl_matches_the_engine():
    S = path(2)
    ref = run_trade(DAYS, S, np.zeros(len(DAYS)), 0.2, 0.0, 0.0, cost=3e-4)
    tau = (DAYS[-1] - DAYS[:-1]) / 365
    pnl = St.straddle_pnl(as_row(S), as_row(ref["holdings"]), torch.tensor([0.2]), as_row(tau), 3e-4)
    assert pnl.item() * S[0] == pytest.approx(ref["total"], rel=1e-4)


def test_deep_policy_wrapper_matches_training_forward():
    """The real-trade wrapper must feed the network exactly the features it was trained on."""
    torch.manual_seed(0)
    model = St.StraddleHedger()
    S, vix = path(3), 20 + np.random.default_rng(4).normal(0, 2, len(DAYS))
    tau = (DAYS[-1] - DAYS[:-1]) / 365
    with torch.no_grad():
        trained = model(as_row(S), as_row(vix), as_row(tau), torch.tensor([0.2]))[0].numpy()
    live = run_trade(DAYS, S, np.zeros(len(DAYS)), 0.2, 0.0, 0.0, hedge=St.deep_policy([model]), vix=vix)["holdings"]
    np.testing.assert_allclose(live, trained, atol=1e-5)


def test_history_windows_end_before_cutoff():
    df = gbm_frame(years=2)
    cutoff = df.index[300]
    w = St.history_windows(df, cutoff)
    last_start = w["S"].shape[0] - 1
    assert df.index[last_start + St.STEPS] < cutoff
    assert torch.allclose(w["S"][:, 0], torch.ones(w["S"].shape[0]))


def test_heston_paths_start_at_the_model_vix():
    pool, params = np.array([0.01, 0.09]), (3.0, 0.04, 1.0, -0.7)
    d = St.heston_paths(500, params, v0_pool=pool, seed=1)
    assert d["S"].shape == (500, St.STEPS + 1) and d["tau"].shape == (500, St.STEPS)
    expected = np.sqrt(heston.avg_variance(pool, params[0], params[1], St.T30)[:, 0])
    np.testing.assert_allclose(np.unique(d["sigma"].numpy().round(5)), expected.round(5), atol=1e-5)


def test_har_monthly_forecasts_ignore_the_future():
    df = gbm_frame(years=6, seed=5)
    dates = list(df.index[600:1200:40])
    k = df.index[1000]
    df2 = df.copy()
    after = df2.index > k
    df2.loc[after, "close"] *= np.random.default_rng(6).uniform(0.8, 1.2, after.sum())
    df2.loc[after, "vix"] = 45.0
    a, b = har_monthly(df, dates), har_monthly(df2, dates)
    early, late = [d for d in dates if d <= k], [d for d in dates if d > k]
    pd.testing.assert_series_equal(a.loc[early], b.loc[early])
    assert not np.allclose(a.loc[late], b.loc[late])
