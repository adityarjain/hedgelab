"""Short ATM SPY straddle, sold monthly at VIX-implied vol and delta-hedged daily at the close.

Per trade (all P&L in dollars per share of SPY, i.e. per straddle on one share):
  total      cash account at settlement: premium (less entry cost) + stock trading + interest + dividends
             - stock trading costs - payoff
  vrp        sum over days of 1/2 * Gamma_t * S_t^2 * (sigma^2 * dt_t - ret_t^2): the gamma-weighted gap between
             implied and realized variance, i.e. the variance risk premium actually captured
  residual   total - vrp + stock_cost + entry_cost: discretisation, higher-order terms, dividend/rate carry
So total = vrp + residual - stock_cost - entry_cost exactly.

Timing: the hedge set at close t uses only data through close t (spot, entry vol, rates, trailing dividends).
Time is calendar time (VIX is a 30-calendar-day vol); dt_t = calendar days to the next session / 365.
r and q are fixed at their entry values for pricing and financing (a 30-day trade).
"""
import numpy as np
import pandas as pd

from hedgelab.pricing import bs_greeks, bs_price

TRADE_FIELDS = ["entry", "settle", "S0", "sigma", "premium", "total", "vrp", "residual", "stock_cost",
                "entry_cost", "realized_vol"]


def dividend_yield(df):
    """Trailing 365-day cash dividends / close, using data through each day only."""
    return df.dividend.rolling("365D").sum() / df.close


def entry_dates(index, start="2005-01-01"):
    """First trading day of each month from `start` (default leaves a year to warm up the dividend yield)."""
    idx = index[index >= start]
    return idx.to_series().groupby([idx.year, idx.month]).min().to_list()


def straddle(S, K, tau, r, q, sigma):
    """Value, delta, gamma of one call + one put (long). Vectorised over S and tau (tau > 0)."""
    c, p = (bs_greeks(S, K, tau, r, sigma, q, call) for call in (True, False))
    value = bs_price(S, K, tau, r, sigma, q, True) + bs_price(S, K, tau, r, sigma, q, False)
    return value, c["delta"] + p["delta"], c["gamma"] + p["gamma"]


def run_trade(days, S, div, sigma, r, q, cost=0.0, entry_cost=0.0, hedge=True, vix=None, ctx=None):
    """One short straddle struck at S[0], settled at S[-1].

    days: calendar day numbers of the sessions (entry ... settle); S, div: closes and cash dividends on those
    sessions; cost: stock cost per side as a fraction of traded notional; entry_cost: fraction of premium.
    hedge: True (Black-Scholes delta at the entry implied vol), False (no hedge), or a policy called at each
    close t as policy(t=, S=S[:t+1], tau=, T=, prev=, K=, sigma=, r=, q=, vix=vix[:t+1], ctx=) -> shares.
    The engine slices the histories, so a policy cannot see past close t. ctx: per-trade values known at entry.
    """
    days, S, div = np.asarray(days, float), np.asarray(S, float), np.asarray(div, float)
    n, K = len(S) - 1, S[0]
    tau = (days[-1] - days[:-1]) / 365
    dt = np.diff(days) / 365
    value, delta, gamma = straddle(S[:-1], K, tau, r, q, sigma)
    if callable(hedge):
        vix = None if vix is None else np.asarray(vix, float)
        h, prev = np.empty(n), 0.0
        for t in range(n):
            prev = h[t] = float(hedge(t=t, S=S[: t + 1], tau=tau[t], T=tau[0], prev=prev, K=K, sigma=sigma, r=r,
                                      q=q, vix=None if vix is None else vix[: t + 1], ctx=ctx or {}))
    else:
        h = delta if hedge else np.zeros(n)  # shares held after the close-t rebalance (short straddle -> +delta)

    premium = value[0]
    cash, prev, stock_cost = premium * (1 - entry_cost), 0.0, 0.0
    for t in range(n):
        trade = h[t] - prev
        cash -= trade * S[t]
        stock_cost += cost * abs(trade) * S[t]
        cash = cash * np.exp(r * dt[t]) + h[t] * div[t + 1]  # interest over the gap, dividend at next session
        prev = h[t]
    stock_cost += cost * abs(prev) * S[n]  # unwind
    payoff = abs(S[n] - K)
    total = cash + prev * S[n] - payoff - stock_cost

    ret = S[1:] / S[:-1] - 1
    vrp = 0.5 * gamma * S[:-1] ** 2 * (sigma**2 * dt - ret**2)
    vrp_sum = vrp.sum() if hedge is True else np.nan  # the decomposition assumes the implied-vol delta hedge
    return {"premium": premium, "total": total, "vrp": vrp_sum,
            "residual": total - vrp_sum + stock_cost + premium * entry_cost,
            "stock_cost": stock_cost, "entry_cost": premium * entry_cost,
            "realized_vol": np.sqrt(np.sum(np.log1p(ret) ** 2) / (tau[0])),
            "holdings": h}  # per-session hedge; dropped by run_all, used by the lookahead test


def mark_to_market(df, days=30, vix_scale=1.0, cost=2e-4, entry_cost=0.0, hedge=True, start="2005-01-01", ctx=None):
    """Daily P&L of the monthly strategy, every open trade marked at each close: the straddle at Black-Scholes
    with that day's VIX (x vix_scale), the hedge at the close, cash with interest and dividends, and costs on the
    day they are paid. Per trade, the daily P&L sums exactly to run_all's total (no interest on costs, as there).
    Returns a Series: date -> P&L per $100 of entry notional, summed over the trades open that day."""
    q_all = dividend_yield(df)
    day_num = (df.index - df.index[0]).days.to_numpy()
    pieces = []
    for t0 in entry_dates(df.index, start):
        window = df.loc[t0 : t0 + pd.Timedelta(days=days)]
        if window.index[-1] - t0 < pd.Timedelta(days=days - 4):
            continue
        i0 = df.index.get_loc(t0)
        sl = slice(i0, i0 + len(window))
        dd, S, div = day_num[sl].astype(float), window.close.to_numpy(), window.dividend.to_numpy()
        vix = window.vix.to_numpy()
        sigma, r, q = vix[0] / 100 * vix_scale, df.rate.iloc[i0], q_all.iloc[i0]
        res = run_trade(dd, S, div, sigma, r, q, cost, entry_cost, hedge, vix=vix,
                        ctx={k: s.loc[t0] for k, s in (ctx or {}).items()})
        h, n, K = res["holdings"], len(S) - 1, S[0]
        tau, dt = (dd[-1] - dd[:-1]) / 365, np.diff(dd) / 365
        value = straddle(S[:-1], K, tau, r, q, vix[:-1] / 100 * vix_scale)[0]  # marked at each day's VIX
        book, cash, prev, spent = np.empty(n + 1), res["premium"] * (1 - entry_cost), 0.0, 0.0
        for t in range(n):
            cash -= (h[t] - prev) * S[t]
            spent += cost * abs(h[t] - prev) * S[t]
            book[t] = cash + h[t] * S[t] - value[t] - spent
            cash = cash * np.exp(r * dt[t]) + h[t] * div[t + 1]
            prev = h[t]
        book[n] = cash + prev * S[n] - abs(S[n] - K) - spent - cost * abs(prev) * S[n]
        pieces.append(pd.Series(np.diff(book, prepend=0.0) / K * 100, index=window.index))
    return pd.concat(pieces).groupby(level=0).sum()


def run_all(df, days=30, vix_scale=1.0, cost=2e-4, entry_cost=0.0, hedge=True, start="2005-01-01", ctx=None,
            vol_col="vix", entries="monthly"):
    """One trade per month (entries="monthly") or one starting every session ("daily", overlapping).
    df: the data.load() frame. Returns one row per trade (TRADE_FIELDS).
    hedge: see run_trade. ctx: {name: Series indexed by date}; each trade's policy gets its entry-date values.
    vol_col: the implied-vol index that prices the option (e.g. vix9d for 9-day, vix3m for 93-day trades);
    trades whose entry has no value (before the index launched) are skipped."""
    q_all = dividend_yield(df)
    day_num = (df.index - df.index[0]).days.to_numpy()
    starts = entry_dates(df.index, start) if entries == "monthly" else list(df.index[df.index >= start])
    rows = []
    for t0 in starts:
        window = df.loc[t0 : t0 + pd.Timedelta(days=days)]
        if window.index[-1] - t0 < pd.Timedelta(days=days - 4):  # not enough history left to settle
            continue
        i0 = df.index.get_loc(t0)
        if np.isnan(df[vol_col].iloc[i0]):
            continue
        sl = slice(i0, i0 + len(window))
        sigma = df[vol_col].iloc[i0] / 100 * vix_scale
        res = run_trade(day_num[sl], df.close.iloc[sl], df.dividend.iloc[sl], sigma, df.rate.iloc[i0],
                        q_all.iloc[i0], cost, entry_cost, hedge, vix=df.vix.iloc[sl].to_numpy(),
                        ctx={k: s.loc[t0] for k, s in (ctx or {}).items()})
        rows.append({"entry": t0, "settle": window.index[-1], "S0": df.close.iloc[i0], "sigma": sigma, **res})
    return pd.DataFrame(rows, columns=TRADE_FIELDS)
