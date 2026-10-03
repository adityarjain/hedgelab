"""Event-driven backtester. The strategy only ever sees prices up to the current bar.

Timing: a strategy sees closes through bar t and returns target weights; the broker fills them at the
bar t+1 close (one-bar delay), paying fee + slippage on traded notional. Returning None keeps positions.
"""
import numpy as np
import pandas as pd


def run(prices, strategy, fee_bps=1.0, slip_bps=2.0, capital=1.0):
    """prices: DataFrame (dates x tickers) of closes. Returns (equity Series, turnover).

    turnover = total traded notional / average equity (one-way, over the whole run).
    """
    px = prices.to_numpy(float)
    n, m = px.shape
    cost = (fee_bps + slip_bps) / 1e4
    cash, shares, pending, traded = capital, np.zeros(m), None, 0.0
    equity = np.empty(n)
    for t in range(n):
        if pending is not None:  # fill last bar's decision at today's close
            eq = cash + shares @ px[t]
            target = pending * eq / px[t]
            notional = np.abs(target - shares) @ px[t]
            cash -= (target - shares) @ px[t] + cost * notional
            shares, traded, pending = target, traded + notional, None
        equity[t] = cash + shares @ px[t]
        w = strategy(prices.iloc[: t + 1])  # history only: nothing after bar t is reachable
        pending = None if w is None else np.asarray(w, float)
    equity = pd.Series(equity, index=prices.index)
    return equity, traded / equity.mean()


def metrics(equity, turnover, periods=252):
    """Annualised stats, rf = 0."""
    r = equity.pct_change().dropna()
    return {
        "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (periods / len(r)) - 1,
        "sharpe": r.mean() / r.std() * np.sqrt(periods),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "turnover": turnover,
    }


def buy_hold():
    """Equal weight on day one, never rebalance."""
    return lambda h: np.full(h.shape[1], 1 / h.shape[1]) if len(h) == 1 else None


def momentum(lookback=126):
    """Hold each asset equal-weight while its price is above its price `lookback` bars ago, else cash."""
    last = [None]

    def strat(h):
        if len(h) <= lookback:
            return None
        up = (h.iloc[-1] > h.iloc[-1 - lookback]).to_numpy()
        if last[0] is not None and (up == last[0]).all():
            return None  # signal unchanged: don't pay to rebalance drift
        last[0] = up
        return up / len(up)
    return strat


def pairs(window=60, entry=2.0, exit=0.5):
    """Two assets. Trade the z-score of the log-price spread: fade it past `entry`, close inside `exit`.

    ponytail: hedge ratio fixed at 1 and 50/50 sizing; rolling-OLS hedge ratio if the pair drifts.
    """
    pos = [0]

    def strat(h):
        if len(h) < window:
            return None
        spread = np.log(h.iloc[-window:, 0]) - np.log(h.iloc[-window:, 1])
        z = (spread.iloc[-1] - spread.mean()) / spread.std()
        new = pos[0]
        if pos[0] == 0 and abs(z) > entry:
            new = -int(np.sign(z))  # spread rich -> short A / long B
        elif pos[0] != 0 and abs(z) < exit:
            new = 0
        if new == pos[0]:
            return None
        pos[0] = new
        return np.array([new, -new]) * 0.5
    return strat
