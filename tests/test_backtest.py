import numpy as np
import pandas as pd
import pytest

from hedgelab.backtest import buy_hold, metrics, momentum, pairs, run


def fake_prices(n=400, m=2, seed=0):
    rng = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, m)), axis=0))
    return pd.DataFrame(p, index=pd.bdate_range("2020-01-01", periods=n), columns=list("AB")[:m])


def scrambled_equity_matches(p, make, k=250):
    """Scramble every price after bar k; equity through bar k must be unchanged."""
    base, _ = run(p, make(p))
    p2 = p.copy()
    p2.iloc[k + 1 :] *= np.random.default_rng(9).uniform(0.5, 2.0, p2.iloc[k + 1 :].shape)
    alt, _ = run(p2, make(p2))
    return base.iloc[: k + 1].equals(alt.iloc[: k + 1])


@pytest.mark.parametrize("make", [lambda p: momentum(20), lambda p: pairs(30, 1.0, 0.2)])
def test_no_lookahead(make):
    assert scrambled_equity_matches(fake_prices(), make)


def test_invariance_test_catches_a_two_bar_peek():
    # peeking 2 bars ahead is observable (a 1-bar peek fills at that same bar, so is not; the next test covers it)
    def leaky(full):
        return lambda h: np.array([1.0, 0.0]) if full.iloc[min(len(h) + 1, len(full) - 1), 0] > full.iloc[len(h) - 1, 0] else np.array([0.0, 1.0])
    assert not scrambled_equity_matches(fake_prices(), leaky)


def test_engine_hands_over_exactly_the_history():
    p, seen = fake_prices(n=50), []
    run(p, lambda h: seen.append((len(h), h.index.equals(p.index[: len(h)]))) or None)
    assert seen == [(t + 1, True) for t in range(len(p))]


def test_buy_hold_matches_price_ratio_without_costs():
    p = fake_prices(m=1)
    eq, _ = run(p, buy_hold(), fee_bps=0, slip_bps=0)
    assert eq.iloc[-1] == pytest.approx(p["A"].iloc[-1] / p["A"].iloc[1])  # filled at bar 1


def test_costs_reduce_equity():
    p = fake_prices()
    free, _ = run(p, momentum(20), fee_bps=0, slip_bps=0)
    paid, turnover = run(p, momentum(20), fee_bps=5, slip_bps=5)
    assert paid.iloc[-1] < free.iloc[-1] and turnover > 0


def test_metrics_on_known_curve():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2])
    m = metrics(eq, 0.0)
    assert m["max_drawdown"] == pytest.approx(0.99 / 1.1 - 1)
