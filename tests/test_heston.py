import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from hedgelab import heston as H
from hedgelab.pricing import bs_price

S, R, Q = 100.0, 0.03, 0.01
P = (0.04, 1.5, 0.05, 0.8, -0.7)  # v0, kappa, theta, xi, rho


def quad_call(K, T, p):
    """Independent reference: adaptive quadrature on the same characteristic function."""
    def prob(j):
        def f(u):
            phi = H.char_fn(u - 1j, T, R, Q, *p) / np.exp((R - Q) * T) if j == 1 else H.char_fn(u, T, R, Q, *p)
            return (np.exp(-1j * u * np.log(K / S)) * phi / (1j * u)).real
        return 0.5 + quad(f, 1e-10, np.inf, limit=2000)[0] / np.pi
    return S * np.exp(-Q * T) * prob(1) - K * np.exp(-R * T) * prob(2)


@pytest.mark.parametrize("T", [9 / 365, 1.0])
def test_fourier_grid_matches_adaptive_quadrature(T):
    Ks = np.array([80.0, 100.0, 125.0])
    np.testing.assert_allclose(H.price(S, Ks, T, R, Q, *P), [quad_call(k, T, P) for k in Ks], atol=1e-4)


def test_reduces_to_black_scholes_without_vol_of_vol():
    assert H.price(S, 100.0, 0.5, R, Q, 0.04, 2.0, 0.04, 1e-4, 0.0)[0] == pytest.approx(
        bs_price(S, 100.0, 0.5, R, 0.2, Q), abs=1e-6)


def test_put_and_call_both_match_monte_carlo():
    T = 0.5
    St, _ = H.simulate(200_000, 100, T, S, R, Q, *P, seed=1)
    for call, payoff in [(True, np.maximum(St[:, -1] - 95, 0)), (False, np.maximum(95 - St[:, -1], 0))]:
        disc = np.exp(-R * T) * payoff
        assert abs(disc.mean() - H.price(S, 95.0, T, R, Q, *P, call=call)[0]) < 3 * disc.std() / np.sqrt(len(disc))


def test_monte_carlo_moments():
    T, (v0, kappa, theta, *_), n = 1.0, P, 100_000
    St, v = H.simulate(n, 50, T, S, R, Q, *P, seed=2)
    fwd = S * np.exp((R - Q) * T)
    assert abs(St[:, -1].mean() - fwd) < 3 * St[:, -1].std() / np.sqrt(n)
    assert v[:, -1].mean() == pytest.approx(theta + (v0 - theta) * np.exp(-kappa * T), rel=0.01)
    assert (v >= 0).all()


def test_correlation_sign_sets_skew_direction():
    Ks = np.array([85.0, 115.0])
    down = H.smile(S, Ks, 0.25, R, Q, P)
    up = H.smile(S, Ks, 0.25, R, Q, (*P[:4], 0.7))
    assert down[0] > down[1] and up[0] < up[1]


def test_term_structure_fit_recovers_kappa_theta_and_daily_variance():
    rng = np.random.default_rng(3)
    taus = np.array([9, 30, 93, 183]) / 365
    v = rng.uniform(0.01, 0.12, 500)
    levels = np.sqrt(H.avg_variance(v, 2.5, 0.045, taus)) * 100 + rng.normal(0, 0.05, (500, 4))
    fit = H.fit_term_structure(levels, taus)
    assert fit["kappa"] == pytest.approx(2.5, rel=0.05) and fit["theta"] == pytest.approx(0.045, rel=0.03)
    assert np.corrcoef(fit["v"], v)[0, 1] > 0.999 and (fit["rmse_vol_points"] < 0.1).all()


def test_rho_xi_recovered_from_a_simulated_history():
    steps = 20_000
    St, v = H.simulate(1, steps, steps / 365, S, 0.0, 0.0, 0.04, 3.0, 0.04, 0.5, -0.6, seed=4)
    rho, xi = H.fit_rho_xi(St[0], v[0], np.arange(steps + 1.0))
    assert rho == pytest.approx(-0.6, abs=0.05) and xi == pytest.approx(0.5, rel=0.1)


def test_implied_carry_recovers_dividend_yield_from_parity():
    rows = []
    for T, q_true in [(30 / 365, 0.012), (90 / 365, 0.018)]:
        for K in [95.0, 100.0, 105.0]:
            for call in (True, False):
                rows.append({"T": T, "K": K, "call": call, "mid": bs_price(S, K, T, R, 0.2, q_true, call)})
    carry = H.implied_carry(pd.DataFrame(rows), S, R)
    assert carry[30 / 365] == pytest.approx(0.012, abs=1e-10) and carry[90 / 365] == pytest.approx(0.018, abs=1e-10)


def test_chain_fit_recovers_parameters():
    rows = []
    for T in [20 / 365, 60 / 365, 150 / 365]:
        Ks = np.linspace(85, 110, 11)
        calls = Ks >= S  # out-of-the-money side, as a market fit would use
        for K, c, mid in zip(Ks, calls, H.price(S, Ks, T, R, Q, *P, call=calls), strict=True):
            rows.append({"K": K, "T": T, "mid": mid, "call": bool(c)})
    quotes = H.market_quotes(pd.DataFrame(rows), S, R, Q)
    params, rmse = H.fit_chain(quotes, S, R, Q)
    assert rmse < 0.01  # vol points
    np.testing.assert_allclose(params, P, rtol=0.1, atol=0.01)
