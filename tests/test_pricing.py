import numpy as np
import pytest

from hedgelab.pricing import (binomial_price, bs_greeks, bs_price, geometric_asian_price,
                           implied_vol, mc_asian, mc_european)

A = dict(S=100, K=105, T=0.75, r=0.04, sigma=0.25, q=0.01)


def test_bs_known_value():
    # textbook: S=K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert bs_price(100, 100, 1, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)


def test_put_call_parity():
    c, p = bs_price(**A, call=True), bs_price(**A, call=False)
    rhs = A["S"] * np.exp(-A["q"] * A["T"]) - A["K"] * np.exp(-A["r"] * A["T"])
    assert c - p == pytest.approx(rhs)


def test_greeks_match_finite_differences():
    h = 1e-4
    g = bs_greeks(**A)
    up, dn = dict(A, S=A["S"] + h), dict(A, S=A["S"] - h)
    assert g["delta"] == pytest.approx((bs_price(**up) - bs_price(**dn)) / (2 * h), rel=1e-5)
    vu, vd = dict(A, sigma=A["sigma"] + h), dict(A, sigma=A["sigma"] - h)
    assert g["vega"] == pytest.approx((bs_price(**vu) - bs_price(**vd)) / (2 * h), rel=1e-5)


def test_implied_vol_roundtrip():
    assert implied_vol(bs_price(**A), **{k: v for k, v in A.items() if k != "sigma"}) == pytest.approx(A["sigma"])


def test_binomial_converges_to_bs():
    assert binomial_price(**A, steps=1000) == pytest.approx(bs_price(**A), abs=0.01)


def test_american_put_at_least_european():
    eu = binomial_price(**A, call=False)
    am = binomial_price(**A, call=False, american=True)
    assert am >= eu


def test_mc_within_3_standard_errors_of_bs():
    price, se = mc_european(**A, n=400_000, seed=1)
    assert abs(price - bs_price(**A)) < 3 * se


def test_asian_cheaper_than_european():
    # averaging lowers volatility, so the Asian call must be cheaper
    asian, _ = mc_asian(**A, seed=2)
    assert asian < bs_price(**A)


def test_geometric_asian_with_one_fixing_is_european():
    assert geometric_asian_price(**A, steps=1) == pytest.approx(bs_price(**A))


def test_control_variate_agrees_and_cuts_error():
    plain, se_plain = mc_asian(**A, n=50_000, seed=3, control=False)
    cv, se_cv = mc_asian(**A, n=50_000, seed=3, control=True)
    assert abs(cv - plain) < 3 * se_plain
    assert se_cv < se_plain / 5  # in practice ~20-30x smaller
