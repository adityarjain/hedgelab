"""Heston (1993) stochastic volatility: Fourier pricing, QE Monte Carlo, and three calibrations.

    dS/S = (r - q) dt + sqrt(v) dW1,    dv = kappa (theta - v) dt + xi sqrt(v) dW2,    d<W1, W2> = rho dt

Units as in pricing.py: decimals, T in years. v is a variance (0.04 = 20% vol). Parameter order: PARAMS.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from hedgelab.pricing import bs_greeks, implied_vol

PARAMS = ("v0", "kappa", "theta", "xi", "rho")
BOUNDS = ([1e-4, 0.05, 1e-4, 0.01, -0.99], [1.0, 20.0, 1.0, 5.0, 0.5])


# ---------- pricing ----------
def char_fn(u, T, r, q, v0, kappa, theta, xi, rho):
    """E[exp(i u ln(S_T / S_0))], 'little trap' form (Albrecher et al., 2007): no branch-cut jumps for long T."""
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (iu + u**2))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (r - q) * iu * T + kappa * theta / xi**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (b - d) / xi**2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


def price(S, K, T, r, q, v0, kappa, theta, xi, rho, call=True, n=4096):
    """Heston price for one maturity, vectorised over strikes K. call may be a bool or a bool array like K.

    P_j = 1/2 + 1/pi * int_0^inf Re[e^{-iu ln(K/S)} phi_j(u) / (iu)] du, trapezoid on a grid whose length
    scales with 1/sqrt(v T): the integrand decays like exp(-u^2 v T / 2), slowly for short-dated options.
    """
    K = np.atleast_1d(np.asarray(K, float))
    v_low = max(min(v0, theta), 1e-4)
    U = min(max(200.0, np.sqrt(80 / (v_low * T))), 20_000.0)
    u = np.linspace(1e-8, U, n)[:, None]
    k = np.log(K / S)[None, :]
    phi2 = char_fn(u, T, r, q, v0, kappa, theta, xi, rho)
    phi1 = char_fn(u - 1j, T, r, q, v0, kappa, theta, xi, rho) / np.exp((r - q) * T)
    rot = np.exp(-1j * u * k) / (1j * u)
    P1 = 0.5 + np.trapezoid((rot * phi1).real, u[:, 0], axis=0) / np.pi
    P2 = 0.5 + np.trapezoid((rot * phi2).real, u[:, 0], axis=0) / np.pi
    c = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    p = c - S * np.exp(-q * T) + K * np.exp(-r * T)
    return np.where(call, c, p)


def smile(S, K, T, r, q, params, call=True):
    """Black-Scholes implied vols of Heston prices (NaN where a price breaks no-arbitrage bounds)."""
    prices = price(S, K, T, r, q, *params, call=call)
    calls = np.broadcast_to(call, prices.shape)
    out = []
    for p, k, c in zip(prices, np.atleast_1d(K), calls, strict=True):
        try:
            out.append(implied_vol(p, S, k, T, r, q, c))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


# ---------- simulation ----------
def simulate(n, steps, T, S0, r, q, v0, kappa, theta, xi, rho, seed=0):
    """Andersen (2008) QE scheme for v, his central discretisation for ln S. Returns S, v, each (n, steps+1).
    v0 may be a scalar or one starting variance per path."""
    rng = np.random.default_rng(seed)
    dt = T / steps
    e = np.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = k4 = 0.5 * dt * (1 - rho**2)
    lnS, v = np.full(n, np.log(S0)), np.broadcast_to(np.asarray(v0, float), (n,)).copy()
    S_out, v_out = [np.exp(lnS)], [v]
    for _ in range(steps):
        m = theta + (v - theta) * e
        s2 = v * xi**2 * e * (1 - e) / kappa + theta * xi**2 * (1 - e) ** 2 / (2 * kappa)
        psi = s2 / m**2
        z, uni = rng.standard_normal(n), rng.random(n)
        # quadratic branch (psi <= 1.5) and exponential branch, computed for all paths then selected
        b2 = np.maximum(2 / psi - 1 + np.sqrt(2 / psi) * np.sqrt(np.maximum(2 / psi - 1, 0)), 0)
        quad = m / (1 + b2) * (np.sqrt(b2) + z) ** 2
        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / m
        expo = np.where(uni <= p, 0.0, np.log(np.maximum(1 - p, 1e-300) / np.maximum(1 - uni, 1e-300)) / beta)
        v_new = np.where(psi <= 1.5, quad, expo)
        lnS = (lnS + (r - q) * dt + k0 + k1 * v + k2 * v_new
               + np.sqrt(np.maximum(k3 * v + k4 * v_new, 0)) * rng.standard_normal(n))
        v = v_new
        S_out.append(np.exp(lnS))
        v_out.append(v)
    return np.stack(S_out, 1), np.stack(v_out, 1)


# ---------- calibration 1: VIX term structure -> kappa, theta, daily v ----------
def avg_variance(v, kappa, theta, tau):
    """Risk-neutral expected average variance over the next tau years: the VIX^2 of tenor tau under Heston."""
    w = (1 - np.exp(-kappa * tau)) / (kappa * tau)
    return theta + (np.asarray(v)[..., None] - theta) * w


def fit_term_structure(levels, taus):
    """levels: (days, tenors) array of vol indexes in vol points (e.g. VIX9D, VIX, VIX3M, VIX6M), no NaNs.

    Global kappa, theta; each day's v is profiled out by closed-form least squares.
    Returns dict(kappa, theta, v (days,), rmse_vol_points (tenors,)).
    """
    Y, taus = (np.asarray(levels, float) / 100) ** 2, np.asarray(taus, float)

    def solve(log_p):
        kappa, theta = np.exp(log_p)
        w = (1 - np.exp(-kappa * taus)) / (kappa * taus)
        v = (Y - theta * (1 - w)) @ w / (w @ w)
        return kappa, theta, v, Y - (theta * (1 - w) + v[:, None] * w)

    best = minimize(lambda lp: np.sum(solve(lp)[3] ** 2), np.log([2.0, 0.04]), method="Nelder-Mead",
                    options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 2000})
    kappa, theta, v, _ = solve(best.x)
    fitted_vol = np.sqrt(np.maximum(avg_variance(v, kappa, theta, taus), 0)) * 100
    rmse = np.sqrt(np.mean((fitted_vol - np.asarray(levels, float)) ** 2, axis=0))
    return {"kappa": kappa, "theta": theta, "v": v, "rmse_vol_points": rmse}


# ---------- calibration 2: history -> rho, xi ----------
def fit_rho_xi(close, v, days):
    """From daily closes, a daily variance series v, and calendar day numbers: rho = corr(log return, dv) and
    xi from E[dv^2] = xi^2 v dt. These are physical-measure estimates (see README limitations)."""
    ret, dv = np.diff(np.log(close)), np.diff(v)
    dt = np.diff(days) / 365
    xi = np.sqrt(np.sum(dv**2) / np.sum(np.maximum(v[:-1], 1e-6) * dt))
    return float(np.corrcoef(ret, dv)[0, 1]), float(xi)


# ---------- calibration 3: one option chain -> all five parameters ----------
def implied_carry(chain, S, r, n_strikes=3):
    """Per-expiry dividend yield implied by put-call parity, C - P = S e^{-qT} - K e^{-rT}, from the
    n_strikes closest-to-spot strikes quoted on both sides (median). Returns {T: q}."""
    out = {}
    for T, g in chain.groupby("T"):
        pairs = g.pivot_table(index="K", columns="call", values="mid").dropna()
        if pairs.empty:
            continue
        pairs = pairs.iloc[np.argsort(np.abs(pairs.index - S))[:n_strikes]]
        disc_fwd = pairs[True] - pairs[False] + pairs.index * np.exp(-r * T)  # = S e^{-qT}
        out[T] = float(-np.log(np.median(disc_fwd) / S) / T)
    return out


def market_quotes(chain, S, r, q):
    """Add market implied vol and vega to a chain (columns K, T, mid, call; optional per-row q, else the scalar
    q); drop quotes that break no-arbitrage bounds (implied_vol raises)."""
    rows = []
    for row in chain.itertuples(index=False):
        qi = getattr(row, "q", q)
        try:
            iv = implied_vol(row.mid, S, row.K, row.T, r, qi, row.call)
        except ValueError:
            continue
        rows.append({**row._asdict(), "iv": iv, "vega": bs_greeks(S, row.K, row.T, r, iv, qi, row.call)["vega"]})
    return pd.DataFrame(rows, columns=[*chain.columns, "iv", "vega"])


def fit_chain(quotes, S, r, q, starts=((0.03, 2.0, 0.04, 0.6, -0.7), (0.02, 5.0, 0.05, 1.2, -0.8),
                                        (0.04, 1.0, 0.03, 0.4, -0.5))):
    """quotes: market_quotes() output (a per-row q column, constant within each expiry, overrides q).
    Minimises vega-weighted price errors (~ implied-vol errors) from several starting points.
    Returns (params array in PARAMS order, rmse in vol points)."""
    groups = [(T, g.q.iloc[0] if "q" in g else q, g.K.to_numpy(), g.call.to_numpy(), g.mid.to_numpy(),
               g.vega.to_numpy()) for T, g in quotes.groupby("T")]

    def resid(p):
        return np.concatenate([(price(S, K, T, r, qT, *p, call=c) - mid) / vega
                               for T, qT, K, c, mid, vega in groups])

    fits = [least_squares(resid, x0, bounds=BOUNDS, x_scale="jac", max_nfev=300) for x0 in starts]
    best = min(fits, key=lambda f: f.cost)
    return best.x, float(np.sqrt(np.mean(best.fun**2)) * 100)
