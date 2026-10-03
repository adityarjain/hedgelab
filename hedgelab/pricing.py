"""Option pricing: Black-Scholes, Greeks, implied vol, binomial tree, Monte Carlo.

Conventions: S spot, K strike, T years, r risk-free rate, sigma volatility,
q continuous dividend yield. All rates are continuously compounded.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, sigma, q):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, q=0.0, call=True):
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, q=0.0, call=True):
    """Delta, gamma, vega (per 1.00 vol), theta (per year), rho (per 1.00 rate)."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    sgn = 1 if call else -1
    pdf = norm.pdf(d1)
    return {
        "delta": sgn * np.exp(-q * T) * norm.cdf(sgn * d1),
        "gamma": np.exp(-q * T) * pdf / (S * sigma * np.sqrt(T)),
        "vega": S * np.exp(-q * T) * pdf * np.sqrt(T),
        "theta": -S * np.exp(-q * T) * pdf * sigma / (2 * np.sqrt(T))
        - sgn * r * K * np.exp(-r * T) * norm.cdf(sgn * d2)
        + sgn * q * S * np.exp(-q * T) * norm.cdf(sgn * d1),
        "rho": sgn * K * T * np.exp(-r * T) * norm.cdf(sgn * d2),
    }


def implied_vol(price, S, K, T, r, q=0.0, call=True):
    # ponytail: Brent root-find on a fixed bracket; raises ValueError if price violates no-arbitrage bounds
    return brentq(lambda s: bs_price(S, K, T, r, s, q, call) - price, 1e-6, 5.0)


def binomial_price(S, K, T, r, sigma, q=0.0, call=True, american=False, steps=500):
    """Cox-Ross-Rubinstein tree, vectorised backward induction."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    sgn = 1 if call else -1
    j = np.arange(steps + 1)
    V = np.maximum(sgn * (S * u ** (steps - j) * d**j - K), 0)
    for n in range(steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j = np.arange(n + 1)
            V = np.maximum(V, sgn * (S * u ** (n - j) * d**j - K))
    return float(V[0])


def mc_european(S, K, T, r, sigma, q=0.0, call=True, n=200_000, seed=None):
    """Terminal-value Monte Carlo with antithetic variates. Returns (price, std_error)."""
    z = np.random.default_rng(seed).standard_normal(n // 2)
    z = np.concatenate([z, -z])
    ST = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    pay = np.maximum((ST - K) if call else (K - ST), 0) * np.exp(-r * T)
    # antithetic pairs are correlated, so take the SE over pair averages
    pairs = 0.5 * (pay[: n // 2] + pay[n // 2 :])
    return float(pairs.mean()), float(pairs.std(ddof=1) / np.sqrt(len(pairs)))


def geometric_asian_price(S, K, T, r, sigma, q=0.0, call=True, steps=252):
    """Closed form for a discretely monitored geometric-average Asian (fixings at T/steps ... T).

    ln(G) is normal, so this is Black-Scholes on a lognormal with adjusted mean/variance.
    """
    n = steps
    m = np.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d2 = (m - np.log(K)) / np.sqrt(v)
    d1 = d2 + np.sqrt(v)
    fwd = np.exp(m + 0.5 * v)
    if call:
        return np.exp(-r * T) * (fwd * norm.cdf(d1) - K * norm.cdf(d2))
    return np.exp(-r * T) * (K * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def mc_asian(S, K, T, r, sigma, q=0.0, call=True, steps=252, n=100_000, seed=None, control=True):
    """Arithmetic-average Asian option (no closed form). Returns (price, std_error).

    control=True uses the geometric Asian as a control variate: the two payoffs are ~99% correlated
    and the geometric has an exact price, so most of the sampling noise cancels.
    """
    dt = T / steps
    z = np.random.default_rng(seed).standard_normal((n, steps))
    log_paths = np.log(S) + np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1)
    disc = np.exp(-r * T)
    sgn = 1 if call else -1
    pay = np.maximum(sgn * (np.exp(log_paths).mean(axis=1) - K), 0) * disc
    if control:
        geo = np.maximum(sgn * (np.exp(log_paths.mean(axis=1)) - K), 0) * disc
        beta = np.cov(pay, geo)[0, 1] / geo.var(ddof=1)
        pay = pay - beta * (geo - geometric_asian_price(S, K, T, r, sigma, q, call, steps))
    return float(pay.mean()), float(pay.std(ddof=1) / np.sqrt(n))


if __name__ == "__main__":
    a = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)
    print("BS call      ", round(bs_price(**a), 4))
    print("Binomial     ", round(binomial_price(**a), 4))
    print("MC (price,se)", mc_european(**a, seed=0))
    print("Greeks       ", {k: round(v, 4) for k, v in bs_greeks(**a).items()})
    print("Asian MC     ", mc_asian(**a, seed=0))
    print("American put ", round(binomial_price(**a, call=False, american=True), 4))
