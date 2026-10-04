"""Hedging policies for real trades (trades.run_trade(hedge=policy)) and the deep hedger's training.

A policy is called at each close t with data through t only (the engine slices S and vix):
    policy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx) -> shares held over [t, t+1]
sigma is the entry implied vol (the option was sold at it); ctx holds per-trade values known at entry.
Holdings hedge one short straddle on one share, so the Black-Scholes hedge holds +delta.
"""
import numpy as np
import torch
from torch import nn

from hedgelab import heston
from hedgelab.pricing import bs_price
from hedgelab.trades import straddle

STEPS = 21  # trading sessions in a 30-calendar-day trade (training paths)
T30 = 30 / 365


def _greeks(S, K, tau, r, q, sigma):
    _, d, g = straddle(np.array([S]), K, tau, r, q, sigma)
    return d[0], g[0]


# ---------- analytical policies ----------
def implied_delta(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
    """Black-Scholes delta at the entry implied vol, as a policy (same holdings as run_trade(hedge=True))."""
    return _greeks(S[-1], K, tau, r, q, sigma)[0]


def every(k, policy):
    """Rebalance only on every k-th session of the trade (t = 0, k, 2k, ...); otherwise keep the position."""
    def wrapped(t, prev, **kw):
        return policy(t=t, prev=prev, **kw) if t % k == 0 else prev
    return wrapped


def forecast_delta(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
    """Black-Scholes delta at the walk-forward forecast of realized vol, not at the vol the option was sold at."""
    return _greeks(S[-1], K, tau, r, q, ctx["forecast_vol"])[0]


def leland(cost, dt=1 / 252):
    """Delta at Leland's (1985) cost-adjusted vol sigma^2 (1 + sqrt(2/pi) k / (sigma sqrt(dt))), k = round trip."""
    def policy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
        adj = sigma * np.sqrt(1 + np.sqrt(2 / np.pi) * 2 * cost / (sigma * np.sqrt(dt)))
        return _greeks(S[-1], K, tau, r, q, adj)[0]
    return policy


def whalley_wilmott(cost, risk_aversion):
    """No-trade band (Whalley & Wilmott, 1997): hold still while within delta +/- H, else trade to the nearest
    edge. H = (3/2 e^{-r tau} cost S Gamma^2 / gamma)^(1/3), gamma = risk_aversion / S0 (scale-free)."""
    def policy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
        d, g = _greeks(S[-1], K, tau, r, q, sigma)
        H = (1.5 * np.exp(-r * tau) * cost * S[-1] * g**2 / (risk_aversion / S[0])) ** (1 / 3)
        return float(np.clip(prev, d - H, d + H))
    return policy


def vix_beta(df, end):
    """OLS slope of the daily change in VIX/100 on the daily log return, on sessions before `end`."""
    d = df.loc[: end].iloc[:-1]
    x, y = np.log(d.close).diff().to_numpy()[1:], (d.vix / 100).diff().to_numpy()[1:]
    return float(np.polyfit(x, y, 1)[0])


def skew_delta(beta):
    """Minimum-variance delta in the spirit of Hull & White (2017): BS delta + vega * d(sigma)/dS, with implied
    vol assumed to move by `beta` per unit log return (so d sigma / dS = beta / S). In equities beta < 0, so the
    hedge holds fewer shares than BS delta."""
    from hedgelab.pricing import bs_greeks

    def policy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
        d, _ = _greeks(S[-1], K, tau, r, q, sigma)
        vega = sum(bs_greeks(S[-1], K, tau, r, sigma, q, c)["vega"] for c in (True, False))
        return d + vega * beta / S[-1]
    return policy


# ---------- deep hedger ----------
def _norm_cdf(x):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


def _features(logm, tau_frac, delta, vol_ratio, prev):
    return torch.stack([logm, tau_frac, delta, vol_ratio, prev], -1)


class StraddleHedger(nn.Module):
    """Inputs: log(S/K) / (sigma sqrt T), tau / T, BS straddle delta at sigma (r = q = 0), VIX_t / (100 sigma),
    current holding. The delta input lets it learn adjustments to delta rather than delta itself."""

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, S, vix, tau, sigma):
        """S, vix: (n, steps+1) with S[:, 0] = K; tau: (n, steps); sigma: (n,). Returns holdings (n, steps)."""
        T = tau[:, :1]
        s = sigma[:, None]
        logm = torch.log(S[:, :-1] / S[:, :1]) / (s * torch.sqrt(T))
        d1 = (torch.log(S[:, :-1] / S[:, :1]) + 0.5 * s**2 * tau) / (s * torch.sqrt(tau))
        delta = 2 * _norm_cdf(d1) - 1
        ratio = vix[:, :-1] / 100 / s
        h, out = torch.zeros(S.shape[0]), []
        for t in range(tau.shape[1]):
            h = self.net(_features(logm[:, t], tau[:, t] / T[:, 0], delta[:, t], ratio[:, t], h)).squeeze(-1)
            out.append(h)
        return torch.stack(out, 1)


def straddle_pnl(S, h, sigma, tau, cost):
    """Torch P&L of one short straddle (K = S0), r = q = 0, premium at sigma, per unit of S0."""
    T = tau[:, 0]
    premium = torch.as_tensor(bs_price(1.0, 1.0, T.numpy(), 0.0, sigma.numpy(), 0.0, True)
                              + bs_price(1.0, 1.0, T.numpy(), 0.0, sigma.numpy(), 0.0, False), dtype=torch.float32)
    S = S / S[:, :1]
    zero = torch.zeros_like(h[:, :1])
    trades = torch.diff(torch.cat([zero, h, zero], 1), dim=1)
    return (premium + (h * torch.diff(S, dim=1)).sum(1) - torch.abs(S[:, -1] - 1)
            - cost * (trades.abs() * S).sum(1))


def heston_paths(n, params, v0_pool, seed=0):
    """Training paths from a calibrated Heston model. Starting variances are drawn from v0_pool (a history of
    fitted daily variances), so every vol regime appears. VIX along the path = model 30-day expected vol."""
    kappa, theta, xi, rho = params
    rng = np.random.default_rng(seed)
    v0 = rng.choice(v0_pool, n)
    S, v = heston.simulate(n, STEPS, T30, 1.0, 0.0, 0.0, v0, kappa, theta, xi, rho, seed=seed)
    vix = np.sqrt(np.maximum(heston.avg_variance(v, kappa, theta, T30), 1e-8))[..., 0] * 100
    tau = np.broadcast_to(T30 * (1 - np.arange(STEPS) / STEPS), (n, STEPS))
    return _tensors(S, vix, tau, vix[:, 0] / 100)


def history_windows(df, end):
    """Every run of STEPS+1 consecutive sessions that finishes before `end`, normalised to S0 = 1, with the
    actual VIX path and calendar time to the window's last session."""
    close, vix = df.close.to_numpy(), df.vix.to_numpy()
    day = (df.index - df.index[0]).days.to_numpy().astype(float)
    last = int(np.searchsorted(df.index, end)) - STEPS - 1
    starts = np.arange(0, max(last, 0))
    idx = starts[:, None] + np.arange(STEPS + 1)
    S = close[idx] / close[starts, None]
    tau = (day[idx[:, -1:]] - day[idx[:, :-1]]) / 365
    return _tensors(S, vix[idx], tau, vix[starts] / 100)


def _tensors(S, vix, tau, sigma):
    f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)  # noqa: E731
    return {"S": f(S), "vix": f(vix), "tau": f(tau), "sigma": f(sigma)}


def train_deep(data, cost, iters=1500, batch=4096, lr=1e-3, alpha=0.95, seed=0):
    """Minimise CVaR_alpha of the per-S0 loss over mini-batches drawn from `data` (heston_paths / history_windows)."""
    torch.manual_seed(seed)
    model = StraddleHedger()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = data["S"].shape[0]
    gen = torch.Generator().manual_seed(seed)
    k = max(1, int(np.ceil(round((1 - alpha) * batch, 9))))
    for _ in range(iters):
        b = torch.randint(0, n, (batch,), generator=gen)
        S, vix, tau, sigma = (data[key][b] for key in ("S", "vix", "tau", "sigma"))
        loss = torch.topk(-straddle_pnl(S, model(S, vix, tau, sigma), sigma, tau, cost), k).values.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def deep_policy(models):
    """Wrap one or more trained StraddleHedgers (averaged) as a real-trade policy."""
    def policy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx):
        x = np.log(S[-1] / K)
        d1 = (x + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
        delta = 2 * _norm_cdf(torch.tensor(d1, dtype=torch.float32)) - 1
        feats = _features(*(torch.tensor([v], dtype=torch.float32) for v in
                            (x / (sigma * np.sqrt(T)), tau / T)), delta.reshape(1),
                          torch.tensor([vix[-1] / 100 / sigma], dtype=torch.float32),
                          torch.tensor([prev], dtype=torch.float32))
        with torch.no_grad():
            return float(np.mean([m.net(feats).item() for m in models]))
    return policy
