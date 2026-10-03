"""Deep hedging (Buehler et al., 2019) of a short European call under GBM with proportional costs.

A small network maps (moneyness, time left, current holding) to the next stock holding and is trained to
minimise CVaR of the hedger's terminal loss. Baselines: Black-Scholes delta and Leland (cost-adjusted vol)
delta, both from hedgelab.pricing. r = 0 and the stock has zero drift.

Shapes: paths S are (n, steps+1); holdings h are (n, steps), h[:, t] held over [t, t+1].
"""
import numpy as np
import torch
from torch import nn

from hedgelab.pricing import bs_greeks, bs_price

S0, K, SIGMA, T, STEPS = 100.0, 100.0, 0.2, 30 / 252, 30
PREMIUM = float(bs_price(S0, K, T, 0.0, SIGMA))


def simulate(n, steps=STEPS, sigma=SIGMA, seed=0):
    """GBM paths, shape (n, steps+1), float32."""
    dt = T / steps
    z = np.random.default_rng(seed).standard_normal((n, steps))
    log_s = np.log(S0) + np.cumsum(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z, axis=1)
    return torch.tensor(np.exp(np.hstack([np.full((n, 1), np.log(S0)), log_s])), dtype=torch.float32)


def pnl(S, h, cost):
    """Hedger short one call: premium - payoff + trading gains - cost * |traded notional|.

    Positions open from 0 at t=0 and are unwound at maturity; both legs pay costs.
    """
    zero = torch.zeros_like(h[:, :1])
    trades = torch.diff(torch.cat([zero, h, zero], 1), dim=1)  # at times 0..steps
    gains = (h * torch.diff(S, dim=1)).sum(1)
    payoff = torch.relu(S[:, -1] - K)
    return PREMIUM + gains - payoff - cost * (trades.abs() * S).sum(1)


def cvar(loss, alpha=0.95):
    """Mean of the worst (1 - alpha) share of losses (expected shortfall). Differentiable."""
    k = max(1, int(np.ceil(round((1 - alpha) * len(loss), 9))))  # round: (1-0.95)*100 = 5.000000000000004
    return torch.topk(loss, k).values.mean()


def _tau(steps):
    return T * (steps - np.arange(steps)) / steps  # time to maturity at each rebalance


def delta_hedge(S, sigma=SIGMA):
    """Black-Scholes delta at each rebalance."""
    tau = _tau(S.shape[1] - 1)
    d = bs_greeks(S[:, :-1].double().numpy(), K, tau, 0.0, sigma)["delta"]
    return torch.tensor(d, dtype=torch.float32)


def leland_hedge(S, cost):
    """Delta at Leland's inflated vol: sigma^2 (1 + sqrt(2/pi) * k / (sigma sqrt(dt))), k = round-trip cost."""
    dt = T / (S.shape[1] - 1)
    sig = SIGMA * np.sqrt(1 + np.sqrt(2 / np.pi) * 2 * cost / (SIGMA * np.sqrt(dt)))
    return delta_hedge(S, sig)


class Hedger(nn.Module):
    """Holding at t = f(log(S_t/K) / (sigma sqrt T), tau_t / T, holding at t-1). Sees nothing after t."""

    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, S):
        n, steps = S.shape[0], S.shape[1] - 1
        h, out = torch.zeros(n), []
        for t in range(steps):
            x = torch.stack([torch.log(S[:, t] / K) / (SIGMA * np.sqrt(T)), torch.full((n,), 1 - t / steps), h], 1)
            h = self.net(x).squeeze(1)
            out.append(h)
        return torch.stack(out, 1)


def train(cost, iters=1500, batch=8192, lr=1e-3, alpha=0.95, seed=0):
    """Fresh simulated paths every step (seeds 10**6 + ...), so test paths (other seeds) are never seen."""
    torch.manual_seed(seed)
    model = Hedger()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(iters):
        S = simulate(batch, seed=10**6 + seed * iters + i)
        loss = cvar(-pnl(S, model(S), cost), alpha)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def report(S, h, cost):
    """P&L stats for holdings h on paths S."""
    p = pnl(S, h, cost)
    zero = torch.zeros_like(h[:, :1])
    turnover = torch.diff(torch.cat([zero, h, zero], 1), dim=1).abs().sum(1).mean()
    return {"mean": p.mean().item(), "std": p.std().item(), "CVaR95": cvar(-p).item(), "turnover": turnover.item()}
