"""Train deep hedgers at several cost levels and compare with delta / Leland on fresh paths.

Usage: python run_example.py  (~3-4 minutes on CPU; writes hedging.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hedgelab.hedging import K, SIGMA, STEPS, T, delta_hedge, leland_hedge, pnl, report, simulate, train
from hedgelab.pricing import bs_greeks

COSTS = [0.0, 0.001, 0.005]
S = simulate(100_000, seed=7)  # test paths; training uses seeds >= 10**6

rows, models = {}, {}
for c in COSTS:
    models[c] = train(c)
    with torch.no_grad():
        h_net = models[c](S)
    for name, h in [("no hedge", torch.zeros(len(S), STEPS)), ("BS delta", delta_hedge(S)),
                    ("Leland delta", leland_hedge(S, c)), ("deep hedge", h_net)]:
        rows[(f"{c:.1%}", name)] = report(S, h, c)
table = pd.DataFrame(rows).T
table.index.names = ["cost", "strategy"]
print(table.round(3).to_string())

# --- plots: P&L distribution at the highest cost, and the learned policy vs BS delta at mid-life
c = COSTS[-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
with torch.no_grad():
    for name, h in [("BS delta", delta_hedge(S)), ("Leland delta", leland_hedge(S, c)), ("deep hedge", models[c](S))]:
        a1.hist(pnl(S, h, c).numpy(), bins=150, range=(-6, 2), histtype="step", lw=1.3, label=name)
a1.set(xlabel="hedger P&L", ylabel="paths", title=f"P&L, short ATM call, {c:.1%} cost per trade")
a1.legend()

t = STEPS // 2
grid = torch.linspace(90, 110, 200)
tau = T * (STEPS - t) / STEPS
a2.plot(grid, bs_greeks(grid.double().numpy(), K, tau, 0.0, SIGMA)["delta"], "k--", lw=1.5, label="BS delta")
for prev in (0.2, 0.5, 0.8):
    x = torch.stack([torch.log(grid / K) / (SIGMA * np.sqrt(T)), torch.full_like(grid, 1 - t / STEPS),
                     torch.full_like(grid, prev)], 1)
    with torch.no_grad():
        a2.plot(grid, models[c].net(x).squeeze(1), lw=1.3, label=f"deep hedge, holding {prev}")
a2.set(xlabel="stock price", ylabel="new holding", title=f"Learned policy at day {t} of {STEPS}, {c:.1%} cost")
a2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("hedging.png", dpi=150)
