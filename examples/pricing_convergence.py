"""Plot Monte Carlo standard error vs number of paths for an arithmetic Asian call.

Plain MC error falls as 1/sqrt(n). The geometric-Asian control variate shifts the whole line down.
Run: python convergence.py   (writes convergence.png)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hedgelab.pricing import mc_asian

A = dict(S=100, K=100, T=1, r=0.05, sigma=0.2, steps=64)  # fewer steps keeps the sweep fast
NS = [1_000, 3_000, 10_000, 30_000, 100_000]

plain = [mc_asian(**A, n=n, seed=0, control=False)[1] for n in NS]
cv = [mc_asian(**A, n=n, seed=0, control=True)[1] for n in NS]

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(NS, plain, "o-", label="plain MC")
ax.loglog(NS, cv, "s-", label="geometric control variate")
ax.set(xlabel="paths", ylabel="standard error ($)", title="Asian call: MC standard error vs paths")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("convergence.png", dpi=150)
print("variance reduction at n=100k: %.0fx" % (plain[-1] ** 2 / cv[-1] ** 2))
