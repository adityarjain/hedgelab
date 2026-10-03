"""Reproduce the results in results/. Usage: python -m hedgelab.run baseline"""
import sys
from pathlib import Path

import pandas as pd

from hedgelab import data, stats, trades

RESULTS = Path(__file__).resolve().parent.parent / "results"
PERIODS = {"2008-09 financial crisis": ("2008-09-01", "2009-06-30"), "2020 Covid crash": ("2020-02-01", "2020-06-30"),
           "2022 bear market": ("2022-01-01", "2022-12-31")}
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (dataviz reference palette, validated)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def per_100(t):
    """P&L per $100 of SPY notional at entry."""
    return t.total / t.S0 * 100


def summarize(t, label):
    p = per_100(t).to_numpy()
    mean, mlo, mhi = stats.mean_ci(p)
    sharpe, slo, shi = stats.sharpe_ci(p)
    return {"strategy": label, "trades": len(p), "mean_per_100": mean, "mean_lo": mlo, "mean_hi": mhi,
            "annual_per_100": mean * 12, "sharpe": sharpe, "sharpe_lo": slo, "sharpe_hi": shi,
            "nw_se": stats.newey_west_se(p), "std": p.std(ddof=1), "worst": p.min(), "win_rate": (p > 0).mean(),
            "worst_month_share_of_total": p.min() / p.sum()}


def baseline():
    df = data.load()
    runs = {"delta-hedged (2bp stock cost)": trades.run_all(df),
            "delta-hedged (no costs)": trades.run_all(df, cost=0.0),
            "unhedged": trades.run_all(df, hedge=False)}
    hedged = runs["delta-hedged (2bp stock cost)"]

    out = hedged.assign(**{f"{c}_per_100": hedged[c] / hedged.S0 * 100
                           for c in ["total", "vrp", "residual", "stock_cost", "premium"]})
    out.to_csv(RESULTS / "baseline_trades.csv", index=False, float_format="%.6g")
    summary = pd.DataFrame([summarize(t, k) for k, t in runs.items()])
    decomp = {f"{c}_share": hedged[c].sum() / hedged.total.sum() for c in ["vrp", "residual", "stock_cost"]}
    summary.loc[0, list(decomp)] = list(decomp.values())
    summary.to_csv(RESULTS / "baseline_summary.csv", index=False, float_format="%.6g")

    rows = []
    in_crisis = pd.Series(False, index=hedged.index)
    for name, (a, b) in PERIODS.items():
        m = hedged.entry.between(a, b)
        in_crisis |= m
        rows.append({"period": name, "trades": int(m.sum()), "mean_per_100": per_100(hedged[m]).mean(),
                     "sum_per_100": per_100(hedged[m]).sum()})
    rows.append({"period": "all other months", "trades": int((~in_crisis).sum()),
                 "mean_per_100": per_100(hedged[~in_crisis]).mean(), "sum_per_100": per_100(hedged[~in_crisis]).sum()})
    pd.DataFrame(rows).to_csv(RESULTS / "baseline_periods.csv", index=False, float_format="%.6g")

    plot(runs, RESULTS / "baseline_pnl.png")
    print(summary[["strategy", "mean_per_100", "mean_lo", "mean_hi", "sharpe", "sharpe_lo", "sharpe_hi",
                   "worst", "win_rate"]].round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print({k: round(v, 3) for k, v in decomp.items()})


def plot(runs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hedged, unhedged = runs["delta-hedged (2bp stock cost)"], runs["unhedged"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 2], facecolor="#fcfcfb")
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    for name, (a, b) in PERIODS.items():
        a1.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f0efec", lw=0)
        a1.text(pd.Timestamp(a), 1.0, name.split(" ", 1)[0], transform=a1.get_xaxis_transform(), fontsize=8,
                color=MUTED, va="top")
    for t, color, label in [(hedged, BLUE, "delta-hedged"), (unhedged, ORANGE, "unhedged")]:
        cum = per_100(t).cumsum()
        a1.plot(t.settle, cum, color=color, lw=2, label=label)
        a1.text(t.settle.iloc[-1], cum.iloc[-1], f"  {label}", color=INK, fontsize=9, va="center")
    a1.set_ylabel("cumulative P&L per $100 notional", color=MUTED, fontsize=9)
    a1.set_title("Selling a 30-day SPY straddle every month, 2005-2025", loc="left", color=INK, fontsize=12)
    a1.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, 0.93))
    a1.set_xlim(right=hedged.settle.iloc[-1] + pd.Timedelta(days=900))  # room for the end labels
    a1.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2006, hedged.settle.dt.year.max() + 1, 2)],
                  [str(y) for y in range(2006, hedged.settle.dt.year.max() + 1, 2)])

    yearly = per_100(hedged).groupby(hedged.settle.dt.year).sum()
    a2.bar(yearly.index, yearly.values, color=BLUE, width=0.75)
    a2.axhline(0, color=MUTED, lw=0.8)
    for yr, v in yearly.items():
        if v < 0:
            a2.text(yr, v, f"{v:.1f}", ha="center", va="top", fontsize=8, color=INK)
    a2.set_ylabel("P&L per $100, by year", color=MUTED, fontsize=9)
    losers = int((yearly < 0).sum())
    note = "losing years labelled" if losers else f"all {len(yearly)} calendar years positive"
    a2.set_title(f"Delta-hedged P&L by calendar year ({note})", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    {"baseline": baseline}[sys.argv[1] if len(sys.argv) > 1 else "baseline"]()
