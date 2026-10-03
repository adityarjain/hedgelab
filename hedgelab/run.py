"""Reproduce the results in results/. Usage: python -m hedgelab.run baseline"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hedgelab import data, heston, stats, trades

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


TENORS = {"vix9d": 9, "vix": 30, "vix3m": 93, "vix6m": 183}  # calendar days


def heston_results():
    # 1) VIX term structure -> kappa, theta, daily v (risk-neutral); 2) history -> rho, xi (physical)
    df = data.load()
    lv = df[list(TENORS)].dropna()
    ts = heston.fit_term_structure(lv.to_numpy(), np.array(list(TENORS.values())) / 365)
    days = (lv.index - lv.index[0]).days.to_numpy()
    rho, xi = heston.fit_rho_xi(df.close.loc[lv.index].to_numpy(), ts["v"], days)
    ts_rows = {"start": lv.index[0].date(), "end": lv.index[-1].date(), "days": len(lv), "kappa": ts["kappa"],
               "theta": ts["theta"], "long_run_vol": np.sqrt(ts["theta"]), "rho": rho, "xi": xi,
               **{f"rmse_vol_points_{k}": e for k, e in zip(TENORS, ts["rmse_vol_points"], strict=True)}}
    pd.Series(ts_rows).rename("value").to_csv(RESULTS / "heston_timeseries.csv", index_label="parameter")

    # 3) one live option chain -> all five parameters
    snap = data.option_chain_snapshot()
    S, r, q, vix = (float(snap[c].iloc[0]) for c in ["spot", "rate", "div_yield", "vix"])
    carry = heston.implied_carry(snap, S, r)  # per-expiry forward from put-call parity, not trailing dividends
    snap["q"] = snap["T"].map(carry).fillna(q)
    quotes = heston.market_quotes(snap.loc[snap.otm, ["K", "T", "mid", "call", "expiry", "q"]], S, r, q)
    params, rmse = heston.fit_chain(quotes, S, r, q)
    quotes["model_iv"] = np.nan
    for T, g in quotes.groupby("T"):  # assign by index: groupby order is not the row order
        quotes.loc[g.index, "model_iv"] = heston.smile(S, g.K.to_numpy(), T, r, g.q.iloc[0], params,
                                                       g.call.to_numpy())
    by_exp = []
    for (expiry, T), g in quotes.groupby(["expiry", "T"]):
        g = g.sort_values("K")
        by_exp.append({"expiry": pd.Timestamp(expiry).date(), "T": T, "quotes": len(g), "implied_q": g.q.iloc[0],
                       "rmse_vol_points": np.sqrt(np.mean((g.model_iv - g.iv) ** 2)) * 100,
                       "atm_iv_market": np.interp(S, g.K, g.iv), "atm_iv_model": np.interp(S, g.K, g.model_iv)})
    by_exp = pd.DataFrame(by_exp)
    by_exp.to_csv(RESULTS / "heston_chain_by_expiry.csv", index=False, float_format="%.6g")
    # 30-day ATM implied vol: interpolate total variance iv^2 T across expiries, compare with VIX
    t30 = 30 / 365
    iv30 = np.sqrt(np.interp(t30, by_exp["T"], by_exp.atm_iv_market**2 * by_exp["T"]) / t30)
    v0, kappa, theta, xi_q, rho_q = params
    chain_row = {"asof": pd.Timestamp(snap["asof"].iloc[0]).date(), "spot": S, "rate": r,
                 "trailing_div_yield": q, "implied_q_min": min(carry.values()), "implied_q_max": max(carry.values()),
                 "quotes": len(quotes), "v0": v0, "kappa": kappa, "theta": theta, "xi": xi_q, "rho": rho_q,
                 "rmse_vol_points": rmse, "feller_2kt_minus_xi2": 2 * kappa * theta - xi_q**2,
                 "vix": vix, "atm_iv_30d": iv30, "atm_iv_30d_over_vix": iv30 / (vix / 100)}
    pd.Series(chain_row).rename("value").to_csv(RESULTS / "heston_chain_fit.csv", index_label="parameter")

    plot_smiles(quotes, S, RESULTS / "heston_smile.png", rmse)
    print(pd.Series(ts_rows).to_string())
    print(pd.Series(chain_row).to_string())
    print(by_exp.round(4).to_string(index=False))


def plot_smiles(quotes, S, path, rmse):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exps = quotes.groupby("expiry")["T"].first().sort_values()
    picks = [exps.index[np.argmin(np.abs(exps.to_numpy() - d / 365))] for d in (20, 45, 90, 180)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharey=True, facecolor="#fcfcfb")
    for ax, expiry in zip(axes.flat, dict.fromkeys(picks), strict=False):
        g = quotes[quotes.expiry == expiry].sort_values("K")
        ax.set_facecolor("#fcfcfb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.plot(g.K / S, g.model_iv * 100, color=ORANGE, lw=2, label="Heston fit", zorder=2)
        ax.scatter(g.K / S, g.iv * 100, s=18, color=BLUE, edgecolor="#fcfcfb", lw=0.8, label="market (mid)", zorder=3)
        ax.set_title(f"{pd.Timestamp(expiry).date()} ({round(g['T'].iloc[0] * 365)} days)", loc="left",
                     color=INK, fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("strike / spot", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("implied vol (%)", color=MUTED, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"SPY implied-vol smile vs one Heston fit across all expiries (RMSE {rmse:.2f} vol pts)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    {"baseline": baseline, "heston": heston_results}[sys.argv[1] if len(sys.argv) > 1 else "baseline"]()
