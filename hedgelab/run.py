"""Reproduce the results in results/. Usage: python -m hedgelab.run baseline"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hedgelab import data, heston, stats, strategies, trades, volatility

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


TEST_START = "2015-01-01"  # everything tuned or trained uses data before this; results are reported after it
COST = 2e-4
WW_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
SEEDS = (0, 1, 2)


def _metrics(p):
    return {"mean": p.mean(), "std": p.std(ddof=1), "cvar95": stats.cvar(p), "worst": p.min(),
            "win_rate": (p > 0).mean(), "sharpe": p.mean() / p.std(ddof=1) * np.sqrt(12)}


def hedges():
    df = data.load()
    base = trades.run_all(df)
    entries = base.entry.tolist()
    tune = (base.settle < TEST_START).to_numpy()

    # strategy 3: walk-forward 30-day vol forecast at each entry
    forecast = volatility.har_monthly(df, entries)
    # strategy 5: band risk aversion chosen on training trades only, by CVaR95
    ww_cvar = {g: stats.cvar(per_100(trades.run_all(df, hedge=strategies.whalley_wilmott(COST, g)))[tune].to_numpy())
               for g in WW_GRID}
    ww_g = min(ww_cvar, key=ww_cvar.get)
    # strategy 6a: Heston calibrated on 2008-2014 only (VIX9D starts 2011, so three tenors)
    lv = df.loc["2008":"2014", ["vix", "vix3m", "vix6m"]].dropna()
    ts = heston.fit_term_structure(lv.to_numpy(), np.array([30, 93, 183]) / 365)
    rho, xi = heston.fit_rho_xi(df.close.loc[lv.index].to_numpy(), ts["v"], (lv.index - lv.index[0]).days.to_numpy())
    h_params = (ts["kappa"], ts["theta"], xi, rho)
    sim = strategies.heston_paths(200_000, h_params, np.maximum(ts["v"], 1e-4), seed=10)
    hist = strategies.history_windows(df, TEST_START)  # strategy 6b
    beta = strategies.vix_beta(df, TEST_START)  # exploratory strategy, training years only
    deep = {"deep hedge (Heston-trained)": [strategies.train_deep(sim, COST, seed=s) for s in SEEDS],
            "deep hedge (history-trained)": [strategies.train_deep(hist, COST, seed=s) for s in SEEDS]}

    policies = {"no hedge": (False, None), "BS delta (implied vol)": (True, None),
                "BS delta (forecast vol)": (strategies.forecast_delta, {"forecast_vol": forecast}),
                "Leland": (strategies.leland(COST), None),
                "Whalley-Wilmott band": (strategies.whalley_wilmott(COST, ww_g), None),
                **{k: (strategies.deep_policy(m), None) for k, m in deep.items()},
                # exploratory, added after the first results to explain the deep hedges (not pre-declared)
                "skew-adjusted delta (exploratory)": (strategies.skew_delta(beta), None)}
    runs = {k: trades.run_all(df, hedge=h, ctx=c) for k, (h, c) in policies.items()}
    test = (base.entry >= TEST_START).to_numpy()
    pnl = {k: per_100(t).to_numpy() for k, t in runs.items()}
    ref = pnl["BS delta (implied vol)"][test]

    summary, paired = [], []
    for k, p in pnl.items():
        cost_paid = (runs[k].stock_cost / runs[k].S0 * 100).to_numpy()
        summary.append({"strategy": k, "period": "2015-2025 (test)", "trades": int(test.sum()), **_metrics(p[test]),
                        "stock_cost": cost_paid[test].mean()})
        if not k.startswith("deep"):  # nothing learned, so the full history is fair too
            summary.append({"strategy": k, "period": "2005-2025 (full)", "trades": len(p), **_metrics(p),
                            "stock_cost": cost_paid.mean()})
        if k != "BS delta (implied vol)":
            for name, f in [("mean", lambda x: x.mean(axis=-1)), ("std", lambda x: x.std(axis=-1, ddof=1)),
                            ("cvar95", stats.cvar)]:
                d, lo, hi, pv = stats.paired_stat(p[test], ref, f)
                paired.append({"strategy": k, "metric": name, "diff_vs_bs_delta": d, "lo": lo, "hi": hi, "p": pv})
    seeds = [{"strategy": k, "seed": s, **_metrics(per_100(trades.run_all(df, hedge=strategies.deep_policy([m])))
                                                      .to_numpy()[test])}
             for k, models in deep.items() for s, m in zip(SEEDS, models, strict=True)]
    fc_rv, fc_iv, fc_f = base.realized_vol.to_numpy() ** 2, base.sigma.to_numpy() ** 2, forecast.to_numpy() ** 2
    qlike = lambda f, y: float(np.mean(y / f - np.log(y / f) - 1))  # noqa: E731
    spy_move = (df.close.reindex(base.settle).to_numpy() / base.S0.to_numpy() - 1)[test] * 100
    direction = []
    for k, p in pnl.items():
        if k in ("BS delta (implied vol)", "no hedge"):
            continue
        d = p[test] - ref
        direction.append({"strategy": k, "corr_with_spy_move": np.corrcoef(spy_move, d)[0, 1],
                          "mean_diff_spy_up": d[spy_move > 0].mean(), "mean_diff_spy_down": d[spy_move < 0].mean(),
                          "share_spy_up": (spy_move > 0).mean()})
    pd.DataFrame(direction).to_csv(RESULTS / "hedges_direction.csv", index=False, float_format="%.6g")
    setup = {"test_start": TEST_START, "stock_cost_per_side": COST, "ww_risk_aversion": ww_g, "vix_beta": beta,
             **{f"ww_train_cvar95_g{g}": v for g, v in ww_cvar.items()},
             "heston_train_kappa": h_params[0], "heston_train_theta": h_params[1], "heston_train_xi": xi,
             "heston_train_rho": rho, "history_windows": int(hist["S"].shape[0]),
             "forecast_qlike_test": qlike(fc_f[test], fc_rv[test]), "vix_qlike_test": qlike(fc_iv[test], fc_rv[test])}

    pd.DataFrame(summary).to_csv(RESULTS / "hedges_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(paired).to_csv(RESULTS / "hedges_paired.csv", index=False, float_format="%.6g")
    pd.DataFrame(seeds).to_csv(RESULTS / "hedges_seeds.csv", index=False, float_format="%.6g")
    pd.Series(setup).rename("value").to_csv(RESULTS / "hedges_setup.csv", index_label="parameter")
    pd.DataFrame({"entry": base.entry, **pnl}).to_csv(RESULTS / "hedges_trades.csv", index=False, float_format="%.6g")
    plot_paired(pd.DataFrame(paired), RESULTS / "hedges_comparison.png")
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(pd.DataFrame(seeds).round(3).to_string(index=False))
    print(pd.Series(setup).to_string())


def robustness():
    df = data.load()
    St = strategies

    # A) the baseline trade itself (BS delta, 2005-2025): does the premium survive pessimistic assumptions?
    base_settings = [
        ("default: VIX x1.00, 2bp, daily hedge, 30d, monthly", {}),
        *[(f"VIX x{s:.2f}", {"vix_scale": s}) for s in (0.80, 0.85, 0.90, 0.95)],
        *[(f"stock cost {c}bp", {"cost": c * 1e-4}) for c in (0, 5, 10)],
        *[(f"option entry cost {e:.0%} of premium", {"entry_cost": e}) for e in (0.01, 0.03)],
        ("pessimistic: VIX x0.85, 5bp, 1% entry", {"vix_scale": 0.85, "cost": 5e-4, "entry_cost": 0.01}),
        ("very pessimistic: VIX x0.80, 10bp, 3% entry", {"vix_scale": 0.80, "cost": 1e-3, "entry_cost": 0.03}),
        *[(f"rebalance every {k} days", {"hedge": St.every(k, St.implied_delta)}) for k in (2, 5)],
        ("9-day options at VIX9D (2011+)", {"days": 9, "vol_col": "vix9d"}),
        ("93-day options at VIX3M (2006+, overlapping)", {"days": 93, "vol_col": "vix3m"}),
        ("30-day, a new trade every day (overlapping)", {"entries": "daily"}),
    ]
    base_rows = []
    for name, kw in base_settings:
        t = trades.run_all(df, **kw)
        p = per_100(t).to_numpy()
        overlap = kw.get("entries") == "daily" or kw.get("days", 30) > 31
        mean, lo, hi = stats.mean_ci(p, block=30 if kw.get("entries") == "daily" else None)
        sharpe = stats.sharpe_ci(p) if not overlap else (np.nan, np.nan, np.nan)
        base_rows.append({"setting": name, "trades": len(p), "first_entry": t.entry.min().date(),
                          "mean_per_100": mean, "mean_lo": lo, "mean_hi": hi, "sharpe": sharpe[0],
                          "sharpe_lo": sharpe[1], "sharpe_hi": sharpe[2], "win_rate": (p > 0).mean(),
                          "worst": p.min(), "premium_survives": lo > 0})
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(RESULTS / "robustness_baseline.csv", index=False, float_format="%.6g")
    print(base_df.round(3).to_string(index=False))

    # B) skew-adjusted vs BS delta on 2015-2025 trades. Pre-declared: survives a setting if the paired std
    #    difference has a 95% interval entirely below zero; the wrong-sign placebo must not.
    beta = St.vix_beta(df, TEST_START)
    betas = {"beta from 2005-2009": St.vix_beta(df.loc["2005":], "2010-01-01"),
             "beta from 2010-2014": St.vix_beta(df.loc["2010":], TEST_START),
             "beta from 2005-2025 (in-sample, for reference)": St.vix_beta(df.loc["2005":], "2026-01-01")}
    skew_settings = [
        ("default (M4)", {}, beta, 1),
        *[(f"VIX x{s:.2f}", {"vix_scale": s}, beta, 1) for s in (0.80, 0.85)],
        *[(f"stock cost {c}bp", {"cost": c * 1e-4}, beta, 1) for c in (0, 5, 10)],
        *[(f"rebalance every {k} days", {}, beta, k) for k in (2, 5)],
        ("9-day options at VIX9D", {"days": 9, "vol_col": "vix9d"}, beta, 1),
        ("93-day options at VIX3M", {"days": 93, "vol_col": "vix3m"}, beta, 1),
        ("30-day, a new trade every day", {"entries": "daily"}, beta, 1),
        *[(name, {}, b, 1) for name, b in betas.items()],
        ("PLACEBO: beta with the wrong sign", {}, -beta, 1),
    ]
    skew_rows = []
    for name, kw, b, k in skew_settings:
        ref = trades.run_all(df, hedge=St.every(k, St.implied_delta) if k > 1 else True, **kw)
        alt = trades.run_all(df, hedge=St.every(k, St.skew_delta(b)), **kw)
        test = (ref.entry >= TEST_START).to_numpy()
        a, r = per_100(alt).to_numpy()[test], per_100(ref).to_numpy()[test]
        block = 30 if kw.get("entries") == "daily" else None
        for metric, f in [("mean", lambda x: x.mean(axis=-1)), ("std", lambda x: x.std(axis=-1, ddof=1)),
                          ("cvar95", stats.cvar)]:
            d, lo, hi, pv = stats.paired_stat(a, r, f, block=block)
            skew_rows.append({"strategy": name, "metric": metric, "beta": b, "trades": int(test.sum()),
                              "diff_vs_bs_delta": d, "lo": lo, "hi": hi, "p": pv})
    skew_df = pd.DataFrame(skew_rows)
    std_rows = skew_df[skew_df.metric == "std"]
    skew_df["survives"] = skew_df.strategy.map(dict(zip(std_rows.strategy, std_rows.hi < 0, strict=True)))
    skew_df.to_csv(RESULTS / "robustness_skew.csv", index=False, float_format="%.6g")
    plot_paired(skew_df, RESULTS / "robustness_skew.png",
                title="Skew-adjusted delta vs BS delta under stress, 2015-2025 trades (95% paired bootstrap CI)")
    print(skew_df.round(4).to_string(index=False))


def plot_paired(paired, path, title="Hedging strategies vs Black-Scholes delta, same 2015-2025 trades "
                                    "(95% paired bootstrap CI)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s for s in paired.strategy.unique() if s != "no hedge"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, facecolor="#fcfcfb")
    titles = {"mean": "Mean P&L (higher is better)", "std": "Std of P&L (lower is better)",
              "cvar95": "CVaR95 loss (lower is better)"}
    for ax, metric in zip(axes, titles, strict=True):
        m = paired[(paired.metric == metric) & paired.strategy.isin(names)].set_index("strategy").loc[names]
        y = np.arange(len(names))[::-1]
        ax.set_facecolor("#fcfcfb")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.hlines(y, m.lo, m.hi, color=BLUE, lw=2)
        ax.scatter(m.diff_vs_bs_delta, y, s=40, color=BLUE, edgecolor="#fcfcfb", lw=1, zorder=3)
        ax.set_title(titles[metric], loc="left", color=INK, fontsize=10)
        ax.set_xlabel("difference vs BS delta, per $100 / month", color=MUTED, fontsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # keeps tick labels from colliding
    axes[0].set_yticks(np.arange(len(names))[::-1], names, color=INK, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    commands = {"baseline": baseline, "heston": heston_results, "hedges": hedges, "robustness": robustness}
    commands[sys.argv[1] if len(sys.argv) > 1 else "baseline"]()
