# hedgelab: project plan

## Context

Goal: one flagship research project rather than several small ones. The four prototypes (pricer, backtester, vol forecaster, deep hedging) are already merged into `~/Documents/PROJECTS/hedgelab` (27 tests passing). This plan turns that package into one research project with one question and one headline result.

## The question

> **If you systematically sell 30-day SPY options and delta-hedge them daily, how much money do you make or lose,
> where does it come from, and can better volatility forecasts or smarter (cost-aware or learned) hedging keep more of it?**

This is a well-studied effect, the **variance risk premium**: option-implied volatility is usually higher than the volatility that actually follows (Carr & Wu 2009; Bakshi & Kapadia 2003). Option sellers earn that gap on average and lose badly in crashes. The project measures it on 20+ years of real data, decomposes it, and tests whether hedging choices change the outcome after costs.

---

## Data (all free)

| Data | Source | Used for |
|---|---|---|
| SPY daily OHLC + dividends, 2004-2025 | Yahoo Finance | underlying paths, realized vol, dividend yield |
| VIX (30-day implied vol), 2004-2025 | Yahoo `^VIX` | implied vol for pricing the options we sell |
| VIX9D / VIX3M / VIX6M | Yahoo `^VIX9D`, `^VIX3M`, `^VIX6M` | implied-vol term structure, used to calibrate Heston daily (check history coverage in M1) |
| 13-week T-bill rate | Yahoo `^IRX` | risk-free rate |
| One live SPY option chain snapshot | Yahoo (`yf.Ticker("SPY").option_chain`) | demonstrating Heston smile calibration on real quotes |

**Biggest limitation, stated up front:** free data has **no historical option prices**. The options we "sell" are priced with Black-Scholes at VIX-implied volatility. VIX is a model-free variance-swap level, which runs slightly above at-the-money implied vol because of skew, so M5 tests scaling it down (×0.85-1.0). Upgrade path: OptionMetrics via WRDS, if your university has access.

Raw data is cached locally (`data/`, gitignored, since Yahoo's terms don't allow redistribution) and validated on load: no missing days, split-adjusted, dates aligned across series.

---

## The trade (main experiment)

- **What:** on the first trading day of each month, sell one 30-day at-the-money SPY **straddle** (a call plus a put, the standard way to sell volatility), priced with Black-Scholes at VIX/100. That gives about 260 non-overlapping trades from 2004 to 2025.
- **Hedge:** delta-hedge with SPY at each daily close until expiry, paying a proportional cost on every stock trade, plus an entry cost on the option (a % of premium, swept in M5).
- **Settle** at payoff at expiry. No daily option marking in the main result, so there's no vega P&L. Daily mark-to-market at that day's VIX is used only for drawdown charts.
- **Secondary runs:** a single call (matches the existing deep-hedging setup); overlapping trades started every day (more data, with Newey-West or block-bootstrap errors); 9-day and 93-day maturities via VIX9D/VIX3M.

### P&L decomposition (the core insight)

For an option hedged at the implied vol σᵢ, the daily hedged P&L is approximately

  **½ Γₜ Sₜ² (σᵢ² Δt − rₜ²)**

where rₜ is the day's return. Summed over the trade's life, this is the **gamma-weighted variance spread**: the variance risk premium actually captured. Each trade's total P&L splits into

  **total = VRP term + discretization/higher-order residual − hedging costs − option entry cost**

which answers "where does the money come from". The test for this: the components must sum to the total exactly.

---

## Hedging strategies compared (same trades, same paths)

| # | Strategy | Why it's in |
|---|---|---|
| 1 | No hedge | scale reference |
| 2 | Black-Scholes delta at implied vol | the market standard; gives a smooth, path-dependent P&L |
| 3 | Black-Scholes delta at **forecast** realized vol (HAR, from `volatility`) | the "which vol should you hedge at" question (Ahmad & Wilmott, 2005) |
| 4 | Leland delta | the classic analytical cost adjustment (already in `hedging.py`) |
| 5 | Whalley-Wilmott no-trade band: rebalance only when the holding leaves Δ ± H, H = (3 e^{−r(T−t)} λ S Γ² / 2γ)^{1/3} | the theoretically optimal shape under costs; the network should find something like it |
| 6a | Deep hedge trained on **Heston paths calibrated to the market** | simulation-to-real: trained on a model, tested on real history |
| 6b | Deep hedge trained on **block-bootstrapped real returns from training years only** | data-driven alternative, strictly walk-forward |

The comparison metrics are mean P&L, standard deviation, CVaR95, max drawdown, turnover, and cost paid. All comparisons are **paired** (the same trades under each hedge), which gives much tighter intervals than comparing two separate averages.

---

## Statistics

- **Confidence intervals:** stationary block bootstrap on per-trade P&L, plus Newey-West errors for overlapping trades.
- **Hedge A vs hedge B:** paired difference per trade, with a block-bootstrap interval and p-value.
- **Sub-periods:** 2008-09, 2020, and 2022 (crises), against calm stretches such as 2017. The headline must say how much of the long-run result one crash wipes out.
- **Multiple testing:** a small, pre-declared set of comparisons, listed in the report before results are produced. Everything else is labelled exploratory.

---

## Milestones

Total about 5 weeks part-time. Each milestone ends with tests passing, a result file, and a short README update.

### M0: Merge into one package *(done)*
Four prototypes became the `hedgelab` package, with 27 tests passing.

### M1: Foundations and data (3-4 days) *(done)*
Delivered: `hedgelab/data.py` (5,534 validated days), `results/data_coverage.csv`, ruff and CI, CLAUDE.md. Finding: Yahoo history starts VIX3M 2006-07, VIX6M 2008-01, **VIX9D 2011-01**, so M3's four-point term-structure calibration covers 2011+ only (three points from 2008).

- Remove off-story code: the equity momentum/pairs strategies in `hedgelab/backtest.py` go. They stay in the old `backtester` repo, and the no-lookahead discipline carries over to the options engine.
- New `hedgelab/data.py`: downloads, local cache, validation, and one aligned daily frame (S, dividends, VIX family, rate).
- Tooling: `ruff` lint; GitHub Actions running fast tests on every push (slow ones marked `@pytest.mark.slow`); `CLAUDE.md`.
- **Done when:** `hedgelab.data.load()` returns a validated frame from 2004-2025, there's a coverage table for every ticker, and CI is green.

### M2: Options trade engine plus the first real result (1 week) *(done)*
Delivered: `trades.py`, `stats.py`, `run.py baseline`, results in `results/baseline_*`. Hedged: 0.70 per $100/month (CI 0.52-0.89), Sharpe 1.82 (CI 1.24-2.63); VRP term = 109% of P&L. Trades start 2005 so the trailing dividend yield has a year of history. Deferred to M4: daily mark-to-market at that day's VIX (drawdowns use per-trade P&L for now). Lookahead tests were strengthened after an injected within-trade leak slipped past the first version.

- New `hedgelab/trades.py`: monthly short straddle, daily hedge, costs, settlement; per-trade P&L with its decomposition; daily mark-to-market for drawdowns. It reuses `bs_price` and `bs_greeks` from `hedgelab/pricing.py`.
- New `hedgelab/stats.py`: block bootstrap, Newey-West, paired tests.
- Tests:
  - the decomposition sums to the total;
  - on synthetic paths where realized vol equals implied, mean P&L ≈ 0;
  - **no lookahead** (scramble the future, past P&L unchanged; same pattern as the existing tests);
  - costs only ever reduce P&L;
  - put-call parity on the legs.
- **First result:** short straddle with a BS-delta hedge from 2004 to 2025. It gives annualized return, Sharpe, worst month, the VRP share of P&L, and a P&L chart with crises marked.

### M3: Heston model (1 week) *(done)*
Delivered: `heston.py`, `data.option_chain_snapshot()`, `run.py heston`, results in `results/heston_*`. Term-structure fit (2011-2025): κ 4.2, θ 0.058, ρ −0.69, ξ 1.67, RMSE 1.0-2.7 vol pts by tenor. Chain fit (2026-10-02): RMSE 0.61 vol pts. Added beyond plan: per-expiry forwards from put-call parity (removed a ~1 vol-pt put/call jump). Finding for M5: 30d ATM IV / VIX = 0.84 on the snapshot date, so the VIX-scaling sweep now starts at 0.80.

- New `hedgelab/heston.py`:
  - semi-closed-form price via the characteristic function, using the stable "little trap" form;
  - Monte Carlo with Andersen's QE scheme;
  - implied-vol smile output, using `implied_vol` from `pricing.py`.
- **Calibration**, two ways:
  - **Time series:** the daily variance level plus mean reversion (κ) and long-run level (θ), fitted to the VIX9D/VIX/VIX3M/VIX6M term structure using E[avg var over τ] = θ + (v − θ)(1 − e^{−κτ})/(κτ). The correlation ρ and vol-of-vol ξ come from joint SPY-return and VIX-change history.
  - **Cross-section:** fit the full parameter set to one live SPY option chain and show the fitted smile against market quotes.
- Tests:
  - the Fourier price matches Monte Carlo within 3 standard errors;
  - Heston reduces to Black-Scholes as ξ → 0;
  - put-call parity;
  - calibration recovers known parameters from a synthetic surface.

### M4: Hedging strategies head-to-head (1 week) *(done)*
Delivered: `strategies.py`, `volatility.har_monthly`, `stats.cvar`/`paired_stat`, `run.py hedges`, results in `results/hedges_*`. Test period 2015-2025 (132 trades). No deep hedge beat BS delta reliably; the Whalley-Wilmott band significantly raised risk at 2 bp. Diagnosis showed both networks under-hedge (gain when SPY falls), i.e. a spot-vol effect. **Exploratory (post-hoc) addition:** skew-adjusted delta, BS delta + vega × β/S with β from 2005-2014. Std −0.32 (p 0.008), CVaR95 −1.08 (p 0.04), worst month −2.30 vs −6.07. M5 must stress-test it before it can be a headline. Still deferred: daily mark-to-market at VIX.

- Strategies 1-6b behind one interface: `strategy(state up to t) -> holding`, so lookahead is impossible by construction.
- The volatility forecaster (`hedgelab/volatility.py`, currently next-day) switches to predicting **30-day realized variance** to match the trade horizon, walk-forward as before.
- Deep hedges extend `Hedger` / `train` in `hedgelab/hedging.py`: add the variance state as an input; 6a trains on calibrated Heston paths, 6b on bootstrapped training-year returns. Both are tested only on held-out real trades. **Several seeds, with the spread reported.**
- **Result:** the strategy comparison table with paired confidence intervals, overall and by sub-period.

### M5: Robustness (3-4 days) *(done)*
Delivered: `run.py robustness`, `trades.run_all(vol_col=, entries=)`, `strategies.every`/`implied_delta`, results in `results/robustness_*`. Baseline premium survives 13/17 settings; all 4 failures are VIX ≤ 0.85 (ATM vol measured at 0.84 × VIX in M3), so ATM straddles at realistic prices capture little of the premium. Skew-adjusted delta survives 11/13 out-of-sample stresses on the pre-declared std rule; the placebo hurts (std +0.76, p < 0.001); it fails for 93-day options (β from 30-day VIX over-corrects) and narrowly for 5-day rebalancing. VRP timing (stretch) not done. Remaining caveat: stresses reuse the 2015-2025 trades the strategy was proposed on; 2026+ data is the clean test.

- Implied-vol proxy scaling: VIX × {0.80, 0.85, 0.9, 0.95, 1.0} (M3 measured 0.84 on one day).
- Costs: stock {0, 2, 5, 10} bp per side, and option entry {0, 1, 3}% of premium.
- Rebalancing every 1, 2 or 5 days, or by band.
- Maturity: 9, 30 and 93 days.
- Overlapping daily-start trades.
- **Skew-adjusted delta (from M4) must face every row of this sweep**, plus β re-estimated on different windows and a placebo (β of the wrong sign), before it is reported as a finding rather than exploratory.
- **Stretch: VRP timing.** Sell only when VIX² exceeds the 30-day variance forecast by a margin, with the threshold chosen on training years only.
- **Result:** a single sensitivity table showing which conclusions survive every variation.

### M6: Write-up and polish (3-4 days) *(done)*
Delivered: `python -m hedgelab.run all`, `trades.mark_to_market` (daily marks at that day's VIX; sums exactly to trade totals, tested), `results/drawdowns.*`, `docs/report.md`, the README rebuilt around the three findings, and CLAUDE.md updated. A full rebuild reproduced the committed results (baseline and hedges byte-identical; Heston chain fit equal at reported precision). Daily marks: BS-delta max drawdown −10.4 vs a −6.07 worst month; skew-adjusted −8.9, though worse than BS delta in August 2024.

- `python -m hedgelab.run all` regenerates every table and figure into `results/`, and README numbers are copied from those files.
- `docs/report.md`: a 6-8 page research note covering question, data, method, results, limitations, and what would change with real option data.
- README: headline result first, then one chart and one table, then how to reproduce.

---

## Final repo layout (target)

```
hedgelab/
  data.py         download, cache, validate, align                     (M1)
  pricing.py      Black-Scholes, Greeks, binomial, Monte Carlo         (exists)
  heston.py       Fourier pricer, QE Monte Carlo, calibration          (M3)
  volatility.py   HAR/GARCH/MLP forecasts, QLIKE, Diebold-Mariano      (exists; retarget to 30d in M4)
  hedging.py      strategies 1-6b, deep hedger                         (exists; extend in M4)
  trades.py       option trade engine, P&L decomposition               (M2)
  stats.py        block bootstrap, Newey-West, paired tests            (M2)
  run.py          one command for all results                          (M6)
tests/            one file per module, slow tests marked
results/          generated CSVs and figures (committed)
docs/             PLAN.md, report.md
```

## Verification (applies to every milestone)

- `.venv/bin/pytest -q` is green, with fast tests in CI and slow tests run locally before each milestone closes.
- Every new engine has a no-lookahead test that has been shown to catch an injected leak (as done for the backtester and the vol forecaster).
- Every accounting identity is tested exactly (decomposition sums to total; costs ≥ 0).
- Every model is checked against an independent result (Heston vs Monte Carlo, Heston→BS limit, put-call parity).
- Every number written into the README or report is checked against `results/*.csv` before it's written.

## Definition of done

- [x] `pip install -e .` then `python -m hedgelab.run all` reproduces every result from scratch.
- [ ] CI is green (check the GitHub Actions tab after pushing), and the slow tests pass locally.
- [x] The README headline has a number with a confidence interval, and every README number is in `results/`.
- [x] The limitations section is honest: no historical option quotes, the VIX proxy, a single underlying.
- [ ] You can explain every formula in the report without notes.

## Risks

| Risk | Mitigation |
|---|---|
| Yahoo data gaps or flakiness | local cache, validation on load, coverage table in M1 |
| VIX ≠ ATM implied vol | sensitivity sweep in M5; stated as a headline limitation |
| Heston calibration unstable | bounded parameters, multiple starts, recovery test on synthetic data |
| Deep hedge doesn't transfer to real data | report it; that gap is a finding, not a failure |
| Scope creep | stretch items stay stretch until M6 is done |
