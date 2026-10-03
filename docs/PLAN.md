# hedgelab: project plan

## Context

You asked for one flagship finance project with real resume weight, not several small ones. The four prototypes (pricer, backtester, vol forecaster, deep hedging) are already merged into `~/Documents/PROJECTS/hedgelab` (27 tests passing). This plan turns that package into one research project with one question and one headline result. **First step on approval: save this plan into the repo as `docs/PLAN.md`.**

## The question

> **If you systematically sell 30-day SPY options and delta-hedge them daily, how much money do you make or lose,
> where does it come from, and can better volatility forecasts or smarter (cost-aware or learned) hedging keep more of it?**

This is a well-studied effect, the **variance risk premium**: option-implied volatility is usually higher than the volatility that actually follows (Carr & Wu 2009; Bakshi & Kapadia 2003). Option sellers earn that gap on average and lose badly in crashes. The project measures it on 20+ years of real data, decomposes it, and tests whether hedging choices change the outcome after costs.

Why this carries weight on a resume:
- **One question, one headline number,** with a confidence interval, not a pile of demos.
- **It touches every core quant skill:** pricing (Black-Scholes, Heston), volatility modelling, hedging, backtesting, statistics, and ML. Each one is there because the question needs it.
- **It's real data,** with honest controls: no lookahead, transaction costs, crisis periods, and robustness checks.
- **It reproduces with one command,** and every number in the README traces to a generated results file.

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

### M4: Hedging strategies head-to-head (1 week)
- Strategies 1-6b behind one interface: `strategy(state up to t) -> holding`, so lookahead is impossible by construction.
- The volatility forecaster (`hedgelab/volatility.py`, currently next-day) switches to predicting **30-day realized variance** to match the trade horizon, walk-forward as before.
- Deep hedges extend `Hedger` / `train` in `hedgelab/hedging.py`: add the variance state as an input; 6a trains on calibrated Heston paths, 6b on bootstrapped training-year returns. Both are tested only on held-out real trades. **Several seeds, with the spread reported.**
- **Result:** the strategy comparison table with paired confidence intervals, overall and by sub-period.

### M5: Robustness (3-4 days)
- Implied-vol proxy scaling: VIX × {0.80, 0.85, 0.9, 0.95, 1.0} (M3 measured 0.84 on one day).
- Costs: stock {0, 2, 5, 10} bp per side, and option entry {0, 1, 3}% of premium.
- Rebalancing every 1, 2 or 5 days, or by band.
- Maturity: 9, 30 and 93 days.
- Overlapping daily-start trades.
- **Stretch: VRP timing.** Sell only when VIX² exceeds the 30-day variance forecast by a margin, with the threshold chosen on training years only.
- **Result:** a single sensitivity table showing which conclusions survive every variation.

### M6: Write-up and polish (3-4 days)
- `python -m hedgelab.run all` regenerates every table and figure into `results/`, and README numbers are copied from those files.
- `docs/report.md`: a 6-8 page research note covering question, data, method, results, limitations, and what would change with real option data.
- README: headline result first, then one chart and one table, then how to reproduce.
- Resume bullets and a 5-minute interview walkthrough (see below).

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

- [ ] `pip install -e .` then `python -m hedgelab.run all` reproduces every result from scratch.
- [ ] CI is green, and the slow tests pass locally.
- [ ] The README headline has a number with a confidence interval, and every README number is in `results/`.
- [ ] The limitations section is honest: no historical option quotes, the VIX proxy, a single underlying.
- [ ] You can explain every formula in the report without notes.

## Resume bullets (fill in the real numbers at the end)

- Built **hedgelab**, an options-hedging research platform in Python. On 21 years of SPY data, it measured the variance risk premium from systematically selling delta-hedged 30-day straddles (**X% annualized, Sharpe Y, 95% CI [a, b]**) and decomposed P&L into the volatility premium, hedging error and costs.
- Implemented Heston stochastic volatility (Fourier pricing, QE Monte Carlo, calibration to the VIX term structure), and compared six hedging strategies, including a cost-aware no-trade band and a CVaR-trained neural hedger, cutting **tail loss by Z%** after costs versus Black-Scholes delta.

## Interview questions this prepares you for

- Why does a delta-hedged short option make money on average, and when does it blow up?
- Why hedge at implied vs realized vol, and what changes?
- What's wrong with using VIX as the option's implied vol?
- How do you know there's no lookahead, and how did you test it?
- Why did the neural hedge win or lose on real data when it was trained on simulations?
- How confident are you in the headline number, and what's the confidence interval?

## Risks

| Risk | Mitigation |
|---|---|
| Yahoo data gaps or flakiness | local cache, validation on load, coverage table in M1 |
| VIX ≠ ATM implied vol | sensitivity sweep in M5; stated as a headline limitation |
| Heston calibration unstable | bounded parameters, multiple starts, recovery test on synthetic data |
| Deep hedge doesn't transfer to real data | report it; that gap is a finding, not a failure |
| Scope creep | stretch items stay stretch until M6 is done |
