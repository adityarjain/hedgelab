# hedgelab

Research project: **if you sell 30-day SPY options and hedge them daily, how much do you make or lose, and do better volatility forecasts or a learned hedge improve it after trading costs?**

**Status:** M3 of 6 done (baseline result and Heston model below). Full roadmap: [docs/PLAN.md](docs/PLAN.md).

## First result: the baseline trade

On the first trading day of every month from 2005 to 2025 (252 trades), sell a 30-day at-the-money SPY straddle priced at VIX-implied vol, delta-hedge it at every daily close paying 2 bp per stock trade, and settle at expiry.

| Strategy | Mean P&L per $100 notional per month (95% CI) | Annualized Sharpe (95% CI) | Worst month | Months profitable |
|---|---|---|---|---|
| **Delta-hedged** | **0.70 (0.52 – 0.89)** | **1.82 (1.24 – 2.63)** | −6.07 | 79% |
| Delta-hedged, no costs | 0.76 (0.59 – 0.96) | 1.99 (1.40 – 2.85) | −6.01 | 81% |
| Unhedged | 0.88 (0.56 – 1.18) | 1.23 (0.71 – 1.87) | −12.72 | 73% |

Intervals are stationary block-bootstrap intervals (blocks of consecutive months, so clustered crashes are resampled together).

![cumulative P&L](results/baseline_pnl.png)

**Where the money comes from.** Implied vol exceeded the volatility that followed in 83% of months. Each trade's P&L splits exactly into a gamma-weighted variance term, ½ Γ S² (σ²_implied Δt − r²) summed daily, plus a residual and costs. Over all trades, the **variance risk premium term is 109% of the total**, the residual is +2%, and stock trading costs take 10%.

**Crises:** the 2008–09 crisis was *profitable* on average (+0.80 per trade over 10 trades), because VIX was already extreme when the options were sold. Covid lost money (−0.60 per trade over 5 trades), because it hit while VIX was low. The single worst month was −6.07, about 3.5% of the 21-year cumulative P&L.

**Why these numbers are an upper bound for now:**
- VIX is a variance-swap level and runs above at-the-money implied vol because of the volatility skew, so pricing the straddle at VIX overstates the premium collected.
- There is no option bid-ask cost yet.

M5 tests both. Until then, treat a Sharpe near 1.8 with all 21 years positive as too good to take at face value. The Heston work below measures the first caveat directly: on one recent day, VIX overstated 30-day at-the-money vol by 16%. Data: [results/baseline_summary.csv](results/baseline_summary.csv), [baseline_periods.csv](results/baseline_periods.csv), [baseline_trades.csv](results/baseline_trades.csv); reproduce with `python -m hedgelab.run baseline`.

## Heston stochastic volatility

Black-Scholes assumes one constant vol. Heston lets variance itself move randomly, mean-revert, and correlate with the stock, which produces the volatility skew seen in real option prices. M4 hedges in this world.

**Correctness.**
- The Fourier pricer matches an independent adaptive-quadrature integration to 10⁻⁴ for maturities from 9 days to 1 year, including short-dated options, where naive Fourier grids fail.
- It reduces to Black-Scholes when vol-of-vol → 0 (agreement to 10⁻⁸).
- It agrees with Andersen-QE Monte Carlo within 3 standard errors.
- Both calibrations recover known parameters from synthetic data.

**Calibration 1: 15 years of the VIX term structure** (VIX9D/VIX/VIX3M/VIX6M, 2011–2025, 3,771 days). The fit gives mean reversion κ = 4.2, long-run vol 24.1%, and from the return/variance history, correlation ρ = −0.69 and vol-of-vol ξ = 1.67. **A one-factor model can't fit the whole curve:** errors are 1.0 vol point at 30 days but 2.4 at 9 days and 2.7 at 6 months. One mean-reversion speed is too rigid, which is the usual motivation for two-factor models.

**Calibration 2: a live SPY option chain** (close of 2026-10-02, 499 out-of-the-money quotes across 13 expiries from 14 to 180 days). Each expiry's forward is backed out from put-call parity, not assumed from trailing dividends. That removed a ~1 vol-point jump where the quotes switch from puts to calls. One parameter set (v₀ = 10.7% vol, κ = 12.2, long-run 19.0%, ξ = 2.05, ρ = −0.60) fits every expiry to **0.61 vol points RMSE**. The fit is 0.2 vol points around 3–4 months and 1.6 at 14 days. Heston can't make the short-dated skew steep enough, the classic argument for adding jumps.

![Heston smile fit](results/heston_smile.png)

**What this says about the baseline.**
- On this date, 30-day at-the-money implied vol was **12.9% vs VIX 15.3% (ratio 0.84)**. The M2 trade prices options at VIX, so it overstates the premium collected; M5 will re-run it at 0.80–1.0 × VIX.
- The two calibrations disagree on mean reversion (12.2 from one day of options vs 4.2 from 15 years of VIX), and the chain fit violates the Feller condition. Both are common in practice: they're different measures (one day's risk-neutral surface vs a 15-year average), and Heston is too simple to be consistent across them.

Data: [results/heston_timeseries.csv](results/heston_timeseries.csv), [heston_chain_fit.csv](results/heston_chain_fit.csv), [heston_chain_by_expiry.csv](results/heston_chain_by_expiry.csv); reproduce with `python -m hedgelab.run heston`. The chain is one snapshot cached locally: free data has no option history, so re-downloading on a new day gives a new fit.

## Data

21 years of daily data (2004-2025, 5,534 trading days) from Yahoo Finance: SPY prices and dividends, VIX, the VIX term structure (VIX9D from 2011, VIX3M from 2006, VIX6M from 2008), and the 3-month T-bill rate. Every load is validated, and the run fails loudly on gaps, impossible price bars, absurd moves, or out-of-range levels. Coverage: [results/data_coverage.csv](results/data_coverage.csv). Raw data is cached locally and not committed.

## Modules

| Module | What it does |
|---|---|
| `hedgelab.data` | download, cache, align across timezones, validate |
| `hedgelab.trades` | monthly short straddle, daily delta hedge, exact P&L decomposition (VRP term / residual / costs) |
| `hedgelab.stats` | stationary block bootstrap, Newey-West, paired comparisons |
| `hedgelab.heston` | Fourier pricer, QE Monte Carlo, calibration to the VIX term structure and to an option chain |
| `hedgelab.run` | regenerates everything in `results/` |
| `hedgelab.pricing` | Black-Scholes, Greeks, implied vol, binomial tree, Monte Carlo (European, Asian with control variate) |
| `hedgelab.volatility` | GARCH / HAR-RV / linear-log / MLP forecasts, walk-forward, QLIKE, Diebold-Mariano |
| `hedgelab.hedging` | CVaR-trained deep hedger vs BS delta and Leland delta under transaction costs |

## Develop

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/pytest -q -m "not slow"                                 # fast suite (what CI runs)
.venv/bin/pytest -q                                               # everything, incl. model training
.venv/bin/pytest -q tests/test_pricing.py::test_put_call_parity   # one test
```

Example scripts (these download data or train models, so they take minutes) are in `examples/`.
