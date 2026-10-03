# hedgelab

Research project: **if you sell 30-day SPY options and hedge them daily, how much do you make or lose, and do better volatility forecasts or a learned hedge improve it after trading costs?**

**Status:** M2 of 6 done (first real result below). Full roadmap: [docs/PLAN.md](docs/PLAN.md).

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

M5 tests both. Until then, treat a Sharpe near 1.8 with all 21 years positive as too good to take at face value. Data: [results/baseline_summary.csv](results/baseline_summary.csv), [baseline_periods.csv](results/baseline_periods.csv), [baseline_trades.csv](results/baseline_trades.csv); reproduce with `python -m hedgelab.run baseline`.

## Data

21 years of daily data (2004-2025, 5,534 trading days) from Yahoo Finance: SPY prices and dividends, VIX, the VIX term structure (VIX9D from 2011, VIX3M from 2006, VIX6M from 2008), and the 3-month T-bill rate. Every load is validated, and the run fails loudly on gaps, impossible price bars, absurd moves, or out-of-range levels. Coverage: [results/data_coverage.csv](results/data_coverage.csv). Raw data is cached locally and not committed.

## Modules

| Module | What it does |
|---|---|
| `hedgelab.data` | download, cache, align across timezones, validate |
| `hedgelab.trades` | monthly short straddle, daily delta hedge, exact P&L decomposition (VRP term / residual / costs) |
| `hedgelab.stats` | stationary block bootstrap, Newey-West, paired comparisons |
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
