# hedgelab

Research project: **if you sell 30-day SPY options and hedge them daily, how much do you make or lose, and do better volatility forecasts or a learned hedge improve it after trading costs?**

**Status:** M1 of 6 done (package merged, data layer, CI). Research results start at M2. Full roadmap: [docs/PLAN.md](docs/PLAN.md).

## Data

21 years of daily data (2004-2025, 5,534 trading days) from Yahoo Finance: SPY prices and dividends, VIX, the VIX term structure (VIX9D from 2011, VIX3M from 2006, VIX6M from 2008), and the 3-month T-bill rate. Every load is validated, and the run fails loudly on gaps, impossible price bars, absurd moves, or out-of-range levels. Coverage: [results/data_coverage.csv](results/data_coverage.csv). Raw data is cached locally and not committed.

## Modules

| Module | What it does |
|---|---|
| `hedgelab.data` | download, cache, align across timezones, validate |
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
