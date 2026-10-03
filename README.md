# hedgelab

Research project: **if you sell 30-day SPY options and hedge them daily, how much do you make or lose, and do better volatility forecasts or a learned hedge improve it after trading costs?**

Status: phase 1 of 5 (code merged into one package; research results coming). Plan:

1. **Merge** the four earlier prototypes into one package. *(done)*
2. **Heston stochastic volatility** with calibration, so hedging is tested where Black-Scholes breaks.
3. **Real-data hedging backtest** on SPY 2004-2025, using VIX as implied volatility to measure the variance risk premium.
4. **Deep hedging** on real and Heston paths, against delta and Leland baselines, with confidence intervals and crisis sub-periods.
5. **Write-up** with the headline result, honest limitations, and CI.

## Modules

| Module | What it does |
|---|---|
| `hedgelab.pricing` | Black-Scholes, Greeks, implied vol, binomial tree, Monte Carlo (European, Asian with control variate) |
| `hedgelab.volatility` | GARCH / HAR-RV / linear-log / MLP forecasts, walk-forward, QLIKE, Diebold-Mariano |
| `hedgelab.hedging` | CVaR-trained deep hedger vs BS delta and Leland delta under transaction costs |
| `hedgelab.backtest` | event-driven backtester: strategy sees only past data, fills one bar later with costs |

## Develop

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/pytest -q tests/test_pricing.py::test_put_call_parity   # one test
```

Example scripts (these download data or train models, so they take minutes) are in `examples/`.
