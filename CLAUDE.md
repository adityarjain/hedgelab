# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single research project, not a library: *if you sell 30-day SPY options and delta-hedge them daily, what do you earn, where does it come from (variance risk premium vs hedging error vs costs), and can better vol forecasts or smarter hedging keep more of it?* `docs/PLAN.md` is the roadmap (milestones M0-M6, with done-criteria). Read it before starting work, and keep code on-story: anything that doesn't serve that question doesn't belong here.

## Commands

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .                                         # lint (config in pyproject.toml)
.venv/bin/pytest -q -m "not slow"                              # fast suite, what CI runs (~3s)
.venv/bin/pytest -q                                            # everything, incl. model training
.venv/bin/pytest -q tests/test_data.py::test_irx_to_rate       # one test
```

CI (`.github/workflows/ci.yml`) runs ruff and the fast suite with CPU-only torch. Mark any test that trains a model or runs a long simulation with `@pytest.mark.slow`.

## Architecture

- `hedgelab/data.py` is the only **package** module that touches the network. `examples/` are pre-merge prototype scripts: they download or simulate their own data (e.g. `volatility_walkforward.py` pulls dividend-adjusted SPY from 2000 via yfinance directly, unlike `data.load()`), take minutes, write PNGs to the working directory, and are not the source of any research result.
- In `data.py`, `load()` = `fetch` (Yahoo, cached as CSV in `data/`, gitignored) → `align` (onto SPY trading days) → `validate` (raises listing every problem). Yahoo stamps SPY in New York time and the VIX family/^IRX in Chicago time, so `_download` normalises everything to tz-naive dates. Don't compare raw timestamps. Index gaps are forward-filled at most `MAX_FILL` days, and the counts go into `df.attrs["filled"]`. VIX9D starts 2011, VIX6M 2008, VIX3M mid-2006; they're NaN before launch by design, so don't fill or backfill them.
- `pricing.py`: Black-Scholes/Greeks/binomial/Monte Carlo. Functions take `(S, K, T, r, sigma, q=0.0, call=True)` and are NumPy-vectorised (`hedging.py` passes 2-D `S` and 1-D `T`).
- `volatility.py`: two layers. The original next-day GARCH/HAR/linear-log/MLP walk-forward (units % and %², used by `examples/` only). `har_monthly` is the one the research uses: a walk-forward 30-day forecast of annualised calendar-time variance (decimal², same convention as VIX), refit at each date on rows whose target window had already closed.
- `hedging.py`: the original toy deep-hedging study (GBM, single call, CVaR loss). No result in `results/` uses it; real-data hedging lives in `strategies.py`. Its module constants define the toy setup, and `PREMIUM` is computed from them once at import.

- `trades.py`: the research engine. `run_all(df)` sells a straddle on the first session of each month and hedges at each close. Its P&L uses an independent cash account, and `vrp`/`residual` decompose it exactly (`total = vrp + residual - stock_cost - entry_cost`). Time is calendar days/365, matching VIX; r and q are frozen at entry. `run_trade` also returns `holdings` (dropped by `run_all`), which the hedge-lookahead test needs.
- `stats.py`: stationary block bootstrap (`mean_ci`, `sharpe_ci`, `paired_diff`) and `newey_west_se`. Use these, not naive SEs; monthly P&L is fat-tailed and clustered.
- `run.py`: `python -m hedgelab.run all` regenerates everything in `results/` (~6 min). Sub-commands: `baseline`, `heston`, `hedges`, `robustness`, `drawdowns`. Chart colours are the validated dataviz palette slots defined at its top. Write-ups (`README.md`, `docs/report.md`, and the local-only, gitignored `docs/interview.md`) quote these CSVs, so re-check the numbers after any re-run (deep-hedge training can shift in the last digit across machines).
- `trades.mark_to_market` marks every open straddle at each day's VIX for drawdowns. Its daily P&L sums exactly to `run_all`'s totals (tested), so keep the two cash accounts in step if either changes.

- `heston.py`: `price` integrates on a grid whose length scales with 1/sqrt(vT). Don't replace it with a fixed grid: short-dated options (9-14 days) need it, and `test_fourier_grid_matches_adaptive_quadrature` guards it. `simulate` is Andersen QE and returns `(S, v)` paths for M4's deep hedge. Chain calibration needs per-expiry forwards from `implied_carry` (put-call parity), not the trailing dividend yield; quotes carry a `q` column that `market_quotes`/`fit_chain` use. `data.option_chain_snapshot()` is one cached day (`data/SPY_chain_<asof>.csv`) and keeps both sides near the money (`otm=False`) only for that parity step.
- pandas gotcha: `snap.asof` is a DataFrame method, so use `snap["asof"]`.
- `strategies.py`: real-trade hedging policies with the signature `policy(t, S, tau, T, prev, K, sigma, r, q, vix, ctx)`. `trades.run_trade(hedge=policy)` slices `S`/`vix` to `[:t+1]` before calling, so policies are lookahead-free by construction; per-trade inputs known at entry go in `ctx` (`run_all(ctx={name: Series})`). The deep hedger trains in torch (`straddle_pnl`, r = q = 0) and runs live through `deep_policy`. `test_torch_training_pnl_matches_the_engine` and `test_deep_policy_wrapper_matches_training_forward` guard the two places training and evaluation could silently diverge; keep both passing if you change features or P&L.
- `run.py robustness` sweeps the baseline (full period) and skew-vs-BS delta (2015+). Pass/fail rules are fixed in code comments before results; don't move them after seeing output. `run_all(entries="daily")` makes ~5,000 overlapping trades, so use `block=30` in the bootstrap for those.
- `run.py hedges` is the train/test experiment: `TEST_START = 2015-01-01`. Anything tuned, calibrated or trained must use data before it (the Whalley-Wilmott grid uses trades *settled* before it). Strategies added after seeing results are labelled "(exploratory)" in their name and the README.

Lookahead in `trades.py` needs two tests, because a leak can live inside a single trade: `test_no_lookahead` (scramble after k; settled trades unchanged and entry pricing of every trade entered by k unchanged) and `test_hedge_uses_only_past_prices`.

## Units (differ by module, so convert at the boundary)

- `data`: VIX family in **vol points** (20 = 20%); `rate` as a decimal, continuously compounded.
- `pricing`, `hedging`: `sigma` and `r` as **decimals** (0.2), T in years. Use `vix / 100` before pricing.
- `volatility`: returns in **percent**, variances in **%²** (daily). Annualised decimal vol = `sqrt(252 * var) / 100`.

## Rules this codebase holds itself to

- **No lookahead, tested:** every engine gets a "scramble the future, the past is unchanged" test, and that test must be shown to catch an injected leak. See `tests/test_volatility.py::test_no_lookahead` and `tests/test_hedging.py::test_hedger_does_not_look_ahead`.
- Models are tested against independent results (closed form, parity, limits, Monte Carlo), not hard-coded outputs. Validation code is tested by feeding it broken data.
- Numbers in README/docs come from files in `results/` (committed); never type a result by hand.
- Raw market data is never committed (`data/` is gitignored; Yahoo terms).
