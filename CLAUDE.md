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

- `hedgelab/data.py` is the only module that touches the network. `load()` = `fetch` (Yahoo, cached as CSV in `data/`, gitignored) → `align` (onto SPY trading days) → `validate` (raises listing every problem). Yahoo stamps SPY in New York time and the VIX family/^IRX in Chicago time, so `_download` normalises everything to tz-naive dates. Don't compare raw timestamps. Index gaps are forward-filled at most `MAX_FILL` days, and the counts go into `df.attrs["filled"]`. VIX9D starts 2011, VIX6M 2008, VIX3M mid-2006; they're NaN before launch by design, so don't fill or backfill them.
- `pricing.py`: Black-Scholes/Greeks/binomial/Monte Carlo. Functions take `(S, K, T, r, sigma, q=0.0, call=True)` and are NumPy-vectorised (`hedging.py` passes 2-D `S` and 1-D `T`).
- `volatility.py`: GARCH/HAR/linear-log/MLP walk-forward forecasts, units in % and %². Currently a next-day target; M4 retargets it to 30-day realized variance.
- `hedging.py`: simulated-world deep hedging (GBM, CVaR loss) with BS-delta and Leland baselines. Module constants (`S0, K, SIGMA, T, STEPS`) define the toy setup; real-data hedging comes in M2/M4.

## Rules this codebase holds itself to

- **No lookahead, tested:** every engine gets a "scramble the future, the past is unchanged" test, and that test must be shown to catch an injected leak. See `tests/test_volatility.py::test_no_lookahead` and `tests/test_hedging.py::test_hedger_does_not_look_ahead`.
- Models are tested against independent results (closed form, parity, limits, Monte Carlo), not hard-coded outputs. Validation code is tested by feeding it broken data.
- Numbers in README/docs come from files in `results/` (committed); never type a result by hand.
- Raw market data is never committed (`data/` is gitignored; Yahoo terms).
