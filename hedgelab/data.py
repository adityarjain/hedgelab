"""Market data: download from Yahoo Finance, cache locally, align on SPY trading days, validate.

`load()` returns one row per SPY trading day (tz-naive date index) with columns:
  open, high, low, close    SPY raw prices (SPY has never split, so raw = split-adjusted)
  dividend                  SPY cash dividend per share on its ex-date, else 0
  vix, vix9d, vix3m, vix6m  CBOE implied-vol indices in vol points (20 = 20%); NaN before each index launched
                            (VIX3M mid-2006, VIX6M 2008, VIX9D 2011)
  rate                      3-month T-bill (^IRX discount yield) as a continuously compounded annual rate

Raw downloads are cached as CSV in data/ (gitignored: Yahoo data is not ours to redistribute).
"""
from pathlib import Path

import numpy as np
import pandas as pd

START, END = "2004-01-01", "2025-12-31"  # yfinance treats END as exclusive
INDEXES = {"^VIX": "vix", "^VIX9D": "vix9d", "^VIX3M": "vix3m", "^VIX6M": "vix6m", "^IRX": "irx"}
CACHE = Path(__file__).resolve().parent.parent / "data"
MAX_FILL = 3  # trading days an index level may be carried forward over a gap in its own history
MAX_GAP_DAYS = 5  # calendar days between SPY sessions; Hurricane Sandy (Oct 2012) is the longest since 2004


def _download(ticker, start, end):
    import yfinance as yf  # imported here so offline use (tests, cached runs) never touches it

    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    # SPY is stamped in New York time, VIX and IRX in Chicago time: align on the calendar date.
    h.index = h.index.tz_localize(None).normalize()
    h.index.name = "date"
    return h


def fetch(ticker, start=START, end=END, refresh=False):
    """Raw Yahoo history for one ticker, read from the CSV cache unless missing or refresh=True."""
    path = CACHE / f"{ticker.lstrip('^')}_{start}_{end}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    h = _download(ticker, start, end)
    CACHE.mkdir(exist_ok=True)
    h.to_csv(path)
    return h


def irx_to_rate(irx):
    """^IRX is a 91-day T-bill discount yield in percent; convert to a continuously compounded rate."""
    price = 1 - irx / 100 * 91 / 360
    return -np.log(price) / (91 / 365)


def align(raw):
    """raw: {ticker: Yahoo history} -> one frame on SPY's trading days. Index gaps after launch are
    forward-filled for at most MAX_FILL days; the number filled per column is in df.attrs["filled"]."""
    spy = raw["SPY"]
    df = pd.DataFrame({"open": spy.Open, "high": spy.High, "low": spy.Low, "close": spy.Close,
                       "dividend": spy.Dividends.fillna(0.0)})
    filled = {}
    for ticker, col in INDEXES.items():
        s = raw[ticker].Close.reindex(df.index)
        f = s.ffill(limit=MAX_FILL)
        filled[col] = int((s.isna() & f.notna()).sum())
        df[col] = f
    df["rate"] = irx_to_rate(df.pop("irx"))
    filled["rate"] = filled.pop("irx")
    df.attrs["filled"] = filled
    return df


def validate(df):
    """Raise ValueError listing every problem found; return df unchanged if clean."""
    problems = []
    idx = df.index
    if not (idx.is_monotonic_increasing and idx.is_unique):
        problems.append("index not sorted/unique")
    gaps = idx.to_series().diff().dt.days
    if (gaps > MAX_GAP_DAYS).any():
        problems.append(f"gaps > {MAX_GAP_DAYS} days before: {list(gaps[gaps > MAX_GAP_DAYS].index.date)}")
    o, h, lo, c = df.open, df.high, df.low, df.close
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        problems.append("non-positive SPY price")
    bad = (lo > np.minimum(o, c) + 1e-6) | (h < np.maximum(o, c) - 1e-6)
    if bad.any():
        first_bad = list(df.index[bad].date[:5])
        problems.append(f"{int(bad.sum())} SPY bars with open/close outside [low, high]: {first_bad}")
    if (df.dividend < 0).any():
        problems.append("negative dividend")
    big = np.log(c).diff().abs() > 0.25
    if big.any():
        problems.append(f"daily SPY move > 25% (bad print?): {list(df.index[big].date)}")
    for col in ["vix", "rate"]:  # required on every day; the term-structure indexes may start later
        missing = df[col].isna()
        if missing.any():
            problems.append(f"{col} missing on {int(missing.sum())} days, first {df.index[missing][0].date()}")
    vols = df[["vix", "vix9d", "vix3m", "vix6m"]]
    if ((vols <= 5) | (vols >= 150)).any().any():
        problems.append("vol index outside (5, 150)")
    if ((df.rate < -0.02) | (df.rate > 0.2)).any():
        problems.append("rate outside (-2%, 20%)")
    for col in ["vix9d", "vix3m", "vix6m"]:  # once launched, no holes left after filling
        s = df[col]
        if s.first_valid_index() is not None and s.loc[s.first_valid_index():].isna().any():
            problems.append(f"{col} has unfilled gaps after launch")
    if problems:
        raise ValueError("data validation failed:\n  " + "\n  ".join(problems))
    return df


def coverage(df):
    """Per column: first/last date with data, days missing after first, days forward-filled."""
    filled = df.attrs.get("filled", {})
    rows = {}
    for col in df.columns:
        s = df[col]
        first = s.first_valid_index()
        rows[col] = {"first": first.date() if first is not None else None,
                     "last": s.last_valid_index().date() if first is not None else None,
                     "missing_after_first": int(s.loc[first:].isna().sum()) if first is not None else len(s),
                     "forward_filled": filled.get(col, 0)}
    return pd.DataFrame(rows).T


def load(start=START, end=END, refresh=False):
    """Validated daily frame of SPY, its dividends, the VIX family and the risk-free rate."""
    raw = {t: fetch(t, start, end, refresh) for t in ["SPY", *INDEXES]}
    return validate(align(raw))
