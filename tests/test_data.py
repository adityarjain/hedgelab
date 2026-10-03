"""Offline tests for the data layer: synthetic Yahoo-shaped frames, no network."""
import numpy as np
import pandas as pd
import pytest

from hedgelab import data

DAYS = pd.bdate_range("2020-01-02", periods=60)


def yahoo(close, tz, index=DAYS, dividends=None):
    """A frame shaped like yfinance history (tz-aware), already date-normalised as _download does."""
    idx = index.tz_localize(tz).tz_localize(None).normalize()
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                         "Dividends": 0.0 if dividends is None else dividends})


def raw_set(**overrides):
    raw = {"SPY": yahoo(np.linspace(300, 330, len(DAYS)), "America/New_York"),
           "^VIX": yahoo(20.0, "America/Chicago"), "^VIX9D": yahoo(18.0, "America/New_York"),
           "^VIX3M": yahoo(22.0, "America/New_York"), "^VIX6M": yahoo(23.0, "America/New_York"),
           "^IRX": yahoo(1.5, "America/Chicago")}
    raw.update(overrides)
    return raw


def test_clean_data_passes_and_timezones_align():
    df = data.validate(data.align(raw_set()))
    assert len(df) == len(DAYS) and not df.isna().any().any()


def test_short_index_gap_is_filled_and_counted():
    vix = yahoo(20.0, "America/Chicago").drop(DAYS[10:12])  # 2 missing days <= MAX_FILL
    df = data.validate(data.align(raw_set(**{"^VIX": vix})))
    assert df.attrs["filled"]["vix"] == 2 and df.vix.iloc[10] == 20.0


def test_long_index_gap_fails_validation():
    vix = yahoo(20.0, "America/Chicago").drop(DAYS[10:16])  # 6 missing days > MAX_FILL
    with pytest.raises(ValueError, match="vix missing"):
        data.validate(data.align(raw_set(**{"^VIX": vix})))


def test_index_launching_later_is_nan_before_launch_and_valid():
    late = yahoo(18.0, "America/New_York", index=DAYS[30:])
    df = data.validate(data.align(raw_set(**{"^VIX9D": late})))
    assert df.vix9d.iloc[:30].isna().all() and df.vix9d.iloc[30:].notna().all()
    assert data.coverage(df).loc["vix9d", "first"] == DAYS[30].date()


def _spike(spy):
    bad = spy.copy()
    bad.loc[DAYS[20], ["Open", "High", "Low", "Close"]] = 1.0  # a consistent bar, just absurdly low
    return bad


@pytest.mark.parametrize("ticker, broken, message", [
    ("SPY", lambda s: s.assign(High=s.Close * 0.98), "outside \\[low, high\\]"),
    ("SPY", _spike, "move > 25%"),
    ("^VIX", lambda s: s.assign(Close=200.0), "vol index outside"),
    ("SPY", lambda s: s.drop(DAYS[20:25]), "gaps > 5 days"),
])
def test_validation_catches_bad_data(ticker, broken, message):
    raw = raw_set()
    raw[ticker] = broken(raw[ticker])
    with pytest.raises(ValueError, match=message):
        data.validate(data.align(raw))


def test_irx_to_rate():
    assert data.irx_to_rate(0.0) == 0.0
    # 5% discount yield on a 91-day bill: price 0.98736, continuously compounded ~5.10%
    assert data.irx_to_rate(5.0) == pytest.approx(0.05102, abs=1e-5)
    assert data.irx_to_rate(-0.1) < 0  # negative bill yields (2008, 2020) stay negative


def test_fetch_reads_cache_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data, "_download", lambda *a: pytest.fail("network used despite cache"))
    yahoo(20.0, "America/Chicago").to_csv(tmp_path / f"VIX_{data.START}_{data.END}.csv")
    assert data.fetch("^VIX").Close.iloc[0] == 20.0
