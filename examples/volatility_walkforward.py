"""Walk-forward next-day variance forecasts on SPY. Usage: python run_example.py (takes a few minutes)."""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

from hedgelab.volatility import evaluate, make_frame, walk_forward

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn MLP convergence chatter

ohlc = yf.Ticker("SPY").history(start="2000-01-01", end="2025-12-31", auto_adjust=True)
ohlc = ohlc.rename(columns=str.lower)[["open", "high", "low", "close"]]
f = make_frame(ohlc)
preds = walk_forward(f)

oos = preds.dropna().index
print(f"out-of-sample: {oos[0].date()} to {oos[-1].date()} ({len(oos)} days)")
fmt = {"p": "{:.1e}".format}
print(evaluate(f, preds).to_string(float_format="{:.4f}".format, formatters=fmt))
print("\nDoes nonlinearity help? MLP vs the linear model on identical features:")
print(evaluate(f, preds, base="linlog").loc[["mlp"]].to_string(float_format="{:.4f}".format, formatters=fmt))
oos_years = oos.year
for name, years in [("2008-2009", [2008, 2009]), ("2020", [2020]), ("ex-crisis", None)]:
    idx = oos[~oos_years.isin([2008, 2009, 2020])] if years is None else oos[oos_years.isin(years)]
    print(f"QLIKE {name}:", evaluate(f.loc[idx], preds.loc[idx])["QLIKE"].round(4).to_dict())

win = preds.loc["2020-01-01":"2020-12-31"].index
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(win, np.sqrt(f.y[win] * 252), lw=0.6, color="0.6", label="next-day GK proxy (annualised vol)")
for k in ("garch", "har", "mlp"):
    ax.plot(win, np.sqrt(preds.loc[win, k] * 252), lw=1.2, label=k)
ax.set(ylabel="annualised vol (%)", title="SPY 2020: next-day volatility forecasts (out-of-sample)")
ax.legend()
fig.tight_layout()
fig.savefig("forecasts_2020.png", dpi=150)
