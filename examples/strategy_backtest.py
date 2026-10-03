"""Run the strategies on real daily closes and print a results table. Usage: python run_example.py"""
import pandas as pd
import yfinance as yf

from hedgelab.backtest import buy_hold, metrics, momentum, pairs, run

START, END = "2010-01-01", "2025-12-31"


def closes(tickers):
    return yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"][tickers].dropna()


rows = {}
etfs = closes(["SPY", "QQQ", "TLT", "GLD"])
for name, strat in [("ETFs buy & hold", buy_hold()), ("ETFs momentum 126d", momentum(126))]:
    rows[name] = metrics(*run(etfs, strat))

kopep = closes(["KO", "PEP"])
rows["KO/PEP buy & hold"] = metrics(*run(kopep, buy_hold()))
rows["KO/PEP pairs"] = metrics(*run(kopep, pairs()))

print(pd.DataFrame(rows).T.round(3).to_string())
