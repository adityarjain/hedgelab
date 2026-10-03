"""Next-day volatility forecasting: GARCH(1,1) vs HAR-RV vs linear-log vs MLP, evaluated walk-forward.

Units: returns in %, variances in %^2. `rv` is a Garman-Klass range estimate from daily OHLC, a noisy proxy
for true (intraday) realized variance. Row t of the feature frame holds data through close t and the target
y = rv[t+1]; a forecast made on row t never sees anything later.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

MODELS = ["garch", "har", "linlog", "mlp"]


# ---------- data ----------
def make_frame(ohlc):
    """ohlc: DataFrame with open/high/low/close. Returns the feature frame (see module docstring)."""
    o, h, lo, c = (ohlc[k] for k in ("open", "high", "low", "close"))
    r = 100 * np.log(c).diff()
    rv = 1e4 * (0.5 * np.log(h / lo) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2)
    return features(r, rv.clip(lower=1e-4))


def features(r, rv):
    f = pd.DataFrame({"r": r, "rv": rv})
    f["rv_w"] = rv.rolling(5).mean()
    f["rv_m"] = rv.rolling(22).mean()
    f["y"] = rv.shift(-1)
    return f.dropna()


# ---------- losses ----------
def qlike(f, y):
    """Per-day QLIKE; robust to noise in the variance proxy. Lower is better."""
    ratio = y / np.maximum(f, 1e-8)
    return ratio - np.log(ratio) - 1


def sqerr(f, y):
    return (f - y) ** 2


def dm_test(loss_a, loss_b, lag=5):
    """Diebold-Mariano with Newey-West variance. stat < 0 means model A has lower loss. Returns (stat, p)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n, dc = len(d), d - d.mean()
    var = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / n)
    return stat, 2 * norm.sf(abs(stat))


# ---------- models ----------
def _garch_g(e2, omega, alpha, beta, h0):
    """g[t] = h[t+1] = omega + alpha*e2[t] + beta*h[t], the one-step-ahead variance made at close t."""
    return lfilter([1.0], [1.0, -beta], omega + alpha * e2, zi=[beta * h0])[0]


def garch_fit(r):
    """Gaussian MLE for GARCH(1,1) with variance targeting. Returns (mu, omega, alpha, beta)."""
    mu, v = r.mean(), r.var()
    e2 = (r - mu) ** 2

    def nll(p):
        a, b = p
        if a + b >= 0.9999:
            return 1e10
        g = _garch_g(e2, v * (1 - a - b), a, b, v)
        h = np.concatenate([[v], g[:-1]])
        return 0.5 * np.sum(np.log(h) + e2 / h)

    a, b = minimize(nll, [0.08, 0.9], bounds=[(1e-4, 0.5), (0.3, 0.9998)], method="L-BFGS-B").x
    return mu, v * (1 - a - b), a, b


def _har_x(f):
    return np.column_stack([np.ones(len(f)), f.rv, f.rv_w, f.rv_m])


def _nn_x(f):
    return np.column_stack([np.log(f.rv), np.log(f.rv_w), np.log(f.rv_m), f.r, np.minimum(f.r, 0)])


def _fit_log_model(model, tr, te):
    """Fit on log(next-day variance), predict on `te`, undo the log with Duan smearing."""
    sc = StandardScaler().fit(_nn_x(tr))
    model.fit(sc.transform(_nn_x(tr)), np.log(tr.y))
    smear = np.mean(np.exp(np.log(tr.y) - model.predict(sc.transform(_nn_x(tr)))))
    return np.exp(model.predict(sc.transform(_nn_x(te)))) * smear


def walk_forward(f, first=1000, step=63, seeds=3, mlp_iter=300):
    """Refit every `step` rows on all rows before the block (expanding window); predict the block.

    Returns a DataFrame of forecasts (NaN before `first`) with columns MODELS.
    """
    out = pd.DataFrame(np.nan, index=f.index, columns=MODELS)
    r = f.r.to_numpy()
    for s in range(first, len(f), step):
        tr, te = f.iloc[:s], f.iloc[s : s + step]
        # GARCH: params from train; the filter is causal, so run it over train+block. Rescaled so its level
        # matches the proxy on the training rows (GK excludes overnight gaps, so it sits below close-to-close).
        mu, om, a, b = garch_fit(r[:s])
        g = _garch_g((r[: s + step] - mu) ** 2, om, a, b, r[:s].var())
        out.loc[te.index, "garch"] = g[s : s + step] * tr.y.mean() / g[:s].mean()
        # HAR-RV: OLS in levels, floored at the smallest training variance
        beta = np.linalg.lstsq(_har_x(tr), tr.y, rcond=None)[0]
        out.loc[te.index, "har"] = np.maximum(_har_x(te) @ beta, tr.y.min())
        # same features as the MLP but linear: isolates what nonlinearity adds
        out.loc[te.index, "linlog"] = _fit_log_model(Ridge(alpha=1.0), tr, te)
        out.loc[te.index, "mlp"] = np.mean(
            [_fit_log_model(MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=mlp_iter, random_state=k),
                            tr, te)
             for k in range(seeds)], axis=0)
    return out


def evaluate(f, preds, base="har"):
    """Table of out-of-sample QLIKE / RMSE plus Diebold-Mariano (on QLIKE) of each model vs `base`."""
    m = preds.notna().all(axis=1)
    y = f.y[m].to_numpy()
    base_loss = qlike(preds[base][m].to_numpy(), y)
    rows = {}
    for k in MODELS:
        p = preds[k][m].to_numpy()
        stat, pv = dm_test(qlike(p, y), base_loss) if k != base else (np.nan, np.nan)
        rows[k] = {"QLIKE": qlike(p, y).mean(), "RMSE_var": np.sqrt(sqerr(p, y).mean()), f"DM_vs_{base}": stat, "p": pv}
    return pd.DataFrame(rows).T
