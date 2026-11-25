"""Volatility pricing (exploratory): put a rupee value on short-term price risk.

Method, fixed before any test result was seen:

* Volatility: exponentially weighted moving average (RiskMetrics style) of squared
  weekly log-price changes, decay ``LAMBDA`` per week, past-only. A change measured
  across missing weeks is spread over the weeks it covers. At least ``MIN_CHANGES``
  observed changes are needed.
* Price distribution h weeks ahead: lognormal, standard deviation sigma * sqrt(h)
  around a point forecast (no change unless stated).
* Volatility band: 10th-90th percentile of that distribution.
* Downside price: expected shortfall below a floor K, E[max(K - P_h, 0)], which is the
  value of a put option (Black-76 formula, forward F, strike K). With K = 90% of
  today's price it is the expected cost, per quintal, of guaranteeing 90% of today's
  price h weeks from now. The backtest compares it with the shortfall that occurred.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy.special import ndtr

from .features import KEY

LAMBDA = 0.94
MIN_CHANGES = 8
Z80 = 1.2815515655446004  # standard normal 90th percentile
ALPHA = 0.2  # 80% interval


def ewma_sigma(grid: pl.DataFrame) -> pl.DataFrame:
    """Weekly volatility (sd of weekly log change) per series and week, using weeks <= that week."""
    out = []
    for key, s in grid.sort([*KEY, "week"]).partition_by(KEY, as_dict=True, maintain_order=True).items():
        y = s.get_column("lp").cast(pl.Float64).fill_null(np.nan).to_numpy()
        sig = np.full(y.size, np.nan)
        v, last, n = math.nan, -1, 0
        for t in range(y.size):
            if not np.isnan(y[t]):
                if last >= 0:
                    k = t - last  # weeks covered by this change
                    x = (y[t] - y[last]) ** 2 / k
                    w = LAMBDA ** k
                    v = x if n == 0 else w * v + (1 - w) * x
                    n += 1
                last = t
            if n >= MIN_CHANGES:
                sig[t] = math.sqrt(v)
        out.append(s.select(*KEY, "week").with_columns(pl.Series("sigma", sig).fill_nan(None)))
    return pl.concat(out)


def put_value(fwd: np.ndarray, strike: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """E[max(K - P, 0)] for lognormal P with mean ``fwd`` and log-sd ``sd`` (Black-76 put, no discounting)."""
    sd = np.maximum(sd, 1e-9)
    d1 = (np.log(fwd / strike) + 0.5 * sd**2) / sd
    return strike * ndtr(-(d1 - sd)) - fwd * ndtr(-d1)


def shortfall_prob(fwd: np.ndarray, strike: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """P(P < K) for lognormal P with mean ``fwd`` and log-sd ``sd``."""
    sd = np.maximum(sd, 1e-9)
    d2 = (np.log(fwd / strike) - 0.5 * sd**2) / sd
    return ndtr(-d2)


def model_lognormal(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, p0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and log-sd of a lognormal matched to the global model's P10-P50-P90 (log changes)."""
    sd = np.maximum((q90 - q10) / (2 * Z80), 1e-9)
    return p0 * np.exp(q50 + 0.5 * sd**2), sd


def pricing_rows(ev: pl.DataFrame, floors: tuple[float, ...] = (1.0, 0.9)) -> pl.DataFrame:
    """Per backtest row and floor: realised shortfall below the floor and its price under
    (a) EWMA volatility around today's price and (b) the global model's band.
    Shortfalls are in % of today's price. Rows without a volatility estimate are dropped."""
    d = ev.filter(pl.col("sigma").is_not_null())
    lp, y, h = (d.get_column(c).to_numpy().astype(float) for c in ("lp", "y", "h"))
    sig = d.get_column("sigma").to_numpy().astype(float)
    q10, q50, q90 = (d.get_column(c).to_numpy().astype(float) for c in ("q10", "q50", "q90"))
    p0, act = np.exp(lp), np.exp(lp + y)
    s_ewma = sig * np.sqrt(h)
    f_mod, s_mod = model_lognormal(q10, q50, q90, p0)
    out = []
    for fl in floors:
        k = fl * p0
        out.append(d.select(*KEY, "week", "h", "selected").with_columns(
            pl.lit(fl).alias("floor"),
            pl.Series("realised", 100 * np.maximum(k - act, 0) / p0),
            pl.Series("price_ewma", 100 * put_value(p0, k, s_ewma) / p0),
            pl.Series("price_model", 100 * put_value(f_mod, k, s_mod) / p0),
            pl.Series("hit", (act < k).astype(float) * 100),
            pl.Series("prob_ewma", 100 * shortfall_prob(p0, k, s_ewma)),
            pl.Series("prob_model", 100 * shortfall_prob(f_mod, k, s_mod)),
        ))
    return pl.concat(out)


def interval_score(lo: pl.Expr, hi: pl.Expr, act: pl.Expr, alpha: float = ALPHA) -> pl.Expr:
    """Winkler interval score (lower is better): width plus 2/alpha times any miss."""
    return (hi - lo) + (2 / alpha) * ((lo - act).clip(lower_bound=0) + (act - hi).clip(lower_bound=0))
