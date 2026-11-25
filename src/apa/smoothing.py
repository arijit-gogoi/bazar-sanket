"""Holt-Winters (triple exponential smoothing) benchmark, re-estimated at every forecast origin.

Method, fixed before any test result was seen:

* statsmodels ``ExponentialSmoothing`` with an additive damped trend and additive
  52-week seasonality, fitted to the log weekly modal price of one market series;
  heuristic initialisation; parameters re-estimated at every forecast origin.
* Fit window: the weeks since the series' last reporting gap longer than
  ``MAX_GAP`` weeks, at most the latest ``MAX_WINDOW`` weeks. Shorter gaps inside the
  window are filled by linear interpolation of the log price. The origin week is
  always observed and nothing after it is used, so there is no lookahead.
* Fallbacks: a window shorter than two seasons uses damped Holt without
  seasonality; a window shorter than ``MIN_HOLT`` weeks uses no change.
* Sensitivity variant (``max_gap=None``), added after the first result showed how
  rarely two unbroken years exist: every gap in the window is interpolated.

Speed: one task per series, run in parallel worker processes (joblib/loky) on all
CPU cores, one thread each.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from .features import HORIZONS, KEY

MAX_GAP = 26
MAX_WINDOW = 312
SEASON = 52
MIN_SEASONAL = 2 * SEASON
MIN_HOLT = 20
N_JOBS = os.cpu_count() or 1


def fit_window(y: np.ndarray, t: int, max_gap: int | None = MAX_GAP) -> int:
    """First index of the fit window that ends at origin t (max_gap=None: ignore gaps)."""
    obs = np.flatnonzero(~np.isnan(y[: t + 1]))
    big = np.flatnonzero(np.diff(obs) - 1 > max_gap) if max_gap is not None else np.array([], dtype=int)
    a = int(obs[big[-1] + 1]) if big.size else int(obs[0])
    return max(a, t - MAX_WINDOW + 1)


def forecast_origin(y: np.ndarray, t: int, max_gap: int | None = MAX_GAP) -> tuple[np.ndarray, str, bool]:
    """Forecast log-price changes for every horizon from origin t (y[t] must be observed).

    Returns (changes, method used, optimiser converged)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if np.isnan(y[t]):
        raise ValueError("origin week must be observed")
    seg = y[fit_window(y, t, max_gap) : t + 1].copy()
    i = np.arange(seg.size)
    ok = ~np.isnan(seg)
    seg = np.interp(i, i[ok], seg[ok])
    zero = np.zeros(len(HORIZONS))
    if seg.size < MIN_HOLT:
        return zero, "no change (short history)", True
    seasonal = seg.size >= MIN_SEASONAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = ExponentialSmoothing(
                seg, trend="add", damped_trend=True,
                seasonal="add" if seasonal else None,
                seasonal_periods=SEASON if seasonal else None,
                initialization_method="heuristic",
            ).fit()
            path = np.asarray(res.forecast(max(HORIZONS)), dtype=float)
        except Exception:  # noqa: BLE001 - any numerical failure falls back to no change
            return zero, "no change (fit failed)", False
    f = path[[h - 1 for h in HORIZONS]] - y[t]
    if not np.all(np.isfinite(f)):
        return zero, "no change (fit failed)", False
    rv = getattr(res, "mle_retvals", None)
    converged = bool(rv.get("success", True)) if isinstance(rv, dict) else True
    return f, "Holt-Winters" if seasonal else "damped Holt (under 2 years of data)", converged


def _series_task(key: tuple, y: np.ndarray, idx: np.ndarray, weeks: list, max_gap: int | None) -> list[tuple]:
    from threadpoolctl import threadpool_limits

    out = []
    with threadpool_limits(limits=1):
        for t, w in zip(idx, weeks):
            f, kind, conv = forecast_origin(y, int(t), max_gap)
            out.append((key, w, f, kind, conv))
    return out


def backtest(grid: pl.DataFrame, origins: pl.DataFrame, max_gap: int | None = MAX_GAP, n_jobs: int = N_JOBS) -> pl.DataFrame:
    """Holt-Winters forecasts from every origin in ``origins`` (KEY + week).

    ``grid`` is the complete weekly grid (KEY, week, lp) used by the main model.
    Returns KEY, week, h, hw (forecast log-price change), hw_method, hw_converged."""
    t0 = time.time()
    want = origins.select(*KEY, "week").unique().group_by(KEY).agg(pl.col("week").sort())
    want_map = {tuple(r[k] for k in KEY): r["week"] for r in want.iter_rows(named=True)}
    tasks = []
    for key, s in grid.sort([*KEY, "week"]).partition_by(KEY, as_dict=True, maintain_order=True).items():
        if key not in want_map:
            continue
        pos = {w: i for i, w in enumerate(s.get_column("week").to_list())}
        y = s.get_column("lp").cast(pl.Float64).fill_null(np.nan).to_numpy()
        weeks = want_map[key]
        tasks.append((key, y, np.array([pos[w] for w in weeks], dtype=int), weeks))
    tasks.sort(key=lambda tk: -tk[2].size)  # most origins first, for load balance
    parts = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(delayed(_series_task)(*tk, max_gap) for tk in tasks)
    rows = [
        {**dict(zip(KEY, key)), "week": w, "h": h, "hw": float(f[j]), "hw_method": kind, "hw_converged": conv}
        for part in parts
        for key, w, f, kind, conv in part
        for j, h in enumerate(HORIZONS)
    ]
    n_fit = sum(tk[2].size for tk in tasks)
    print(f"holt-winters (max gap {max_gap}): {n_fit} origin fits over {len(tasks)} series on {n_jobs} workers in {time.time() - t0:.0f}s", flush=True)
    return pl.DataFrame(rows).with_columns(pl.col("h").cast(pl.Int32))

