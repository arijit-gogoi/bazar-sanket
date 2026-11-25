"""Quantile models, a parallel fit/predict task runner, and the rolling-origin backtest.

Global model: one scikit-learn HistGradientBoostingRegressor per horizon and
quantile (10th, 50th, 90th percentile), trained on all series together. It
accepts missing feature values natively, so gaps need no imputation.

Backtest protocol (no lookahead): at each refit week T the model is trained
only on examples whose target week t+h is <= T, then used for origins
T .. T+refit_every-1. Features at an origin use only weeks <= that origin.

Speed: every (refit, horizon) fit is an independent task. Tasks run in
parallel worker processes (joblib/loky), one thread each, on all CPU cores.
Feature matrices are shared with workers through memory-mapped .npy files.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, HORIZONS, KEY

QUANTILES = (0.1, 0.5, 0.9)
N_JOBS = os.cpu_count() or 1


def make(q: float, local: bool = False, categorical: list[int] | None = None) -> HistGradientBoostingRegressor:
    if local:  # small-data settings for single-series ("local") models
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.05, max_iter=100, max_leaf_nodes=8,
            min_samples_leaf=5, l2_regularization=1.0, early_stopping=False, random_state=0,
        )
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=0.05, max_iter=250, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=categorical or None,
        early_stopping=False, random_state=0,
    )


# --------------------------------------------------------------------------- shared arrays
class ArrayStore:
    """Temporary .npy files that worker processes open with mmap (no copying, no hashing)."""

    def __init__(self, tag: str):
        self.dir = Path(tempfile.mkdtemp(prefix=f"apa_{tag}_"))

    def put(self, name: str, arr: np.ndarray) -> str:
        path = self.dir / f"{name}.npy"
        np.save(path, np.ascontiguousarray(arr))
        return str(path)

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def frame_arrays(g: pl.DataFrame, feats: list[str], store: ArrayStore, tag: str) -> dict:
    """Numpy views of a feature frame, saved for workers."""
    x = g.select(feats).to_numpy().astype(np.float64)
    out = {
        "xpath": store.put(f"{tag}_X", x),
        "week": g.get_column("week").to_numpy().astype("datetime64[D]"),
        "lp_ok": g.get_column("lp").is_not_null().to_numpy(),
        "feats": list(feats),
    }
    for h in HORIZONS:
        y = g.get_column(f"y{h}").cast(pl.Float64).fill_null(np.nan).to_numpy()
        out[f"y{h}"] = y
        out[f"ypath{h}"] = store.put(f"{tag}_y{h}", y)
    return out


def _fit_predict(xpath, ypath, train, test, cols, cat, local):
    from threadpoolctl import threadpool_limits

    X = np.load(xpath, mmap_mode="r")
    y = np.load(ypath, mmap_mode="r")
    with threadpool_limits(limits=1):
        xtr = np.asarray(X[train])
        xte = np.asarray(X[test])
        if cols is not None:
            xtr, xte = xtr[:, cols], xte[:, cols]
        keep = ~np.all(np.isnan(xtr), axis=0)
        kept = np.flatnonzero(keep)
        cat_k = [int(np.flatnonzero(kept == c)[0]) for c in (cat or []) if c in kept] if not local else None
        xtr, xte = xtr[:, keep], xte[:, keep]
        ytr = np.asarray(y[train])
        models = [make(q, local=local, categorical=cat_k).fit(xtr, ytr) for q in QUANTILES]
        p = np.column_stack([m.predict(xte) for m in models])
    return np.sort(p, axis=1), len(train)  # sort guards against quantile crossing


def task(arr: dict, h: int, train: np.ndarray, test: np.ndarray, local: bool = False, cols: list[str] | None = None) -> dict:
    feats = arr["feats"]
    col_idx = [feats.index(c) for c in cols] if cols is not None else None
    space = cols if cols is not None else feats
    cat = [space.index(c) for c in CATEGORICAL if c in space]
    return {"xpath": arr["xpath"], "ypath": arr[f"ypath{h}"], "train": train, "test": test,
            "cols": col_idx, "cat": cat, "local": local}


def run_tasks(tasks: list[dict], n_jobs: int = N_JOBS) -> list[tuple[np.ndarray, int]]:
    """Run fit/predict tasks in parallel, biggest first; results in input order."""
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i]["train"]))
    res = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(delayed(_fit_predict)(**tasks[i]) for i in order)
    out: list = [None] * len(tasks)
    for i, r in zip(order, res):
        out[i] = r
    return out


def train_index(arr: dict, h: int, cutoff: dt.date, extra: np.ndarray | None = None) -> np.ndarray:
    """Examples usable at cutoff week: current price observed, target observed, t+h <= cutoff."""
    m = arr["lp_ok"] & ~np.isnan(arr[f"y{h}"]) & (arr["week"] <= np.datetime64(cutoff) - np.timedelta64(7 * h, "D"))
    if extra is not None:
        m &= extra
    return np.flatnonzero(m)


# --------------------------------------------------------------------------- backtest
def rolling_backtest(
    g: pl.DataFrame,
    feats: list[str],
    eval_mask: pl.Expr,
    test_start: dt.date,
    last_week: dt.date,
    refit_every: int = 4,
    n_jobs: int = N_JOBS,
) -> pl.DataFrame:
    """Rolling-origin evaluation. Returns one row per (series, origin, horizon)."""
    t0 = time.time()
    g = g.with_row_index("_r")
    store = ArrayStore("bt")
    try:
        arr = frame_arrays(g, feats, store, "bt")
        ev = g.select(eval_mask.fill_null(False).alias("_ev")).to_series().to_numpy()
        wk = arr["week"]
        refits = []
        t = test_start
        while t <= last_week - dt.timedelta(weeks=1):
            refits.append(t)
            t += dt.timedelta(weeks=refit_every)
        tasks, meta = [], []
        for T in refits:
            T64 = np.datetime64(T)
            block_end = T64 + np.timedelta64(7 * (refit_every - 1), "D")
            for h in HORIZONS:
                y_ok = ~np.isnan(arr[f"y{h}"])
                test = np.flatnonzero(
                    ev & arr["lp_ok"] & y_ok & (wk >= T64) & (wk <= block_end)
                    & (wk <= np.datetime64(last_week) - np.timedelta64(7 * h, "D"))
                )
                if test.size == 0:
                    continue
                tasks.append(task(arr, h, train_index(arr, h, T), test))
                meta.append((T, h))
        results = run_tasks(tasks, n_jobs)
    finally:
        store.close()
    out = []
    for (T, h), tk, (p, n_train) in zip(meta, tasks, results):
        te = g[tk["test"]]
        out.append(
            te.select(
                *KEY, "week", "lp",
                pl.lit(h).alias("h"),
                pl.lit(T).alias("refit_week"),
                pl.col(f"y{h}").alias("y"),
                pl.col(f"sn{h}").alias("sn"),
            ).with_columns(
                pl.Series("q10", p[:, 0]), pl.Series("q50", p[:, 1]), pl.Series("q90", p[:, 2]),
                pl.lit(n_train).alias("n_train"),
            )
        )
    print(f"backtest: {len(tasks)} fit tasks x {len(QUANTILES)} quantiles on {n_jobs} workers in {time.time() - t0:.0f}s", flush=True)
    return pl.concat(out)


def score(res: pl.DataFrame, by: list[str], flat: float = 0.05) -> pl.DataFrame:
    """Accuracy of model P50 vs naive (no change) and seasonal naive.

    MAPE and MAE are on the price level (Rs/quintal). Direction uses three
    classes: down (< -5%), flat, up (> +5%) over the horizon.
    """
    p0 = pl.col("lp").exp()
    act = (pl.col("lp") + pl.col("y")).exp()

    def cls(e: pl.Expr) -> pl.Expr:
        return pl.when(e > np.log1p(flat)).then(1).when(e < np.log1p(-flat)).then(-1).otherwise(0)

    sn = pl.col("sn").fill_null(0.0)
    r = res.with_columns(
        act.alias("actual"),
        p0.alias("p0"),
        (pl.col("lp") + pl.col("q50")).exp().alias("f50"),
        (pl.col("lp") + pl.col("q10")).exp().alias("f10"),
        (pl.col("lp") + pl.col("q90")).exp().alias("f90"),
        (pl.col("lp") + sn).exp().alias("fsn"),
        cls(pl.col("y")).alias("c_act"),
        cls(pl.col("q50")).alias("c_mod"),
        cls(sn).alias("c_sn"),
    )
    return (
        r.group_by(by)
        .agg(
            pl.len().alias("n"),
            ((pl.col("f50") - pl.col("actual")).abs() / pl.col("actual")).mean().mul(100).alias("mape_model"),
            ((pl.col("p0") - pl.col("actual")).abs() / pl.col("actual")).mean().mul(100).alias("mape_naive"),
            ((pl.col("fsn") - pl.col("actual")).abs() / pl.col("actual")).mean().mul(100).alias("mape_snaive"),
            (pl.col("f50") - pl.col("actual")).abs().mean().alias("mae_model"),
            (pl.col("p0") - pl.col("actual")).abs().mean().alias("mae_naive"),
            (pl.col("fsn") - pl.col("actual")).abs().mean().alias("mae_snaive"),
            (pl.col("c_mod") == pl.col("c_act")).mean().mul(100).alias("dir_model"),
            (pl.col("c_act") == 0).mean().mul(100).alias("dir_naive"),
            (pl.col("c_sn") == pl.col("c_act")).mean().mul(100).alias("dir_snaive"),
            ((pl.col("actual") >= pl.col("f10")) & (pl.col("actual") <= pl.col("f90"))).mean().mul(100).alias("coverage80"),
            ((pl.col("f90") - pl.col("f10")) / pl.col("p0")).mean().mul(100).alias("band_width_pct"),
            pl.col("sn").is_null().mean().mul(100).alias("snaive_fallback_pct"),
        )
        .with_columns(
            (100 * (1 - pl.col("mae_model") / pl.col("mae_naive"))).alias("skill_vs_naive"),
            (100 * (1 - pl.col("mae_model") / pl.col("mae_snaive"))).alias("skill_vs_snaive"),
        )
        .sort(by)
    )
