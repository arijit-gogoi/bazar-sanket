"""Quantile models and the rolling-origin backtest.

Global model: one scikit-learn HistGradientBoostingRegressor per horizon and
quantile (10th, 50th, 90th percentile), trained on all series together. It
accepts missing feature values natively, so gaps need no imputation.

Backtest protocol (no lookahead): at each refit week T the model is trained
only on examples whose target week t+h is <= T, then used for origins
T .. T+refit_every-1. Features at an origin use only weeks <= that origin.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, HORIZONS, KEY

QUANTILES = (0.1, 0.5, 0.9)


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


class QuantileModel:
    """Three quantile regressors sharing one feature set. Columns that are entirely
    missing in the training data are dropped (and ignored at prediction)."""

    def __init__(self, feats: list[str], local: bool = False):
        self.feats = list(feats)
        self.local = local
        self.cols: list[str] = []
        self.models: list = []

    def fit(self, train: pl.DataFrame, h: int) -> "QuantileModel":
        x_all = train.select(self.feats).to_numpy().astype(np.float64)
        keep = ~np.all(np.isnan(x_all), axis=0)
        self.cols = [f for f, k in zip(self.feats, keep) if k]
        x = x_all[:, keep]
        y = train.get_column(f"y{h}").to_numpy().astype(np.float64)
        cat = [i for i, f in enumerate(self.cols) if f in CATEGORICAL] if not self.local else None
        self.models = [make(q, local=self.local, categorical=cat).fit(x, y) for q in QUANTILES]
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        x = df.select(self.cols).to_numpy().astype(np.float64)
        p = np.column_stack([m.predict(x) for m in self.models])
        return np.sort(p, axis=1)  # guard against quantile crossing


def fit_quantiles(train: pl.DataFrame, feats: list[str], h: int, local: bool = False) -> QuantileModel:
    return QuantileModel(feats, local=local).fit(train, h)


def predict_quantiles(qm: QuantileModel, df: pl.DataFrame, feats: list[str] | None = None) -> np.ndarray:
    return qm.predict(df)


def train_rows(g: pl.DataFrame, h: int, cutoff: dt.date) -> pl.DataFrame:
    """Examples usable at cutoff week: current price observed, target observed, t+h <= cutoff."""
    return g.filter(
        pl.col("lp").is_not_null()
        & pl.col(f"y{h}").is_not_null()
        & (pl.col("week") <= pl.lit(cutoff) - pl.duration(weeks=h))
    )


def rolling_backtest(
    g: pl.DataFrame,
    feats: list[str],
    eval_mask: pl.Expr,
    test_start: dt.date,
    last_week: dt.date,
    refit_every: int = 4,
    log=print,
) -> pl.DataFrame:
    """Rolling-origin evaluation. Returns one row per (series, origin, horizon)."""
    out = []
    refits = []
    t = test_start
    while t <= last_week - dt.timedelta(weeks=1):
        refits.append(t)
        t += dt.timedelta(weeks=refit_every)
    t0 = time.time()
    for T in refits:
        block_end = T + dt.timedelta(weeks=refit_every - 1)
        for h in HORIZONS:
            tr = train_rows(g, h, T)
            models = fit_quantiles(tr, feats, h)
            te = g.filter(
                eval_mask
                & pl.col("week").is_between(T, block_end)
                & (pl.col("week") <= pl.lit(last_week) - pl.duration(weeks=h))
                & pl.col("lp").is_not_null()
                & pl.col(f"y{h}").is_not_null()
            )
            if te.height == 0:
                continue
            p = predict_quantiles(models, te, feats)
            out.append(
                te.select(
                    *KEY, "week", "lp",
                    pl.lit(h).alias("h"),
                    pl.lit(T).alias("refit_week"),
                    pl.col(f"y{h}").alias("y"),
                    pl.col(f"sn{h}").alias("sn"),
                ).with_columns(
                    pl.Series("q10", p[:, 0]), pl.Series("q50", p[:, 1]), pl.Series("q90", p[:, 2]),
                    pl.lit(tr.height).alias("n_train"),
                )
            )
        log(f"refit {T}: done ({time.time() - t0:.0f}s elapsed)")
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
