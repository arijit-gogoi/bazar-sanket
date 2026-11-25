"""Scores for comparing point forecasts and 80% bands on identical backtest rows.

Every forecast is a log-price change from the origin week (column ``lp``) to the
target week; ``y`` is the change that happened. Errors are on the price level.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .volatility import interval_score

MODEL = "Global model (P50)"
HW = "Holt-Winters"
HW_ALL = "Holt-Winters, all gaps interpolated (sensitivity check)"
HW_MUL = "Holt-Winters, multiplicative on rupee price (check)"
LGBM = "Global model, LightGBM (P50)"
PREV = "Global model set before testing (P50)"
COMBO = "Average of global model and Holt-Winters"
NAIVE = "No change"
SNAIVE = "Seasonal naive"
FLAT = 0.05  # direction classes: down < -5%, flat, up > +5%


def _cls(e: pl.Expr) -> pl.Expr:
    return pl.when(e > np.log1p(FLAT)).then(1).when(e < np.log1p(-FLAT)).then(-1).otherwise(0)


def point_scores(ev: pl.DataFrame, by: list[str], methods: dict[str, pl.Expr]) -> pl.DataFrame:
    """One row per group and method: n, MAPE (%), MAE (Rs/q), direction accuracy (%),
    skill vs no change (% reduction in MAE). ``methods`` must include NAIVE."""
    act = (pl.col("lp") + pl.col("y")).exp()
    parts = []
    for name, e in methods.items():
        f = (pl.col("lp") + e).exp()
        parts.append(
            ev.group_by(by)
            .agg(
                pl.len().alias("n"),
                ((f - act).abs() / act).mean().mul(100).alias("mape"),
                (f - act).abs().mean().alias("mae"),
                (_cls(e) == _cls(pl.col("y"))).mean().mul(100).alias("direction"),
            )
            .with_columns(pl.lit(name).alias("method"))
        )
    out = pl.concat(parts)
    base = out.filter(pl.col("method") == NAIVE).select(*by, pl.col("mae").alias("_mae0"))
    order = {m: i for i, m in enumerate(methods)}
    return (
        out.join(base, on=by, how="left")
        .with_columns((100 * (1 - pl.col("mae") / pl.col("_mae0"))).alias("skill_vs_naive"))
        .drop("_mae0")
        .with_columns(pl.col("method").replace_strict(order, return_dtype=pl.Int32).alias("_o"))
        .sort([*by, "_o"])
        .drop("_o")
    )


def band_scores(ev: pl.DataFrame, by: list[str], bands: dict[str, tuple[pl.Expr, pl.Expr]]) -> pl.DataFrame:
    """One row per group and band method: coverage of the actual price (%), mean width and
    mean interval score (both % of today's price; lower score is better)."""
    p0 = pl.col("lp").exp()
    act = (pl.col("lp") + pl.col("y")).exp()
    parts = []
    for name, (lo, hi) in bands.items():
        lo_p, hi_p = (pl.col("lp") + lo).exp(), (pl.col("lp") + hi).exp()
        parts.append(
            ev.group_by(by)
            .agg(
                pl.len().alias("n"),
                ((act >= lo_p) & (act <= hi_p)).mean().mul(100).alias("coverage"),
                ((hi_p - lo_p) / p0).mean().mul(100).alias("width_pct"),
                (interval_score(lo_p, hi_p, act) / p0).mean().mul(100).alias("interval_score_pct"),
            )
            .with_columns(pl.lit(name).alias("method"))
        )
    order = {m: i for i, m in enumerate(bands)}
    return (
        pl.concat(parts)
        .with_columns(pl.col("method").replace_strict(order, return_dtype=pl.Int32).alias("_o"))
        .sort([*by, "_o"])
        .drop("_o")
    )
