"""Features and targets for the global (pooled, cross-market) weekly price model.

One series = state x market x commodity (dominant variety). Every series is put
on a complete Monday-based weekly grid, so missing weeks stay explicit (null).

No-lookahead rule: a feature for origin week t uses only weeks <= t. Only the
targets y_h = log price(t+h) - log price(t) look forward, and they are labels.
"""

from __future__ import annotations

import datetime as dt
import math

import polars as pl

KEY = ["state", "market_key", "commodity"]
HORIZONS = (1, 2, 3, 4)
LAGS = (1, 2, 4, 8, 13, 26, 52)
REGIONS = {"as": "Assam markets", "nb": "neighbouring-State markets"}

OWN_FEATURES = [
    *[f"d{k}" for k in LAGS],
    "dev4", "dev13", "dev52", "vol13", "obs13", "obs52", "rng", "n_days",
    "woy_sin", "woy_cos", "commodity_code", "is_assam",
]
REGIONAL_FEATURES = [
    *[f"{r}_rd{k}" for r in REGIONS for k in (1, 4, 13, 52)],
    "as_cover", "nb_cover", "sp_dev26",
]
FEATURES = OWN_FEATURES + REGIONAL_FEATURES
CATEGORICAL = ["commodity_code"]


def dominant_variety(wk: pl.LazyFrame) -> pl.LazyFrame:
    """Keep, per state x market x commodity, the variety with most observed weeks."""
    best = (
        wk.filter(~pl.col("outlier"))
        .group_by([*KEY, "variety"])
        .agg(pl.len().alias("nw"))
        .sort(["nw", "variety"], descending=[True, False])
        .group_by(KEY, maintain_order=True)
        .first()
        .select([*KEY, "variety"])
    )
    return wk.join(best, on=[*KEY, "variety"], how="inner")


def weekly_grid(wk: pl.LazyFrame, end_week: dt.date) -> pl.DataFrame:
    """Complete weekly grid per series from its first observation to end_week.
    Outlier weeks are treated as missing."""
    obs = (
        wk.filter(~pl.col("outlier") & (pl.col("week") <= end_week))
        .select([*KEY, "variety", "week", "price", "pmin", "pmax", "n_days"])
        .collect()
    )
    spans = obs.group_by(KEY).agg(pl.col("week").min().alias("w0"), pl.col("variety").first())
    grid = (
        spans.with_columns(pl.date_ranges("w0", pl.lit(end_week), interval="1w").alias("week"))
        .explode("week")
        .drop("w0")
    )
    return (
        grid.join(obs.drop("variety"), on=[*KEY, "week"], how="left")
        .sort([*KEY, "week"])
        .with_columns(pl.col("price").log().alias("lp"))
    )


def _regional(g: pl.DataFrame) -> pl.DataFrame:
    """Chain-linked regional price indices per commodity, for Assam ('as') and
    neighbouring States ('nb'): weekly median of observed log changes, cumulated."""
    d = g.select(
        "commodity", "week", "d1", "lp",
        pl.when(pl.col("state") == "Assam").then(pl.lit("as")).otherwise(pl.lit("nb")).alias("region"),
    )
    agg = d.group_by("commodity", "region", "week").agg(
        pl.col("d1").median().alias("rd1"),
        pl.col("d1").count().alias("rn"),
        pl.col("lp").median().alias("rlvl"),
    )
    weeks = g.select(pl.col("week").unique()).sort("week")
    combos = agg.select("commodity", "region").unique()
    full = combos.join(weeks, how="cross").join(agg, on=["commodity", "region", "week"], how="left").sort(
        "commodity", "region", "week"
    )
    over = ["commodity", "region"]
    full = full.with_columns(pl.col("rd1").fill_null(0.0).cum_sum().over(over).alias("R"))
    full = full.with_columns(
        *[(pl.col("R") - pl.col("R").shift(k).over(over)).alias(f"rd{k}") for k in (4, 13, 52)],
        (pl.col("rn").fill_null(0) > 0).cast(pl.Float64).rolling_mean(13, min_samples=1).over(over).alias("cover"),
    )
    out = None
    for r in REGIONS:
        part = full.filter(pl.col("region") == r).select(
            "commodity", "week",
            *[pl.col(f"rd{k}").alias(f"{r}_rd{k}") for k in (1, 4, 13, 52)],
            pl.col("cover").alias(f"{r}_cover"),
            pl.col("rlvl").alias(f"{r}_lvl"),
        )
        out = part if out is None else out.join(part, on=["commodity", "week"], how="full", coalesce=True)
    return out


def build(g: pl.DataFrame, commodity_codes: dict[str, int]) -> pl.DataFrame:
    """Add features (past-only) and targets (future labels) to a weekly grid."""
    lp = pl.col("lp")
    g = g.sort([*KEY, "week"])  # all window expressions below assume week order within a series
    g = g.with_columns(*[(lp - lp.shift(k).over(KEY)).alias(f"d{k}") for k in LAGS])
    g = g.with_columns(
        (lp - lp.rolling_mean(4, min_samples=2).over(KEY)).alias("dev4"),
        (lp - lp.rolling_mean(13, min_samples=4).over(KEY)).alias("dev13"),
        (lp - lp.rolling_mean(52, min_samples=10).over(KEY)).alias("dev52"),
        pl.col("d1").rolling_std(13, min_samples=4).over(KEY).alias("vol13"),
        lp.is_not_null().cast(pl.Float64).rolling_mean(13, min_samples=1).over(KEY).alias("obs13"),
        lp.is_not_null().cast(pl.Float64).rolling_mean(52, min_samples=1).over(KEY).alias("obs52"),
        (pl.col("pmax") / pl.col("pmin")).log().alias("rng"),
        (2 * math.pi * pl.col("week").dt.week() / 52.18).sin().alias("woy_sin"),
        (2 * math.pi * pl.col("week").dt.week() / 52.18).cos().alias("woy_cos"),
        pl.col("commodity").replace_strict(commodity_codes, return_dtype=pl.Float64).alias("commodity_code"),
        (pl.col("state") == "Assam").cast(pl.Float64).alias("is_assam"),
    )
    reg = _regional(g)
    g = g.join(reg, on=["commodity", "week"], how="left").sort([*KEY, "week"])  # joins need not keep order
    g = g.with_columns((lp - pl.col("as_lvl")).alias("sp"))
    g = g.with_columns((pl.col("sp") - pl.col("sp").rolling_mean(26, min_samples=4).over(KEY)).alias("sp_dev26"))
    # targets (labels only) and seasonal-naive baseline (past-only)
    g = g.with_columns(
        *[(lp.shift(-h).over(KEY) - lp).alias(f"y{h}") for h in HORIZONS],
        *[(lp.shift(52 - h).over(KEY) - lp.shift(52).over(KEY)).alias(f"sn{h}") for h in HORIZONS],
    )
    return g.with_columns(pl.col("n_days").cast(pl.Float64))
