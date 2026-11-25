"""Long-horizon (season-ahead) price intelligence.

Seasonal profile: for each series-year with enough months observed, the
deviation of each month's mean log price from that year's mean. The profile
for a commodity is the median deviation across series-years (pooled over Assam
and neighbouring-State markets), with the inter-quartile range as a measure of
year-to-year variation.

Out-of-sample check: the profile estimated only from years before Y is used to
predict the monthly shape of year Y in Assam markets, and compared with a flat
(no-seasonality) outlook.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .features import KEY

MIN_MONTHS = 8


def monthly(g: pl.DataFrame) -> pl.DataFrame:
    m = (
        g.filter(pl.col("lp").is_not_null())
        .with_columns(pl.col("week").dt.year().alias("year"), pl.col("week").dt.month().alias("month"))
        .group_by([*KEY, "year", "month"])
        .agg(pl.col("lp").mean().alias("mlp"))
    )
    full_years = (
        m.group_by([*KEY, "year"]).agg(pl.len().alias("nm")).filter(pl.col("nm") >= MIN_MONTHS)
    )
    return (
        m.join(full_years, on=[*KEY, "year"], how="inner")
        .with_columns((pl.col("mlp") - pl.col("mlp").mean().over([*KEY, "year"])).alias("dev"))
    )


def profile(mon: pl.DataFrame, years_lt: int | None = None) -> pl.DataFrame:
    m = mon if years_lt is None else mon.filter(pl.col("year") < years_lt)
    return (
        m.group_by("commodity", "month")
        .agg(
            pl.col("dev").median().alias("p50"),
            pl.col("dev").quantile(0.25).alias("p25"),
            pl.col("dev").quantile(0.75).alias("p75"),
            pl.len().alias("n"),
            pl.struct(KEY + ["year"]).n_unique().alias("series_years"),
        )
        .sort("commodity", "month")
    )


def backtest(mon: pl.DataFrame, test_years: list[int]) -> pl.DataFrame:
    rows = []
    for y in test_years:
        prof = profile(mon, years_lt=y).select("commodity", "month", pl.col("p50").alias("pred"))
        act = mon.filter((pl.col("year") == y) & (pl.col("state") == "Assam"))
        j = act.join(prof, on=["commodity", "month"], how="inner")
        if j.height == 0:
            continue
        for c in j.get_column("commodity").unique().sort().to_list():
            jc = j.filter(pl.col("commodity") == c)
            a = jc.get_column("dev").to_numpy()
            p = jc.get_column("pred").to_numpy()
            corr = float(np.corrcoef(a, p)[0, 1]) if len(a) > 2 and np.std(p) > 0 else float("nan")
            rows.append({
                "year": y, "commodity": c, "n_months": len(a),
                "n_series": jc.select(KEY).unique().height,
                "mae_profile_pct": float(np.mean(np.abs(np.expm1(a) - np.expm1(p))) * 100),
                "mae_flat_pct": float(np.mean(np.abs(np.expm1(a))) * 100),
                "corr": corr,
            })
    return pl.DataFrame(rows)
