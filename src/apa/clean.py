"""Cleaning, harmonisation and weekly aggregation of Agmarknet daily price records.

All transformations are lazy (polars LazyFrame) until a caller collects.
Rules are deliberately simple and documented, because AMIU staff must be able
to audit them:

* market names: lower-case, whitespace collapsed, trailing " APMC" removed
  (Agmarknet 2.0 renamed most Assam markets "<name> APMC" in Dec 2025), plus a
  short explicit alias table for other renames seen in the data;
* grade is ignored (its labels changed at the same migration); a series is
  state x market x commodity x variety;
* weekly price = median of daily modal prices in the Monday-based week;
* outlier = weekly price more than 3x above or below the median of the series'
  previous 8 observed weeks (trailing window, so no future data is used).
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import polars as pl

MARKET_ALIASES = {
    # observed renames (old name -> canonical key); keys are post-normalisation
    "uparhali pmy": "p.o. uparhali guwahati",
    "gandhimaidan smy, hojai": "gandhimaidan",
    "tinsukia market": "tinsukia",
}
SERIES = ["state", "market_key", "commodity", "variety"]
OUTLIER_LOG_RATIO = math.log(3.0)


def market_key(col: str = "market") -> pl.Expr:
    base = (
        pl.col(col)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.replace(r"\s+apmc$", "")
    )
    return base.replace(MARKET_ALIASES)


def load_daily(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .filter(
            pl.col("date").is_not_null()
            & pl.col("modal_price").is_not_null()
            & (pl.col("modal_price") > 0)
        )
        .with_columns(
            pl.col("state").str.strip_chars(),
            pl.col("commodity").str.strip_chars(),
            pl.col("variety").str.strip_chars(),
            pl.col("district").str.strip_chars().str.to_titlecase(),
            market_key().alias("market_key"),
        )
    )


def market_labels(daily: pl.LazyFrame) -> pl.LazyFrame:
    """Readable label per market key: most frequent raw name without ' APMC'."""
    return daily.group_by("state", "market_key").agg(
        pl.col("market").str.replace(r"\s+APMC$", "").mode().first().alias("market_label"),
        pl.col("district").mode().first().alias("district"),
    )


def weekly(daily: pl.LazyFrame) -> pl.LazyFrame:
    return (
        daily.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by([*SERIES, "week"])
        .agg(
            pl.col("modal_price").median().alias("price"),
            pl.col("min_price").median().alias("pmin"),
            pl.col("max_price").median().alias("pmax"),
            pl.col("date").n_unique().alias("n_days"),
            pl.len().alias("n_rows"),
        )
    )


def flag_outliers(wk: pl.LazyFrame) -> pl.LazyFrame:
    return (
        wk.sort([*SERIES, "week"])
        .with_columns(pl.col("price").log().alias("lp"))
        .with_columns(
            pl.col("lp").shift(1).rolling_median(window_size=8, min_samples=3).over(SERIES).alias("ref_lp")
        )
        .with_columns(
            ((pl.col("lp") - pl.col("ref_lp")).abs() > OUTLIER_LOG_RATIO).fill_null(False).alias("outlier")
        )
    )


def last_full_week(max_date: dt.date) -> dt.date:
    """Monday of the last complete Monday-Sunday week on or before max_date."""
    monday = max_date - dt.timedelta(days=max_date.weekday())
    return monday if max_date.weekday() == 6 else monday - dt.timedelta(weeks=1)


def coverage(wk: pl.LazyFrame, start: dt.date, end: dt.date, state: str = "Assam") -> pl.DataFrame:
    """Weekly coverage of every series of `state` in [start, end] (week starts)."""
    n_weeks = (end - start).days // 7 + 1
    return (
        wk.filter(
            (pl.col("state") == state)
            & pl.col("week").is_between(start, end)
            & ~pl.col("outlier")
        )
        .group_by(SERIES)
        .agg(
            pl.col("week").n_unique().alias("weeks"),
            pl.col("n_days").sum().alias("days"),
            pl.col("week").min().alias("first_week"),
            pl.col("week").max().alias("last_week"),
        )
        .with_columns((pl.col("weeks") / n_weeks).alias("coverage"))
        .collect()
    )


def select_pairs(cov: pl.DataFrame, k: int = 3) -> pl.DataFrame:
    """Selection rule (fixed before any modelling):
    1. per market x commodity keep the variety with most covered weeks;
    2. rank pairs by covered weeks, then reporting days, then name;
    3. keep the top k, whatever commodities they are.
    """
    dom = (
        cov.sort(["weeks", "days", "variety"], descending=[True, True, False])
        .group_by(["state", "market_key", "commodity"], maintain_order=True)
        .first()
    )
    return dom.sort(["weeks", "days", "market_key", "commodity"], descending=[True, True, False, False]).head(k)
