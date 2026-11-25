import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", app_title="Assam mandi data profile and pair selection")


@app.cell
def _():
    import datetime as dt
    import json

    import marimo as mo
    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    from apa import clean, paths, viz

    viz.theme()
    return clean, dt, json, mo, paths, pl, plt, sns, viz


@app.cell
def _(mo):
    mo.md(
        """
        # 1 · Data profile and selection of pilot series

        Source: Agmarknet daily mandi prices (Directorate of Marketing & Inspection),
        published on data.gov.in (OGD resource `35985678-0d79-46b4-9ed6-6f13308a1d24`,
        Government Open Data License – India). This notebook measures how complete the
        public Assam record is, harmonises market names, and applies the fixed selection
        rule: **the three Assam market × commodity series with the most weeks reported in
        the last 156 complete weeks**, whatever they are.
        """
    )
    return


@app.cell
def _(clean, paths, pl):
    daily = clean.load_daily(paths.RAW_PARQUET)
    wk = clean.flag_outliers(clean.weekly(daily)).collect().lazy()
    meta = (
        pl.scan_parquet(paths.RAW_PARQUET)
        .select(
            pl.len().alias("rows"),
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
            pl.col("retrieved_at").min().alias("retrieved_from"),
            pl.col("retrieved_at").max().alias("retrieved_to"),
        )
        .collect()
        .row(0, named=True)
    )
    by_state = (
        pl.scan_parquet(paths.RAW_PARQUET)
        .group_by("state")
        .agg(pl.len().alias("rows"), pl.col("date").min().alias("first"), pl.col("date").max().alias("last"))
        .sort("rows", descending=True)
        .collect()
    )
    by_state
    return by_state, daily, meta, wk


@app.cell
def _(clean, daily, dt, pl):
    assam_daily = daily.filter(pl.col("state") == "Assam")
    max_date = assam_daily.select(pl.col("date").max()).collect().item()
    win_end = clean.last_full_week(max_date)
    win_start = win_end - dt.timedelta(weeks=155)
    raw_names = assam_daily.select(pl.col("market").n_unique()).collect().item()
    keys = assam_daily.select(pl.col("market_key").n_unique()).collect().item()
    apmc_renamed = (
        assam_daily.select("market", "market_key").unique()
        .group_by("market_key").agg(pl.col("market").n_unique().alias("names"))
        .filter(pl.col("names") > 1).collect().height
    )
    assam_rows = assam_daily.select(pl.len()).collect().item()
    assam_first = assam_daily.select(pl.col("date").min()).collect().item()
    (max_date, win_start, win_end, raw_names, keys, apmc_renamed)
    return (apmc_renamed, assam_daily, assam_first, assam_rows, keys, max_date, raw_names, win_end, win_start)


@app.cell
def _(assam_daily, pl):
    # reporting timeline: distinct (harmonised) Assam markets reporting each week
    timeline = (
        assam_daily.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
        .group_by("week")
        .agg(pl.col("market_key").n_unique().alias("markets"), pl.len().alias("rows"))
        .sort("week")
        .collect()
    )
    yearly = (
        assam_daily.group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.len().alias("rows"), pl.col("market_key").n_unique().alias("markets"))
        .sort("year")
        .collect()
    )
    yearly
    return timeline, yearly


@app.cell
def _(assam_first, max_date, meta, mo, pl, plt, sns, timeline, viz):
    _all_weeks = pl.DataFrame(
        {"week": pl.date_range(timeline["week"].min(), timeline["week"].max(), "1w", eager=True)}
    )
    _t = _all_weeks.join(timeline, on="week", how="left").with_columns(pl.col("markets").fill_null(0))
    _fig, _ax = plt.subplots(figsize=(9.5, 3.2))
    sns.lineplot(data=_t.to_pandas(), x="week", y="markets", ax=_ax, color=viz.PALETTE["model"], linewidth=0.8)
    _ax.fill_between(_t["week"].to_numpy(), _t["markets"].to_numpy(), color=viz.PALETTE["band"], alpha=0.5)
    _ax.set(xlabel="", ylabel="markets reporting", title="Assam markets reporting prices each week, Agmarknet record")
    fig_timeline = viz.save(
        _fig, "f1_reporting_timeline",
        f"Source: Agmarknet via data.gov.in OGD API (resource 35985678…), Assam, {assam_first}–{max_date}, retrieved {meta['retrieved_to'][:10]}. Market names harmonised.",
    )
    mo.image(fig_timeline)
    return (fig_timeline,)


@app.cell
def _(pl, timeline):
    # reporting gaps: runs of >= 4 consecutive weeks with no Assam price at all
    _all = pl.DataFrame({"week": pl.date_range(timeline["week"].min(), timeline["week"].max(), "1w", eager=True)})
    _t = (
        _all.join(timeline, on="week", how="left")
        .with_columns(pl.col("markets").fill_null(0))
        .with_columns((pl.col("markets") == 0).alias("empty"))
        .with_columns((pl.col("empty") != pl.col("empty").shift(1)).fill_null(True).cum_sum().alias("run"))
    )
    gaps = (
        _t.filter(pl.col("empty"))
        .group_by("run")
        .agg(pl.col("week").min().alias("start"), pl.col("week").max().alias("end"), pl.len().alias("weeks"))
        .filter(pl.col("weeks") >= 4)
        .sort("start")
        .drop("run")
    )
    gap_stats = {
        "weeks_total": _t.height,
        "weeks_empty": int(_t["empty"].sum()),
        "weeks_lt5_markets": int((_t["markets"] < 5).sum()),
        "max_markets_week": int(_t["markets"].max()),
        "median_markets_week": float(_t["markets"].median()),
    }
    gaps
    return gap_stats, gaps


@app.cell
def _(clean, win_end, win_start, wk):
    cov = clean.coverage(wk, win_start, win_end, state="Assam")
    selected = clean.select_pairs(cov, k=3)
    top = clean.select_pairs(cov, k=25)
    selected
    return cov, selected, top


@app.cell
def _(cov, pl):
    pairs = (
        cov.sort(["weeks", "days"], descending=[True, True])
        .group_by(["market_key", "commodity"], maintain_order=True).first()
    )
    cov_stats = pairs.select(
        pl.len().alias("n_pairs"),
        pl.col("coverage").median().mul(100).alias("median_cov_pct"),
        (pl.col("coverage") >= 0.5).sum().alias("n_ge50"),
        (pl.col("coverage") >= 0.25).sum().alias("n_ge25"),
        pl.col("coverage").max().mul(100).alias("max_cov_pct"),
    ).row(0, named=True)
    cov_stats
    return cov_stats, pairs


@app.cell
def _(daily, meta, mo, pl, plt, sns, top, viz, win_end, win_start, wk):
    from apa.clean import market_labels as _ml
    _lab = _ml(daily).collect()
    _top = top.join(_lab, on=["state", "market_key"], how="left").with_columns(
        (pl.col("market_label") + " · " + pl.col("commodity")).alias("series")
    ).with_row_index("rank")
    _w = (
        wk.filter(
            (pl.col("state") == "Assam") & pl.col("week").is_between(win_start, win_end) & ~pl.col("outlier")
        )
        .join(_top.lazy().select("market_key", "commodity", "variety", "series", "rank"),
              on=["market_key", "commodity", "variety"], how="inner")
        .select("series", "rank", "week")
        .with_columns(pl.lit(1).alias("v"))
        .collect()
    )
    _weeks = pl.date_range(win_start, win_end, "1w", eager=True)
    _mat = (
        _top.select("series", "rank").join(pl.DataFrame({"week": _weeks}), how="cross")
        .join(_w.select("series", "week", "v"), on=["series", "week"], how="left")
        .with_columns(pl.col("v").fill_null(0))
        .sort("rank", "week")
    )
    _pd = _mat.to_pandas().pivot(index="series", columns="week", values="v")
    _pd = _pd.loc[_top.sort("rank")["series"].to_list()]
    _fig, _ax = plt.subplots(figsize=(9.5, 5.6))
    sns.heatmap(_pd, cmap=["#f1f1f1", viz.PALETTE["model"]], cbar=False, ax=_ax, linewidths=0)
    _ticks = list(range(0, len(_weeks), 13))
    _ax.set_xticks([t + 0.5 for t in _ticks])
    _ax.set_xticklabels([_weeks[t].strftime("%b %Y") for t in _ticks], rotation=0, fontsize=7)
    _ax.set(xlabel="", ylabel="", title="Weeks with a reported price: 25 best-covered Assam series (last 156 weeks)")
    _ax.tick_params(axis="y", labelsize=7)
    for _i in range(3):
        _ax.get_yticklabels()[_i].set_fontweight("bold")
    fig_coverage = viz.save(
        _fig, "f2_coverage_heatmap",
        f"Source: Agmarknet via data.gov.in OGD API, weeks {win_start}–{win_end}, retrieved {meta['retrieved_to'][:10]}. Top 3 rows (bold) = selected pilot series.",
    )
    mo.image(fig_coverage)
    return (fig_coverage,)


@app.cell
def _(assam_daily, pl, selected, win_end, win_start):
    # same-day price dispersion across Assam markets for the selected commodities
    _sel_c = selected.get_column("commodity").unique().to_list()
    dispersion = (
        assam_daily.filter(pl.col("commodity").is_in(_sel_c) & pl.col("date").is_between(win_start, win_end))
        .group_by("commodity", "date", "market_key")
        .agg(pl.col("modal_price").median())
        .group_by("commodity", "date")
        .agg(pl.len().alias("n"), pl.col("modal_price").max().alias("hi"), pl.col("modal_price").min().alias("lo"))
        .filter(pl.col("n") >= 3)
        .with_columns((pl.col("hi") / pl.col("lo") - 1).mul(100).alias("spread_pct"))
        .group_by("commodity")
        .agg(pl.len().alias("days"), pl.col("spread_pct").median().alias("median_spread_pct"), pl.col("n").median().alias("median_markets"))
        .sort("commodity")
        .collect()
    )
    dispersion
    return (dispersion,)


@app.cell
def _(assam_daily, dt, pl, win_end):
    # renamed markets (several raw names -> one harmonised market) and markets gone silent
    renames = (
        assam_daily.group_by("market_key", "market")
        .agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"), pl.len().alias("rows"))
        .sort("market_key", "first")
        .group_by("market_key", maintain_order=True)
        .agg(pl.col("market"), pl.col("first"), pl.col("last"))
        .filter(pl.col("market").list.len() > 1)
        .collect()
    )
    _last = assam_daily.group_by("market_key").agg(
        pl.col("date").max().alias("last"), pl.col("market").last().alias("name")
    ).collect()
    _since = win_end - dt.timedelta(weeks=26)
    silent = _last.filter(pl.col("last").is_between(_since, win_end - dt.timedelta(weeks=3))).sort("last")
    renames
    return renames, silent


@app.cell
def _(pl, wk):
    outliers = wk.select(
        pl.col("outlier").sum().alias("n_outlier"), pl.len().alias("n_weeks")
    ).collect().row(0, named=True)
    outliers
    return (outliers,)


@app.cell
def _(
    apmc_renamed, assam_first, assam_rows, by_state, cov_stats, daily, dispersion, fig_coverage, fig_timeline,
    gap_stats, gaps, json, keys, max_date, meta, outliers, paths, pl, raw_names, renames, selected, silent, win_end,
    win_start, yearly,
):
    from apa.clean import market_labels as _ml2
    _lab = _ml2(daily).collect()
    _sel = selected.join(_lab, on=["state", "market_key"], how="left")
    selection = {
        "rule": "Top 3 Assam market x commodity series by number of weeks with a reported price in the last 156 complete weeks; ties broken by reporting days. Fixed before any modelling.",
        "window": [str(win_start), str(win_end)],
        "pairs": [
            {"market_key": r["market_key"], "market": r["market_label"], "district": r["district"],
             "commodity": r["commodity"], "variety": r["variety"], "weeks": r["weeks"], "days": r["days"],
             "coverage_pct": round(100 * r["coverage"], 1)}
            for r in _sel.iter_rows(named=True)
        ],
    }
    (paths.OUT / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    _yr = {int(r["year"]): {"rows": r["rows"], "markets": r["markets"]} for r in yearly.iter_rows(named=True)}
    profile_metrics = {
        "source": "Agmarknet (DMI, MoA&FW) via data.gov.in OGD API, resource 35985678-0d79-46b4-9ed6-6f13308a1d24",
        "rows_total": meta["rows"],
        "date_first": str(meta["first"]), "date_last": str(meta["last"]),
        "retrieved": meta["retrieved_to"][:10],
        "states": by_state.get_column("state").to_list(),
        "rows_by_state": {r["state"]: r["rows"] for r in by_state.iter_rows(named=True)},
        "assam_rows": assam_rows, "assam_first": str(assam_first), "assam_last": str(max_date),
        "assam_market_names_raw": raw_names, "assam_markets_harmonised": keys,
        "assam_markets_renamed": apmc_renamed,
        "assam_yearly": _yr,
        "gaps": gaps.to_dicts(),
        "gap_stats": gap_stats,
        "renames": renames.to_dicts(),
        "silent_markets": silent.to_dicts(),
        "window": [str(win_start), str(win_end)],
        "coverage": cov_stats,
        "dispersion": dispersion.to_dicts(),
        "outliers": outliers,
        "figures": {"timeline": fig_timeline, "coverage": fig_coverage},
        "selection": selection,
    }
    (paths.OUT / "metrics_profile.json").write_text(json.dumps(profile_metrics, indent=2, default=str), encoding="utf-8")
    profile_metrics
    return (profile_metrics,)


if __name__ == "__main__":
    app.run()
