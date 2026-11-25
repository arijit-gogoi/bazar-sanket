import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", app_title="Assam price forecasting prototype: LightGBM, multiplicative Holt-Winters, more commodities")


@app.cell
def _():
    import datetime as dt
    import json
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns

    from apa import clean, compare, features, model, paths, smoothing, viz

    viz.theme()
    RUN_T0 = time.time()
    return RUN_T0, clean, compare, dt, features, json, mo, model, np, paths, pl, plt, smoothing, sns, time, viz


@app.cell
def _(mo):
    mo.md(
        """
        # 4 · LightGBM, multiplicative Holt-Winters and more commodities

        Added on request after the first results. Settings were fixed before each run.
        After seeing these results the applicant chose part B's scikit-learn model (trained
        on all Assam markets of the good-data commodities) as the advisory model of
        `nb/forecast.py`; this notebook checks that the two are identical.

        * **A. Main backtest rows (pilot commodities):** the advisory model, LightGBM on the
          same data and tasks, and the configuration set before testing (pilot commodities plus
          neighbouring-State markets, `out/backtest_prev.parquet`), on identical rows.
        * **B. More commodities:** every Assam commodity with good data, by a coverage rule
          fixed before any model was run on them: at least 1,000 market-weeks with a price in
          the last 156 complete weeks and at least 100 in the last 52 (the test year).
          Assam markets only (neighbouring-State prices were fetched for the three pilot
          commodities only). Global model (scikit-learn and LightGBM), Holt-Winters on the log
          price, textbook multiplicative Holt-Winters on the rupee price, no change and
          seasonal naive, all on identical rows. Accuracy only: storage and shelf-life rules
          exist for the pilot commodities alone, so no advisories are made for the others.
        """
    )
    return


@app.cell
def _(clean, paths, pl):
    # good-data rule: coverage only, computed on all Assam commodities (names as published)
    _assam = clean.load_daily(paths.RAW_PARQUET).filter(pl.col("state") == "Assam")
    end_week = clean.last_full_week(_assam.select(pl.col("date").max()).collect().item())
    _wk = clean.flag_outliers(clean.weekly(_assam)).collect().lazy()
    comm_cov = clean.commodity_coverage(_wk, end_week)
    ext_commodities = sorted(comm_cov.filter(pl.col("good_data")).get_column("commodity").to_list())
    retrieved = pl.scan_parquet(paths.RAW_PARQUET).select(pl.col("retrieved_at").max()).collect().item()[:10]
    (end_week, ext_commodities)
    return comm_cov, end_week, ext_commodities, retrieved


@app.cell
def _(ext_bt, ext_commodities, features, json, paths, pl):
    # ---- A. main backtest rows (pilot commodities). The advisory model of nb/forecast.py must be exactly
    # part B's scikit-learn model; LightGBM comes from part B; the earlier configuration from backtest_prev.parquet
    _fc = json.loads((paths.OUT / "metrics_forecast.json").read_text(encoding="utf-8"))
    assert _fc["advisory_model"]["commodities"] == ext_commodities, "advisory model uses other commodities"
    _sel = json.loads((paths.OUT / "selection.json").read_text(encoding="utf-8"))
    pilot_commodities = sorted({p["commodity"] for p in _sel["pairs"]})
    _K = [*features.KEY, "week", "h"]
    _bt = pl.read_parquet(paths.OUT / "backtest.parquet").with_columns(pl.col("h").cast(pl.Int32))
    _prev = pl.read_parquet(paths.OUT / "backtest_prev.parquet").with_columns(pl.col("h").cast(pl.Int32)).select(
        *_K, pl.col("q10").alias("p10"), pl.col("q50").alias("p50"), pl.col("q90").alias("p90"))
    _x = ext_bt.filter(pl.col("commodity").is_in(pilot_commodities)).select(
        *_K, pl.col("q10").alias("x10"), pl.col("q50").alias("x50"), pl.col("q90").alias("x90"), "l10", "l50", "l90")
    main_ev = _bt.join(_x, on=_K, how="inner").join(_prev, on=_K, how="inner")
    assert main_ev.height == _bt.height == _x.height == _prev.height, "rows differ between models"
    _d = main_ev.select(*[(pl.col(f"q{q}") - pl.col(f"x{q}")).abs().max().alias(f"d{q}") for q in (10, 50, 90)]).row(0)
    assert max(_d) < 1e-9, f"advisory model differs from the tested 11-commodity model: {_d}"
    main_ev = main_ev.drop("x10", "x50", "x90")
    main_ev.height
    return main_ev, pilot_commodities


@app.cell
def _(compare, main_ev, pl):
    _M = {compare.MODEL: pl.col("q50"), compare.LGBM: pl.col("l50"), compare.PREV: pl.col("p50"),
          compare.NAIVE: pl.lit(0.0), compare.SNAIVE: pl.col("sn").fill_null(0.0)}
    _B = {compare.MODEL: (pl.col("q10"), pl.col("q90")), compare.LGBM: (pl.col("l10"), pl.col("l90")),
          compare.PREV: (pl.col("p10"), pl.col("p90"))}
    _all = main_ev.with_columns(pl.lit("all").alias("all"))
    main_points = compare.point_scores(_all, ["all"], _M)
    main_points_h = compare.point_scores(main_ev, ["h"], _M)
    main_points_c = compare.point_scores(main_ev, ["commodity"], _M)
    main_bands = compare.band_scores(_all, ["all"], _B)
    main_points
    return main_bands, main_points, main_points_c, main_points_h


@app.cell
def _(clean, dt, end_week, ext_commodities, features, model, paths, pl):
    # ---- B. good-data commodities, Assam markets only; both learners on the same tasks
    _codes = {c: i for i, c in enumerate(ext_commodities)}
    _daily = clean.load_daily(paths.RAW_PARQUET).filter(
        (pl.col("state") == "Assam") & pl.col("commodity").is_in(ext_commodities))
    _wk = clean.flag_outliers(clean.weekly(_daily)).collect().lazy()
    ext_grid = features.weekly_grid(features.dominant_variety(_wk), end_week)
    _GX = features.build(ext_grid, _codes)
    ext_test_start = end_week - dt.timedelta(weeks=51)
    _mask = pl.col("state") == "Assam"
    _K = [*features.KEY, "week", "h"]
    _hgb = model.rolling_backtest(_GX, features.FEATURES, _mask, ext_test_start, end_week, refit_every=4, learner="hgb")
    _lgb = model.rolling_backtest(_GX, features.FEATURES, _mask, ext_test_start, end_week, refit_every=4, learner="lgbm")
    ext_bt = _hgb.with_columns(pl.col("h").cast(pl.Int32)).join(
        _lgb.with_columns(pl.col("h").cast(pl.Int32)).select(
            *_K, pl.col("q10").alias("l10"), pl.col("q50").alias("l50"), pl.col("q90").alias("l90")),
        on=_K, how="inner")
    assert ext_bt.height == _hgb.height == _lgb.height
    ext_bt.write_parquet(paths.OUT / "backtest_extended.parquet")
    (ext_bt.height, ext_bt.select(features.KEY).unique().height)
    return ext_bt, ext_grid, ext_test_start


@app.cell
def _(ext_bt, ext_grid, features, pl, smoothing):
    _origins = ext_bt.select(*features.KEY, "week")
    _hw = smoothing.backtest(ext_grid, _origins)
    _hwm = smoothing.backtest(ext_grid, _origins, form="mul").rename(
        {"hw": "hw_mul", "hw_method": "hw_mul_method", "hw_converged": "hw_mul_converged"})
    _K = [*features.KEY, "week", "h"]
    ext_ev = ext_bt.join(_hw, on=_K, how="left").join(_hwm, on=_K, how="left")
    assert ext_ev.height == ext_bt.height and ext_ev.get_column("hw").null_count() == 0 and ext_ev.get_column("hw_mul").null_count() == 0
    _o = ext_ev.filter(pl.col("h") == 1)
    ext_hw_methods = pl.concat([
        _o.group_by(pl.col("hw_method").alias("method")).agg(pl.len().alias("origins"))
        .with_columns(pl.lit("log").alias("form")),
        _o.group_by(pl.col("hw_mul_method").alias("method")).agg(pl.len().alias("origins"))
        .with_columns(pl.lit("mul").alias("form")),
    ]).sort("form", "origins", descending=[False, True])
    ext_hw_methods
    return ext_ev, ext_hw_methods


@app.cell
def _(compare, ext_ev, pilot_commodities, pl):
    EXT_METHODS = {
        compare.MODEL: pl.col("q50"),
        compare.LGBM: pl.col("l50"),
        compare.HW: pl.col("hw"),
        compare.HW_MUL: pl.col("hw_mul"),
        compare.NAIVE: pl.lit(0.0),
        compare.SNAIVE: pl.col("sn").fill_null(0.0),
    }
    _B = {compare.MODEL: (pl.col("q10"), pl.col("q90")), compare.LGBM: (pl.col("l10"), pl.col("l90"))}
    _all = ext_ev.with_columns(pl.lit("all").alias("all"))
    ext_points = compare.point_scores(_all, ["all"], EXT_METHODS)
    ext_points_h = compare.point_scores(ext_ev, ["h"], EXT_METHODS)
    ext_points_c = compare.point_scores(ext_ev, ["commodity"], EXT_METHODS)
    ext_points_group = compare.point_scores(
        ext_ev.with_columns(pl.when(pl.col("commodity").is_in(pilot_commodities)).then(pl.lit("pilot"))
                            .otherwise(pl.lit("other")).alias("group")), ["group"], EXT_METHODS)
    ext_points_hw_rows = compare.point_scores(
        ext_ev.filter(pl.col("hw_method") == "Holt-Winters").with_columns(pl.lit("all").alias("all")), ["all"], EXT_METHODS)
    ext_bands = compare.band_scores(_all, ["all"], _B)
    ext_points
    return EXT_METHODS, ext_bands, ext_points, ext_points_c, ext_points_group, ext_points_h, ext_points_hw_rows


@app.cell
def _(compare, ext_points, ext_points_c, mo, pl, plt, retrieved, sns, viz):
    _pal = {compare.MODEL: viz.PALETTE["model"], compare.LGBM: "#2a9d8f", compare.HW: "#7b4ea3", compare.HW_MUL: "#c9a0dc"}
    _short = {compare.MODEL: "global model, scikit-learn", compare.LGBM: "global model, LightGBM",
              compare.HW: "Holt-Winters (log price)", compare.HW_MUL: "Holt-Winters, multiplicative"}
    _n = {r["commodity"]: r["n"] for r in ext_points_c.filter(pl.col("method") == compare.NAIVE).iter_rows(named=True)}
    _order = sorted(_n, key=lambda c: -_n[c])
    _d = pl.concat([
        ext_points.filter(pl.col("method").is_in(list(_pal))).select(pl.lit("All commodities").alias("commodity"), "method", "skill_vs_naive"),
        ext_points_c.filter(pl.col("method").is_in(list(_pal))).select("commodity", "method", "skill_vs_naive"),
    ]).with_columns(pl.col("method").replace_strict(_short))
    _fig, _ax = plt.subplots(figsize=(9.5, 5.4))
    sns.barplot(data=_d.to_pandas(), y="commodity", x="skill_vs_naive", hue="method", ax=_ax, orient="h",
                order=["All commodities", *_order], hue_order=[_short[k] for k in _pal],
                palette={_short[k]: v for k, v in _pal.items()})
    _ax.axvline(0, color="#333", lw=0.8)
    _ax.set(xlabel="% lower absolute error than no change (negative = worse)", ylabel="",
            title="Out-of-sample accuracy by commodity, Assam markets, 1–4 weeks ahead")
    _ax.tick_params(axis="y", labelsize=7.5)
    _ax.legend(fontsize=7, title="", loc="upper right", ncols=2)
    _ax.grid(axis="x", color="#ddd", lw=0.5)
    fig_ext = viz.save(_fig, "f8_more_commodities", f"Source: Agmarknet via data.gov.in OGD API, retrieved {retrieved}. Rolling-origin backtest, 12 months to latest complete week; identical rows for every method; commodities ordered by number of forecasts.")
    mo.image(fig_ext)
    return (fig_ext,)


@app.cell
def _(
    RUN_T0, comm_cov, end_week, ext_bands, ext_bt, ext_commodities, ext_hw_methods, ext_points, ext_points_c,
    ext_points_group, ext_points_h, ext_points_hw_rows, ext_test_start, fig_ext, json, main_bands, main_ev,
    main_points, main_points_c, main_points_h, paths, pilot_commodities, retrieved, time,
):
    from apa import clean as _clean
    ext_metrics = {
        "retrieved": retrieved,
        "end_week": str(end_week),
        "test_start": str(ext_test_start),
        "rule": {"min_weeks_3y": _clean.GOOD_DATA_MIN_WEEKS_3Y, "min_weeks_1y": _clean.GOOD_DATA_MIN_WEEKS_1Y,
                 "window_3y_weeks": 156, "window_1y_weeks": 52},
        "commodity_coverage": comm_cov.to_dicts(),
        "ext_commodities": ext_commodities,
        "pilot_commodities": pilot_commodities,
        "lgbm_settings": {"objective": "quantile", "learning_rate": 0.05, "n_estimators": 250, "num_leaves": 31,
                          "min_child_samples": 40, "reg_lambda": 1.0},
        "main": {"n_rows": main_ev.height, "points_all": main_points.to_dicts(), "points_by_h": main_points_h.to_dicts(),
                 "points_by_commodity": main_points_c.to_dicts(), "bands_all": main_bands.to_dicts()},
        "ext": {"n_rows": ext_bt.height, "n_series": ext_bt.select("market_key", "commodity").unique().height,
                "n_origins": ext_bt.filter(ext_bt["h"] == 1).height,
                "points_all": ext_points.to_dicts(), "points_by_h": ext_points_h.to_dicts(),
                "points_by_commodity": ext_points_c.to_dicts(), "points_by_group": ext_points_group.to_dicts(),
                "points_hw_rows": ext_points_hw_rows.to_dicts(), "bands_all": ext_bands.to_dicts(),
                "hw_methods": ext_hw_methods.to_dicts()},
        "figures": {"more_commodities": fig_ext},
        "runtime_min": round((time.time() - RUN_T0) / 60, 1),
    }
    (paths.OUT / "metrics_extended.json").write_text(json.dumps(ext_metrics, indent=2, default=str), encoding="utf-8")
    ext_metrics["runtime_min"]
    return (ext_metrics,)


if __name__ == "__main__":
    app.run()
