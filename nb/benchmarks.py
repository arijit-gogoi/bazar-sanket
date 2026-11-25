import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", app_title="Assam price forecasting prototype: other methods tried")


@app.cell
def _():
    import json
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import seaborn as sns

    from apa import clean, compare, features, paths, smoothing, viz, volatility

    viz.theme()
    RUN_T0 = time.time()
    return RUN_T0, clean, compare, features, json, mo, np, paths, pl, plt, smoothing, sns, time, viz, volatility


@app.cell
def _(mo):
    mo.md(
        """
        # 3 · Other methods tried: Holt-Winters and volatility pricing

        Both are evaluated on exactly the same out-of-sample rows as the global model
        (`out/backtest.parquet` from `nb/forecast.py`). Method choices were fixed before
        any result was seen (see `src/apa/smoothing.py` and `src/apa/volatility.py`).

        * **Holt-Winters** (triple exponential smoothing, statsmodels): additive damped
          trend + additive 52-week seasonality on the log price of each market series,
          re-estimated at every forecast origin. A pre-set equal-weight average of the
          global model and Holt-Winters is also scored.
        * **Volatility pricing:** an exponentially weighted (RiskMetrics-style) estimate of
          weekly price volatility turns into (a) an 80% price band and (b) a rupee price
          for downside risk: the expected shortfall below a floor (the value of a put
          option), e.g. the expected cost of guaranteeing 90% of today's price. The
          backtest compares predicted with realised shortfalls.
        """
    )
    return


@app.cell
def _(clean, features, json, paths, pl):
    _sel = json.loads((paths.OUT / "selection.json").read_text(encoding="utf-8"))
    _commodities = sorted({p["commodity"] for p in _sel["pairs"]})
    _daily = clean.load_daily(paths.RAW_PARQUET).filter(pl.col("commodity").is_in(_commodities))
    _wk = clean.flag_outliers(clean.weekly(_daily)).collect().lazy()
    end_week = clean.last_full_week(
        _daily.filter(pl.col("state") == "Assam").select(pl.col("date").max()).collect().item()
    )
    # same weekly grid as nb/forecast.py (dominant variety per series), Assam series only
    grid = features.weekly_grid(features.dominant_variety(_wk), end_week).filter(pl.col("state") == "Assam")
    bt = pl.read_parquet(paths.OUT / "backtest.parquet").with_columns(pl.col("h").cast(pl.Int32))
    retrieved = pl.scan_parquet(paths.RAW_PARQUET).select(pl.col("retrieved_at").max()).collect().item()[:10]
    _chk = bt.join(grid.select("state", "market_key", "commodity", "week", pl.col("lp").alias("lp_grid")),
                   on=["state", "market_key", "commodity", "week"], how="left")
    assert _chk.select(((pl.col("lp") - pl.col("lp_grid")).abs() < 1e-9).all()).item(), "grid differs from backtest"
    (bt.height, grid.height, str(end_week))
    return bt, end_week, grid, retrieved


@app.cell
def _(bt, grid, smoothing):
    _origins = bt.select("state", "market_key", "commodity", "week")
    hw = smoothing.backtest(grid, _origins)
    # sensitivity variant: interpolate every gap (added after seeing how few unbroken windows exist)
    hw_all = smoothing.backtest(grid, _origins, max_gap=None).rename(
        {"hw": "hw_all", "hw_method": "hw_all_method", "hw_converged": "hw_all_converged"})
    return hw, hw_all


@app.cell
def _(bt, features, grid, hw, hw_all, pl, volatility):
    sigma = volatility.ewma_sigma(grid)
    ev = (
        bt.join(hw, on=[*features.KEY, "week", "h"], how="left")
        .join(hw_all, on=[*features.KEY, "week", "h"], how="left")
        .join(sigma, on=[*features.KEY, "week"], how="left")
        .with_columns(((pl.col("q50") + pl.col("hw")) / 2).alias("combo"))
    )
    assert ev.height == bt.height and ev.get_column("hw").null_count() == 0 and ev.get_column("hw_all").null_count() == 0
    _o = ev.filter(pl.col("h") == 1)
    hw_methods = pl.concat([
        _o.group_by(pl.col("hw_method").alias("method")).agg(pl.len().alias("origins"), pl.col("hw_converged").mean().mul(100).alias("converged_pct"))
        .with_columns(pl.lit("primary").alias("variant")),
        _o.group_by(pl.col("hw_all_method").alias("method")).agg(pl.len().alias("origins"), pl.col("hw_all_converged").mean().mul(100).alias("converged_pct"))
        .with_columns(pl.lit("all gaps interpolated").alias("variant")),
    ]).sort("variant", "origins", descending=[True, True])
    hw_methods
    return ev, hw_methods


@app.cell
def _(ev, pl):
    # how fast does the spread of price changes grow with the horizon? (random walk: in proportion to h)
    _ms = ev.group_by("h").agg((pl.col("y") ** 2).mean().alias("ms")).sort("h")
    _m1 = _ms.filter(pl.col("h") == 1).get_column("ms").item()
    var_ratio = {int(_r["h"]): _r["ms"] / _m1 for _r in _ms.iter_rows(named=True)}
    var_ratio
    return (var_ratio,)


@app.cell
def _(compare, ev, pl):
    METHODS = {
        compare.MODEL: pl.col("q50"),
        compare.HW: pl.col("hw"),
        compare.HW_ALL: pl.col("hw_all"),
        compare.COMBO: pl.col("combo"),
        compare.NAIVE: pl.lit(0.0),
        compare.SNAIVE: pl.col("sn").fill_null(0.0),
    }
    _all = ev.with_columns(pl.lit("all").alias("all"))
    points_all = compare.point_scores(_all, ["all"], METHODS)
    points_by_h = compare.point_scores(ev, ["h"], METHODS)
    points_selected = compare.point_scores(ev.filter(pl.col("selected")), ["commodity"], METHODS)
    points_by_commodity = compare.point_scores(ev, ["commodity"], METHODS)
    points_all
    return METHODS, points_all, points_by_commodity, points_by_h, points_selected


@app.cell
def _(compare, ev, pl):
    # post-hoc check (added after the results): does the average gain from Holt-Winters itself, or from pulling the
    # global model halfway towards no change? Holt-Winters often fell back to damped Holt or no change.
    _m = {
        compare.MODEL: pl.col("q50"),
        "Global model halved (post-hoc check)": pl.col("q50") * 0.5,
        compare.COMBO: pl.col("combo"),
        compare.HW: pl.col("hw"),
        compare.NAIVE: pl.lit(0.0),
    }
    combo_check = compare.point_scores(ev.with_columns(pl.lit("all").alias("all")), ["all"], _m)
    combo_check_hw_rows = compare.point_scores(
        ev.filter(pl.col("hw_method") == "Holt-Winters").with_columns(pl.lit("all").alias("all")), ["all"], _m)
    combo_check
    return combo_check, combo_check_hw_rows


@app.cell
def _(compare, ev, pl, volatility):
    _z = volatility.Z80
    _s = pl.col("sigma") * pl.col("h").cast(pl.Float64).sqrt()
    BANDS = {
        "Global model quantiles (P10-P90)": (pl.col("q10"), pl.col("q90")),
        "Volatility band around today's price": (-_z * _s, _z * _s),
        "Volatility band around Holt-Winters forecast": (pl.col("hw") - _z * _s, pl.col("hw") + _z * _s),
    }
    _ev = ev.filter(pl.col("sigma").is_not_null()).with_columns(pl.lit("all").alias("all"))
    bands_all = compare.band_scores(_ev, ["all"], BANDS)
    bands_by_h = compare.band_scores(_ev, ["h"], BANDS)
    band_rows_pct = 100 * _ev.height / ev.height
    bands_all
    return BANDS, band_rows_pct, bands_all, bands_by_h


@app.cell
def _(ev, pl, volatility):
    _rows = volatility.pricing_rows(ev)
    _agg = [
        pl.len().alias("n"),
        pl.col("realised").mean(), pl.col("price_ewma").mean(), pl.col("price_model").mean(),
        pl.col("hit").mean(), pl.col("prob_ewma").mean(), pl.col("prob_model").mean(),
    ]
    pricing_by_h = _rows.group_by("floor", "h").agg(*_agg).sort("floor", "h", descending=[True, False])
    pricing_all = _rows.group_by("floor").agg(*_agg).sort("floor", descending=True)
    pricing_by_commodity = _rows.filter(pl.col("floor") == 0.9).group_by("commodity").agg(*_agg).sort("commodity")
    pricing_all
    return pricing_all, pricing_by_commodity, pricing_by_h


@app.cell
def _(compare, mo, pl, plt, points_by_h, pricing_by_h, retrieved, sns, viz):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
    _pal = {compare.MODEL: viz.PALETTE["model"], compare.HW: "#7b4ea3", compare.HW_ALL: "#b9a3d6",
            compare.COMBO: "#5aa0c8", compare.NAIVE: viz.PALETTE["naive"]}
    _short = {compare.MODEL: "global model", compare.HW: "Holt-Winters", compare.HW_ALL: "Holt-Winters, all gaps filled",
              compare.COMBO: "average of global model and Holt-Winters", compare.NAIVE: "no change"}
    _pb = points_by_h.filter(pl.col("method").is_in(list(_pal))).with_columns(pl.col("method").replace_strict(_short))
    sns.barplot(data=_pb.to_pandas(), x="h", y="mape", hue="method", ax=_a1,
                palette={_short[k]: v for k, v in _pal.items()}, hue_order=[_short[k] for k in _pal])
    for _c in _a1.containers:
        _a1.bar_label(_c, fmt="%.1f", fontsize=5.5, padding=1)
    _a1.set(xlabel="forecast horizon (weeks ahead)", ylabel="mean absolute % error",
            title="Point forecasts, all Assam series (lower is better)")
    _a1.legend(fontsize=6.5, title="", ncols=2, loc="upper left")
    _a1.set_ylim(0, _pb.get_column("mape").max() * 1.45)
    _p = pricing_by_h.filter(pl.col("floor") == 0.9).unpivot(
        index="h", on=["realised", "price_ewma", "price_model"], variable_name="what", value_name="pct")
    _lab = {"realised": "actual shortfall (backtest)", "price_ewma": "priced from EWMA volatility",
            "price_model": "priced from global model band"}
    _p = _p.with_columns(pl.col("what").replace_strict(_lab))
    sns.barplot(data=_p.to_pandas(), x="h", y="pct", hue="what", ax=_a2,
                palette=["#333333", "#d08c3a", viz.PALETTE["model"]], hue_order=list(_lab.values()))
    for _c in _a2.containers:
        _a2.bar_label(_c, fmt="%.2f", fontsize=5.5, padding=1)
    _a2.set(xlabel="weeks ahead", ylabel="% of today's price",
            title="Price of a floor at 90% of today's price")
    _a2.legend(fontsize=6.5, title="", loc="upper left")
    _a2.set_ylim(0, _p.get_column("pct").max() * 1.45)
    fig_bench = viz.save(_fig, "f7_other_methods", f"Source: Agmarknet via data.gov.in OGD API, retrieved {retrieved}. Same out-of-sample rows as the global model; 12 months to latest complete week.")
    mo.image(fig_bench)
    return (fig_bench,)


@app.cell
def _(end_week, ev, json, np, paths, pl, sigma, volatility):
    # price risk now, for the selected series (latest complete week), using the published advisory bands
    _fc = json.loads((paths.OUT / "metrics_forecast.json").read_text(encoding="utf-8"))
    _sel = ev.filter(pl.col("selected")).select("market_key", "commodity").unique()
    _now = (
        sigma.join(_sel, on=["market_key", "commodity"], how="semi")
        .filter(pl.col("week") <= end_week).sort("week").group_by("market_key", "commodity").agg(pl.all().last())
    )
    current_risk = []
    for _a in _fc["current_advisory"]:
        _r = _now.filter(pl.col("commodity") == _a["commodity"]).row(0, named=True)
        _p0, _s = float(_a["price_now"]), float(_r["sigma"])
        _out = {"commodity": _a["commodity"], "market": _a["market"], "week": _a["week"], "price_now": _p0,
                "sigma_week_pct": 100 * _s, "floor": 0.9, "by_h": {}}
        for _h, (_b10, _b50, _b90) in _a["bands"].items():
            _hh = int(_h)
            _k = np.array([0.9 * _p0])
            _se = np.array([_s * np.sqrt(_hh)])
            _fm, _sm = volatility.model_lognormal(np.log([_b10 / _p0]), np.log([_b50 / _p0]), np.log([_b90 / _p0]), np.array([_p0]))
            _out["by_h"][_h] = {
                "price_ewma_rs": float(volatility.put_value(np.array([_p0]), _k, _se)[0]),
                "price_model_rs": float(volatility.put_value(_fm, _k, _sm)[0]),
                "prob_ewma_pct": float(100 * volatility.shortfall_prob(np.array([_p0]), _k, _se)[0]),
                "prob_model_pct": float(100 * volatility.shortfall_prob(_fm, _k, _sm)[0]),
            }
        current_risk.append(_out)
    current_risk
    return (current_risk,)


@app.cell
def _(
    RUN_T0, band_rows_pct, bands_all, bands_by_h, combo_check, combo_check_hw_rows, current_risk, end_week, ev, fig_bench, hw_methods, json, paths,
    points_all, points_by_commodity, points_by_h, points_selected, pricing_all, pricing_by_commodity, pricing_by_h,
    retrieved, smoothing, time, var_ratio, volatility,
):
    bench_metrics = {
        "retrieved": retrieved,
        "end_week": str(end_week),
        "n_rows": ev.height,
        "n_origins": ev.filter(ev["h"] == 1).height,
        "hw_settings": {"max_gap_weeks": smoothing.MAX_GAP, "max_window_weeks": smoothing.MAX_WINDOW,
                        "season_weeks": smoothing.SEASON, "min_seasonal_weeks": smoothing.MIN_SEASONAL,
                        "min_holt_weeks": smoothing.MIN_HOLT},
        "vol_settings": {"lambda": volatility.LAMBDA, "min_changes": volatility.MIN_CHANGES},
        "hw_methods": hw_methods.to_dicts(),
        "points_all": points_all.to_dicts(),
        "points_by_h": points_by_h.to_dicts(),
        "points_selected": points_selected.to_dicts(),
        "points_by_commodity": points_by_commodity.to_dicts(),
        "band_rows_pct": band_rows_pct,
        "bands_all": bands_all.to_dicts(),
        "bands_by_h": bands_by_h.to_dicts(),
        "pricing_all": pricing_all.to_dicts(),
        "pricing_by_h": pricing_by_h.to_dicts(),
        "pricing_by_commodity": pricing_by_commodity.to_dicts(),
        "var_ratio": var_ratio,
        "combo_check": combo_check.to_dicts(),
        "combo_check_hw_rows": combo_check_hw_rows.to_dicts(),
        "current_risk": current_risk,
        "figures": {"other_methods": fig_bench},
        "runtime_min": round((time.time() - RUN_T0) / 60, 1),
    }
    (paths.OUT / "metrics_benchmarks.json").write_text(json.dumps(bench_metrics, indent=2, default=str), encoding="utf-8")
    bench_metrics["runtime_min"]
    return (bench_metrics,)


if __name__ == "__main__":
    app.run()
