import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium", app_title="Assam price forecasting prototype: backtest, stress test, advisory")


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

    from apa import advisory, clean, features, model, paths, seasonal, viz

    viz.theme()
    RUN_T0 = time.time()
    return RUN_T0, advisory, clean, dt, features, json, mo, model, np, paths, pl, plt, seasonal, sns, time, viz


@app.cell
def _(mo):
    mo.md(
        """
        # 2 · Forecasting prototype

        * **Short horizon (1–4 weeks):** one global quantile model (scikit-learn
          `HistGradientBoostingRegressor`, quantile loss) trained on every market of the
          selected commodities in Assam and neighbouring States. Outputs a 10–50–90
          percentile band of the price change.
        * **Benchmarks:** no-change (naive) and same-weeks-last-year (seasonal naive).
        * **Protocol:** rolling origin, refit every 4 weeks, training only on targets
          observed before the refit week (no lookahead).
        * **Stress test:** how much does learning from other markets help when local
          history is short, gappy, or absent?
        * **Decision layer and season-ahead outlook** for the selected series.
        """
    )
    return


@app.cell
def _(clean, json, paths, pl):
    selection = json.loads((paths.OUT / "selection.json").read_text(encoding="utf-8"))
    sel_pairs = pl.DataFrame(selection["pairs"])
    commodities = sorted(sel_pairs.get_column("commodity").unique().to_list())
    codes = {c: i for i, c in enumerate(commodities)}
    daily = clean.load_daily(paths.RAW_PARQUET).filter(pl.col("commodity").is_in(commodities))
    labels = clean.market_labels(daily).collect()
    wk = clean.flag_outliers(clean.weekly(daily)).collect().lazy()
    end_week = clean.last_full_week(
        daily.filter(pl.col("state") == "Assam").select(pl.col("date").max()).collect().item()
    )
    retrieved = pl.scan_parquet(paths.RAW_PARQUET).select(pl.col("retrieved_at").max()).collect().item()[:10]
    (commodities, end_week, retrieved)
    return codes, commodities, daily, end_week, labels, retrieved, sel_pairs, selection, wk


@app.cell
def _(codes, end_week, features, pl, sel_pairs, wk):
    grid = features.weekly_grid(features.dominant_variety(wk), end_week)
    G = features.build(grid, codes)
    sel_keys = sel_pairs.select(pl.lit("Assam").alias("state"), "market_key", "commodity").with_columns(
        pl.lit(True).alias("selected")
    )
    G = G.join(sel_keys, on=["state", "market_key", "commodity"], how="left").with_columns(
        pl.col("selected").fill_null(False)
    )
    data_summary = (
        G.filter(pl.col("lp").is_not_null())
        .group_by(pl.when(pl.col("state") == "Assam").then(pl.lit("Assam")).otherwise(pl.lit("Neighbouring States")).alias("region"), "commodity")
        .agg(pl.struct("state", "market_key").n_unique().alias("series"), pl.len().alias("observed_weeks"),
             pl.col("week").min().alias("first_week"))
        .sort("region", "commodity")
    )
    data_summary
    return G, data_summary


@app.cell
def _(G, dt, end_week, features, model, pl):
    test_start = end_week - dt.timedelta(weeks=51)
    feats = features.FEATURES
    eval_mask = (pl.col("state") == "Assam")
    bt = model.rolling_backtest(G, feats, eval_mask, test_start, end_week, refit_every=4)
    bt = bt.join(G.select("state", "market_key", "commodity", "week", "selected"), on=["state", "market_key", "commodity", "week"], how="left")
    return bt, eval_mask, feats, test_start


@app.cell
def _(bt, model, paths, pl):
    bt.write_parquet(paths.OUT / "backtest.parquet")
    score_sel = model.score(bt.filter(pl.col("selected")), ["commodity", "h"])
    score_sel_all_h = model.score(bt.filter(pl.col("selected")), ["commodity"])
    score_assam_h = model.score(bt, ["h"])
    score_assam = model.score(bt.with_columns(pl.lit("all").alias("all")), ["all"])
    score_assam_c = model.score(bt, ["commodity"])
    score_sel
    return score_assam, score_assam_c, score_assam_h, score_sel, score_sel_all_h


@app.cell
def _(mo, score_assam_h):
    mo.vstack([mo.md("**All Assam series of the selected commodities, by horizon**"), score_assam_h])
    return


@app.cell
def _(bt, labels, mo, pl, plt, retrieved, sns, test_start, viz):
    # fan chart: 2-week-ahead band vs actual, selected series
    _h = 2
    _d = (
        bt.filter(pl.col("selected") & (pl.col("h") == _h))
        .join(labels, on=["state", "market_key"], how="left")
        .with_columns(
            (pl.col("week") + pl.duration(weeks=_h)).alias("target_week"),
            (pl.col("lp") + pl.col("y")).exp().alias("actual"),
            (pl.col("lp") + pl.col("q10")).exp().alias("p10"),
            (pl.col("lp") + pl.col("q50")).exp().alias("p50"),
            (pl.col("lp") + pl.col("q90")).exp().alias("p90"),
            pl.col("lp").exp().alias("naive"),
        )
        .sort("commodity", "target_week")
    )
    _cs = _d.get_column("commodity").unique().sort().to_list()
    _fig, _axes = plt.subplots(len(_cs), 1, figsize=(9.5, 2.6 * len(_cs)), sharex=True)
    for _ax, _c in zip(_axes, _cs):
        _x = _d.filter(pl.col("commodity") == _c)
        _ax.fill_between(_x["target_week"].to_numpy(), _x["p10"].to_numpy(), _x["p90"].to_numpy(),
                         color=viz.PALETTE["band"], alpha=0.6, label="forecast band (P10–P90)", step=None)
        _ax.plot(_x["target_week"].to_numpy(), _x["p50"].to_numpy(), color=viz.PALETTE["model"], lw=1.2, label="forecast P50")
        _ax.plot(_x["target_week"].to_numpy(), _x["naive"].to_numpy(), color=viz.PALETTE["naive"], lw=1, ls="--", label="no-change benchmark")
        sns.scatterplot(x=_x["target_week"].to_numpy(), y=_x["actual"].to_numpy(), ax=_ax, s=14, color=viz.PALETTE["actual"], label="actual", zorder=5)
        _mk = _x.get_column("market_label").first()
        _ax.set(ylabel="Rs/quintal", title=f"{_c}, {_mk}: forecast made 2 weeks earlier vs actual")
        _ax.legend(fontsize=7, loc="upper left", ncols=4)
    fig_fan = viz.save(_fig, "f3_fan_selected", f"Source: Agmarknet via data.gov.in OGD API, retrieved {retrieved}. Out-of-sample forecasts, origins from {test_start}; gaps = weeks without a reported price.")
    mo.image(fig_fan)
    return (fig_fan,)


@app.cell
def _(mo, pl, plt, retrieved, score_assam_h, score_sel, sns, viz):
    _rows = []
    for _r in score_assam_h.iter_rows(named=True):
        for _m, _lab in (("mape_model", "global model (P50)"), ("mape_snaive", "seasonal naive"), ("mape_naive", "no change")):
            _rows.append({"group": "All Assam series", "h": _r["h"], "method": _lab, "MAPE": _r[_m]})
    _p = pl.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(7.5, 3.2))
    sns.barplot(data=_p.to_pandas(), x="h", y="MAPE", hue="method", ax=_ax,
                palette=[viz.PALETTE["model"], viz.PALETTE["snaive"], viz.PALETTE["naive"]])
    _ax.set(xlabel="forecast horizon (weeks ahead)", ylabel="mean absolute % error",
            title="Out-of-sample error, all Assam markets of the 3 commodities (lower is better)")
    _ax.legend(fontsize=8, title="")
    for _cont in _ax.containers:
        _ax.bar_label(_cont, fmt="%.1f", fontsize=6.5, padding=1)
    fig_mape = viz.save(_fig, "f4_mape_by_horizon", f"Source: Agmarknet via data.gov.in OGD API, retrieved {retrieved}. Rolling-origin backtest, 12 months to latest complete week.")
    mo.image(fig_mape)
    return (fig_mape,)


@app.cell
def _(G, dt, end_week, features, model, np, pl, test_start):
    # ---- Incomplete-data stress test: Assam history before the test year is masked;
    # one fit at test start; identical evaluation rows in every scenario; all fits in parallel.
    _assam = pl.col("state") == "Assam"
    _pre = pl.col("week") < test_start
    _scen = ["S0 full history", "S1 only 26 weeks of local history", "S2 half of local weeks missing", "S3 no local history at all"]
    _base = G.select("state", "market_key", "commodity", "variety", "week", "price", "pmin", "pmax", "n_days").sort(
        ["state", "market_key", "commodity", "week"])
    _codes = {c: i for i, c in enumerate(sorted(G.get_column("commodity").unique().to_list()))}
    _keep_half = np.random.default_rng(7).random(_base.height) < 0.5
    _store = model.ArrayStore("stress")
    _tasks, _meta, _frames = [], [], {}
    try:
        for _sc in _scen:
            if _sc.startswith("S0"):
                _drop = pl.lit(False)
            elif _sc.startswith("S1"):
                _drop = _assam & _pre & (pl.col("week") < test_start - dt.timedelta(weeks=26))
            elif _sc.startswith("S2"):
                _drop = _assam & _pre & pl.lit(pl.Series("drop", ~_keep_half))
            else:
                _drop = _assam & _pre
            _g = _base.with_columns(
                *[pl.when(_drop).then(None).otherwise(pl.col(_c)).alias(_c) for _c in ("price", "pmin", "pmax", "n_days")]
            ).with_columns(pl.col("price").log().alias("lp"))
            _g = features.build(_g, _codes).with_columns(
                pl.concat_str(["market_key", "commodity"], separator="|").rank("dense").alias("_sid"))
            _frames[_sc] = _g
            _arr = model.frame_arrays(_g, features.FEATURES, _store, _sc[:2])
            _is_as = (_g.get_column("state") == "Assam").to_numpy()
            _sid = _g.get_column("_sid").to_numpy()
            _wk = _arr["week"]
            for _h in features.HORIZONS:
                _yok = ~np.isnan(_arr[f"y{_h}"])
                _test = np.flatnonzero(_is_as & _arr["lp_ok"] & _yok & (_wk >= np.datetime64(test_start))
                                       & (_wk <= np.datetime64(end_week) - np.timedelta64(7 * _h, "D")))
                _train = model.train_index(_arr, _h, test_start)
                if _test.size == 0 or _train.size == 0:
                    continue
                _tasks.append(model.task(_arr, _h, _train, _test))
                _meta.append((_sc, _h, "pooled", _test))
                if _sc.startswith("S3"):
                    continue
                _trm = np.zeros(_wk.size, dtype=bool)
                _trm[_train] = True
                for _s in np.unique(_sid[_test]):
                    _own_tr = np.flatnonzero(_trm & _is_as & (_sid == _s))
                    if _own_tr.size >= 8:
                        _own_te = _test[_sid[_test] == _s]
                        _tasks.append(model.task(_arr, _h, _own_tr, _own_te, local=True, cols=features.OWN_FEATURES))
                        _meta.append((_sc, _h, "local", _own_te))
        _res = model.run_tasks(_tasks)
    finally:
        _store.close()
    _parts = {}
    for (_sc, _h, _kind, _idx), (_p, _n) in zip(_meta, _res):
        if _kind == "pooled":
            _parts[(_sc, _h)] = {"idx": _idx, "pooled": _p[:, 1], "local": np.full(_idx.size, np.nan)}
    for (_sc, _h, _kind, _idx), (_p, _n) in zip(_meta, _res):
        if _kind == "local":
            _d = _parts[(_sc, _h)]
            _d["local"][np.searchsorted(_d["idx"], _idx)] = _p[:, 1]
    stress_rows = []
    for (_sc, _h), _d in _parts.items():
        _te = _frames[_sc][_d["idx"]]
        stress_rows.append(_te.select(
            "market_key", "commodity", "week", "lp", pl.col(f"y{_h}").alias("y"),
            pl.lit(_h).alias("h"), pl.lit(_sc).alias("scenario"),
        ).with_columns(
            pl.Series("q50_pooled", _d["pooled"]),
            pl.Series("q50_local", _d["local"]).fill_nan(None),
        ))
    stress = pl.concat(stress_rows, how="vertical_relaxed")
    return (stress,)


@app.cell
def _(pl, stress):
    _act = (pl.col("lp") + pl.col("y")).exp()

    def _ape(pred):
        return ((pl.col("lp") + pred).exp() - _act).abs() / _act * 100

    # compare on identical rows: those where the actual and current price exist (same set in all scenarios)
    stress_score = (
        stress.group_by("scenario")
        .agg(
            pl.len().alias("n"),
            _ape(pl.lit(0.0)).mean().alias("mape_naive"),
            _ape(pl.col("q50_pooled")).mean().alias("mape_pooled"),
            _ape(pl.col("q50_local")).filter(pl.col("q50_local").is_not_null()).mean().alias("mape_local"),
            _ape(pl.col("q50_pooled")).filter(pl.col("q50_local").is_not_null()).mean().alias("mape_pooled_same_rows"),
            _ape(pl.lit(0.0)).filter(pl.col("q50_local").is_not_null()).mean().alias("mape_naive_same_rows"),
            pl.col("q50_local").is_not_null().mean().mul(100).alias("local_available_pct"),
        )
        .sort("scenario")
    )
    stress_score
    return (stress_score,)


@app.cell
def _(mo, pl, plt, retrieved, sns, stress_score, viz):
    _rows = []
    for _r in stress_score.iter_rows(named=True):
        _sc = _r["scenario"].split(" ", 1)[1]
        if _r["mape_local"] is not None:  # compare on the forecasts where a local model exists
            _rows.append({"scenario": _sc, "method": "no change", "MAPE": _r["mape_naive_same_rows"]})
            _rows.append({"scenario": _sc, "method": "local model (own history only)", "MAPE": _r["mape_local"]})
            _rows.append({"scenario": _sc, "method": "global model (learns from other markets)", "MAPE": _r["mape_pooled_same_rows"]})
        else:  # no local history: all forecasts
            _rows.append({"scenario": _sc, "method": "no change", "MAPE": _r["mape_naive"]})
            _rows.append({"scenario": _sc, "method": "global model (learns from other markets)", "MAPE": _r["mape_pooled"]})
    _p = pl.DataFrame(_rows)
    _fig, _ax = plt.subplots(figsize=(8.5, 3.4))
    sns.barplot(data=_p.to_pandas(), x="scenario", y="MAPE", hue="method", ax=_ax,
                palette=[viz.PALETTE["naive"], viz.PALETTE["snaive"], viz.PALETTE["model"]])
    for _cont in _ax.containers:
        _ax.bar_label(_cont, fmt="%.1f", fontsize=6.5, padding=1)
    _ax.set(xlabel="", ylabel="mean absolute % error (1–4 weeks)", title="Stress test: forecasting Assam prices with incomplete local history")
    _ax.tick_params(axis="x", labelsize=7.5)
    _ax.legend(fontsize=7.5, title="")
    fig_stress = viz.save(_fig, "f5_stress_test", f"Source: Agmarknet via data.gov.in OGD API, retrieved {retrieved}. Local history masked before the test year only. First three groups: forecasts where a local model could be fitted; last group: all forecasts.")
    mo.image(fig_stress)
    return (fig_stress,)


@app.cell
def _(advisory, bt, features, np, pl):
    # ---- Decision backtest on the selected series ----
    _w = (
        bt.filter(pl.col("selected"))
        .with_columns(pl.col("lp").exp().alias("p0"), (pl.col("lp") + pl.col("y")).exp().alias("act"),
                      *[(pl.col("lp") + pl.col(q)).exp().alias(q) for q in ("q10", "q50", "q90")])
    )
    dec_rows = []
    for (_c, _wk0), _grp in _w.group_by(["commodity", "week"]):
        _rule = advisory.rule_for(_c)
        _need = list(range(1, _rule.max_hold_weeks + 1))
        _byh = {int(r["h"]): r for r in _grp.iter_rows(named=True)}
        if not all(h in _byh for h in _need):
            continue  # evaluate only where every allowed holding period has an observed outcome
        _p0 = _byh[1]["p0"]
        _bands = {h: (_byh[h]["q10"], _byh[h]["q50"], _byh[h]["q90"]) for h in _need}
        _d = advisory.decide(_p0, _bands, _rule)
        _net = {h: _byh[h]["act"] * (1 - _rule.weekly_loss) ** h for h in _need}
        _real = _p0 if _d["signal"] == "SELL" else _net[_d["h"]]
        dec_rows.append({
            "commodity": _c, "week": _wk0, "signal": _d["signal"], "hold_weeks": _d["h"], "p0": _p0,
            "rule_value": _real, "sell_now": _p0, "always_hold_max": _net[_rule.max_hold_weeks],
            "oracle": max([_p0, *_net.values()]),
        })
    decisions = pl.DataFrame(dec_rows).sort("commodity", "week")
    decision_score = (
        decisions.group_by("commodity")
        .agg(
            pl.len().alias("weeks"),
            (pl.col("signal") != "SELL").sum().alias("hold_signals"),
            ((pl.col("rule_value") - pl.col("sell_now")) / pl.col("sell_now")).mean().mul(100).alias("rule_vs_sell_pct"),
            ((pl.col("always_hold_max") - pl.col("sell_now")) / pl.col("sell_now")).mean().mul(100).alias("hold_max_vs_sell_pct"),
            ((pl.col("oracle") - pl.col("sell_now")) / pl.col("sell_now")).mean().mul(100).alias("oracle_vs_sell_pct"),
            (pl.col("rule_value") - pl.col("sell_now")).mean().alias("rule_vs_sell_rs"),
            ((pl.col("signal") != "SELL") & (pl.col("rule_value") < pl.col("sell_now"))).sum().alias("hold_regrets"),
        )
        .sort("commodity")
    )
    _ = np, features
    decision_score
    return decision_score, decisions


@app.cell
def _(G, seasonal):
    mon = seasonal.monthly(G)
    season_profile = seasonal.profile(mon)
    _years = sorted(mon.filter(mon["state"] == "Assam").get_column("year").unique().to_list())
    season_bt = seasonal.backtest(mon, [y for y in _years if y >= 2015])
    season_bt
    return mon, season_bt, season_profile


@app.cell
def _(mo, np, pl, plt, retrieved, season_profile, sns, viz):
    _p = season_profile.with_columns(
        (pl.col("p50").exp() - 1).mul(100).alias("median"),
        (pl.col("p25").exp() - 1).mul(100).alias("q25"),
        (pl.col("p75").exp() - 1).mul(100).alias("q75"),
    )
    _cs = _p.get_column("commodity").unique().sort().to_list()
    _fig, _axes = plt.subplots(1, len(_cs), figsize=(9.5, 2.9), sharey=True)
    _months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for _ax, _c in zip(_axes, _cs):
        _x = _p.filter(pl.col("commodity") == _c).sort("month")
        _ax.fill_between(_x["month"].to_numpy(), _x["q25"].to_numpy(), _x["q75"].to_numpy(), color=viz.PALETTE["band"], alpha=0.6, label="middle half of years")
        sns.lineplot(x=_x["month"].to_numpy(), y=_x["median"].to_numpy(), ax=_ax, color=viz.PALETTE["model"], marker="o", markersize=3, label="typical year")
        _ax.axhline(0, color="#777", lw=0.6)
        _ax.set_xticks(np.arange(1, 13))
        _ax.set_xticklabels(_months, fontsize=7)
        _ax.set(title=_c, xlabel="", ylabel="% vs year average")
        _ax.legend(fontsize=6.5, loc="upper left") if _c == _cs[0] else _ax.get_legend().remove()
    fig_season = viz.save(_fig, "f6_seasonal_profiles", f"Source: Agmarknet via data.gov.in OGD API, retrieved {retrieved}. Pooled Assam + neighbouring-State markets, series-years with >= 8 months of data.")
    mo.image(fig_season)
    return (fig_season,)


@app.cell
def _(G, advisory, end_week, features, labels, model, np, pl):
    # ---- Current advisory: models fitted on everything known at the latest complete week ----
    _Gs = G.sort(["state", "market_key", "commodity", "week"])
    _rows = (
        _Gs.with_row_index("_r").filter(pl.col("selected") & pl.col("lp").is_not_null())
        .group_by("market_key", "commodity").agg(pl.col("_r").sort_by("week").last())
    )
    _now_idx = np.sort(_rows.get_column("_r").to_numpy())
    _store = model.ArrayStore("now")
    try:
        _arr = model.frame_arrays(_Gs, features.FEATURES, _store, "now")
        _res = model.run_tasks([model.task(_arr, _h, model.train_index(_arr, _h, end_week), _now_idx) for _h in features.HORIZONS])
    finally:
        _store.close()
    _now = _Gs[_now_idx]
    current = []
    for _i, _r in enumerate(_now.iter_rows(named=True)):
        _c, _p0 = _r["commodity"], float(_r["price"])
        _bands = {_h: tuple(float(_p0 * np.exp(v)) for v in _res[_k][0][_i]) for _k, _h in enumerate(features.HORIZONS)}
        _rule = advisory.rule_for(_c)
        _d = advisory.decide(_p0, _bands, _rule)
        _lab = labels.filter((pl.col("state") == "Assam") & (pl.col("market_key") == _r["market_key"])).row(0, named=True)
        current.append({
            "commodity": _c, "market": _lab["market_label"], "district": _lab["district"],
            "week": str(_r["week"]), "trained_to": str(end_week),
            "price_now": _p0, "bands": {str(h): b for h, b in _bands.items()},
            "signal": _d["signal"], "hold_weeks": _d["h"],
            "reason_en": advisory.reason_en(_c, _p0, _d, _rule),
            "rule": {"max_hold_weeks": _rule.max_hold_weeks, "weekly_loss_pct": 100 * _rule.weekly_loss,
                     "storable": _rule.storable, "note": _rule.note},
        })
    current = sorted(current, key=lambda a: a["commodity"])
    current
    return (current,)


@app.cell
def _(G, end_week, labels, pl):
    # nearby-market prices in the latest complete week (same commodities, Assam)
    nearby = (
        G.filter((pl.col("state") == "Assam") & (pl.col("week") == end_week) & pl.col("price").is_not_null())
        .join(labels, on=["state", "market_key"], how="left")
        .select("commodity", "market_label", "district", "price")
        .sort("commodity", "price", descending=[False, True])
    )
    nearby
    return (nearby,)


@app.cell
def _(
    RUN_T0, bt, current, data_summary, decision_score, end_week, fig_fan, fig_mape, fig_season, fig_stress, json,
    nearby, paths, retrieved, score_assam, score_assam_c, score_assam_h, score_sel, score_sel_all_h, season_bt,
    season_profile, stress_score, test_start, time,
):
    forecast_metrics = {
        "retrieved": retrieved,
        "end_week": str(end_week),
        "test_start": str(test_start),
        "n_forecasts_assam": bt.height,
        "n_series_assam": bt.select("market_key", "commodity").unique().height,
        "data_summary": data_summary.to_dicts(),
        "score_selected_by_h": score_sel.to_dicts(),
        "score_selected": score_sel_all_h.to_dicts(),
        "score_assam_by_h": score_assam_h.to_dicts(),
        "score_assam": score_assam.to_dicts(),
        "score_assam_by_commodity": score_assam_c.to_dicts(),
        "stress": stress_score.to_dicts(),
        "decision": decision_score.to_dicts(),
        "season_backtest": season_bt.to_dicts(),
        "season_profile": season_profile.to_dicts(),
        "current_advisory": current,
        "nearby_latest_week": nearby.to_dicts(),
        "figures": {"fan": fig_fan, "mape": fig_mape, "stress": fig_stress, "season": fig_season},
        "runtime_min": round((time.time() - RUN_T0) / 60, 1),
    }
    (paths.OUT / "metrics_forecast.json").write_text(json.dumps(forecast_metrics, indent=2, default=str), encoding="utf-8")
    forecast_metrics["runtime_min"]
    return (forecast_metrics,)


if __name__ == "__main__":
    app.run()
