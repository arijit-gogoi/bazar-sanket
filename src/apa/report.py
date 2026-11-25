"""Build the template context (formatted strings) from out/metrics.json.

Sentences whose direction depends on results (better / worse than a benchmark)
are generated here from the numbers, so the text can never contradict them.
"""

from __future__ import annotations

import datetime as dt
import html

from .advisory import rupees


def num(x) -> str:
    v = int(round(float(x)))
    s = str(abs(v))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if v < 0 else "") + s


def f1(x) -> str:
    return f"{float(x):.1f}"


def f0(x) -> str:
    return f"{float(x):.0f}"


def dlong(s) -> str:
    d = dt.date.fromisoformat(str(s)[:10])
    return f"{d.day} {d.strftime('%B %Y')}"


def dshort(s) -> str:
    d = dt.date.fromisoformat(str(s)[:10])
    return f"{d.day} {d.strftime('%b %Y')}"


def mon_year(s) -> str:
    d = dt.date.fromisoformat(str(s)[:10])
    return d.strftime("%B %Y")


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def better(a: float, b: float, lower_is_better: bool = True) -> str:
    """'lower' / 'higher' wording helper: is a better than b?"""
    return (a < b) if lower_is_better else (a > b)


def build_context(m: dict) -> dict[str, str]:
    p, f = m["profile"], m["forecast"]
    c: dict[str, str] = {}

    # ---------------- profile ----------------
    c["rows_total"] = num(p["rows_total"])
    c["retrieved"] = dlong(p["retrieved"])
    c["assam_rows"] = num(p["assam_rows"])
    c["assam_first_year"] = str(p["assam_first"])[:4]
    c["assam_first"] = dlong(p["assam_first"])
    c["assam_last"] = dlong(p["assam_last"])
    c["n_states"] = str(len(p["states"]))
    c["n_comp_states"] = str(len(p["states"]) - 1)
    comp = [s for s in p["states"] if s != "Assam"]
    c["comp_states"] = ", ".join(comp[:-1]) + " and " + comp[-1]
    c["assam_markets_raw"] = str(p["assam_market_names_raw"])
    c["assam_markets_harm"] = str(p["assam_markets_harmonised"])
    c["assam_markets_renamed"] = str(p["assam_markets_renamed"])
    gs = p["gap_stats"]
    c["weeks_total"] = num(gs["weeks_total"])
    c["weeks_empty"] = num(gs["weeks_empty"])
    c["weeks_empty_pct"] = f0(100 * gs["weeks_empty"] / gs["weeks_total"])
    c["median_markets_week"] = f0(gs["median_markets_week"])
    longest = max(p["gaps"], key=lambda g: g["weeks"])
    c["longest_gap"] = f"{mon_year(longest['start'])} to {mon_year(longest['end'])} ({longest['weeks']} weeks)"
    recent = [g for g in p["gaps"] if str(g["start"]) >= "2025-01-01"]
    c["recent_gap"] = (
        f"{mon_year(recent[-1]['start'])} to {mon_year(recent[-1]['end'])}" if recent else "none"
    )
    c["n_gaps"] = str(len(p["gaps"]))
    c["timeline_sentence"] = (
        f"Since {c['assam_first_year']}, {c['weeks_empty_pct']}% of weeks ({c['weeks_empty']} of {c['weeks_total']}) have no Assam "
        f"price at all in the public record; the longest gap ran from {c['longest_gap']}, and the most recent from {c['recent_gap']}."
    )
    cov = p["coverage"]
    c["n_pairs"] = num(cov["n_pairs"])
    c["median_cov"] = f0(cov["median_cov_pct"])
    c["n_ge50"] = str(cov["n_ge50"])
    c["max_cov"] = f0(cov["max_cov_pct"])
    c["renamed_sentence"] = (
        f"Market names are unstable: {c['assam_markets_raw']} names map to {c['assam_markets_harm']} actual markets, because "
        f"{c['assam_markets_renamed']} markets were renamed (mostly when Agmarknet 2.0 appended \"APMC\"), which silently breaks naive time series."
    )
    disp = sorted(p["dispersion"], key=lambda d: d["commodity"])
    c["disp_sentence"] = ", ".join(f"{d['commodity'].lower()} {f0(d['median_spread_pct'])}%" for d in disp)
    c["disp_min"] = f0(min(d["median_spread_pct"] for d in disp))
    c["disp_max"] = f0(max(d["median_spread_pct"] for d in disp))
    o = p["outliers"]
    c["outlier_pct"] = f1(100 * o["n_outlier"] / o["n_weeks"])
    sel = p["selection"]["pairs"]
    c["sel_market"] = sel[0]["market"].replace("(", " (")
    c["sel_district"] = sel[0]["district"]
    names = [s["commodity"].lower() for s in sel]
    c["sel_commodities"] = ", ".join(names[:-1]) + " and " + names[-1]
    c["sel_cov"] = f0(sel[0]["coverage_pct"])
    c["sel_weeks"] = str(sel[0]["weeks"])
    c["win_start"] = dlong(p["window"][0])
    c["win_end"] = dlong(p["window"][1])

    # ---------------- forecast ----------------
    c["runtime_min"] = f0(max(1, round(f["runtime_min"])))
    c["test_start"] = dlong(f["test_start"])
    c["end_week"] = dlong(f["end_week"])
    c["n_series_assam"] = str(f["n_series_assam"])
    c["n_forecasts_assam"] = num(f["n_forecasts_assam"])
    weeks_test = (dt.date.fromisoformat(f["end_week"]) - dt.date.fromisoformat(f["test_start"])).days // 7 + 1
    c["n_fits"] = str(-(-weeks_test // 4) * 4 * 3)  # refits x horizons x quantiles
    sa = f["score_assam"][0]
    for k in ("mape_model", "mape_naive", "mape_snaive", "dir_model", "dir_naive", "dir_snaive",
              "coverage80", "band_width_pct", "skill_vs_naive", "skill_vs_snaive", "snaive_fallback_pct"):
        c[f"all_{k}"] = f1(sa[k])
    c["all_n"] = num(sa["n"])
    for r in f["score_assam_by_h"]:
        h = r["h"]
        for k in ("mape_model", "mape_naive", "mape_snaive", "dir_model", "dir_naive", "coverage80", "skill_vs_naive", "n"):
            c[f"h{h}_{k}"] = f1(r[k]) if k != "n" else num(r[k])
    for r in f["score_selected"]:
        key = r["commodity"].split("(")[0].replace(" ", "").lower()
        for k in ("mape_model", "mape_naive", "mape_snaive", "dir_model", "dir_naive", "coverage80", "skill_vs_naive", "n"):
            c[f"sel_{key}_{k}"] = f1(r[k]) if k != "n" else num(r[k])
    # stress test
    st = {r["scenario"][:2]: r for r in f["stress"]}
    for k, r in st.items():
        c[f"{k}_naive"] = f1(r["mape_naive"])
        c[f"{k}_pooled"] = f1(r["mape_pooled"])
        c[f"{k}_local"] = f1(r["mape_local"]) if r["mape_local"] is not None else "n/a"
        c[f"{k}_pooled_same"] = f1(r["mape_pooled_same_rows"]) if r["mape_pooled_same_rows"] is not None else "n/a"
        c[f"{k}_n"] = num(r["n"])
    c["stress_sentence"] = stress_sentence(st)
    from . import report_text as rt
    c["res_overall"] = rt.overall(f)
    c["res_short"] = rt.res_short(f)
    c["reading_short"] = rt.reading_short(f)
    c["res_band_dir"] = rt.band_dir(f)
    c["res_horizons"] = rt.horizons(f)
    c["res_selected"] = rt.selected(f)
    c["stress_short"] = rt.stress_short(f)
    c["decision_short"] = rt.decision_short(f)
    c["season_short"] = rt.season_short(f)
    c["results_reading"] = " ".join([c["res_overall"], c["res_band_dir"], c["res_horizons"], c["res_selected"]])
    c["switch_note"] = rt.switch_note(f)
    c["switch_limit"] = rt.switch_limit(f)
    c["adv_k"] = str(len(f["advisory_model"]["commodities"]))
    c["stress_reading"] = rt.stress_reading(f)
    c["decision_reading"] = rt.decision_reading(f)
    c["season_reading"] = rt.season_reading(f)
    # decision layer
    for r in f["decision"]:
        key = r["commodity"].split("(")[0].replace(" ", "").lower()
        c[f"dec_{key}_weeks"] = str(r["weeks"])
        c[f"dec_{key}_holds"] = str(r["hold_signals"])
        c[f"dec_{key}_rule"] = f1(r["rule_vs_sell_pct"])
        c[f"dec_{key}_holdmax"] = f1(r["hold_max_vs_sell_pct"])
        c[f"dec_{key}_oracle"] = f1(r["oracle_vs_sell_pct"])
        c[f"dec_{key}_regrets"] = str(r["hold_regrets"])
    # seasonal
    sb = f["season_backtest"]
    if sb:
        import statistics as stt
        c["season_mae_profile"] = f1(stt.mean(r["mae_profile_pct"] for r in sb))
        c["season_mae_flat"] = f1(stt.mean(r["mae_flat_pct"] for r in sb))
        corrs = [r["corr"] for r in sb if r["corr"] == r["corr"]]
        c["season_corr"] = f"{stt.median(corrs):.2f}" if corrs else "n/a"
        c["season_years"] = f"{min(r['year'] for r in sb)}–{max(r['year'] for r in sb)}"
        c["season_n"] = str(len(sb))
    # current advisory (latest complete week)
    for a in f["current_advisory"]:
        key = a["commodity"].split("(")[0].replace(" ", "").lower()
        c[f"cur_{key}_price"] = rupees(a["price_now"])
        c[f"cur_{key}_signal"] = a["signal"]
        c[f"cur_{key}_week"] = dlong(a["week"])
        c[f"cur_{key}_reason"] = html.escape(a["reason_en"])
        for h, b in a["bands"].items():
            c[f"cur_{key}_h{h}_lo"] = rupees(b[0])
            c[f"cur_{key}_h{h}_mid"] = rupees(b[1])
            c[f"cur_{key}_h{h}_hi"] = rupees(b[2])
    return c


def stress_sentence(st: dict) -> str:
    s1, s3 = st.get("S1"), st.get("S3")
    parts = []
    if s1 and s1["mape_local"] is not None:
        if s1["mape_pooled_same_rows"] < s1["mape_local"]:
            parts.append(
                f"In the prototype's stress test (pre-set model), with only 26 weeks of local history the global model's error was "
                f"{f1(s1['mape_pooled_same_rows'])}% against {f1(s1['mape_local'])}% for a model trained on local data alone"
            )
        else:
            parts.append(
                f"In the prototype's stress test (pre-set model), with only 26 weeks of local history the global model's error "
                f"({f1(s1['mape_pooled_same_rows'])}%) did not beat a local-only model ({f1(s1['mape_local'])}%)"
            )
    if s3:
        rel = "below" if s3["mape_pooled"] < s3["mape_naive"] else "above"
        parts.append(
            f"with no Assam history at all, it still forecast Assam prices with {f1(s3['mape_pooled'])}% error, "
            f"{rel} the {f1(s3['mape_naive'])}% of a no-change forecast"
        )
    return "; ".join(parts) + "." if parts else ""
