"""Report context for the other methods tried (Holt-Winters, volatility pricing).

All numbers come from out/metrics.json ("benchmarks", written by nb/benchmarks.py).
Sentences that compare methods are generated from the numbers, so the wording
follows the results.
"""

from __future__ import annotations

import html

from .compare import COMBO, HW, HW_ALL, HW_MUL, MODEL, NAIVE, SNAIVE
from .report import f0, f1, num


def _tr(cells: list[str]) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def _vs_naive(skill: float) -> str:
    """Skill is % reduction in mean absolute error versus no change."""
    if abs(skill) < 0.05:
        return "the same absolute error as no change"
    return f"{f1(abs(skill))}% {'lower' if skill > 0 else 'higher'} absolute error than no change"


def _rs(x: float) -> str:
    return "₹" + num(round(x / 10.0) * 10)


def bench_context(m: dict) -> dict[str, str]:
    b = m["benchmarks"]
    pa = {r["method"]: r for r in b["points_all"]}
    n_or = b["n_origins"]
    use = {(r["variant"], r["method"]): r["origins"] for r in b["hw_methods"]}
    hw_k = use.get(("primary", "Holt-Winters"), 0)
    hw_dh = use.get(("primary", "damped Holt (under 2 years of data)"), 0)
    hw_nc = use.get(("primary", "no change (short history)"), 0)
    hwall_k = use.get(("all gaps interpolated", "Holt-Winters"), 0)
    best = min(pa.values(), key=lambda r: r["mae"])["method"]
    combo_h = [r["h"] for r in b["points_by_h"] if r["method"] == COMBO and r["skill_vs_naive"] > 0]
    pr = {r["floor"]: r for r in b["pricing_all"]}[0.9]
    pr_h = {r["h"]: r for r in b["pricing_by_h"] if r["floor"] == 0.9}
    ba = {r["method"]: r for r in b["bands_all"]}
    band_model = ba["Global model quantiles (P10-P90)"]
    band_vol = ba["Volatility band around today's price"]
    best_band = min(b["bands_all"], key=lambda r: r["interval_score_pct"])["method"]
    ratio = pr["price_ewma"] / pr["realised"] if pr["realised"] > 0 else float("nan")
    hmax = max(pr_h)
    ratio_hmax = pr_h[hmax]["price_ewma"] / pr_h[hmax]["realised"]
    if ratio > 1.1:
        pricing_dir = "overstated"
    elif ratio < 0.9:
        pricing_dir = "understated"
    else:
        pricing_dir = "matched"

    c: dict[str, str] = {}
    def short(skill: float) -> str:
        return f"{f1(abs(skill))}% {'better' if skill > 0 else 'worse'} than no change"

    chk = {r["method"]: r for r in b["combo_check"]}
    halved = chk["Global model halved (post-hoc check)"]["skill_vs_naive"]
    damping_explains = halved >= pa[COMBO]["skill_vs_naive"]
    combo_clause = (
        f"a pre-set average with the global model {short(pa[COMBO]['skill_vs_naive'])}"
        + (f", though damping the model alone did better ({f1(halved)}%, post-hoc)" if damping_explains else "")
    )
    c["other_short"] = (
        f"Other methods: Holt-Winters {short(pa[HW]['skill_vs_naive'])}; {combo_clause}; volatility pricing of a 90% "
        f"price floor {pricing_dir} shortfalls "
        f"({f1(pr['price_ewma'])}% vs {f1(pr['realised'])}%)."
    )

    # ---------------------------------------------------------------- PDF readings
    hw_txt = (
        f"Holt-Winters (statsmodels, re-estimated at every forecast origin; additive on the log price, so seasonal effects "
        f"are multiplicative on the rupee price) needs two years of weekly "
        f"prices per market. Only {num(hw_k)} of {num(n_or)} forecast origins ({f1(100 * hw_k / n_or)}%) had two years without a "
        f"reporting gap longer than {b['hw_settings']['max_gap_weeks']} weeks; elsewhere it fell back to damped Holt ({num(hw_dh)}) or no change ({num(hw_nc)}). "
        f"Overall it had {_vs_naive(pa[HW]['skill_vs_naive'])} (MAPE {f1(pa[HW]['mape'])}% vs {f1(pa[NAIVE]['mape'])}%). "
        f"Filling every gap by interpolation (a check added after the first result) allowed seasonal fits at {num(hwall_k)} "
        f"origins but gave {_vs_naive(pa[HW_ALL]['skill_vs_naive'])}. The textbook multiplicative form on the rupee price "
        f"(added on request) gave {_vs_naive(pa[HW_MUL]['skill_vs_naive'])}. Per-market smoothing cannot borrow from other "
        f"markets; the global model can."
    )
    n_h = len({r["h"] for r in b["points_by_h"]})
    if len(combo_h) == n_h:
        combo_where = " (better than no change at every horizon)"
    elif combo_h:
        hs = ", ".join(str(h) for h in combo_h)
        combo_where = f" (better than no change at {hs} week{'s' if combo_h[-1] > 1 else ''} ahead)"
    else:
        combo_where = ""
    hwr = {r["method"]: r for r in b["combo_check_hw_rows"]}
    if damping_explains:
        why = (
            f"A post-hoc check shows where the gain comes from: halving the global model's forecast change, with no Holt-Winters "
            f"at all, gave {_vs_naive(halved)}. The average mostly damps the model towards no change. Where full seasonal "
            f"Holt-Winters could be fitted ({num(hwr[HW]['n'])} forecasts), it had {_vs_naive(hwr[HW]['skill_vs_naive'])}, while "
            f"the global model had {_vs_naive(hwr[MODEL]['skill_vs_naive'])}. Damping is a candidate to test in the pilot, on new data."
        )
    else:
        why = (
            f"A post-hoc check: halving the global model's forecast change alone gave {_vs_naive(halved)}, so Holt-Winters adds "
            f"to the gain. The pilot would confirm this on new data before any switch."
        )
    combo_txt = (
        f"An equal-weight average of the global model and Holt-Winters, fixed before testing, had "
        f"{_vs_naive(pa[COMBO]['skill_vs_naive'])}{combo_where}; the global model alone had "
        f"{_vs_naive(pa[MODEL]['skill_vs_naive'])}. " + why
    )
    c["other_reading_points"] = hw_txt + " " + combo_txt
    band_txt = (
        f"For 80% price ranges, the {'global model' if best_band.startswith('Global') else 'volatility'} bands scored best "
        f"(interval score {f1(band_model['interval_score_pct'])}% of price vs {f1(band_vol['interval_score_pct'])}% for a "
        f"volatility band around today's price)."
    )
    r1 = pr_h[1]["price_ewma"] / pr_h[1]["realised"] if pr_h[1]["realised"] > 0 else float("nan")
    if 0.8 <= r1 <= 1.25:
        h1_txt = f"One week ahead the two were close ({f1(pr_h[1]['price_ewma'])}% vs {f1(pr_h[1]['realised'])}%)"
    else:
        h1_txt = f"One week ahead the price was {r1:.1f} times the outcome ({f1(pr_h[1]['price_ewma'])}% vs {f1(pr_h[1]['realised'])}%)"
    vr = b["var_ratio"][str(hmax)]
    if ratio_hmax > 1.25 and vr < 0.75 * hmax:
        why = (f", because weekly vegetable prices partly revert after jumps: price changes over {hmax} weeks varied only {vr:.1f} "
               f"times as much as over one week, not {hmax} times as volatility scaled with time assumes")
    else:
        why = ""
    pricing_txt = (
        f"Volatility pricing values downside risk as a put option: the expected shortfall below a price floor. For a floor at 90% of today's price, prices from recent volatility predicted an average shortfall of "
        f"{f1(pr['price_ewma'])}% of today's price ({f1(pr['price_model'])}% from the global model's band); the shortfall that "
        f"actually occurred averaged {f1(pr['realised'])}%. {h1_txt}; {hmax} weeks ahead the price was {ratio_hmax:.1f} times the "
        f"outcome{why}. Such prices need calibration on backtests like this before any use."
    )
    ex = next((r for r in b["current_risk"] if r["commodity"] == "Green Chilli"), b["current_risk"][0])
    e2 = ex["by_h"]["2"]
    example_txt = (
        f"Example for the week of the mock-ups: a floor at 90% of this week's {ex['commodity'].lower()} price at {ex['market']} "
        f"({_rs(ex['price_now'])} per quintal), two weeks ahead, is priced at about {_rs(e2['price_ewma_rs'])} per quintal from recent "
        f"volatility and {_rs(e2['price_model_rs'])} from the global model's band."
    )
    c["other_reading_risk"] = band_txt + " " + pricing_txt + " " + example_txt
    c["other_intro"] = (
        f"Two further methods were tried on exactly the same {num(b['n_rows'])} out-of-sample forecasts, with settings fixed before "
        f"testing (code: <code>apa/smoothing.py</code>, <code>apa/volatility.py</code>, notebook <code>nb/benchmarks.py</code>)."
    )

    # ---------------------------------------------------------------- tables
    label = {MODEL: "Global model (P50)", HW: "Holt-Winters", HW_ALL: "Holt-Winters, all gaps filled (check)",
             HW_MUL: "Holt-Winters, multiplicative (check)",
             COMBO: "Average: global model + Holt-Winters", NAIVE: "No change", SNAIVE: "Seasonal naive"}
    t = ["<table><tr><th>Method (all horizons)</th><th class='n'>MAPE</th><th class='n'>Skill vs no change</th>"
         "<th class='n'>Direction correct</th></tr>"]
    rows = list(b["points_all"])
    k = next(i for i, r in enumerate(rows) if r["method"] == COMBO) + 1
    rows.insert(k, chk["Global model halved (post-hoc check)"])
    label["Global model halved (post-hoc check)"] = "Global model, change halved (post-hoc check)"
    for r in rows:
        name = html.escape(label.get(r["method"], r["method"]))
        t.append(_tr([f"<td>{name}</td>", f"<td class='n'>{f1(r['mape'])}%</td>",
                      f"<td class='n'>{r['skill_vs_naive']:+.1f}%</td>", f"<td class='n'>{f1(r['direction'])}%</td>"]))
    t.append("</table>")
    c["table_other_points"] = "\n".join(t)

    blabel = {"Global model quantiles (P10-P90)": "Global model (P10–P90)",
              "Volatility band around today's price": "Volatility, around today's price",
              "Volatility band around Holt-Winters forecast": "Volatility, around Holt-Winters"}
    t = ["<table><tr><th>80% range</th><th class='n'>Coverage</th><th class='n'>Width</th><th class='n'>Interval score</th></tr>"]
    for r in b["bands_all"]:
        t.append(_tr([f"<td>{html.escape(blabel.get(r['method'], r['method']))}</td>", f"<td class='n'>{f1(r['coverage'])}%</td>",
                      f"<td class='n'>{f1(r['width_pct'])}%</td>", f"<td class='n'>{f1(r['interval_score_pct'])}%</td>"]))
    t.append("</table>")
    c["table_other_bands"] = "\n".join(t)

    t = ["<table><tr><th>Weeks ahead</th><th class='n'>Actual shortfall</th><th class='n'>Priced from volatility</th>"
         "<th class='n'>Priced from model band</th></tr>"]
    for h in sorted(pr_h):
        r = pr_h[h]
        t.append(_tr([f"<td>{h}</td>", f"<td class='n'>{r['realised']:.2f}%</td>", f"<td class='n'>{r['price_ewma']:.2f}%</td>",
                      f"<td class='n'>{r['price_model']:.2f}%</td>"]))
    t.append("</table>")
    c["table_other_pricing"] = "\n".join(t)
    c["pipeline_min"] = f0(max(1, round(m["forecast"]["runtime_min"] + b["runtime_min"]
                                        + m.get("extended", {}).get("runtime_min", 0))))
    c["other_band_rows"] = f"{f1(b['band_rows_pct'])}"
    c["vol_lambda"] = f"{b['vol_settings']['lambda']:.2f}"
    c["vol_min_changes"] = str(b["vol_settings"]["min_changes"])
    return c
