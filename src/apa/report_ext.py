"""Report context for the extension added on request: LightGBM as the learner, the textbook
multiplicative Holt-Winters method, and more commodities chosen by a good-data rule.

All numbers come from out/metrics.json ("extended", written by nb/extended.py).
Comparative sentences are generated from the numbers, so the wording follows the results.
"""

from __future__ import annotations

import html

from .compare import HW, HW_MUL, LGBM, MODEL, NAIVE, PREV, SNAIVE
from .report import f1, num


def _tr(cells: list[str]) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def _skill(s: float) -> str:
    return f"{s:+.1f}%"


def _vs(s: float) -> str:
    if abs(s) < 0.05:
        return "the same absolute error as no change"
    return f"{f1(abs(s))}% {'lower' if s > 0 else 'higher'} absolute error than no change"


def _short(s: float) -> str:
    return f"{f1(abs(s))}% {'better' if s > 0 else 'worse'} than no change"


def _join(xs: list[str]) -> str:
    return ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]


def ext_context(m: dict) -> dict[str, str]:
    e = m["extended"]
    rule = e["rule"]
    comms = e["ext_commodities"]
    k = len(comms)
    main = {r["method"]: r for r in e["main"]["points_all"]}
    mb = {r["method"]: r for r in e["main"]["bands_all"]}
    xa = {r["method"]: r for r in e["ext"]["points_all"]}
    xb = {r["method"]: r for r in e["ext"]["bands_all"]}
    byc: dict[str, dict[str, dict]] = {}
    for r in e["ext"]["points_by_commodity"]:
        byc.setdefault(r["commodity"], {})[r["method"]] = r
    use = {(r["form"], r["method"]): r["origins"] for r in e["ext"]["hw_methods"]}
    n_or = e["ext"]["n_origins"]
    hw_seasonal = use.get(("log", "Holt-Winters"), 0)
    beat = {mm: sum(1 for c in comms if byc[c][mm]["skill_vs_naive"] > 0) for mm in (MODEL, LGBM, HW, HW_MUL)}
    c: dict[str, str] = {}

    # ---------------------------------------------------------------- intro (PDF)
    c["ext_intro"] = (
        "Added on request after the first results; settings were fixed before each run. "
        "<b>LightGBM</b>, a second gradient-boosting library, replaces scikit-learn as the learner of the global model "
        f"(same features, quantile loss and refit weeks; tree settings mirrored, not tuned). <b>More commodities</b>: every "
        f"Assam commodity with good data, by a coverage rule fixed before any model was run on them: at least "
        f"{num(rule['min_weeks_3y'])} market-weeks with a price in the last {rule['window_3y_weeks']} weeks and at least "
        f"{num(rule['min_weeks_1y'])} in the test year. {k} commodities qualify: {html.escape(_join(comms))}. "
        f"Assam markets only (neighbouring-State prices were fetched for the pilot commodities only). After these results "
        f"the applicant chose the scikit-learn model trained on these {k} commodities as the advisory model (Section E); "
        f"advice is still given only for the pilot commodities, the only ones with storage and shelf-life rules. The list of "
        f"all downloaded commodities, with the reason each was or was not used, is <code>COMMODITIES.md</code> in the repository."
    )

    # ---------------------------------------------------------------- tables
    t = ["<table><tr><th>Main backtest rows</th><th class='n'>MAPE</th><th class='n'>Skill vs no change</th>"
         "<th class='n'>Direction</th><th class='n'>80% band coverage</th><th class='n'>Interval score</th></tr>"]
    lab = {MODEL: f"{k} crops, scikit-learn (advisory model)", LGBM: f"{k} crops, LightGBM",
           PREV: "Set before testing: 3 crops + neighbouring States", NAIVE: "No change", SNAIVE: "Seasonal naive"}
    for mm in (MODEL, LGBM, PREV, NAIVE, SNAIVE):
        r = main[mm]
        b = mb.get(mm)
        t.append(_tr([f"<td>{lab[mm]}</td>", f"<td class='n'>{f1(r['mape'])}%</td>",
                      f"<td class='n'>{_skill(r['skill_vs_naive'])}</td>", f"<td class='n'>{f1(r['direction'])}%</td>",
                      f"<td class='n'>{f1(b['coverage']) + '%' if b else '–'}</td>",
                      f"<td class='n'>{f1(b['interval_score_pct']) + '%' if b else '–'}</td>"]))
    t.append("</table>")
    c["table_ext_main"] = "\n".join(t)

    order = sorted(comms, key=lambda x: -byc[x][NAIVE]["n"])
    t = ["<table><tr><th>Commodity (Assam)</th><th class='n'>Forecasts</th><th class='n'>MAPE, no change</th>"
         "<th class='n'>Global, scikit-learn</th><th class='n'>Global, LightGBM</th><th class='n'>Holt-Winters (log)</th>"
         "<th class='n'>Holt-Winters (mult.)</th></tr>"]

    def row(name: str, d: dict, bold: bool = False) -> str:
        cells = [f"<td>{'<b>' if bold else ''}{html.escape(name)}{'</b>' if bold else ''}</td>",
                 f"<td class='n'>{num(d[NAIVE]['n'])}</td>", f"<td class='n'>{f1(d[NAIVE]['mape'])}%</td>"]
        cells += [f"<td class='n'>{_skill(d[mm]['skill_vs_naive'])}</td>" for mm in (MODEL, LGBM, HW, HW_MUL)]
        return _tr(cells)

    t.append(row(f"All {k} commodities", xa, bold=True))
    for cm in order:
        t.append(row(cm, byc[cm]))
    t.append("</table>")
    c["table_ext_commodities"] = "\n".join(t)

    # ---------------------------------------------------------------- reading (PDF)
    d_main = main[LGBM]["skill_vs_naive"] - main[MODEL]["skill_vs_naive"]
    if abs(d_main) < 0.3:
        main_txt = "about the same accuracy as"
    elif d_main > 0:
        main_txt = "slightly better accuracy than" if d_main < 1.5 else "better accuracy than"
    else:
        main_txt = "slightly lower accuracy than" if d_main > -1.5 else "lower accuracy than"
    r1 = (
        f"On the main backtest rows ({num(e['main']['n_rows'])} forecasts), trained on the same {k} commodities, LightGBM had "
        f"{main_txt} scikit-learn: {_vs(main[LGBM]['skill_vs_naive'])} against {_vs(main[MODEL]['skill_vs_naive'])}; its 80% "
        f"bands held {f1(mb[LGBM]['coverage'])}% of actual prices (scikit-learn {f1(mb[MODEL]['coverage'])}%). The model set "
        f"before testing, trained on the three pilot crops with neighbouring-State markets, had {_vs(main[PREV]['skill_vs_naive'])}."
    )
    r2 = (
        f"Across the {k} commodities ({num(e['ext']['n_rows'])} forecasts, {num(e['ext']['n_series'])} market series), "
        f"the global model had {_vs(xa[MODEL]['skill_vs_naive'])} with scikit-learn and {_vs(xa[LGBM]['skill_vs_naive'])} "
        f"with LightGBM; it beat no change for {beat[MODEL]} of {k} commodities with scikit-learn and {beat[LGBM]} with "
        f"LightGBM."
    )
    hwr = {r["method"]: r for r in e["ext"]["points_hw_rows"]}
    form_gap = abs(xa[HW]["skill_vs_naive"] - xa[HW_MUL]["skill_vs_naive"])
    worse_still = hwr[HW]["skill_vs_naive"] < xa[HW]["skill_vs_naive"]
    r_hw = (
        f"Holt-Winters had {_vs(xa[HW]['skill_vs_naive'])} on the log price and {_vs(xa[HW_MUL]['skill_vs_naive'])} in the "
        f"textbook multiplicative form{', so the form made little difference' if form_gap < 1 else ''}. A full seasonal fit "
        f"was possible at only {num(hw_seasonal)} of {num(n_or)} forecast origins; on those {num(hwr[HW]['n'])} forecasts "
        f"Holt-Winters did {'worse still' if worse_still else 'better'} ({_vs(hwr[HW]['skill_vs_naive'])}), while the "
        f"global model had {_vs(hwr[MODEL]['skill_vs_naive'])}."
    )
    lgbm_better = main[LGBM]["skill_vs_naive"] > main[MODEL]["skill_vs_naive"] and xa[LGBM]["skill_vs_naive"] > xa[MODEL]["skill_vs_naive"]
    r3 = (
        f"After these results the applicant chose the scikit-learn model trained on the {k} commodities as the advisory model. "
        f"Because it was chosen after seeing this test, its test accuracy is optimistic, and this test cannot separate the effect "
        f"of more commodities from that of leaving out neighbouring-State data; shadow mode will check it on new data. "
        + ("LightGBM is a candidate for that comparison." if lgbm_better else "LightGBM was not adopted.")
    )
    c["ext_reading_lgbm"] = r1
    c["ext_reading_comm"] = r2 + " " + r_hw
    c["ext_reading_end"] = r3

    # best-covered commodities that the rule left out, with the reason (from the counts)
    exc = sorted((r for r in e["commodity_coverage"] if not r["good_data"]), key=lambda r: -r["weeks_3y"])[:8]
    t = ["<table><tr><th>Not modelled (best covered)</th><th class='n'>Market-weeks, 156 weeks</th>"
         "<th class='n'>Market-weeks, test year</th><th>Reason</th></tr>"]
    for r in exc:
        why = []
        if r["weeks_3y"] < rule["min_weeks_3y"]:
            why.append(f"under {num(rule['min_weeks_3y'])} in 156 weeks"
                       + (" (few prices before the test year)" if r["weeks_1y"] >= 0.6 * r["weeks_3y"] else ""))
        if r["weeks_1y"] < rule["min_weeks_1y"]:
            why.append(f"under {num(rule['min_weeks_1y'])} in the test year")
        t.append(_tr([f"<td>{html.escape(r['commodity'])}</td>", f"<td class='n'>{num(r['weeks_3y'])}</td>",
                      f"<td class='n'>{num(r['weeks_1y'])}</td>", f"<td>{'; '.join(why)}</td>"]))
    t.append("</table>")
    c["table_ext_excluded"] = "\n".join(t)

    # ---------------------------------------------------------------- section 9 bullet
    def bw(s: float) -> str:
        return f"{f1(abs(s))}% {'better' if s > 0 else 'worse'}"

    hm = xa[HW_MUL]["skill_vs_naive"]
    hw_mul_txt = f"{f1(abs(hm))}%" if (hm > 0) == (xa[HW]["skill_vs_naive"] > 0) else bw(hm)
    c["ext_short"] = (
        f"All {k} crops: the model was {_short(xa[MODEL]['skill_vs_naive'])}, better for {beat[MODEL]} of {k}; "
        f"LightGBM {bw(xa[LGBM]['skill_vs_naive'])}; Holt-Winters {bw(xa[HW]['skill_vs_naive'])} (multiplicative {hw_mul_txt})."
    )
    c["ext_k"] = str(k)
    c["ext_rows"] = num(e["ext"]["n_rows"])
    _ = xb, SNAIVE
    return c
