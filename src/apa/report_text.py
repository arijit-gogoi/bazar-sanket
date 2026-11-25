"""Result sentences generated from the numbers, so the text always matches them.

Each function returns plain text (no HTML). Wording follows the direction and
size of each comparison (better / worse / about the same).
"""

from __future__ import annotations

from .report import f1, num


def _skill(model_mae: float, base_mae: float) -> float:
    return 100 * (1 - model_mae / base_mae)


def _vs(red: float, base: str, tie: float = 2.0) -> str:
    if abs(red) < tie:
        return f"essentially the same as {base} (absolute error {f1(abs(red))}% {'lower' if red > 0 else 'higher'})"
    return f"{f1(abs(red))}% {'less' if red > 0 else 'more'} absolute error than {base}"


def overall(f: dict) -> str:
    s = f["score_assam"][0]
    red = _skill(s["mae_model"], s["mae_naive"])
    red_sn = _skill(s["mae_model"], s["mae_snaive"])
    return (
        f"Across {num(s['n'])} out-of-sample forecasts for {f['n_series_assam']} Assam market series (one to four weeks ahead), the model's "
        f"median forecast had a mean absolute percentage error (MAPE) of {f1(s['mape_model'])}%, against {f1(s['mape_naive'])}% for no change "
        f"and {f1(s['mape_snaive'])}% for seasonal naive: {_vs(red, 'no change')}, and {_vs(red_sn, 'seasonal naive')}."
    )


def res_short(f: dict) -> str:
    s = f["score_assam"][0]
    pv = f["score_prev"][0]
    return (
        f"Accuracy over 12 months out of sample ({num(s['n'])} forecasts, {f['n_series_assam']} Assam series, one to four weeks ahead): "
        f"MAPE {f1(s['mape_model'])}% for the model (chosen after this test; pre-set model {f1(pv['mape_model'])}%), "
        f"{f1(s['mape_naive'])}% for no change and {f1(s['mape_snaive'])}% for seasonal naive; "
        f"80% bands contained {f1(s['coverage80'])}% of actual prices; direction called correctly in {f1(s['dir_model'])}% of cases "
        f"versus {f1(s['dir_naive'])}% for no change."
    )


def band_dir(f: dict) -> str:
    s = f["score_assam"][0]
    cov = s["coverage80"]
    if cov < 72:
        cal = "too narrow; the planned conformal adjustment would widen them"
    elif cov > 88:
        cal = "somewhat too wide, i.e. cautious"
    else:
        cal = "close to the 80% target"
    return (
        f"Its 80% price bands contained {f1(cov)}% of actual prices ({cal}). It called the direction of the move (down, flat or up by "
        f"more than 5%) correctly in {f1(s['dir_model'])}% of cases, against {f1(s['dir_naive'])}% for no change, which always predicts \"flat\"."
    )


def horizons(f: dict) -> str:
    rows = sorted(f["score_assam_by_h"], key=lambda r: r["h"])
    sk = [_skill(r["mae_model"], r["mae_naive"]) for r in rows]
    h1, rest = sk[0], sk[1:]
    txt = f"Against no change, absolute error was {f1(abs(h1))}% {'lower' if h1 > 0 else 'higher'} one week ahead"
    if all(v > 0 for v in rest):
        by = f"by {f1(min(rest))}–{f1(max(rest))}%" if min(rest) >= 0.05 else f"by up to {f1(max(rest))}%"
        txt += f" {'and' if h1 > 0 else 'but'} lower at two to four weeks{' too' if h1 > 0 else ''} ({by})."
    elif all(v <= 0 for v in rest):
        txt += f" and {f1(min(abs(v) for v in rest))}–{f1(max(abs(v) for v in rest))}% higher at two to four weeks."
    else:
        txt += f" and within {f1(max(abs(v) for v in rest))}% either way at two to four weeks."
    return txt


def selected(f: dict) -> str:
    rows = f["score_selected"]
    mk = f["current_advisory"][0]["market"].replace("(", " (") if f.get("current_advisory") else "selected"
    win = [r for r in rows if r["mape_model"] < r["mape_naive"]]
    lose = [r for r in rows if r["mape_model"] >= r["mape_naive"]]

    def d(r):
        return f"{r['commodity'].lower()} ({f1(r['mape_model'])}% vs {f1(r['mape_naive'])}%)"

    if not lose:
        return f"For all three selected {mk} series the model beat no change: " + "; ".join(d(r) for r in rows) + "."
    if not win:
        return (f"For none of the three selected {mk} series did the model beat no change (" + "; ".join(d(r) for r in rows)
                + "), so the skill gate would withhold all three.")
    return (
        f"Of the three selected {mk} series, the model beat no change for " + " and ".join(d(r) for r in win)
        + "; " + " and ".join(d(r) for r in lose) + f" would be withheld by the skill gate."
    )


def stress_short(f: dict) -> str:
    st = {r["scenario"][:2]: r for r in f["stress"]}
    out = []
    s1, s2, s3 = st.get("S1"), st.get("S2"), st.get("S3")
    if s1 and s1["mape_local"] is not None:
        nv = f" (no change: {f1(s1['mape_naive_same_rows'])}%)" if s1.get("mape_naive_same_rows") is not None else ""
        out.append(
            f"with only 26 weeks of local history, MAPE was {f1(s1['mape_pooled_same_rows'])}% for the global model versus "
            f"{f1(s1['mape_local'])}% for a local-only model on the same forecasts{nv}"
        )
    if s2 and s2["mape_local"] is not None:
        out.append(f"with half the local weeks missing, {f1(s2['mape_pooled_same_rows'])}% versus {f1(s2['mape_local'])}%")
    if s3:
        out.append(
            f"with no Assam history at all, learning only from neighbouring States, {f1(s3['mape_pooled'])}% "
            f"(no change: {f1(s3['mape_naive'])}%)"
        )
    return ("Stress test: " + "; ".join(out) + ".") if out else ""


def stress_reading(f: dict) -> str:
    st = {r["scenario"][:2]: r for r in f["stress"]}
    s0, s1, s2, s3 = (st.get(k) for k in ("S0", "S1", "S2", "S3"))
    txt = []

    def cmp(a, b):
        if abs(a - b) <= 0.05 * b:
            return "about the same as"
        return "lower than" if a < b else "higher than"

    if s1 and s1["mape_local"] is not None:
        txt.append(
            f"With only 26 weeks of local history, the global model's error ({f1(s1['mape_pooled_same_rows'])}%) was "
            f"{cmp(s1['mape_pooled_same_rows'], s1['mape_local'])} the local model's ({f1(s1['mape_local'])}%) on the same forecasts."
        )
    if s2 and s2["mape_local"] is not None:
        txt.append(
            f"With half of the local weeks removed at random, it was {cmp(s2['mape_pooled_same_rows'], s2['mape_local'])} the local "
            f"model ({f1(s2['mape_pooled_same_rows'])}% vs {f1(s2['mape_local'])}%)."
        )
    if s3 and s0:
        txt.append(
            f"With no Assam history at all, a local model is impossible; the global model learned only from neighbouring States and "
            f"forecast Assam prices with {f1(s3['mape_pooled'])}% error, against {f1(s0['mape_pooled'])}% with full history and "
            f"{f1(s3['mape_naive'])}% for no change."
        )
        if s3["mape_pooled"] <= s0["mape_pooled"]:
            txt.append(
                "That old, patchy Assam history added no accuracy here is itself informative: recent and comparable-market data carry "
                "most of the signal, so a short but regular AMIU feed can be more valuable than a long, broken archive."
            )
    txt.append(
        "In short, learning from comparable markets is what makes forecasting possible where local data are short or missing, "
        "and local data refine it as they accumulate."
    )
    return " ".join(txt)


def decision_short(f: dict) -> str:
    rows = f["decision"]
    ohi = max(r["oracle_vs_sell_pct"] for r in rows)
    if all(r["hold_signals"] == 0 for r in rows):
        lo = min(r["hold_max_vs_sell_pct"] for r in rows)
        hi = max(r["hold_max_vs_sell_pct"] for r in rows)
        if hi < 0:
            blind = f"blind holding would have lost {f1(-hi)}–{f1(-lo)}%"
        else:
            blind = f"blind holding would have changed value by {lo:+.1f}% to {hi:+.1f}%"
        return (
            f"Decision rule: it advised selling promptly in every test week for these perishables; {blind}; "
            f"perfect foresight shows up to {f1(ohi)}% available to better forecasts."
        )
    parts = [f"{r['commodity'].lower()} {r['rule_vs_sell_pct']:+.1f}% (blind holding {r['hold_max_vs_sell_pct']:+.1f}%)" for r in rows]
    return "Decision rule, value realised versus selling at once: " + "; ".join(parts) + "."


def decision_reading(f: dict) -> str:
    txt = []
    rows = f["decision"]
    if all(r["hold_signals"] == 0 for r in rows):
        lo = min(r["hold_max_vs_sell_pct"] for r in rows)
        hi = max(r["hold_max_vs_sell_pct"] for r in rows)
        txt.append(
            f"The rule never advised holding in any of the {sum(r['weeks'] for r in rows)} test weeks; blindly holding for the maximum "
            f"period would have changed the value realised by {lo:+.1f}% to {hi:+.1f}%, depending on the crop."
        )
        rows = []
    for r in rows:
        c = r["commodity"].lower()
        if r["hold_signals"] == 0:
            txt.append(
                f"For {c}, the rule never advised holding in {r['weeks']} test weeks; blindly holding for the maximum period would have "
                f"changed the value realised by {r['hold_max_vs_sell_pct']:+.1f}%."
            )
            continue
        sign = "gained" if r["rule_vs_sell_pct"] > 0 else "lost"
        txt.append(
            f"For {c}, it advised holding in {r['hold_signals']} of {r['weeks']} weeks and {sign} {f1(abs(r['rule_vs_sell_pct']))}% on "
            f"average versus selling at once ({r['hold_regrets']} holds ended below the sell-now value); always holding would have given "
            f"{r['hold_max_vs_sell_pct']:+.1f}%."
        )
    txt.append(
        "Perfect foresight (last column) shows the value that better forecasts could unlock. For perishable vegetables the right default is "
        "to sell promptly unless a rise is both likely and large; the rule's value will be larger for storable crops such as potato or "
        "mustard, which the pilot will add."
    )
    return " ".join(txt)


def season_short(f: dict) -> str:
    sb = f["season_backtest"]
    if not sb:
        return ""
    mp = sum(r["mae_profile_pct"] for r in sb) / len(sb)
    mf = sum(r["mae_flat_pct"] for r in sb) / len(sb)
    better = sum(1 for r in sb if r["mae_profile_pct"] < r["mae_flat_pct"])
    return (
        f"Season-ahead check: profiles from earlier years only had {f1(mp)} points average error in monthly price deviations, "
        f"versus {f1(mf)} assuming no seasonality (better in {better} of {len(sb)} commodity-year tests)."
    )


def season_reading(f: dict) -> str:
    sb = f["season_backtest"]
    if not sb:
        return "Not enough Assam series-years with eight or more months of data to test the seasonal profile."
    mp = sum(r["mae_profile_pct"] for r in sb) / len(sb)
    mf = sum(r["mae_flat_pct"] for r in sb) / len(sb)
    tail = (
        "The seasonal shape is therefore a usable basis for crop-planning cards (which months are usually dear or cheap), while the price "
        "level a season ahead still needs the additional drivers listed below."
        if mp < mf
        else "The seasonal shape alone is therefore not yet reliable enough for planning advice; cards would show it only as context."
    )
    return season_short(f) + " " + tail


def _adv_crops(f: dict) -> int:
    return len(f["advisory_model"]["commodities"])


def switch_note(f: dict) -> str:
    """Section E: which model gives the advice, and why its test accuracy is optimistic."""
    s, pv = f["score_assam"][0], f["score_prev"][0]
    k = _adv_crops(f)

    def err(r: dict) -> str:
        red = _skill(r["mae_model"], r["mae_naive"])
        return f"absolute error {f1(abs(red))}% {'lower' if red > 0 else 'higher'} than no change (MAPE {f1(r['mape_model'])}%)"

    return (
        f"Model choice: the advisory model is trained on every Assam market of the {k} crops with good data (Assam markets only). "
        f"The applicant chose it after this 12-month test, in which it had {err(s)}; the model set before testing, trained on the "
        f"three pilot crops with neighbouring-State markets, had {err(pv)}. Because the choice was made after seeing the test, "
        f"these results are optimistic; shadow mode will check them on new data."
    )


def switch_limit(f: dict) -> str:
    """Section J bullet."""
    pv = f["score_prev"][0]
    return (
        f"The advisory model ({_adv_crops(f)} Assam crops) was chosen after the 12-month test, so its test accuracy is optimistic; "
        f"the model set before testing had MAPE {f1(pv['mape_model'])}%. Shadow mode will check the choice on new data."
    )


def reading_short(f: dict) -> str:
    s = f["score_assam"][0]
    red = _skill(s["mae_model"], s["mae_naive"])
    h1 = _skill(f["score_assam_by_h"][0]["mae_model"], f["score_assam_by_h"][0]["mae_naive"])
    claims = []
    if abs(red) < 2:
        claims.append("matches the no-change benchmark on average")
    elif red > 0:
        claims.append("beats the no-change benchmark on average")
    else:
        claims.append("does not yet beat the no-change benchmark on average")
    if h1 > 0:
        claims.append("is better one week ahead")
    wins = [r for r in f["score_selected"] if r["mape_model"] < r["mape_naive"]]
    if wins:
        claims.append("is better for some series")
    if s["dir_model"] > s["dir_naive"]:
        claims.append("calls the direction of moves more often")
    if 72 <= s["coverage80"] <= 88:
        claims.append("gives well-calibrated ranges")
    body = ", ".join(claims[:-1]) + " and " + claims[-1]
    return (
        f"Weekly vegetable prices are hard to forecast and no change is a strong benchmark. On public data, the model {body}; "
        f"the global approach keeps working when local history is missing. "
        "This is why publication is gated per series. AMIU data with arrivals should add skill, and will be tested the same way."
    )
