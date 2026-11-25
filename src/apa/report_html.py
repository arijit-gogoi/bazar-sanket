"""HTML-specific context for the Proof of Work: tables, mock-ups, generated readings.

All numbers come from out/metrics.json. Sentences that compare the model with a
benchmark are generated from the numbers, so wording follows the results.
"""

from __future__ import annotations

import html
import math

from .advisory import MAX_DOWNSIDE, MIN_GAIN, rule_for, rupees
from .report import MONTHS, dlong, dshort, f0, f1, num

REPO_URL = "https://github.com/arijit-gogoi/bazar-sanket"
SITE_URL = "https://arijit-gogoi.github.io/bazar-sanket/"

AS_NAME = {"Green Chilli": "কেঁচা জলকীয়া", "Cabbage": "বন্ধাকবি", "Pumpkin": "ৰঙালাউ"}
AS_MARKET = {"Pamohi(Garchuk)": "পামোহী (গৰচুক)"}
GSM7 = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)


def esc(s) -> str:
    return html.escape(str(s))


def sms_meta(text: str) -> str:
    if all(ch in GSM7 for ch in text):
        n = len(text)
        parts = 1 if n <= 160 else -(-n // 153)
        enc = "GSM 7-bit"
    else:
        n = len(text.encode("utf-16-le")) // 2
        parts = 1 if n <= 70 else -(-n // 67)
        enc = "Unicode (UCS-2)"
    return f"{n} characters · {enc} · {parts} SMS part{'s' if parts > 1 else ''}"


def signal_en(sig: str, h: int) -> str:
    return {
        "SELL": "SELL NOW",
        "HOLD": f"HOLD UP TO {h} WEEK{'S' if h > 1 else ''}",
        "STORE": f"STORE UP TO {h} WEEKS",
    }[sig]


def signal_as(sig: str, h: int) -> str:
    return {
        "SELL": "এতিয়াই বিক্ৰী কৰক",
        "HOLD": f"{h} সপ্তাহলৈ ধৰি ৰাখিব পাৰে",
        "STORE": f"{h} সপ্তাহলৈ সংৰক্ষণ কৰিব পাৰে",
    }[sig]


def plain_rs(x: float) -> str:
    return rupees(x).replace("₹", "")


def _tr(cells: list[str]) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def html_context(m: dict, c: dict) -> dict[str, str]:
    p, f = m["profile"], m["forecast"]
    h: dict[str, str] = {
        "today": dlong(str(m["rendered_at"])[:10]),
        "repo_url": REPO_URL,
        "site_url": SITE_URL,
    }

    # ---------------------------------------------------------------- section A
    rows = []
    groups = sorted(p["renames"], key=lambda r: (-len(r["market"]), r["market_key"]))
    for r in groups[:8]:
        names = [
            f"{esc(n)} <span class='small'>({str(a)[:4]}–{str(b)[:4]})</span>"
            for n, a, b in zip(r["market"], r["first"], r["last"])
        ]
        rows.append(f"<tr><td>{esc(r['market'][0].replace(' APMC', ''))}</td><td>{' → '.join(names)}</td></tr>")
    h["renames_rows"] = "\n".join(rows)
    comp = {k: v for k, v in p["rows_by_state"].items() if k != "Assam"}
    h["comp_rows"] = num(sum(comp.values()))
    top = max(comp.items(), key=lambda kv: kv[1])
    h["comp_note"] = (
        f"{esc(top[0])} contributes the most ({num(top[1])} records); several North-Eastern States report only a few markets."
    )
    t = ["<table><tr><th>Region</th><th>Commodity</th><th class='n'>Market series</th><th class='n'>Weeks with a price</th><th>First week</th></tr>"]
    for r in f["data_summary"]:
        t.append(
            f"<tr><td>{esc(r['region'])}</td><td>{esc(r['commodity'])}</td><td class='n'>{num(r['series'])}</td>"
            f"<td class='n'>{num(r['observed_weeks'])}</td><td>{str(r['first_week'])[:10]}</td></tr>"
        )
    t.append("</table>")
    h["data_summary_table"] = "\n".join(t)

    # ---------------------------------------------------------------- mock-ups
    cur = {a["commodity"]: a for a in f["current_advisory"]}
    sel_scores = {r["commodity"]: r for r in f["score_selected"]}
    order = [s["commodity"] for s in p["selection"]["pairs"] if s["commodity"] in cur]

    def band(a, hh):
        return a["bands"][str(hh)]

    def sms_as(a):
        b1 = band(a, 1)
        return (
            f"বজাৰ সংকেত: {AS_NAME.get(a['commodity'], a['commodity'])}, {AS_MARKET.get(a['market'], a['market'])}\n"
            f"এই সপ্তাহৰ দাম: {rupees(a['price_now'])}/কুইণ্টল\n"
            f"অহা সপ্তাহৰ সম্ভাৱ্য দাম: {rupees(b1[0])}–{plain_rs(b1[2])}\n"
            f"পৰামৰ্শ: {signal_as(a['signal'], a['hold_weeks'])}\n"
            f"এয়া পূৰ্বাভাস, নিশ্চয়তা নহয়। -AMIU"
        )

    def sms_en(a):
        b1 = band(a, 1)
        return (
            f"Bazar Sanket: {a['commodity']}, {a['market']}\n"
            f"Price this week: Rs {plain_rs(a['price_now'])}/qtl\n"
            f"Likely next week: Rs {plain_rs(b1[0])}-{plain_rs(b1[2])}\n"
            f"Advice: {signal_en(a['signal'], a['hold_weeks'])}\n"
            f"Outlook only, not a guarantee. -AMIU"
        )

    a1, a2 = cur[order[0]], cur[order[-1]]
    h["sms_as_1"] = esc(sms_as(a1))
    h["sms_as_1_meta"] = sms_meta(sms_as(a1))
    h["sms_en_1"] = esc(sms_en(a1))
    h["sms_en_1_meta"] = sms_meta(sms_en(a1))
    h["sms_en_2"] = esc(sms_en(a2))
    h["sms_en_2_meta"] = sms_meta(sms_en(a2))

    wa = max((cur[cn] for cn in order), key=lambda a: a["rule"]["max_hold_weeks"])
    h["wa_title_as"] = f"{AS_NAME.get(wa['commodity'], '')} · {AS_MARKET.get(wa['market'], wa['market'])}"
    h["wa_title_en"] = f"{esc(wa['commodity'])} · {esc(wa['market'])}"
    h["wa_p0"] = rupees(wa["price_now"])
    lo = min(min(b[0] for b in wa["bands"].values()), wa["price_now"])
    hi = max(max(b[2] for b in wa["bands"].values()), wa["price_now"])
    span = (hi - lo) or 1.0

    def xpos(v):
        return 4 + 92 * (v - lo) / span

    bars = []
    for hh in sorted(wa["bands"], key=int):
        b = wa["bands"][hh]
        bars.append(
            f"<div class='small'>In {hh} week{'s' if int(hh) > 1 else ''}: <b>{rupees(b[0])}–{plain_rs(b[2])}</b> "
            f"(most likely {rupees(b[1])})</div>"
            f"<div class='bandbar'><div class='b' style='left:{xpos(b[0]):.1f}%;width:{max(xpos(b[2]) - xpos(b[0]), 1):.1f}%'></div>"
            f"<div class='m' style='left:{xpos(b[1]):.1f}%'></div><div class='t' style='left:{xpos(wa['price_now']):.1f}%'></div></div>"
        )
    h["wa_bars"] = "\n".join(bars) + (
        "<div class='small' style='margin-top:-2mm'>Band = likely range (P10–P90) · thick mark = most likely · "
        "thin black mark = today's price</div>"
    )
    h["wa_signal"] = wa["signal"]
    h["wa_signal_text"] = signal_en(wa["signal"], wa["hold_weeks"])
    h["wa_reason"] = esc(wa["reason_en"])
    if wa["signal"] == "SELL":
        wb1 = band(wa, 1)
        h["wa_reason_as"] = (
            f"অহা সপ্তাহত দাম {rupees(wb1[0])}ৰ পৰা {rupees(wb1[2])}ৰ ভিতৰত থকাৰ সম্ভাৱনা। "
            "ৰাখি থ'লে লোকচানৰ আশংকা আছে, সেয়েহে এতিয়াই বিক্ৰী কৰাটো ভাল।"
        )
    else:
        bh = band(wa, wa["hold_weeks"])
        h["wa_reason_as"] = (
            f"{wa['hold_weeks']} সপ্তাহত দাম {rupees(bh[0])}ৰ পৰা {rupees(bh[2])}ৰ ভিতৰত থকাৰ সম্ভাৱনা। "
            "লোকচান বাদ দিও ৰাখিলে লাভ হ'ব পাৰে।"
        )

    b1 = band(a1, 1)
    as_name = AS_NAME.get(a1["commodity"], a1["commodity"])
    as_mkt = AS_MARKET.get(a1["market"], a1["market"])
    menu_as = ", ".join(f"{AS_NAME.get(cn, cn)}ৰ বাবে {i + 1} টিপক" for i, cn in enumerate(order))
    menu_en = ", ".join(f"for {cn.lower()} press {i + 1}" for i, cn in enumerate(order))
    h["ivr_as"] = (
        f"<b>আৰম্ভণি:</b> নমস্কাৰ। অসম কৃষি বিশ্ববিদ্যালয়ৰ বজাৰ সংকেতলৈ আপোনাক স্বাগতম। {menu_as}।<br>"
        f"<b>১ টিপিলে:</b> {as_mkt} বজাৰত {as_name}ৰ এই সপ্তাহৰ দাম প্ৰতি কুইণ্টল {plain_rs(a1['price_now'])} টকা। "
        f"অহা সপ্তাহত দাম {plain_rs(b1[0])} টকাৰ পৰা {plain_rs(b1[2])} টকাৰ ভিতৰত থকাৰ সম্ভাৱনা। "
        f"পৰামৰ্শ: {signal_as(a1['signal'], a1['hold_weeks'])}। এয়া পূৰ্বাভাস, নিশ্চয়তা নহয়।<br>"
        "<b>শেষত:</b> পুনৰ শুনিবলৈ ৯ টিপক। কৃষি সম্প্ৰসাৰণ সহায়কৰ সৈতে কথা পাতিবলৈ ০ টিপক।"
        "<br><span style='font-family:Segoe UI;color:#777'>Draft translation, native review pending</span>"
    )
    h["ivr_en"] = (
        f"<b>Greeting:</b> Namaskar. Welcome to Bazar Sanket from Assam Agricultural University. {menu_en.capitalize()}.<br>"
        f"<b>On 1:</b> At {esc(a1['market'])} market, {a1['commodity'].lower()} is selling at {plain_rs(a1['price_now'])} "
        f"rupees per quintal this week. Next week the price is likely to be between {plain_rs(b1[0])} and {plain_rs(b1[2])} rupees. "
        f"Advice: {signal_en(a1['signal'], a1['hold_weeks']).lower()}. This is an outlook, not a guarantee.<br>"
        "<b>Close:</b> Press 9 to hear again. Press 0 to reach your Agriculture Extension Assistant."
    )

    dash, console = [], []
    for cn in order:
        a = cur[cn]
        s = sel_scores[cn]
        gate_ok = s["mape_model"] < s["mape_naive"]
        status = (
            f"<span class='ok'>Published</span> · error {f1(s['mape_model'])}% vs {f1(s['mape_naive'])}% for no change"
            if gate_ok
            else f"<span class='warn'>Withheld</span> · error {f1(s['mape_model'])}% vs {f1(s['mape_naive'])}% for no change"
        )

        def bb(hh, a=a):
            b = a["bands"].get(str(hh))
            return f"{rupees(b[0])}–{plain_rs(b[2])}" if b else "–"

        sig = f"<span class='signal sig-{a['signal']}'>{signal_en(a['signal'], a['hold_weeks'])}</span>"
        dash.append(
            f"<tr><td>{esc(cn)}<br><span class='as small'>{AS_NAME.get(cn, '')}</span></td>"
            f"<td class='n'>{rupees(a['price_now'])}</td><td>{bb(1)}</td><td>{bb(2)}</td><td>{bb(4)}</td>"
            f"<td>{sig if gate_ok else '<span class=small>current price only</span>'}</td><td class='small'>{status}</td></tr>"
        )
        fresh = a["week"] == f["end_week"]
        checks = [
            ("✓" if gate_ok else "✗") + f" skill gate ({f1(s['mape_model'])}% vs {f1(s['mape_naive'])}%)",
            "✓ reported this week" if fresh else f"! last report {dshort(a['week'])}",
            "✓ no event notes",
        ]
        draft = (f"{esc(cn)}, {esc(a['market'])}: next week {bb(1)} per quintal" if gate_ok else
                 f"{esc(cn)}, {esc(a['market'])}: this week {rupees(a['price_now'])} per quintal; no forecast sent (below benchmark)")
        csig = sig if gate_ok else "<span class='signal' style='background:#777'>PRICE ONLY</span>"
        console.append(
            f"<tr><td>{draft}</td><td>{csig}</td>"
            f"<td class='small'>{'<br>'.join(checks)}</td>"
            "<td><span class='btn pri'>Approve</span><span class='btn'>Edit</span><span class='btn'>Hold</span></td></tr>"
        )
    h["dash_rows"] = "\n".join(dash)
    h["console_rows"] = "\n".join(console)

    nrows = ["<tr><th>Commodity</th><th>Market</th><th class='n'>₹/q</th></tr>"]
    for cn in order:
        for r in [r for r in f["nearby_latest_week"] if r["commodity"] == cn][:3]:
            nrows.append(f"<tr><td>{esc(cn)}</td><td>{esc(r['market_label'])}</td><td class='n'>{plain_rs(r['price'])}</td></tr>")
    h["nearby_rows"] = "\n".join(nrows)

    dq = ["<tr><th>Alert</th><th>Detail</th></tr>"]
    for s_ in p["silent_markets"][:3]:
        dq.append(f"<tr><td>Market silent</td><td>{esc(s_['name'].replace(' APMC', ''))}: no price since {dshort(s_['last'])}</td></tr>")
    tri = [r for r in p["renames"] if len(r["market"]) >= 3]
    for r in tri[:2]:
        dq.append(f"<tr><td>Name variants</td><td>{esc(' / '.join(r['market']))} treated as one market (AMIU to confirm)</td></tr>")
    dq.append(f"<tr><td>Implausible price</td><td>{c['outlier_pct']}% of weekly prices flagged (&gt;3× jump); check unit or entry</td></tr>")
    h["dq_rows"] = "\n".join(dq)

    prof = f["season_profile"]
    sbt = f["season_backtest"]
    cards = []
    for cn in order:
        pr = sorted([r for r in prof if r["commodity"] == cn], key=lambda r: r["month"])
        if not pr:
            continue
        lows = sorted(pr, key=lambda r: r["p50"])[:2]
        highs = sorted(pr, key=lambda r: -r["p50"])[:2]
        swing = 100 * (math.exp(max(r["p50"] for r in pr) - min(r["p50"] for r in pr)) - 1)
        spread = 100 * (math.exp(sorted(r["p75"] - r["p25"] for r in pr)[len(pr) // 2] / 2) - 1)
        tests = [r for r in sbt if r["commodity"] == cn]
        test_txt = ""
        if tests:
            mp = sum(r["mae_profile_pct"] for r in tests) / len(tests)
            mf = sum(r["mae_flat_pct"] for r in tests) / len(tests)
            test_txt = (
                f"Out-of-sample check on {len(tests)} past year{'s' if len(tests) > 1 else ''}: monthly error {f1(mp)} "
                f"points vs {f1(mf)} assuming no seasonality."
            )
        cards.append(
            f"<div class='box small'><b style='font-size:10pt'>{esc(cn)}</b> <span class='as'>{AS_NAME.get(cn, '')}</span><br>"
            f"Usually cheapest: <b>{MONTHS[lows[0]['month'] - 1]}, {MONTHS[lows[1]['month'] - 1]}</b><br>"
            f"Usually dearest: <b>{MONTHS[highs[0]['month'] - 1]}, {MONTHS[highs[1]['month'] - 1]}</b><br>"
            f"Dearest month is typically about {f0(swing)}% above the cheapest<br>"
            f"Between years, a given month's price varies by about ±{f0(spread)}% (middle half of years)<br>"
            f"<span style='color:#555'>{test_txt}</span><br>"
            "<i>Planning use:</i> plan sowing and harvest so that sales fall in dearer months where the crop allows; treat as a range, not a promise.</div>"
        )
    h["crop_cards"] = "\n".join(cards)

    # ---------------------------------------------------------------- result tables
    th = (
        "<tr><th>{0}</th><th class='n'>Forecasts</th><th class='n'>MAPE model</th><th class='n'>MAPE seasonal naive</th>"
        "<th class='n'>MAPE no change</th><th class='n'>Skill vs no change</th><th class='n'>Direction model</th>"
        "<th class='n'>Direction no change</th><th class='n'>Band coverage (target 80%)</th></tr>"
    )

    def row(label, r):
        return _tr([
            f"<td>{label}</td>", f"<td class='n'>{num(r['n'])}</td>", f"<td class='n'>{f1(r['mape_model'])}%</td>",
            f"<td class='n'>{f1(r['mape_snaive'])}%</td>", f"<td class='n'>{f1(r['mape_naive'])}%</td>",
            f"<td class='n'>{r['skill_vs_naive']:+.1f}%</td>", f"<td class='n'>{f1(r['dir_model'])}%</td>",
            f"<td class='n'>{f1(r['dir_naive'])}%</td>", f"<td class='n'>{f1(r['coverage80'])}%</td>",
        ])

    t = ["<table>", th.format("Horizon")]
    for r in f["score_assam_by_h"]:
        t.append(row(f"{r['h']} week{'s' if r['h'] > 1 else ''} ahead", r))
    t.append(row("<b>All horizons</b>", f["score_assam"][0]))
    t.append("</table>")
    h["table_by_h"] = "\n".join(t)
    t = ["<table>", th.format("Series (all horizons)")]
    for r in f["score_selected"]:
        t.append(row(esc(r["commodity"]), r))
    t.append("</table>")
    h["table_sel"] = "\n".join(t)

    t = [
        "<table><tr><th>Assam history before the test year</th><th class='n'>Forecasts with a local model</th><th class='n'>No change (same)</th>"
        "<th class='n'>Local model</th><th class='n'>Global model (same)</th><th class='n'>No change, all forecasts</th><th class='n'>Global model, all forecasts</th></tr>"
    ]
    for r in f["stress"]:
        loc = f"{f1(r['mape_local'])}%" if r["mape_local"] is not None else "not possible"
        same = f"{f1(r['mape_pooled_same_rows'])}%" if r["mape_pooled_same_rows"] is not None else "–"
        nsame = f"{f1(r['mape_naive_same_rows'])}%" if r.get("mape_naive_same_rows") is not None else "–"
        nloc = num(r["n"] * r["local_available_pct"] / 100) if r["mape_local"] is not None else "0"
        t.append(_tr([
            f"<td>{esc(r['scenario'][3:])}</td>", f"<td class='n'>{nloc} of {num(r['n'])}</td>", f"<td class='n'>{nsame}</td>",
            f"<td class='n'>{loc}</td>", f"<td class='n'>{same}</td>", f"<td class='n'>{f1(r['mape_naive'])}%</td>",
            f"<td class='n'>{f1(r['mape_pooled'])}%</td>",
        ]))
    t.append(
        "</table><p class='small'>MAPE over horizons 1–4 weeks. A local model is fitted only for market series with at least 8 usable past weeks; (same) columns compare on exactly those forecasts.</p>"
    )
    h["table_stress"] = "\n".join(t)

    t = ["<table><tr><th>Crop</th><th class='n'>Max. holding (weeks)</th><th class='n'>Loss per week held</th><th>Storable</th><th>Basis (illustrative)</th></tr>"]
    for cn in order:
        r_ = rule_for(cn)
        t.append(_tr([
            f"<td>{esc(cn)}</td>", f"<td class='n'>{r_.max_hold_weeks}</td>", f"<td class='n'>{100 * r_.weekly_loss:.0f}%</td>",
            f"<td>{'yes' if r_.storable else 'no'}</td>", f"<td class='small'>{esc(r_.note)}</td>",
        ]))
    t.append(
        f"</table><p class='small'>Thresholds: minimum expected gain {100 * MIN_GAIN:.0f}% of today's price; "
        f"maximum pessimistic loss {100 * MAX_DOWNSIDE:.0f}%.</p>"
    )
    h["table_rules"] = "\n".join(t)

    t = [
        "<table><tr><th>Crop</th><th class='n'>Weeks evaluated</th><th class='n'>Hold advised</th><th class='n'>Rule vs selling at once</th>"
        "<th class='n'>Always hold max. vs selling at once</th><th class='n'>Perfect foresight (upper bound)</th><th class='n'>Holds that lost value</th></tr>"
    ]
    for r in f["decision"]:
        t.append(_tr([
            f"<td>{esc(r['commodity'])}</td>", f"<td class='n'>{r['weeks']}</td>", f"<td class='n'>{r['hold_signals']}</td>",
            f"<td class='n'>{r['rule_vs_sell_pct']:+.1f}%</td>", f"<td class='n'>{r['hold_max_vs_sell_pct']:+.1f}%</td>",
            f"<td class='n'>{r['oracle_vs_sell_pct']:+.1f}%</td>", f"<td class='n'>{r['hold_regrets']}</td>",
        ]))
    t.append(
        "</table><p class='small'>Average value realised per quintal relative to selling at that week's price, after the assumed weekly "
        "losses. Evaluated only in weeks where every allowed holding period has an observed price.</p>"
    )
    h["table_decision"] = "\n".join(t)

    t = [
        "<table><tr><th>Commodity</th><th class='n'>Test years</th><th class='n'>Assam series-years</th>"
        "<th class='n'>Error: profile from earlier years</th><th class='n'>Error: no seasonality</th><th class='n'>Median correlation</th></tr>"
    ]
    for cn in order:
        rs_ = [r for r in sbt if r["commodity"] == cn]
        if not rs_:
            continue
        cors = sorted(r["corr"] for r in rs_ if r["corr"] == r["corr"])
        t.append(_tr([
            f"<td>{esc(cn)}</td>", f"<td class='n'>{min(r['year'] for r in rs_)}–{max(r['year'] for r in rs_)} ({len(rs_)})</td>",
            f"<td class='n'>{sum(r['n_series'] for r in rs_)}</td>",
            f"<td class='n'>{f1(sum(r['mae_profile_pct'] for r in rs_) / len(rs_))}</td>",
            f"<td class='n'>{f1(sum(r['mae_flat_pct'] for r in rs_) / len(rs_))}</td>",
            f"<td class='n'>{cors[len(cors) // 2]:.2f}</td>" if cors else "<td class='n'>–</td>",
        ]))
    t.append(
        "</table><p class='small'>Error: mean absolute difference, in percentage points, between predicted and actual deviation of each "
        "month's price from that year's average; Assam series-years with at least 8 months of data.</p>"
    )
    h["table_season"] = "\n".join(t)
    return h
