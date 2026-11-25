"""Decision layer: turn a forecast price band into a Hold / Sell / Store signal.

The rule is deliberately transparent so that AMIU can audit and tune it:

* the farmer can keep produce for at most `max_hold_weeks` (safe shelf life);
* each week held costs `weekly_loss` of the value (weight and quality loss,
  storage rent, interest);
* holding h weeks is advised only if the median outcome after losses beats
  selling today by at least MIN_GAIN, and the pessimistic outcome (10th
  percentile) after losses is not worse than selling today by more than
  MAX_DOWNSIDE;
* otherwise the advice is to sell now.

Crop parameters below are ILLUSTRATIVE ASSUMPTIONS for the prototype. They must
be replaced with values agreed with AAU before any farmer sees an advisory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropRule:
    max_hold_weeks: int
    weekly_loss: float
    storable: bool
    note: str


RULES = {
    "Green Chilli": CropRule(1, 0.10, False, "highly perishable; about one week at ambient temperature"),
    "Cabbage": CropRule(2, 0.05, False, "semi-perishable; up to about two weeks in cool, shaded storage"),
    "Pumpkin": CropRule(4, 0.02, True, "stores for weeks when mature and undamaged"),
}
DEFAULT_RULE = CropRule(1, 0.10, False, "treated as perishable")
MIN_GAIN = 0.05
MAX_DOWNSIDE = 0.10


def rule_for(commodity: str) -> CropRule:
    return RULES.get(commodity, DEFAULT_RULE)


def decide(p0: float, bands: dict[int, tuple[float, float, float]], rule: CropRule) -> dict:
    """bands: horizon -> (p10, p50, p90) in Rs/quintal."""
    best = None
    for h in range(1, rule.max_hold_weeks + 1):
        if h not in bands:
            continue
        p10, p50, p90 = bands[h]
        keep = (1 - rule.weekly_loss) ** h
        gain = p50 * keep - p0
        down = p10 * keep - p0
        if gain >= MIN_GAIN * p0 and down >= -MAX_DOWNSIDE * p0 and (best is None or gain > best["gain"]):
            best = {"h": h, "gain": gain, "down": down, "p10": p10, "p50": p50, "p90": p90, "loss_pct": 100 * (1 - keep)}
    if best is None:
        b1 = bands.get(1)
        return {"signal": "SELL", "h": 0, "gain": 0.0, "band": b1}
    signal = "STORE" if rule.storable and best["h"] >= 3 else "HOLD"
    return {"signal": signal, **best, "band": (best["p10"], best["p50"], best["p90"])}


def rupees(x: float) -> str:
    """Indian digit grouping, rounded to nearest 10 rupees."""
    v = int(round(x / 10.0) * 10)
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
    return ("-" if v < 0 else "") + "₹" + s


def reason_en(commodity: str, p0: float, d: dict, rule: CropRule) -> str:
    if d["signal"] == "SELL":
        if d.get("band"):
            p10, p50, p90 = d["band"]
            return (
                f"Next week's price is likely between {rupees(p10)} and {rupees(p90)} per quintal "
                f"(most likely {rupees(p50)}). Holding is unlikely to pay after losses of about "
                f"{100 * rule.weekly_loss:.0f}% a week, so selling now is advised."
            )
        return "No reliable forecast this week; selling at the best current price is advised."
    word = "storing" if d["signal"] == "STORE" else "holding"
    return (
        f"Price in {d['h']} week{'s' if d['h'] > 1 else ''} is likely between {rupees(d['p10'])} and "
        f"{rupees(d['p90'])} (most likely {rupees(d['p50'])}). After about {d['loss_pct']:.0f}% losses, "
        f"{word} may add about {rupees(d['gain'])} per quintal over today's {rupees(p0)}."
    )
