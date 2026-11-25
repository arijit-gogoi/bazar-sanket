"""Render the proposal answers and the Proof of Work from generated metrics.

Every number in answers.md and proof-of-work.html comes from out/metrics.json,
which merges the outputs of nb/profile.py and nb/forecast.py. Templates use
string.Template placeholders (${name}); a missing value is an error, never a
silent blank.

Usage (from project root):
    uv run python -I -X utf8 src/render.py            # answers.md + pow/proof-of-work.html
    uv run python -I -X utf8 src/render.py --pdf      # also print proof-of-work.pdf with Chrome
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
TPL = ROOT / "src" / "templates"
WORD_LIMIT = 500


# --------------------------------------------------------------------------- formatting
def num(x: float | int) -> str:
    """Indian digit grouping for integers."""
    v = int(round(x))
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


def rs(x: float) -> str:
    return "₹" + num(round(x / 10.0) * 10)


def pct(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def date_long(s: str) -> str:
    d = dt.date.fromisoformat(str(s)[:10])
    return f"{d.day} {d.strftime('%B %Y')}"


def date_short(s: str) -> str:
    d = dt.date.fromisoformat(str(s)[:10])
    return f"{d.day} {d.strftime('%b %Y')}"


# --------------------------------------------------------------------------- word count
def section_word_counts(md: str) -> list[tuple[str, int]]:
    """Split on '## ' headings; count whitespace-separated tokens of each body."""
    out = []
    parts = re.split(r"^## ", md, flags=re.M)
    for p in parts[1:]:
        title, _, body = p.partition("\n")
        out.append((title.strip(), len(body.split())))
    return out


def check_answers(md: str) -> list[tuple[str, int]]:
    counts = section_word_counts(md)
    if len(counts) != 12:
        raise SystemExit(f"V1 violated: expected 12 sections, found {len(counts)}")
    for i, (title, n) in enumerate(counts, 1):
        if not title.startswith(f"{i}."):
            raise SystemExit(f"V1 violated: section {i} out of order: {title!r}")
        if n > WORD_LIMIT:
            raise SystemExit(f"V1 violated: section {title!r} has {n} words > {WORD_LIMIT}")
    return counts


# --------------------------------------------------------------------------- chrome
def chrome() -> Path:
    cands = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    ]
    for c in cands:
        if c.exists():
            return c
    raise SystemExit("Chrome not found")


def print_pdf(html: Path, pdf: Path) -> None:
    cmd = [
        str(chrome()), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw", "--virtual-time-budget=10000",
        f"--print-to-pdf={pdf}", html.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    if not pdf.exists():
        raise SystemExit("PDF not produced")


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    from apa.report import build_context

    prof = json.loads((OUT / "metrics_profile.json").read_text(encoding="utf-8"))
    fc = json.loads((OUT / "metrics_forecast.json").read_text(encoding="utf-8"))
    metrics = {"profile": prof, "forecast": fc, "rendered_at": dt.datetime.now().isoformat(timespec="seconds")}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    ctx = build_context(metrics)

    def render(name: str, extra: dict | None = None) -> str:
        t = Template((TPL / name).read_text(encoding="utf-8"))
        return t.substitute({**ctx, **(extra or {})})

    s3 = render("section3.md.tmpl").strip()
    s9 = render("section9.md.tmpl").strip()
    answers = render("answers.md.tmpl", {"section3": s3, "section9": s9})
    counts = check_answers(answers)
    (ROOT / "answers.md").write_text(answers, encoding="utf-8")
    print("answers.md written; words per section:")
    for t, n in counts:
        print(f"  {n:4d}  {t}")

    html = render("pow.html.tmpl")
    pow_dir = ROOT / "pow"
    pow_dir.mkdir(exist_ok=True)
    (pow_dir / "proof-of-work.html").write_text(html, encoding="utf-8")
    print("pow/proof-of-work.html written")

    if args.pdf:
        pdf = ROOT / "proof-of-work.pdf"
        print_pdf(pow_dir / "proof-of-work.html", pdf)
        size = pdf.stat().st_size / 1e6
        if size > 10:
            raise SystemExit(f"V2 violated: PDF is {size:.1f} MB > 10 MB")
        print(f"proof-of-work.pdf written ({size:.2f} MB)")
        docs = ROOT / "docs"
        (docs / "fig").mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, docs / "proof-of-work.pdf")


if __name__ == "__main__":
    main()
