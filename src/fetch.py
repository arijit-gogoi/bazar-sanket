"""Fetch Agmarknet daily mandi prices from the data.gov.in Open Government Data (OGD) API.

Source dataset
    "Variety-wise Daily Market Prices Data of Commodity"
    Directorate of Marketing & Inspection (DMI), Ministry of Agriculture & Farmers Welfare
    data.gov.in resource 35985678-0d79-46b4-9ed6-6f13308a1d24
    Licence: Government Open Data License - India (attribution required)

The API key is read from the DATA_GOV_IN_API_KEY environment variable or from a
git-ignored `.env` file in the project root. It is never printed or written to disk
by this script. Without a key the public sample key is used (10 rows per call).

Every API response page is cached under data/raw/pages/ together with its query
filters and retrieval timestamp, so reruns are resumable and the retrieval date of
every row can be reported.

Usage (from project root):
    uv run python -I src/fetch.py probe
    uv run python -I src/fetch.py days --state Assam --start 2023-11-25 --end 2026-11-24
    uv run python -I src/fetch.py pull --state "West Bengal" --commodity Potato --since 2016-01-01
    uv run python -I src/fetch.py consolidate
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PAGES = RAW / "pages"
RESOURCE = "35985678-0d79-46b4-9ed6-6f13308a1d24"
BASE = f"https://api.data.gov.in/resource/{RESOURCE}"
SAMPLE_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"  # public sample key published by data.gov.in
SOURCE_LABEL = (
    "Agmarknet (DMI, MoA&FW) via data.gov.in OGD API, resource "
    f"{RESOURCE} 'Variety-wise Daily Market Prices Data of Commodity'"
)


def api_key() -> tuple[str, bool]:
    """Return (key, is_personal). Never log the key."""
    key = os.environ.get("DATA_GOV_IN_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"\s*DATA_GOV_IN_API_KEY\s*=\s*['\"]?([^'\"\s]+)", line)
            if m:
                key = m.group(1)
    if key:
        return key, key != SAMPLE_KEY
    return SAMPLE_KEY, False


class Client:
    def __init__(self, pause: float = 0.4):
        self.key, self.personal = api_key()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "assam-price-advisory-research/0.1 (open-data research prototype)"
        self.pause = pause
        self.calls = 0

    def get(self, filters: dict[str, str], offset: int, limit: int, sort_desc: bool = True) -> dict:
        params: dict[str, str | int] = {"api-key": self.key, "format": "json", "offset": offset, "limit": limit}
        for k, v in filters.items():
            params[f"filters[{k}]"] = v
        if sort_desc:
            params["sort[Arrival_Date]"] = "desc"
        delay = 5.0
        for attempt in range(7):
            time.sleep(self.pause)
            try:
                r = self.session.get(BASE, params=params, timeout=120)
            except requests.RequestException as exc:  # network hiccup
                print(f"  network error ({type(exc).__name__}); retry in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay *= 2
                continue
            self.calls += 1
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    print(f"  non-JSON 200 response; retry in {delay:.0f}s", flush=True)
                    time.sleep(delay)
                    delay *= 2
                    continue
                if str(data.get("status", "ok")).lower() == "ok" or "records" in data:
                    return data
                print(f"  API message: {str(data.get('message'))[:120]}; retry in {delay:.0f}s", flush=True)
            else:
                print(f"  HTTP {r.status_code}: {r.text[:120]!r}; retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
        raise RuntimeError(f"giving up after retries: filters={filters} offset={offset} limit={limit}")


def slug(filters: dict[str, str]) -> str:
    text = "__".join(f"{k}={v}" for k, v in sorted(filters.items()))
    safe = re.sub(r"[^A-Za-z0-9=_.-]+", "-", text)[:120]
    return f"{safe}__{hashlib.sha1(text.encode()).hexdigest()[:8]}"


def save_page(dirpath: Path, offset: int, filters: dict, limit: int, data: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    page = {
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": SOURCE_LABEL,
        "filters": filters,
        "offset": offset,
        "limit": limit,
        "total": data.get("total"),
        "count": data.get("count"),
        "records": data.get("records", []),
    }
    (dirpath / f"{offset:08d}.json").write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")


def load_page(dirpath: Path, offset: int, limit: int | None = None) -> dict | None:
    """Return a cached page, or None if absent or cached with a different page size."""
    f = dirpath / f"{offset:08d}.json"
    if f.exists():
        page = json.loads(f.read_text(encoding="utf-8"))
        if limit is None or page.get("limit") == limit:
            return page
    return None


def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%d/%m/%Y").date()


def page_size(client: Client, requested: int | None) -> int:
    if not client.personal:
        return 10
    return requested or 1000


# --------------------------------------------------------------------------- probe
def cmd_probe(args) -> None:
    client = Client()
    print(f"key: {'personal' if client.personal else 'public sample (10 rows/call)'}")
    f = {"State": "Assam"}
    first = client.get(f, 0, 1)
    print(f"Assam total rows: {first.get('total')}; latest: {first['records'][0]['Arrival_Date'] if first.get('records') else '-'}")
    if not client.personal:
        return
    for lim in (10000, 5000, 2000, 1000, 500, 100):
        t0 = time.time()
        d = client.get(f, 0, lim)
        n = len(d.get("records", []))
        print(f"limit={lim:>6}: got {n} rows in {time.time() - t0:.1f}s")
        if n == lim:
            break
    for off in (9990, 10000, 20000, 100000):
        try:
            d = client.get(f, off, 10)
            recs = d.get("records", [])
            print(f"offset={off:>6}: {len(recs)} rows, first date {recs[0]['Arrival_Date'] if recs else '-'}")
        except RuntimeError as exc:
            print(f"offset={off:>6}: FAILED ({exc})")


# --------------------------------------------------------------------------- days
def cmd_days(args) -> None:
    """One query per calendar day for a state: robust against deep-offset limits."""
    client = Client(pause=args.pause)
    limit = page_size(client, args.limit)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    days.reverse()  # newest first: the most useful data arrives first
    base = PAGES / "days" / re.sub(r"[^A-Za-z0-9]+", "-", args.state)
    rows = 0
    for i, day in enumerate(days, 1):
        filters = {"State": args.state, "Arrival_Date": day.strftime("%d/%m/%Y")}
        if args.commodity:
            filters["Commodity"] = args.commodity
        d_dir = base / (day.isoformat() + (f"_{args.commodity}" if args.commodity else ""))
        offset = 0
        while True:
            page = load_page(d_dir, offset, limit)
            if page is None:
                data = client.get(filters, offset, limit, sort_desc=False)
                save_page(d_dir, offset, filters, limit, data)
                page = load_page(d_dir, offset)
            n = len(page["records"])
            rows += n
            total = int(page.get("total") or 0)
            offset += limit
            if n < limit or offset >= total:
                break
        if i % 25 == 0 or i == len(days):
            print(f"[days {args.state}] {i}/{len(days)} days done, {rows} rows, {client.calls} API calls", flush=True)


# --------------------------------------------------------------------------- pull
def pull(client: Client, filters: dict[str, str], since: dt.date, limit: int, max_offset: int, depth: int = 0) -> set[str]:
    """Page newest-first until rows are older than `since`. Returns markets seen.

    If the offset cap is hit before reaching `since`, re-pull per market (partition).
    """
    d_dir = PAGES / "pull" / slug(filters)
    offset = 0
    markets: set[str] = set()
    reached = False
    total = None
    while offset < max_offset:
        page = load_page(d_dir, offset, limit)
        if page is None:
            t0 = time.time()
            data = client.get(filters, offset, limit, sort_desc=True)
            save_page(d_dir, offset, filters, limit, data)
            page = load_page(d_dir, offset)
            recs = page["records"]
            span = f"{recs[-1]['Arrival_Date']}..{recs[0]['Arrival_Date']}" if recs else "-"
            print(f"  {'  ' * depth}page offset={offset} rows={len(recs)} {span} ({time.time() - t0:.0f}s)", flush=True)
        recs = page["records"]
        total = int(page.get("total") or 0)
        for r in recs:
            markets.add(r["Market"])
        if not recs:
            reached = True
            break
        oldest = min(parse_date(r["Arrival_Date"]) for r in recs)
        offset += limit
        if oldest < since or len(recs) < limit or offset >= total:
            reached = True
            break
    pad = "  " * depth
    print(f"{pad}pull {filters}: total={total}, pages={offset // limit}, reached_since={reached}, markets={len(markets)}", flush=True)
    if not reached and "Market" not in filters:
        for m in sorted(markets):
            pull(client, {**filters, "Market": m}, since, limit, max_offset, depth + 1)
    return markets


def cmd_pull(args) -> None:
    client = Client(pause=args.pause)
    limit = page_size(client, args.limit)
    since = dt.date.fromisoformat(args.since)
    commodities = args.commodity or [None]
    for c in commodities:
        f = {"State": args.state}
        if c:
            f["Commodity"] = c
        if args.market:
            f["Market"] = args.market
        pull(client, f, since, limit, args.max_offset)
    print(f"done: {client.calls} API calls")


# --------------------------------------------------------------------------- consolidate
def cmd_consolidate(args) -> None:
    import polars as pl

    rows = []
    for f in PAGES.rglob("*.json"):
        page = json.loads(f.read_text(encoding="utf-8"))
        ts = page.get("retrieved_at")
        for r in page.get("records", []):
            rows.append({**r, "retrieved_at": ts})
    if not rows:
        sys.exit("no cached pages found")
    num = ["Min_Price", "Max_Price", "Modal_Price"]
    df = (
        pl.DataFrame(rows, infer_schema_length=None)
        .lazy()
        .with_columns(
            pl.col("Arrival_Date").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("date"),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in num],
        )
        .rename({
            "State": "state", "District": "district", "Market": "market", "Commodity": "commodity",
            "Commodity_Code": "commodity_code", "Variety": "variety", "Grade": "grade",
            "Min_Price": "min_price", "Max_Price": "max_price", "Modal_Price": "modal_price",
        })
        .drop("Arrival_Date")
        .group_by([
            "date", "state", "district", "market", "commodity", "commodity_code", "variety", "grade",
            "min_price", "max_price", "modal_price",
        ])
        .agg(pl.col("retrieved_at").min())
        .sort(["state", "commodity", "market", "variety", "date"])
        .collect()
    )
    out = RAW / "ogd_prices.parquet"
    df.write_parquet(out)
    print(f"wrote {out.relative_to(ROOT)}: {df.height} rows, {df['date'].min()} .. {df['date'].max()}, "
          f"states={df['state'].n_unique()}, commodities={df['commodity'].n_unique()}, markets={df['market'].n_unique()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    d = sub.add_parser("days")
    d.add_argument("--state", required=True)
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--commodity")
    d.add_argument("--limit", type=int)
    d.add_argument("--pause", type=float, default=0.4)
    q = sub.add_parser("pull")
    q.add_argument("--state", required=True)
    q.add_argument("--commodity", action="append")
    q.add_argument("--market")
    q.add_argument("--since", default="2016-01-01")
    q.add_argument("--limit", type=int)
    q.add_argument("--max-offset", type=int, default=10000)
    q.add_argument("--pause", type=float, default=0.4)
    sub.add_parser("consolidate")
    args = p.parse_args()
    {"probe": cmd_probe, "days": cmd_days, "pull": cmd_pull, "consolidate": cmd_consolidate}[args.cmd](args)


if __name__ == "__main__":
    main()
