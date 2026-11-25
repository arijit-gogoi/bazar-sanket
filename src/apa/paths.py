"""Project paths. Every script and notebook resolves files from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PARQUET = ROOT / "data" / "raw" / "ogd_prices.parquet"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "out"
FIG = OUT / "fig"
METRICS = OUT / "metrics.json"

for _d in (PROCESSED, OUT, FIG):
    _d.mkdir(parents=True, exist_ok=True)
