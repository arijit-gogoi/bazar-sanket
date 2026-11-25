"""Regenerate every metric, figure, proposal answer and the PDF from the cached raw data.

Usage (from project root):
    uv run python -I src/pipeline.py            # profile, forecast, other methods, render + PDF
    uv run python -I src/pipeline.py --no-pdf   # skip the Chrome PDF step

Data refresh (src/fetch.py) is separate because it needs a data.gov.in API key.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    steps = [
        ["nb/profile.py"],
        ["nb/forecast.py"],
        ["nb/benchmarks.py"],
        ["src/render.py"] + ([] if "--no-pdf" in sys.argv else ["--pdf"]),
    ]
    t0 = time.time()
    for step in steps:
        print(f"== {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, "-I", "-X", "utf8", *step], cwd=ROOT, check=True)
    print(f"== done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
