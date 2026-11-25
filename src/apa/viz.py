"""Figure helpers: one seaborn theme, one save function, a source footnote on every figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from .paths import FIG  # noqa: E402

PALETTE = {"model": "#1f6f8b", "naive": "#9a9a9a", "snaive": "#d08c34", "actual": "#222222", "band": "#9fc5d6"}


def theme() -> None:
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
    })


def save(fig, name: str, source: str) -> str:
    """Save PNG with a source footnote; returns the relative path used in reports."""
    fig.text(0.01, 0.005, source, fontsize=6.5, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"out/fig/{name}.png"
