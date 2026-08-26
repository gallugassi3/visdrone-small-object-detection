"""Horizontal bar chart of per-class mAP50 gain, imgsz 640 -> 1024.

Parses the per-class block of analysis/compare_baseline_640_2k_vs_highres_1024_2k.txt
(the official comparison written by scripts/compare_runs.py). The delta drawn is the
file's own "delta (B-A)" column, computed there from unrounded values; subtracting the two
printed 3-decimal columns can differ from it by 0.001, so the script checks the two agree
to within that rounding and refuses to plot if they do not.

Usage (from anywhere; paths resolve from the project root):
    python scripts/delta_chart.py
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Headless backend: this script only writes a file, never opens a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPARE_PATH = PROJECT_ROOT / "analysis" / "compare_baseline_640_2k_vs_highres_1024_2k.txt"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "per_class_delta.png"

PER_CLASS_MARKER = "per-class mAP50"
# "<class>  <a>  <b>  <signed delta>"; rows with "-" placeholders are skipped
CLASS_ROW = re.compile(r"^(?P<name>[a-z][\w-]*)\s+(?P<a>\d\.\d+)\s+(?P<b>\d\.\d+)\s+(?P<delta>[+-]\d\.\d+)\s*$")
ROUNDING_TOLERANCE = 0.0015  # both columns print 3 decimals, so B-A can differ from delta by 0.001

# Same style tokens as scripts/size_chart.py
SERIES_COLOR = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#ffffff"

FIG_SIZE_IN = (9.0, 5.6)
DPI = 150
BAR_HEIGHT = 0.62
LABEL_PAD_PT = 4
X_HEADROOM = 0.30  # fraction of the widest bar kept free for the labels
SUBTITLE = "identical data & training budget; only imgsz changed"


@dataclass(frozen=True)
class ClassDelta:
    name: str
    map50_a: float
    map50_b: float
    delta: float


def parse_per_class(path: Path) -> list[ClassDelta]:
    rows: list[ClassDelta] = []
    in_block = False
    for line in path.read_text().splitlines():
        if line.strip() == PER_CLASS_MARKER:
            in_block = True
            continue
        if not in_block:
            continue
        match = CLASS_ROW.match(line)
        if match is None:
            continue
        row = ClassDelta(
            name=match["name"],
            map50_a=float(match["a"]),
            map50_b=float(match["b"]),
            delta=float(match["delta"]),
        )
        if abs((row.map50_b - row.map50_a) - row.delta) > ROUNDING_TOLERANCE:
            raise ValueError(f"{path.name}: {row.name} delta {row.delta:+.3f} does not match B-A")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no per-class rows found after '{PER_CLASS_MARKER}'")
    return rows


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline vertical grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.xaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)


def plot(rows: list[ClassDelta]) -> plt.Figure:
    # barh draws index 0 at the bottom, so ascending order puts the largest gain on top
    ordered = sorted(rows, key=lambda r: r.delta)
    y = list(range(len(ordered)))
    deltas = [r.delta for r in ordered]

    fig, ax = plt.subplots(figsize=FIG_SIZE_IN, dpi=DPI, facecolor=SURFACE)
    style_axes(ax)
    bars = ax.barh(y, deltas, height=BAR_HEIGHT, color=SERIES_COLOR, linewidth=0)
    ax.set_yticks(y, [r.name for r in ordered])
    ax.axvline(0, color=INK_MUTED, linewidth=1)

    # Delta at the bar tip in primary ink; before/after values in muted ink so the
    # reader can see the base each gain came from without a second chart
    ax.bar_label(bars, labels=[f"{d:+.3f}" for d in deltas], padding=LABEL_PAD_PT, fontsize=9, color=INK_PRIMARY)
    widest = max(abs(d) for d in deltas)
    for yi, row in zip(y, ordered):
        offset = widest * 0.12 if row.delta >= 0 else -widest * 0.12
        ax.text(
            row.delta + offset, yi, f"{row.map50_a:.3f} → {row.map50_b:.3f}",
            va="center", ha="left" if row.delta >= 0 else "right", fontsize=8, color=INK_MUTED,
        )

    lo = min(0.0, min(deltas)) - widest * X_HEADROOM * (1 if min(deltas) < 0 else 0)
    hi = max(0.0, max(deltas)) + widest * X_HEADROOM * (1 if max(deltas) > 0 else 0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Δ mAP50 (imgsz 1024 − imgsz 640)", color=INK_SECONDARY)

    ax.set_title(
        "Per-class mAP50 gain from imgsz 640 → 1024: YOLO11n on VisDrone-2k (val)",
        color=INK_PRIMARY, fontsize=12, loc="left", pad=18,
    )
    ax.text(0, 1.02, SUBTITLE, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom")
    fig.text(
        0.01, 0.01, f"Source: analysis/{COMPARE_PATH.name} (scripts/compare_runs.py, each model at its own training imgsz).",
        color=INK_MUTED, fontsize=8, ha="left", va="bottom",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main() -> None:
    rows = parse_per_class(COMPARE_PATH)
    fig = plot(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor=SURFACE)
    print(f"Parsed {len(rows)} classes; saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
