"""Grouped bar chart: detection rate per GT size bucket, baseline 640 vs high-res 1024.

Reads the two size-sensitivity CSVs under analysis/ and renders a single PNG for
the README / notes. The "all" aggregate row is not a size bucket, so it goes in
the subtitle instead of the axis: mixing a total among its parts makes the bars
misread as comparable.

Usage (from anywhere; paths resolve from the project root):
    python scripts/size_chart.py
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Headless backend: this script only writes a file, never opens a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.ticker import PercentFormatter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
OUTPUT_PATH = PROJECT_ROOT / "assets" / "size_sensitivity_comparison.png"

# (csv filename, legend label) in plotting order; the first series is the reference
SERIES: tuple[tuple[str, str], ...] = (
    ("size_sensitivity_baseline_640_2k.csv", "Baseline (imgsz 640)"),
    ("size_sensitivity_highres_1024_2k.csv", "High-res (imgsz 1024)"),
)
AGGREGATE_BUCKET = "all"

# Categorical slots 1 and 2 of the validated palette (CVD dE 24.7 on white)
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#ffffff"

FIG_SIZE_IN = (9.0, 5.2)
DPI = 150
BAR_WIDTH = 0.36  # two bars fill 0.72 of the slot; the rest is air between groups
BAR_GAP = 0.03  # surface-colored gap between paired bars so they read as two marks
VALUE_LABEL_PAD_PT = 3
Y_MAX = 1.0
Y_HEADROOM = 0.08  # keeps the tallest value label clear of the top edge
CAPTION = (
    "Bucket = ground-truth box side length measured at the 640 scale for both models, "
    "so each pair of bars compares the same objects."
)


@dataclass(frozen=True)
class SizeSensitivity:
    """One model's detection rate per size bucket, in CSV row order."""

    label: str
    buckets: tuple[str, ...]
    gt_counts: tuple[int, ...]
    rates: tuple[float, ...]
    overall_rate: float


def read_size_csv(path: Path, label: str) -> SizeSensitivity:
    buckets: list[str] = []
    gt_counts: list[int] = []
    rates: list[float] = []
    overall_rate: float | None = None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["bucket"] == AGGREGATE_BUCKET:
                overall_rate = float(row["rate"])
                continue
            buckets.append(row["bucket"])
            gt_counts.append(int(row["gt_count"]))
            rates.append(float(row["rate"]))
    if overall_rate is None:
        raise ValueError(f"{path.name}: missing '{AGGREGATE_BUCKET}' row")
    return SizeSensitivity(label, tuple(buckets), tuple(gt_counts), tuple(rates), overall_rate)


def load_series() -> list[SizeSensitivity]:
    series = [read_size_csv(ANALYSIS_DIR / filename, label) for filename, label in SERIES]
    # Bars are grouped by index, so a bucket mismatch would silently pair wrong rows
    reference = series[0]
    for other in series[1:]:
        if other.buckets != reference.buckets or other.gt_counts != reference.gt_counts:
            raise ValueError(f"bucket definitions differ between {reference.label} and {other.label}")
    return series


def style_axes(ax: plt.Axes) -> None:
    """Recessive chrome: hairline horizontal grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, Y_MAX + Y_HEADROOM)


def plot(series: list[SizeSensitivity]) -> plt.Figure:
    reference = series[0]
    n_buckets = len(reference.buckets)
    n_series = len(series)
    fig, ax = plt.subplots(figsize=FIG_SIZE_IN, dpi=DPI, facecolor=SURFACE)
    style_axes(ax)

    group_centers = list(range(n_buckets))
    # Center the group of bars on each tick: offsets are symmetric around zero
    offsets = [(i - (n_series - 1) / 2) * (BAR_WIDTH + BAR_GAP) for i in range(n_series)]
    for s, offset, color in zip(series, offsets, SERIES_COLORS):
        x = [c + offset for c in group_centers]
        bars = ax.bar(x, s.rates, width=BAR_WIDTH, color=color, label=s.label, linewidth=0)
        # Value labels use text ink, not the series color: identity comes from the bar
        ax.bar_label(
            bars,
            labels=[f"{r:.1%}" for r in s.rates],
            padding=VALUE_LABEL_PAD_PT,
            fontsize=8.5,
            color=INK_PRIMARY,
        )

    # GT count under each bucket so the reader sees the >64px bucket is tiny (n=260)
    tick_labels = [f"{b}\n(n={n:,})" for b, n in zip(reference.buckets, reference.gt_counts)]
    ax.set_xticks(group_centers, tick_labels)
    ax.set_xlabel("Ground-truth box size", color=INK_SECONDARY)
    ax.set_ylabel("Detection rate (GT boxes matched)", color=INK_SECONDARY)

    overall = " vs ".join(f"{s.overall_rate:.1%}" for s in series)
    ax.set_title(
        "Detection rate by object size: YOLO11n on VisDrone (2k subset, val)",
        color=INK_PRIMARY,
        fontsize=12,
        loc="left",
        pad=18,
    )
    ax.text(
        0, 1.02, f"Overall detection rate: {overall}",
        transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5, va="bottom",
    )
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_SECONDARY)

    fig.text(0.01, 0.01, CAPTION, color=INK_MUTED, fontsize=8, ha="left", va="bottom")
    # Leave room at the bottom for the caption so tight_layout doesn't overlap it
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main() -> None:
    series = load_series()
    fig = plot(series)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor=SURFACE)
    print(f"Saved: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
