"""Compare two training runs by validating each at its own training resolution.

Each run's best.pt is evaluated with model.val() on the same data split, with
imgsz read from that run's args.yaml. Both imgsz and data are passed explicitly
even though Ultralytics would inherit imgsz from the checkpoint: the smoke
checkpoint proved that the inherited `data` can silently point at a different
dataset, and an explicit imgsz makes the protocol visible in the code.
See analysis/imgsz_eval_notes.md for the underlying investigation.

Usage (from anywhere; the script chdirs to the project root):
    python scripts/compare_runs.py baseline_640_2k highres_1024_2k
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
VAL_OUTPUT_DIR = PROJECT_ROOT / "runs" / "val"  # keeps val artifacts out of runs/detect
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
WEIGHTS_RELPATH = Path("weights") / "best.pt"
ARGS_FILENAME = "args.yaml"

# Relative on purpose: train.py passes the same string, so resolving it from the
# project root guarantees val sees exactly the dataset training used.
DATA_YAML = "visdrone2k.yaml"
VAL_SPLIT = "val"

OVERALL_METRICS: tuple[tuple[str, str], ...] = (
    # (row label, RunMetrics attribute)
    ("precision", "precision"),
    ("recall", "recall"),
    ("mAP50", "map50"),
    ("mAP50-95", "map50_95"),
)
VALUE_FMT = "{:.3f}"
DELTA_FMT = "{:+.3f}"
MISSING = "-"
LABEL_COL_WIDTH = 18
MIN_VALUE_COL_WIDTH = 16
COL_PADDING = 2  # blank columns between adjacent run names


@dataclass(frozen=True)
class RunMetrics:
    """The subset of DetMetrics needed for the comparison table."""

    name: str
    imgsz: int
    precision: float
    recall: float
    map50: float
    map50_95: float
    per_class_map50: dict[str, float]  # class name -> AP50; absent if the class had no val GT


def read_train_imgsz(run_dir: Path) -> int:
    """Training resolution recorded by Ultralytics in the run's args.yaml."""
    args = yaml.safe_load((run_dir / ARGS_FILENAME).read_text())
    imgsz = args["imgsz"]
    # Ultralytics accepts [h, w] too; the validator uses a single int (max_dim=1)
    if isinstance(imgsz, (list, tuple)):
        imgsz = max(imgsz)
    return int(imgsz)


def extract_metrics(name: str, imgsz: int, metrics: object) -> RunMetrics:
    """Pull scalar and per-class results out of a DetMetrics object.

    ap50 is only populated for classes that appear in the val GT, so it must be
    paired with ap_class_index rather than indexed by class id directly.
    """
    box = metrics.box
    per_class = {
        metrics.names[int(cls_idx)]: float(ap50)
        for cls_idx, ap50 in zip(box.ap_class_index, box.ap50)
    }
    return RunMetrics(
        name=name,
        imgsz=imgsz,
        precision=float(box.mp),
        recall=float(box.mr),
        map50=float(box.map50),
        map50_95=float(box.map),
        per_class_map50=per_class,
    )


def evaluate_run(name: str) -> RunMetrics:
    """Validate a run's best.pt at its own training imgsz on the shared val split."""
    run_dir = RUNS_DIR / name
    imgsz = read_train_imgsz(run_dir)
    model = YOLO(str(run_dir / WEIGHTS_RELPATH))
    metrics = model.val(
        data=DATA_YAML,
        imgsz=imgsz,
        split=VAL_SPLIT,
        project=str(VAL_OUTPUT_DIR),
        name=name,
        exist_ok=True,
    )
    return extract_metrics(name, imgsz, metrics)


def _format_row(label: str, a: float | None, b: float | None, width: int) -> str:
    a_str = VALUE_FMT.format(a) if a is not None else MISSING
    b_str = VALUE_FMT.format(b) if b is not None else MISSING
    delta = DELTA_FMT.format(b - a) if a is not None and b is not None else MISSING
    return (
        f"{label:<{LABEL_COL_WIDTH}}"
        f"{a_str:>{width}}{b_str:>{width}}{delta:>{width}}"
    )


def format_table(a: RunMetrics, b: RunMetrics) -> str:
    """Side-by-side table; delta is B minus A so a positive delta favors run B."""
    # Widen the value columns so long run names never touch each other
    width = max(MIN_VALUE_COL_WIDTH, len(a.name) + COL_PADDING, len(b.name) + COL_PADDING)
    header = [
        f"Validation: data={DATA_YAML} split={VAL_SPLIT}, each model at its own training imgsz",
        "",
        f"{'':<{LABEL_COL_WIDTH}}{a.name:>{width}}{b.name:>{width}}{'delta (B-A)':>{width}}",
        f"{'':<{LABEL_COL_WIDTH}}{f'(imgsz={a.imgsz})':>{width}}{f'(imgsz={b.imgsz})':>{width}}",
        "-" * (LABEL_COL_WIDTH + 3 * width),
    ]
    overall = [
        _format_row(label, getattr(a, attr), getattr(b, attr), width)
        for label, attr in OVERALL_METRICS
    ]
    # Union of class names, ordered by first appearance, so a class missing
    # from one run still gets a row rather than silently disappearing
    class_names = list(dict.fromkeys([*a.per_class_map50, *b.per_class_map50]))
    per_class = [
        _format_row(cls, a.per_class_map50.get(cls), b.per_class_map50.get(cls), width)
        for cls in class_names
    ]
    return "\n".join([*header, *overall, "", "per-class mAP50", *per_class])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_a", help="run name under runs/detect (baseline)")
    parser.add_argument("run_b", help="run name under runs/detect (candidate)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Relative dataset paths (DATA_YAML and the `path:` inside it) resolve
    # against the cwd; pin it so the script behaves the same from any shell
    os.chdir(PROJECT_ROOT)

    # Fail before spending minutes validating run A if run B is incomplete
    for name in (args.run_a, args.run_b):
        for required in (RUNS_DIR / name / ARGS_FILENAME, RUNS_DIR / name / WEIGHTS_RELPATH):
            if not required.exists():
                sys.exit(f"Missing: {required}")

    metrics_a = evaluate_run(args.run_a)
    metrics_b = evaluate_run(args.run_b)
    table = format_table(metrics_a, metrics_b)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ANALYSIS_DIR / f"compare_{args.run_a}_vs_{args.run_b}.txt"
    output_path.write_text(table + "\n")

    print(f"\n{table}\n")
    print(f"Saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
