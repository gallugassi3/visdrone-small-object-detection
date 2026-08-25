"""Quantify detection rate as a function of object size.

Runs a trained model over the validation split, matches predictions to
ground truth class-agnostically (greedy, IoU >= 0.5), and reports the
fraction of GT boxes that were detected in each size bucket.

Size is sqrt(area) measured at the model's input scale rather than at the
original resolution, because the input scale is what the network actually
sees: a 24 px pedestrian in a 1920 px frame is an 8 px object at imgsz=640.

Usage (from the project root):
    python analysis/size_sensitivity.py
"""
from __future__ import annotations

import bisect
import csv
import math
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "baseline_640_smoke" / "weights" / "best.pt"
DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone-2k"
VAL_IMAGES = DATASET_ROOT / "images" / "val"
VAL_LABELS = DATASET_ROOT / "labels" / "val"
OUTPUT_CSV = PROJECT_ROOT / "analysis" / "size_sensitivity.csv"

IMGSZ = 640                 # must match the training resolution being analyzed
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5         # standard "detected" criterion (mAP50 convention)

# Upper edges (exclusive) of the size buckets, in pixels at IMGSZ scale.
# The last bucket is open-ended. 8 px is the P3 stride of YOLO11: objects
# below it may contain no anchor point at all, which is the regime we want
# to isolate.
BUCKET_EDGES: tuple[int, ...] = (8, 16, 32, 64)

CSV_COLUMNS = ("size_bucket", "gt_count", "detected_count", "detection_rate")


def bucket_names(edges: tuple[int, ...]) -> list[str]:
    """Human-readable bucket labels: '<8px', '8-16px', ..., '>64px'."""
    names = [f"<{edges[0]}px"]
    names += [f"{lo}-{hi}px" for lo, hi in zip(edges, edges[1:])]
    names.append(f">{edges[-1]}px")
    return names


def bucket_index(size_px: float) -> int:
    """Index of the bucket containing size_px; edges are exclusive upper bounds."""
    return bisect.bisect_right(BUCKET_EDGES, size_px)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Read a YOLO label file into an (N, 4) xyxy array in original pixels.

    A missing file is a valid background image, not an error. Boxes are
    clipped to the image because the VisDrone converter does not clip them,
    and the predictor's output is always inside the frame; unclipped GT
    would otherwise under-count IoU at the borders.
    """
    if not label_path.exists() or not label_path.read_text().strip():
        return np.zeros((0, 4), dtype=np.float32)

    rows = np.loadtxt(label_path, dtype=np.float32, ndmin=2)

    xc, yc, w, h = (rows[:, i] for i in range(1, 5))
    boxes = np.stack(
        [(xc - w / 2) * img_w, (yc - h / 2) * img_h,
         (xc + w / 2) * img_w, (yc + h / 2) * img_h],
        axis=1,
    )
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_h)
    return boxes


def sizes_at_input_scale(boxes_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """sqrt(area) of each box after letterboxing the image so its long side is IMGSZ."""
    scale = IMGSZ / max(img_w, img_h)
    widths = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * scale
    heights = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) * scale
    return np.sqrt(widths * heights)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy arrays -> (N, M)."""
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (inter_x2 - inter_x1).clip(0) * (inter_y2 - inter_y1).clip(0)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    # A zero-area GT (degenerate label) would divide by zero; treat it as no overlap
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(
    pred_xyxy: np.ndarray, pred_conf: np.ndarray, gt_xyxy: np.ndarray
) -> np.ndarray:
    """Return a bool mask over GT boxes: True where a prediction claimed it.

    Standard detection-evaluation greedy matching: predictions are visited in
    descending confidence, each takes the highest-IoU unmatched GT box if that
    IoU clears the threshold. Class labels are ignored so that a correctly
    localized but misclassified object still counts as "seen" — this script
    measures whether the network can find objects of a given size, not
    whether it names them correctly.
    """
    matched = np.zeros(len(gt_xyxy), dtype=bool)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return matched

    ious = iou_matrix(pred_xyxy, gt_xyxy)
    for pred_idx in np.argsort(-pred_conf):
        candidates = np.where(matched, -1.0, ious[pred_idx])
        best_gt = int(np.argmax(candidates))
        if candidates[best_gt] >= IOU_THRESHOLD:
            matched[best_gt] = True
    return matched


def format_rate(detected: int, total: int) -> str:
    return f"{detected / total:.3f}" if total else ""


def main() -> None:
    for required in (MODEL_PATH, VAL_IMAGES, VAL_LABELS):
        if not required.exists():
            sys.exit(f"Missing: {required}")

    names = bucket_names(BUCKET_EDGES)
    gt_counts = np.zeros(len(names), dtype=np.int64)
    det_counts = np.zeros(len(names), dtype=np.int64)

    model = YOLO(str(MODEL_PATH))
    # stream=True yields one result at a time instead of holding 548 in memory
    results = model.predict(
        source=str(VAL_IMAGES), imgsz=IMGSZ, conf=CONF_THRESHOLD,
        stream=True, verbose=False,
    )

    n_images = 0
    for result in results:
        n_images += 1
        img_h, img_w = result.orig_shape
        gt_xyxy = load_gt_boxes(VAL_LABELS / f"{Path(result.path).stem}.txt", img_w, img_h)
        pred_xyxy = result.boxes.xyxy.cpu().numpy()
        pred_conf = result.boxes.conf.cpu().numpy()

        matched = greedy_match(pred_xyxy, pred_conf, gt_xyxy)
        for size_px, was_found in zip(sizes_at_input_scale(gt_xyxy, img_w, img_h), matched):
            idx = bucket_index(float(size_px))
            gt_counts[idx] += 1
            det_counts[idx] += int(was_found)

    total_gt, total_det = int(gt_counts.sum()), int(det_counts.sum())
    rows = [(name, int(g), int(d), format_rate(int(d), int(g)))
            for name, g, d in zip(names, gt_counts, det_counts)]
    rows.append(("all", total_gt, total_det, format_rate(total_det, total_gt)))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)

    print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Images: {n_images}  imgsz={IMGSZ}  conf>={CONF_THRESHOLD}  IoU>={IOU_THRESHOLD}\n")
    print(f"{'size bucket':<12}{'GT count':>10}{'detected':>10}{'det. rate':>11}")
    for name, g, d, rate in rows:
        print(f"{name:<12}{g:>10}{d:>10}{rate or '-':>11}")
    print(f"\nSaved: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
