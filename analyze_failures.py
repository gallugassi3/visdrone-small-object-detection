"""Categorize model errors on val images: misses, false alarms, misclassifications.

For each image, predictions are matched to ground-truth boxes (IoU >= 0.5,
greedy by confidence). Every GT box and prediction then falls into exactly
one outcome:
  - detected:        matched, correct class
  - misclassified:   matched box, wrong class (counted per GT)
  - missed:          GT box with no matching prediction
  - false_positive:  prediction with no matching GT box

Usage:
    python analyze_failures.py runs/detect/baseline_640_2k/weights/best.pt
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from draw_boxes import CLASS_NAMES, label_path_for

VAL_DIR = Path("datasets/VisDrone-2k/images/val")
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


def yolo_labels_to_xyxy(label_path: Path, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO label file into (boxes[N,4] xyxy pixels, classes[N])."""
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls_id, xc, yc, w, h = line.split()
            xc, w = float(xc) * img_w, float(w) * img_w
            yc, h = float(yc) * img_h, float(h) * img_h
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
            classes.append(int(cls_id))
    return np.array(boxes).reshape(-1, 4), np.array(classes, dtype=int)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two box sets, shapes (N,4) and (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def analyze_image(model: YOLO, image_path: Path) -> Counter:
    """Return outcome counts for one image (keys include per-class misses)."""
    result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]
    pred_boxes = result.boxes.xyxy.cpu().numpy()
    pred_classes = result.boxes.cls.cpu().numpy().astype(int)
    order = result.boxes.conf.cpu().numpy().argsort()[::-1]  # match greedily by confidence
    pred_boxes, pred_classes = pred_boxes[order], pred_classes[order]

    img_h, img_w = result.orig_shape
    gt_boxes, gt_classes = yolo_labels_to_xyxy(label_path_for(image_path), img_w, img_h)

    counts: Counter = Counter()
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)

    for p in range(len(pred_boxes)):
        candidates = np.where(~gt_taken & (ious[p] >= IOU_THRESHOLD))[0]
        if len(candidates) == 0:
            counts["false_positive"] += 1
            continue
        g = candidates[ious[p, candidates].argmax()]
        gt_taken[g] = True
        if pred_classes[p] == gt_classes[g]:
            counts["detected"] += 1
        else:
            counts["misclassified"] += 1
            counts[f"confused:{CLASS_NAMES[gt_classes[g]]}->{CLASS_NAMES[pred_classes[p]]}"] += 1

    for g in np.where(~gt_taken)[0]:
        counts["missed"] += 1
        counts[f"missed:{CLASS_NAMES[gt_classes[g]]}"] += 1
    return counts


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python analyze_failures.py <weights.pt>")
    model = YOLO(sys.argv[1])

    totals: Counter = Counter()
    images = sorted(VAL_DIR.glob("*.jpg"))
    for i, image_path in enumerate(images, 1):
        totals += analyze_image(model, image_path)
        if i % 100 == 0:
            print(f"...{i}/{len(images)} images")

    total_gt = totals["detected"] + totals["misclassified"] + totals["missed"]
    print(f"\n=== Outcome summary ({len(images)} val images, {total_gt} GT boxes) ===")
    for key in ("detected", "misclassified", "missed", "false_positive"):
        share = f" ({totals[key] / total_gt:.1%} of GT)" if key != "false_positive" else ""
        print(f"{key:>15}: {totals[key]}{share}")

    print("\n=== Top confusions (GT -> predicted) ===")
    for key, n in totals.most_common():
        if key.startswith("confused:"):
            print(f"{key.removeprefix('confused:'):>30}: {n}")

    print("\n=== Misses by class ===")
    for name in CLASS_NAMES:
        print(f"{name:>16}: {totals[f'missed:{name}']}")


if __name__ == "__main__":
    main()