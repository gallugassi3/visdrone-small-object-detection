"""Quantify detection rate as a function of object size.

Loads a trained model, runs prediction on all val images, matches
predictions to ground-truth boxes per image (class-agnostic greedy IoU
matching), and buckets GT boxes by their pixel size at the model's input
scale. Small objects shrink below the network's stride after resizing,
so detection rate is expected to collapse in the smallest buckets.

Size is defined as sqrt(box area) after scaling to MODEL_INPUT_SIZE,
which keeps buckets comparable across models evaluated at different
inference resolutions.

Usage:
    python analysis/size_sensitivity.py [weights_path]
"""
import csv
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/detect/baseline_640_smoke/weights/best.pt"
VAL_IMAGES_DIR = Path("datasets/VisDrone-2k/images/val")
VAL_LABELS_DIR = Path("datasets/VisDrone-2k/labels/val")
ANALYSIS_DIR = Path("analysis")
# Output name derives from the run being evaluated so results never overwrite
OUTPUT_CSV_TEMPLATE = "size_sensitivity_{run}.csv"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
MODEL_INPUT_SIZE = 640  # bucket sizes are always computed at the 640 scale

# Bucket edges in pixels (at MODEL_INPUT_SIZE scale). Object "size" is
# sqrt(area), so a 8x8 object has size 8.
BUCKET_EDGES = [0, 8, 16, 32, 64, float("inf")]
BUCKET_LABELS = ["<8px", "8-16px", "16-32px", "32-64px", ">64px"]


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Read YOLO-format labels into an (N, 4) array of xyxy pixel boxes."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            if not line.strip():
                continue
            _, xc, yc, w, h = line.split()
            xc, w = float(xc) * img_w, float(w) * img_w
            yc, h = float(yc) * img_h, float(h) * img_h
            boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return np.array(boxes).reshape(-1, 4)


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


def gt_sizes_at_input_scale(gt_boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Object size (sqrt of area) after letterbox scaling to MODEL_INPUT_SIZE.

    Boxes are clipped to the image first: VisDrone's converter does not clip,
    and out-of-frame area would inflate sizes.
    """
    x1 = np.clip(gt_boxes[:, 0], 0, img_w)
    y1 = np.clip(gt_boxes[:, 1], 0, img_h)
    x2 = np.clip(gt_boxes[:, 2], 0, img_w)
    y2 = np.clip(gt_boxes[:, 3], 0, img_h)
    scale = MODEL_INPUT_SIZE / max(img_w, img_h)
    widths = (x2 - x1) * scale
    heights = (y2 - y1) * scale
    return np.sqrt(np.clip(widths * heights, 0, None))


def bucket_index(size: float) -> int:
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= size < BUCKET_EDGES[i + 1]:
            return i
    return len(BUCKET_LABELS) - 1


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    run_name = Path(weights).parent.parent.name  # runs/detect/<run>/weights/best.pt
    output_csv = ANALYSIS_DIR / OUTPUT_CSV_TEMPLATE.format(run=run_name)

    model = YOLO(weights)

    gt_total = np.zeros(len(BUCKET_LABELS), dtype=int)
    gt_detected = np.zeros(len(BUCKET_LABELS), dtype=int)

    images = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    for i, image_path in enumerate(images, 1):
        result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]
        img_h, img_w = result.orig_shape
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        conf_order = result.boxes.conf.cpu().numpy().argsort()[::-1]
        pred_boxes = pred_boxes[conf_order]

        label_path = VAL_LABELS_DIR / f"{image_path.stem}.txt"
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)
        if len(gt_boxes) == 0:
            continue

        sizes = gt_sizes_at_input_scale(gt_boxes, img_w, img_h)
        buckets = np.array([bucket_index(s) for s in sizes])

        ious = iou_matrix(pred_boxes, gt_boxes)
        gt_taken = np.zeros(len(gt_boxes), dtype=bool)
        for p in range(len(pred_boxes)):
            candidates = np.where(~gt_taken & (ious[p] >= IOU_THRESHOLD))[0]
            if len(candidates):
                g = candidates[ious[p, candidates].argmax()]
                gt_taken[g] = True

        for b in range(len(BUCKET_LABELS)):
            mask = buckets == b
            gt_total[b] += int(mask.sum())
            gt_detected[b] += int((mask & gt_taken).sum())

        if i % 100 == 0:
            print(f"...{i}/{len(images)} images")

    print(f"\n=== Detection rate by object size @ {MODEL_INPUT_SIZE}px scale "
          f"(model: {run_name}, conf={CONF_THRESHOLD}, IoU>={IOU_THRESHOLD}) ===")
    print(f"{'bucket':>10} {'gt_count':>9} {'detected':>9} {'rate':>7}")
    rows = []
    for b, name in enumerate(BUCKET_LABELS):
        rate = gt_detected[b] / gt_total[b] if gt_total[b] else 0.0
        print(f"{name:>10} {gt_total[b]:>9} {gt_detected[b]:>9} {rate:>6.1%}")
        rows.append({"bucket": name, "gt_count": gt_total[b],
                     "detected": gt_detected[b], "rate": f"{rate:.4f}"})

    total = gt_total.sum()
    detected = gt_detected.sum()
    print(f"{'all':>10} {total:>9} {detected:>9} {detected / total:>6.1%}")
    rows.append({"bucket": "all", "gt_count": int(total),
                 "detected": int(detected), "rate": f"{detected / total:.4f}"})

    ANALYSIS_DIR.mkdir(exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "gt_count", "detected", "rate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()