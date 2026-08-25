"""Watch a metric being born: full matching walkthrough on ONE real image.

Runs the trained model on a single val image and prints every step the
evaluator performs: predictions sorted by confidence, greedy IoU matching,
the TP/FP/FN verdict for each box, and the resulting precision/recall.
Saves walkthrough.png with color-coded boxes:
  green = TP, red = FP, orange dashed = FN (missed ground truth).

Usage:
    python walkthrough.py [image_path]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from draw_boxes import CLASS_NAMES, label_path_for
from analyze_failures import yolo_labels_to_xyxy, iou_matrix

WEIGHTS = "runs/detect/baseline_640_2k/weights/best.pt"
DEFAULT_IMAGE = Path("datasets/VisDrone-2k/images/val/0000021_00500_d_0000002.jpg")
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

GREEN, RED, ORANGE = (80, 200, 80), (60, 60, 230), (0, 165, 255)


def dashed_rect(img, p1, p2, color, dash=8):
    """Dashed rectangle for missed GT boxes (FN)."""
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, 2)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, 2)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, 2)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, 2)


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    model = YOLO(WEIGHTS)

    result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]
    img = result.orig_img.copy()
    img_h, img_w = result.orig_shape

    pred_boxes = result.boxes.xyxy.cpu().numpy()
    pred_conf = result.boxes.conf.cpu().numpy()
    pred_cls = result.boxes.cls.cpu().numpy().astype(int)
    order = pred_conf.argsort()[::-1]
    pred_boxes, pred_conf, pred_cls = pred_boxes[order], pred_conf[order], pred_cls[order]

    gt_boxes, gt_cls = yolo_labels_to_xyxy(label_path_for(image_path), img_w, img_h)
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)

    print(f"\nImage: {image_path.name} | GT objects: {len(gt_boxes)} | "
          f"predictions at conf>={CONF_THRESHOLD}: {len(pred_boxes)}\n")
    print(f"{'#':>3} {'conf':>5} {'class':>12} {'best IoU':>8}  verdict")
    print("-" * 55)

    tp = fp = 0
    for p in range(len(pred_boxes)):
        best_iou = ious[p].max() if len(gt_boxes) else 0.0
        candidates = np.where(~gt_taken & (ious[p] >= IOU_THRESHOLD))[0]
        x1, y1, x2, y2 = pred_boxes[p].astype(int)
        name = CLASS_NAMES[pred_cls[p]]

        if len(candidates) == 0:
            fp += 1
            reason = "duplicate (GT already taken)" if best_iou >= IOU_THRESHOLD else "miss (no GT overlap)"
            print(f"{p+1:>3} {pred_conf[p]:>5.2f} {name:>12} {best_iou:>8.2f}  FP - {reason}")
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)
        else:
            g = candidates[ious[p, candidates].argmax()]
            gt_taken[g] = True
            tp += 1
            print(f"{p+1:>3} {pred_conf[p]:>5.2f} {name:>12} {best_iou:>8.2f}  TP - matched GT#{g} ({CLASS_NAMES[gt_cls[g]]})")
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)

    fn = int((~gt_taken).sum())
    for g in np.where(~gt_taken)[0]:
        x1, y1, x2, y2 = gt_boxes[g].astype(int)
        dashed_rect(img, (max(x1, 0), max(y1, 0)), (min(x2, img_w - 1), min(y2, img_h - 1)), ORANGE)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    print("-" * 55)
    print(f"\nTP={tp}  FP={fp}  FN={fn}")
    print(f"Precision = {tp}/({tp}+{fp}) = {precision:.2f}")
    print(f"Recall    = {tp}/({tp}+{fn}) = {recall:.2f}")
    print("\nLegend in walkthrough.png: green=TP, red=FP, orange dashed=FN (missed GT)")

    cv2.imwrite("walkthrough.png", img)
    print("Saved: walkthrough.png")


if __name__ == "__main__":
    main()