"""Visualize YOLO-format ground-truth boxes on a VisDrone image.

Usage:
    python draw_boxes.py [image_path]
If no path is given, a default train image is used.
"""
import sys
from pathlib import Path

import cv2

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# צבע BGR קבוע לכל קלאס, כדי שהעין תלמד לזהות (רכב=ירוק, הולך רגל=אדום...)
CLASS_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 0, 128), (255, 128, 0),
]

DEFAULT_IMAGE = Path("datasets/VisDrone/images/train/0000003_00231_d_0000016.jpg")


def label_path_for(image_path: Path) -> Path:
    """images/train/x.jpg -> labels/train/x.txt"""
    parts = [p if p != "images" else "labels" for p in image_path.parts]
    return Path(*parts).with_suffix(".txt")


def yolo_line_to_corners(line: str, img_w: int, img_h: int):
    """Parse one YOLO label line into (class_id, x1, y1, x2, y2) in pixels."""
    cls_id, xc, yc, w, h = line.split()
    cls_id = int(cls_id)
    # denormalize: x-values scale by width, y-values by height
    xc, w = float(xc) * img_w, float(w) * img_w
    yc, h = float(yc) * img_h, float(h) * img_h
    x1, y1 = round(xc - w / 2), round(yc - h / 2)
    x2, y2 = round(xc + w / 2), round(yc + h / 2)
    return cls_id, x1, y1, x2, y2


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]          # shape is (H, W, C) — rows first!

    lbl_path = label_path_for(image_path)
    if not lbl_path.exists():
        sys.exit(f"No label file found: {lbl_path}")

    lines = lbl_path.read_text().strip().splitlines()
    print(f"Image: {image_path.name}  ({img_w}x{img_h}), objects: {len(lines)}")

    for line in lines:
        cls_id, x1, y1, x2, y2 = yolo_line_to_corners(line, img_w, img_h)
        color = CLASS_COLORS[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, CLASS_NAMES[cls_id], (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    out_path = Path("output.jpg")
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()