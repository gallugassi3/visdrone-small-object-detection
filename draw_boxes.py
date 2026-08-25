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

# Fixed BGR color per class for consistent identification across images
CLASS_COLORS = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0),
    (255, 0, 0), (255, 0, 255), (128, 0, 255), (0, 0, 128), (255, 128, 0),
]

DEFAULT_IMAGE = Path("datasets/VisDrone/images/train/0000003_00231_d_0000016.jpg")

BOX_THICKNESS = 1       # thin lines: VisDrone scenes often contain 100+ objects
FONT_SCALE = 0.35
OUTPUT_PATH = Path("output.jpg")


def label_path_for(image_path: Path) -> Path:
    """Map an image path to its label path (images/x.jpg -> labels/x.txt)."""
    parts = [p if p != "images" else "labels" for p in image_path.parts]
    return Path(*parts).with_suffix(".txt")


def yolo_line_to_corners(
    line: str, img_w: int, img_h: int
) -> tuple[int, int, int, int, int]:
    """Parse one YOLO label line into (class_id, x1, y1, x2, y2) in pixels.

    YOLO format: "class x_center y_center width height", normalized to [0, 1].
    X values are denormalized by image width, Y values by image height.
    """
    cls_str, xc_str, yc_str, w_str, h_str = line.split()
    cls_id = int(cls_str)
    xc, w = float(xc_str) * img_w, float(w_str) * img_w
    yc, h = float(yc_str) * img_h, float(h_str) * img_h
    x1, y1 = round(xc - w / 2), round(yc - h / 2)
    x2, y2 = round(xc + w / 2), round(yc + h / 2)
    return cls_id, x1, y1, x2, y2


def draw_annotations(img, lines: list[str]) -> None:
    """Draw one labeled box per YOLO label line onto the image, in place."""
    img_h, img_w = img.shape[:2]  # OpenCV shape is (H, W, C)
    for line in lines:
        cls_id, x1, y1, x2, y2 = yolo_line_to_corners(line, img_w, img_h)
        color = CLASS_COLORS[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(
            img,
            CLASS_NAMES[cls_id],
            (x1, max(y1 - 3, 10)),  # keep text inside the frame near the top edge
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            BOX_THICKNESS,
        )


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE

    img = cv2.imread(str(image_path))
    if img is None:  # cv2.imread returns None instead of raising on failure
        sys.exit(f"Could not read image: {image_path}")

    label_path = label_path_for(image_path)
    if not label_path.exists():
        sys.exit(f"No label file found: {label_path}")

    lines = label_path.read_text().strip().splitlines()
    img_h, img_w = img.shape[:2]
    print(f"Image: {image_path.name} ({img_w}x{img_h}), objects: {len(lines)}")

    draw_annotations(img, lines)

    cv2.imwrite(str(OUTPUT_PATH), img)
    print(f"Saved: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()