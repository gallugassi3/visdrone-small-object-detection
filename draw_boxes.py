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
OUTPUT_PATH = Path("output.png")  # PNG: JPEG compression smears 1-px box edges


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
    if not 0 <= cls_id < len(CLASS_NAMES):
        raise ValueError(
            f"class id {cls_id} outside valid range 0-{len(CLASS_NAMES) - 1}"
        )
    xc, w = float(xc_str) * img_w, float(w_str) * img_w
    yc, h = float(yc_str) * img_h, float(h_str) * img_h
    x1, y1 = round(xc - w / 2), round(yc - h / 2)
    x2, y2 = round(xc + w / 2), round(yc + h / 2)
    return cls_id, x1, y1, x2, y2


def draw_annotations(img, lines: list[str], label_path: Path) -> None:
    """Draw one labeled box per YOLO label line onto the image, in place."""
    img_h, img_w = img.shape[:2]  # OpenCV shape is (H, W, C)
    clamped = 0

    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            cls_id, x1, y1, x2, y2 = yolo_line_to_corners(line, img_w, img_h)
        except (ValueError, IndexError) as exc:
            sys.exit(f"{label_path}:{lineno}: bad label line {line!r} ({exc})")

        # VisDrone's converter does not clip boxes at image borders;
        # surface the fact instead of hiding it
        cx1, cy1 = max(x1, 0), max(y1, 0)
        cx2, cy2 = min(x2, img_w - 1), min(y2, img_h - 1)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            clamped += 1

        color = CLASS_COLORS[cls_id]
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), color, BOX_THICKNESS)
        cv2.putText(
            img,
            CLASS_NAMES[cls_id],
            (cx1, max(cy1 - 3, 10)),  # keep text inside the frame near the top edge
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            BOX_THICKNESS,
        )

    if clamped:
        print(f"Note: {clamped} boxes extended past image borders (clamped for display)")


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

    draw_annotations(img, lines, label_path)

    cv2.imwrite(str(OUTPUT_PATH), img)
    print(f"Saved: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()