"""Side-by-side comparison: ground truth vs. model predictions on val images.

Produces one panel image per input: [GT | model A] or [GT | model A | model B]
when a second weights path is given. Panels feed the README and the
failure-analysis writeup.

Usage:
    python predict_compare.py runs/detect/baseline_640_2k/weights/best.pt
    python predict_compare.py <weights_640> <weights_1024>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from draw_boxes import draw_annotations, label_path_for

# Hand-picked val images spanning the difficulty spectrum:
# dense urban, medium street, sparse park, small-object-heavy
DEMO_IMAGES = [
    "0000001_02999_d_0000005.jpg",
    "0000001_05499_d_0000010.jpg",
    "0000021_00500_d_0000002.jpg",
    "0000022_01251_d_0000007.jpg",
]

VAL_DIR = Path("datasets/VisDrone-2k/images/val")
OUT_DIR = Path("comparisons")
CONF_THRESHOLD = 0.25   # YOLO default; the threshold a deployed system would use
HEADER_H = 28


def annotate_gt(image_path: Path) -> np.ndarray:
    """Return the image with ground-truth boxes drawn."""
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"Could not read image: {image_path}")
    label_path = label_path_for(image_path)
    lines = label_path.read_text().strip().splitlines() if label_path.exists() else []
    draw_annotations(img, lines, label_path)
    return img


def annotate_pred(model: YOLO, image_path: Path) -> np.ndarray:
    """Return the image with model predictions drawn (Ultralytics renderer)."""
    result = model.predict(str(image_path), conf=CONF_THRESHOLD, verbose=False)[0]
    return result.plot(line_width=1, font_size=0.35)


def add_header(img: np.ndarray, text: str) -> np.ndarray:
    """Add a title strip above the image so panels are self-explanatory."""
    header = np.full((HEADER_H, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    return np.vstack([header, img])


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict_compare.py <weights.pt> [weights_b.pt]")

    models = [(Path(w).parent.parent.name, YOLO(w)) for w in sys.argv[1:]]
    OUT_DIR.mkdir(exist_ok=True)

    for name in DEMO_IMAGES:
        image_path = VAL_DIR / name
        panels = [add_header(annotate_gt(image_path), "ground truth")]
        panels += [add_header(annotate_pred(m, image_path), run_name)
                   for run_name, m in models]

        combined = np.hstack(panels)
        out_path = OUT_DIR / f"{image_path.stem}_compare.png"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()