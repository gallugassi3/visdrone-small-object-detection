"""Pick and render val images where the 1024 model still fails visibly.

Candidate images are chosen from the labels alone (no inference): the ones
with the most <8px GT boxes (size at the 640 scale, as in
analysis/size_sensitivity.py) and the ones with the most bicycle /
awning-tricycle instances. Only that shortlist is run through the model, on
CPU, because the GPU is reserved for training. Each shortlisted image is
scored by its visible failures and the top one per category is rendered
with result.plot(); missed and misclassified GT boxes are then overlaid so
the failure is visible without cross-referencing the label file:
  orange dashed box            = GT box no prediction matched (missed)
  magenta box, "GT:<class>"    = matched, but the prediction's class is wrong

Usage (module form so the root-level helpers import):
    python -m scripts.failure_cases
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from analyze_failures import iou_matrix, yolo_labels_to_xyxy
from draw_boxes import CLASS_NAMES, label_path_for

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "highres_1024_2k" / "weights" / "best.pt"
VAL_IMAGES_DIR = PROJECT_ROOT / "datasets" / "VisDrone-2k" / "images" / "val"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_TEMPLATE = "failure_case_{index}.png"

DEVICE = "cpu"  # GPU is busy training; this is a handful of images
IMGSZ = 1024  # the run's training resolution (runs/detect/highres_1024_2k/args.yaml)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# Same size definition as analysis/size_sensitivity.py so "<8px" means the same thing
SIZE_REFERENCE_SCALE = 640
TINY_SIZE_PX = 8
BICYCLE_ID = CLASS_NAMES.index("bicycle")
AWNING_TRICYCLE_ID = CLASS_NAMES.index("awning-tricycle")
RARE_CLASS_IDS = (BICYCLE_ID, AWNING_TRICYCLE_ID)

SHORTLIST_PER_CATEGORY = 4  # how many label-ranked candidates per category get inference
N_CASES = 3

# Overlay styling (BGR). Thin lines: these scenes hold 100+ objects.
MISSED_COLOR = (0, 165, 255)
MISCLASSIFIED_COLOR = (255, 0, 255)
OVERLAY_THICKNESS = 1
DASH_LENGTH_PX = 4
TAG_FONT_SCALE = 0.4
PLOT_LINE_WIDTH = 1
PLOT_FONT_SIZE = 10


@dataclass(frozen=True)
class Candidate:
    """Label-only statistics used to shortlist an image before any inference."""

    image_path: Path
    n_gt: int
    n_tiny: int
    n_bicycle: int
    n_awning: int


@dataclass
class ImageOutcome:
    """Per-image matching result, with the indices needed to draw the overlay."""

    candidate: Candidate
    result: object  # ultralytics Results
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    missed_idx: np.ndarray
    misclassified: dict[int, int] = field(default_factory=dict)  # gt index -> predicted class
    n_false_positive: int = 0

    @property
    def n_missed(self) -> int:
        return len(self.missed_idx)

    def missed_tiny(self, tiny_mask: np.ndarray) -> int:
        return int(tiny_mask[self.missed_idx].sum())

    def rare_failures(self) -> int:
        rare = np.isin(self.gt_classes, RARE_CLASS_IDS)
        missed_rare = int(rare[self.missed_idx].sum())
        misclassified_rare = sum(1 for g in self.misclassified if rare[g])
        return missed_rare + misclassified_rare


def gt_sizes_at_reference_scale(gt_boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """sqrt(area) after letterboxing to SIZE_REFERENCE_SCALE, boxes clipped to the frame."""
    x1 = np.clip(gt_boxes[:, 0], 0, img_w)
    y1 = np.clip(gt_boxes[:, 1], 0, img_h)
    x2 = np.clip(gt_boxes[:, 2], 0, img_w)
    y2 = np.clip(gt_boxes[:, 3], 0, img_h)
    scale = SIZE_REFERENCE_SCALE / max(img_w, img_h)
    return np.sqrt(np.clip((x2 - x1) * (y2 - y1), 0, None)) * scale


def tiny_mask_for(gt_boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    return gt_sizes_at_reference_scale(gt_boxes, img_w, img_h) < TINY_SIZE_PX


def scan_candidates() -> list[Candidate]:
    """Rank every val image from its labels; PIL reads only the header for the size."""
    candidates = []
    for image_path in sorted(VAL_IMAGES_DIR.glob("*.jpg")):
        with Image.open(image_path) as im:
            img_w, img_h = im.size
        gt_boxes, gt_classes = yolo_labels_to_xyxy(label_path_for(image_path), img_w, img_h)
        if len(gt_boxes) == 0:
            continue
        candidates.append(
            Candidate(
                image_path=image_path,
                n_gt=len(gt_boxes),
                n_tiny=int(tiny_mask_for(gt_boxes, img_w, img_h).sum()),
                n_bicycle=int((gt_classes == BICYCLE_ID).sum()),
                n_awning=int((gt_classes == AWNING_TRICYCLE_ID).sum()),
            )
        )
    return candidates


def sequence_id(image_path: Path) -> str:
    # VisDrone names are <sequence>_<frame>_d_<id>; consecutive frames of one
    # sequence are near-duplicates, so a failure set should not repeat a sequence
    return image_path.stem.split("_")[0]


def top_per_sequence(items: list, key, limit: int) -> list:
    """Highest-`key` items, keeping at most one per video sequence."""
    chosen, seen = [], set()
    for item in sorted(items, key=key, reverse=True):
        path = item.image_path if isinstance(item, Candidate) else item.candidate.image_path
        seq = sequence_id(path)
        if seq in seen:
            continue
        seen.add(seq)
        chosen.append(item)
        if len(chosen) == limit:
            break
    return chosen


def shortlist(candidates: list[Candidate]) -> list[Candidate]:
    rankings = [
        top_per_sequence(candidates, lambda c: c.n_tiny, SHORTLIST_PER_CATEGORY),
        top_per_sequence(candidates, lambda c: c.n_bicycle, SHORTLIST_PER_CATEGORY),
        top_per_sequence(candidates, lambda c: c.n_awning, SHORTLIST_PER_CATEGORY),
    ]
    # dict preserves order and dedups an image that tops more than one list
    return list({c.image_path: c for ranking in rankings for c in ranking}.values())


def evaluate(model: YOLO, candidate: Candidate) -> ImageOutcome:
    """Greedy-by-confidence IoU matching, class checked after the match (as analyze_failures)."""
    result = model.predict(
        str(candidate.image_path), imgsz=IMGSZ, conf=CONF_THRESHOLD, device=DEVICE, verbose=False
    )[0]
    img_h, img_w = result.orig_shape
    pred_boxes = result.boxes.xyxy.cpu().numpy()
    pred_classes = result.boxes.cls.cpu().numpy().astype(int)
    order = result.boxes.conf.cpu().numpy().argsort()[::-1]
    pred_boxes, pred_classes = pred_boxes[order], pred_classes[order]

    gt_boxes, gt_classes = yolo_labels_to_xyxy(label_path_for(candidate.image_path), img_w, img_h)
    ious = iou_matrix(pred_boxes, gt_boxes)
    gt_taken = np.zeros(len(gt_boxes), dtype=bool)
    misclassified: dict[int, int] = {}
    n_false_positive = 0
    for p in range(len(pred_boxes)):
        options = np.where(~gt_taken & (ious[p] >= IOU_THRESHOLD))[0]
        if len(options) == 0:
            n_false_positive += 1
            continue
        g = int(options[ious[p, options].argmax()])
        gt_taken[g] = True
        if pred_classes[p] != gt_classes[g]:
            misclassified[g] = int(pred_classes[p])

    return ImageOutcome(
        candidate=candidate,
        result=result,
        gt_boxes=gt_boxes,
        gt_classes=gt_classes,
        missed_idx=np.where(~gt_taken)[0],
        misclassified=misclassified,
        n_false_positive=n_false_positive,
    )


def dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple) -> None:
    for x in range(x1, x2, DASH_LENGTH_PX * 2):
        cv2.line(img, (x, y1), (min(x + DASH_LENGTH_PX, x2), y1), color, OVERLAY_THICKNESS)
        cv2.line(img, (x, y2), (min(x + DASH_LENGTH_PX, x2), y2), color, OVERLAY_THICKNESS)
    for y in range(y1, y2, DASH_LENGTH_PX * 2):
        cv2.line(img, (x1, y), (x1, min(y + DASH_LENGTH_PX, y2)), color, OVERLAY_THICKNESS)
        cv2.line(img, (x2, y), (x2, min(y + DASH_LENGTH_PX, y2)), color, OVERLAY_THICKNESS)


def clipped_corners(box: np.ndarray, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    # VisDrone labels are not clipped at the frame edge; drawing needs in-frame ints
    x1, y1, x2, y2 = box.astype(int)
    return max(x1, 0), max(y1, 0), min(x2, img_w - 1), min(y2, img_h - 1)


def render(outcome: ImageOutcome, output_path: Path) -> None:
    img = outcome.result.plot(line_width=PLOT_LINE_WIDTH, font_size=PLOT_FONT_SIZE)
    img_h, img_w = img.shape[:2]
    for g in outcome.missed_idx:
        x1, y1, x2, y2 = clipped_corners(outcome.gt_boxes[g], img_w, img_h)
        dashed_rect(img, x1, y1, x2, y2, MISSED_COLOR)
    for g in outcome.misclassified:
        x1, y1, x2, y2 = clipped_corners(outcome.gt_boxes[g], img_w, img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), MISCLASSIFIED_COLOR, OVERLAY_THICKNESS)
        # Tag below the box so it does not fight the prediction label drawn above it
        cv2.putText(
            img, f"GT:{CLASS_NAMES[outcome.gt_classes[g]]}", (x1, min(y2 + 12, img_h - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, TAG_FONT_SCALE, MISCLASSIFIED_COLOR, 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), img)


def describe(outcome: ImageOutcome) -> str:
    c = outcome.candidate
    img_h, img_w = outcome.result.orig_shape
    tiny = tiny_mask_for(outcome.gt_boxes, img_w, img_h)
    n_pred = len(outcome.result.boxes)
    lines = [
        f"{c.image_path.name} ({img_w}x{img_h}): GT={c.n_gt} (tiny<{TINY_SIZE_PX}px={c.n_tiny}, "
        f"bicycle={c.n_bicycle}, awning-tricycle={c.n_awning}), predictions={n_pred}",
        f"  missed={outcome.n_missed} (tiny {outcome.missed_tiny(tiny)}), "
        f"misclassified={len(outcome.misclassified)}, false_positive={outcome.n_false_positive}",
    ]
    missed_by_class: dict[str, int] = {}
    for g in outcome.missed_idx:
        name = CLASS_NAMES[outcome.gt_classes[g]]
        missed_by_class[name] = missed_by_class.get(name, 0) + 1
    lines.append("  missed by class: " + ", ".join(f"{k}={v}" for k, v in sorted(missed_by_class.items(), key=lambda kv: -kv[1])))
    confusions: dict[str, int] = {}
    for g, pred_cls in outcome.misclassified.items():
        key = f"{CLASS_NAMES[outcome.gt_classes[g]]}->{CLASS_NAMES[pred_cls]}"
        confusions[key] = confusions.get(key, 0) + 1
    if confusions:
        lines.append("  confusions: " + ", ".join(f"{k}={v}" for k, v in sorted(confusions.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines)


def select_cases(outcomes: list[ImageOutcome]) -> list[ImageOutcome]:
    """One image per failure story, from distinct sequences: tiny misses, rare-class errors, wrong classes."""
    def tiny_score(o: ImageOutcome) -> int:
        img_h, img_w = o.result.orig_shape
        return o.missed_tiny(tiny_mask_for(o.gt_boxes, img_w, img_h))

    stories = (
        tiny_score,
        ImageOutcome.rare_failures,
        lambda o: len(o.misclassified),
    )
    chosen: list[ImageOutcome] = []
    for score in stories:
        used = {sequence_id(o.candidate.image_path) for o in chosen}
        remaining = [o for o in outcomes if sequence_id(o.candidate.image_path) not in used]
        chosen.extend(top_per_sequence(remaining, score, 1))
    return chosen[:N_CASES]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    candidates = scan_candidates()
    picked = shortlist(candidates)
    print(f"Scanned {len(candidates)} val images; running inference on {len(picked)} shortlisted "
          f"(device={DEVICE}, imgsz={IMGSZ}, conf={CONF_THRESHOLD}, IoU>={IOU_THRESHOLD})\n")

    model = YOLO(str(WEIGHTS))
    outcomes = [evaluate(model, c) for c in picked]
    for outcome in outcomes:
        print(describe(outcome), "\n")

    ASSETS_DIR.mkdir(exist_ok=True)
    print("=== Selected cases ===")
    for index, outcome in enumerate(select_cases(outcomes), 1):
        output_path = ASSETS_DIR / OUTPUT_TEMPLATE.format(index=index)
        render(outcome, output_path)
        print(f"[{index}] {outcome.candidate.image_path.name} -> {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
