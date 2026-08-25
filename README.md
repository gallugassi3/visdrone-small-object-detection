# VisDrone Small-Object Detection

Training YOLO11n to detect pedestrians and vehicles in drone imagery, focusing
on the core challenge of aerial detection: **most objects are tiny**. In this
dataset, 70% of validation objects are under 16 pixels at the model's 640px
input scale — and the baseline detects fewer than 8% of objects below 8px.

> **Status:** baseline trained; resolution experiment (640 vs 1024) in progress.
> Comparison results, visual demos, and failure analysis landing next.

## The problem

Object detectors downsample their input aggressively (YOLO11's finest detection
head operates at stride 8). An object smaller than the stride may contain no
anchor point at all — it becomes invisible to the training loss, not merely
hard to classify. Drone imagery is the worst case for this: a pedestrian seen
from altitude occupies 7-15 pixels.

Measured on our 5-epoch smoke model (VisDrone val, 548 images, 38,759 objects):

| Object size @640 input | Share of dataset | Detection rate |
|---|---|---|
| < 8 px | 31% | **7.8%** |
| 8-16 px | 39% | 35.0% |
| 16-32 px | 22% | 65.3% |
| 32-64 px | 7% | 86.9% |
| > 64 px | 1% | 89.2% |

The failure is a step function around the stride, not a gradual "small is
hard" slope. Error categorization confirms it: **62.6% of ground-truth objects
are missed entirely, versus only 6.3% misclassified** — a 10:1 ratio. The
model's problem is seeing, not understanding.

## The experiment

If sub-stride invisibility is the dominant failure, raising input resolution
(imgsz 640 → 1024) should lift the recall ceiling for small classes — each
object gets 1.6x more pixels, shifting a large share of the <8px bucket above
the stride. We train two otherwise-identical YOLO11n models and compare.

Controls: same 2,000-image train subset (random, seed=42), full val set, SGD
lr0=0.01 pinned (Ultralytics' `optimizer=auto` silently switches optimizers by
run length), identical augmentation, conf=0.25 at the comparison operating
point.

**Results: _in progress — landing here._**

## Dataset

[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset): 8,629 drone
images, 10 classes, ~471K boxes. Notable properties we measured and work with:

- **Extreme density:** median 42 objects/image, max 902.
- **Extreme imbalance:** car is 42% of boxes; awning-tricycle under 1% (45:1).
  Since mAP macro-averages classes, the four rarest classes are 7% of the data
  but 40% of the metric.
- **Confusable class pairs:** van→car is the largest confusion (822 cases vs
  70 reverse — the frequent class absorbs the rare one's ambiguous cases);
  riders "steal" detections from their vehicles (motor→pedestrian: 254).
- **Annotation quirks:** 2,597 train boxes extend past image borders; a few
  duplicate labels. Surfaced by our tooling rather than silently clipped.

Training uses a reproducible 2,000-image subset (seed=42) for iteration speed;
val/test remain full so metrics stay comparable across experiments.

## Repository guide

| Path | What it is |
|---|---|
| `train.py`, `train_1024.py` | The two experiment configurations |
| `analyze_failures.py` | Categorizes errors: miss / confusion / false positive |
| `analysis/size_sensitivity.py` | Detection rate as a function of object size |
| `predict_compare.py` | Side-by-side GT vs prediction panels |
| `draw_boxes.py` | Ground-truth visualization (pure OpenCV, no framework) |
| `EXPERIMENTS.md` | Run log: every experiment, its numbers, its conclusion |
| `analysis/` | Saved measurements and methodology notes |

## What I'd do next

_Sections below fill in after the resolution experiment:_ failure cases the
1024 model still misses, tiling/SAHI as the production-scale alternative to
raw resolution, P2 head trade-offs, and per-class strategies the resolution
lever cannot fix (rare-class confusion needs data, not pixels).