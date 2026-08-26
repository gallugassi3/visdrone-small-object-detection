# Experiment Log

Every training run gets a row; details and conclusions live in the sections below the table.

| # | Run | Change vs previous | mAP50 | mAP50-95 | Epoch time |
|---|-----|--------------------|-------|----------|------------|
| 1 | baseline_640_smoke | — (pipeline check: full train, 5 ep, optimizer=auto) | 0.206 | 0.114 | ~8.5 min |
| 2 | baseline_640_2k | 2k subset, 80 ep, SGD pinned, batch=16 | 0.258 | 0.142 | ~1:25 |
| 3 | highres_1024_2k | imgsz 640→1024, batch=6 | **0.370** | **0.216** | ~3:20 |
| 4 | lowscale_1024_2k | scale aug 0.5→0.2 | TBD | TBD | ~3:20 |

Official metrics: `scripts/compare_runs.py` (unified val protocol on best.pt).

---

## Run 1 — baseline_640_smoke: the failure is non-detection

Pipeline verified; clean underfit (mAP rising linearly, no saturation).

- **Dominant failure is invisibility, not confusion:** 73–93% of small/thin classes predicted
  as background (bicycle 93%, tricycle 85%, people 84%, motor 82%, pedestrian 73%) vs car at 36%.
  Confirms the stride ceiling: objects under ~8px at 640 rarely get anchor assignment.
- Only major inter-class confusion: van→car (42%).
- **Process lesson:** optimizer=auto silently picked AdamW(lr=0.0007), ignoring lr0 —
  SGD lr0=0.01 pinned from here on for cross-run comparability.
- Hardware note: GPU power-throttles 120W→40W at ~90°C.

## Run 2 — baseline_640_2k: training time cannot fix sub-stride size

Clean convergence, no overfit, near-plateau by ep70 (+0.005 over the last 10 epochs).

- **The key asymmetry:** 80 epochs lifted mid/large classes 13–19pt (bus miss 63%→48%,
  tricycle 85%→66%) but pedestrian only 4pt (73%→69% missed). More training helps what the
  network can see; it cannot help what is below the stride. **Only resolution can.**
- bicycle finally registers (AP50 0.009→0.047) but stays near-worst.
- van→car confusion unchanged — structural, not a training artifact.
- close_mosaic step visible in train loss at ep70, with no val effect (val never had mosaic).

## Run 3 — highres_1024_2k: hypothesis confirmed, +43%

Ran the full 80 epochs; best ep79 (0.370), flat from ep70 — converged.

**Headline: mAP50 +43% (0.258→0.370) on identical data and training budget.** mAP50-95 rose
relatively more (+52%): boxes are both found more often and localized better.

Per-class ΔmAP50 (from the official comparison):

| motor | pedestrian | van | bus | people | truck | car | bicycle | tricycle | awning-tri. |
|---|---|---|---|---|---|---|---|---|---|
| +0.166 | +0.161 | +0.126 | +0.123 | +0.113 | +0.111 | +0.100 | +0.077 (×2.6) | +0.076 | +0.057 |

- **Reading the ranking honestly:** the two smallest-object classes lead, but class ordering is
  not strictly by size — class is a noisy proxy for object size. The strictly size-ranked gain
  shows where size is measured directly:

| Size bucket (@640) | <8px | 8-16px | 16-32px | 32-64px | >64px |
|---|---|---|---|---|---|
| 640 → 1024 | 13.6%→28.4% (×2.1) | +15.0pt | +7.9pt | +3.3pt | **92.7%→92.7% (±0)** |

  The unchanged >64px bucket is a perfect internal control: objects already above the
  threshold gained nothing because they had nothing to gain.
- Overall class-agnostic detection: 46.4%→58.9%.
- **Cost:** inference 1.6→6.9ms (×4.3) — the measured price of the resolution lever.
- **New bottleneck exposed:** vs the real baseline, FP rises 7.7k→8.8k (+14%) and
  misclassification share grows 6.3%→8.4%, with newly symmetric twin-class confusion
  (motor↔bicycle 134:134, pedestrian→people 30→263). The model now sees enough pixels
  to hesitate between lookalikes — a problem resolution cannot fix.

## Run 4 — lowscale_1024_2k (running)

Single variable vs Run 3: scale augmentation 0.5→0.2. **Hypothesis:** random downscaling by
up to 2x pushes already-tiny objects below assignment; gentler scaling should preserve more
of the <8px bucket during training. A null result is also informative (mosaic/scale interplay
may compensate).

---

## Standing decisions

- **Train subset:** 2,000 images, seed=42, deliberately fixed across all experiments;
  val/test kept full so metrics stay comparable. Cost: rare classes lose ~2/3 of their examples.
- **Optimizer pinned to SGD lr0=0.01** — optimizer=auto switches by iteration count and would
  confound runs of different lengths.
- **Headline metric: mAP50** (mAP50-95 double-penalizes small objects for identical pixel errors).
- **Official numbers come from scripts/compare_runs.py only**; results.csv at the best epoch
  agrees. Never mix end-of-training console tables with compare_runs outputs.
- **Size buckets are always computed at the 640 scale** for both models — same objects, same
  buckets; only the model changes.
- **Greedy IoU matching is a documented lower bound** (adversarial test in
  analysis/logic_review.md); it matches COCO-style behavior and biases both models equally,
  so deltas are unaffected.
- **Known data quirks:** 2,597 train boxes extend past borders (converter does not clip);
  4 duplicate labels auto-removed; per-class AP for bus/tricycle/awning-tricycle is noisy
  (251–1,045 val instances) — deltas under ~1 mAP there are not meaningful.

## Open questions

- ~~Does resolution lift the recall ceiling for tiny classes?~~ **Answered by Run 3:**
  pedestrian ceiling 0.48→0.65; <8px detection doubled.
- **Lookalike discrimination is the new dominant bottleneck** (motor↔bicycle 134:134).
  Levers: class rebalancing, more rare-class data. Resolution won't fix this.
- bicycle: ×2.6 but still second-worst (0.124, after awning-tricycle 0.116) — rarity +
  thinness + rider-steals-detection all still in play.
- Does scale=0.5 waste part of the resolution gain? (Run 4, landing)
- van→car eased at 1024 (0.46→0.40) but remains the largest single confusion.