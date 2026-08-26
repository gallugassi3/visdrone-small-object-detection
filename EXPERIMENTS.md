# Experiment Log

Every training run gets a row. Conclusions drive the next experiment.

| Run | Setup | mAP50 | mAP50-95 | Epoch time | Conclusion |
|-----|-------|-------|----------|------------|------------|
| baseline_640_smoke | full train (6,471 img), 5 epochs, imgsz=640, batch=8, optimizer=auto | 0.206 | 0.114 | ~8.5 min | Pipeline verified. Clean underfit: mAP rising linearly with no saturation. Confusion matrix shows the dominant failure is non-detection, not misclassification — 73–93% of small/thin classes (bicycle 93%, tricycle 85%, people 84%, motor 82%, pedestrian 73%) predicted as background, vs car at 36%. Confirms the stride ceiling: objects below ~8 px at 640 rarely get anchor assignment. Only major inter-class confusion: van→car (42%). optimizer=auto silently picked AdamW(lr=0.0007), ignoring lr0 — pinning SGD from now on for cross-run comparability. GPU power-throttles 120W→40W at ~90°C. |
| baseline_640_2k | 2k random train subset (seed=42), full val, 80 epochs, imgsz=640, batch=16, SGD lr0=0.01, patience=25 | 0.258 | 0.142 | ~1:25 | Clean convergence, no overfit, near-plateau by ep70 (+0.005 over last 10 epochs). Long training lifted mid/large classes 13-19pt (bus miss 63%→48%, tricycle 85%→66%) but pedestrian only 4pt (73%→69% missed) — training time cannot fix sub-stride size; only resolution can. bicycle finally registers (AP50 0.009→0.047) but remains worst. van→car confusion unchanged (structural). close_mosaic step visible in train loss at ep70 with no val effect. |
| highres_1024_2k | identical to baseline except imgsz=1024, batch=6 | 0.370 | 0.216 | ~3:20 | **Hypothesis confirmed decisively: mAP50 +43% (0.258→0.370) on identical data and budget.** Ran the full 80 epochs; best ep79 (0.370), flat from ep70 (0.369→0.368) — converged; longer same-config training would add nothing. Every class improved: motor +0.166, pedestrian +0.161 (+58%), van +0.126, bus +0.123, people +0.113, truck +0.113, car +0.099, tricycle +0.076, bicycle 0.047→0.124 (×2.6), awning-tricycle +0.058 — the two smallest-object classes lead, but the ordering is not strictly by size; >64px size bucket unchanged at 92.7% — a perfect internal control. Detection rate <8px: 13.6%→28.4% (×2.1); overall class-agnostic detection 46.4%→58.9%. mAP50-95 rose relatively more (+52%) — boxes are both found more and localized better. Cost: inference 1.6→6.9ms (×4.3). New bottleneck exposed: FP 5.7k→8.8k and symmetric twin-class confusion (motor↔bicycle 134:134, pedestrian→people 30→263) — the model now sees enough pixels to hesitate between lookalikes. Official numbers from scripts/compare_runs.py (unified val protocol on best.pt); results.csv agrees at the best epoch (0.370). |

## Standing decisions
- **Train subset:** 2,000 images sampled with seed=42 for iteration speed (deadline-driven);
  val/test kept full so metrics stay comparable. Cost: rare classes lose ~2/3 of their examples.
- **Optimizer pinned to SGD, lr0=0.01** — optimizer=auto switches AdamW/SGD by iteration count,
  which would confound comparisons between runs of different lengths.
- **Headline metric: mAP50** (mAP50-95 double-penalizes small objects for identical pixel errors).
- **Official metrics come from scripts/compare_runs.py** — a unified val protocol on best.pt with
  explicit data/imgsz. The per-epoch results.csv agrees at the best epoch (0.370 for the 1024 run);
  quote the compare file rather than console output so every number has a tracked source.
- **Size-sensitivity buckets are always computed at the 640 scale** for both models — same objects,
  same buckets, only the model changes.
- **Known data quirks:** 2,597 train boxes extend past image borders (converter does not clip);
  4 duplicate labels auto-removed by the scanner; per-class val AP for bus/tricycle/awning-tricycle
  is noisy (251–1,045 val instances) — deltas under ~1 mAP on these are not meaningful.

## Open questions
- ~~How much does imgsz=1024 lift the recall ceiling for pedestrian/people?~~ **Answered:** pedestrian
  recall ceiling 0.48→0.65, people similar; detection rate <8px doubled. The size lever works and
  is now measured end to end.
- **New dominant bottleneck: lookalike discrimination, not size.** motor↔bicycle confusion is
  perfectly symmetric (134:134), pedestrian→people jumped 30→263, FP rose 54%. The model sees
  enough to detect but not enough (or not trained enough) to distinguish twins. Candidate levers:
  class rebalancing, scale augmentation tuning, more rare-class data — resolution won't fix this.
- bicycle: improved ×2.6 (AP50 0.124) but still second-worst after awning-tricycle (0.116); F1 peaks at only ~0.21 at low confidence.
  Rarity + thinness + rider-steals-detection all still in play.
- Default scale=0.5 augmentation shrinks already-tiny objects below existence; on 1024 it may be
  wasting part of the resolution gain. Candidate for the next controlled run (scale=0.2).
- van→car confusion eased at 1024 (0.46→0.40) but remains the largest single confusion — structural
  similarity + frequency imbalance.