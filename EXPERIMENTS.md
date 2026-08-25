# Experiment Log

Every training run gets a row. Conclusions drive the next experiment.

| Run | Setup | mAP50 | mAP50-95 | Epoch time | Conclusion |
|-----|-------|-------|----------|------------|------------|
| baseline_640_smoke | full train (6,471 img), 5 epochs, imgsz=640, batch=8, optimizer=auto | 0.206 | 0.114 | ~8.5 min | Pipeline verified. Clean underfit: mAP rising linearly with no saturation. Confusion matrix shows the dominant failure is non-detection, not misclassification — 73–93% of small/thin classes (bicycle 93%, tricycle 85%, people 84%, motor 82%, pedestrian 73%) predicted as background, vs car at 36%. Confirms the stride ceiling: objects below ~8 px at 640 rarely get anchor assignment. Only major inter-class confusion: van→car (42%). optimizer=auto silently picked AdamW(lr=0.0007), ignoring lr0 — pinning SGD from now on for cross-run comparability. GPU power-throttles 120W→40W at ~90°C. |
| baseline_640_2k | 2k random train subset (seed=42), full val, 80 epochs, imgsz=640, batch=16, SGD lr0=0.01, patience=25 | TBD | TBD | TBD | TBD |

## Standing decisions
- **Train subset:** 2,000 images sampled with seed=42 for iteration speed (deadline-driven);
  val/test kept full so metrics stay comparable. Cost: rare classes lose ~2/3 of their examples.
- **Optimizer pinned to SGD, lr0=0.01** — optimizer=auto switches AdamW/SGD by iteration count,
  which would confound comparisons between runs of different lengths.
- **Headline metric: mAP50** (mAP50-95 double-penalizes small objects for identical pixel errors).
- **Known data quirks:** 2,597 train boxes extend past image borders (converter does not clip);
  4 duplicate labels auto-removed by the scanner; per-class val AP for bus/tricycle/awning-tricycle
  is noisy (251–1,045 val instances) — deltas under ~1 mAP on these are not meaningful.

## Open questions
- How much does imgsz=1024 lift the recall ceiling for pedestrian/people? (next experiment)
- bicycle: 93% undetected and近 zero predictions even at low confidence — size+thinness+rarity
  compounded, or rider stealing the detection? Inspect visually after the long baseline.
- van→car confusion (42%): revisit only if it persists at higher resolution.