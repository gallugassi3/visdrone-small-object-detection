# Logic review of the measurement and data code

Written 2026-08-26. Every item below was exercised with a small synthetic input whose answer was
worked out by hand before running, driven through the *real* project functions (imported from
`analyze_failures.py`, `draw_boxes.py`, `analysis/size_sensitivity.py`, `scripts/walkthrough.py`,
`scripts/failure_cases.py`, `scripts/compare_runs.py`, `predict_compare.py`, `make_subset.py`).
Where a tool only exposes a `main()`, its module globals (`YOLO`, data dirs, `sys.argv`) were
patched so the tool's own matching loop ran on a fake model. Item 8 ran both full-val tools on
CPU with the GPU hidden. Harness files live outside the repo; no project code was changed.

Verdict key: **CONFIRMED CORRECT** / **BUG** / **EDGE CASE**.

Summary: **no BUG found.** Eight CONFIRMED CORRECT, with six EDGE CASEs documented; one side
finding (item 8) affects a published sentence in `EXPERIMENTS.md`.

---

## 1. `iou_matrix` — CONFIRMED CORRECT

Two copies exist (`analyze_failures.py:43`, `analysis/size_sensitivity.py:54`); `inspect.getsource`
shows they are byte-identical, and both were tested.

| case (xyxy) | hand answer | got |
|---|---|---|
| `[0,0,10,10]` vs itself | 1.0 | 0.99999999999 (the `+1e-9` in the denominator) |
| `[0,0,10,10]` vs `[20,20,30,30]` | 0.0 | 0.0 |
| `[0,0,10,10]` vs `[10,0,20,10]` (touching edge) | 0.0 | 0.0 |
| `[0,0,10,10]` vs `[0,0,10,5]` (contained half) | 50/100 = 0.5 | 0.499999999995 |
| `[0,0,10,10]` vs `[5,0,15,10]` (shifted half) | 50/150 = 0.3333 | 0.33333333333 |
| symmetry, `(0,4)` vs `(2,4)` input -> shape `(0,2)` | — | pass |

**Box-format audit.** Every prediction consumer reads `result.boxes.xyxy` (`analyze_failures.py:58`,
`analysis/size_sensitivity.py:103`, `scripts/walkthrough.py:66`, `scripts/failure_cases.py:171`); no
`.xywh` appears anywhere. All three label parsers convert normalised xywh to pixel xyxy before
anything touches IoU (item 3). No mixing.

**EDGE CASE — zero-area box.** `[5,5,5,5]` vs itself gives IoU 0.0, not 1.0 (0/(0+0-0+1e-9)). A
zero-width GT box can therefore never be matched. Checked the data: **0 of 38,759** val boxes have
`w == 0` or `h == 0`, so it does not affect any number.

## 2. Greedy matching — CONFIRMED CORRECT (identical in all four tools), limitation documented

Synthetic image 20x20, two GT boxes of class 0: A = `[0,0,10,10]`, B = `[5,0,15,10]`
(label file `0 0.25 0.25 0.5 0.5` / `0 0.5 0.25 0.5 0.5`). Two predictions:
P1 = `[2,0,12,10]` conf 0.9, P2 = `[0,0,10,10]` conf 0.8.

IoU matrix (rows P1, P2; cols A, B) = `[[0.667, 0.538], [1.000, 0.333]]`.

- Hand-computed greedy-by-confidence: P1 takes A (its best, 0.667). P2 finds A taken and B at
  0.333 < 0.5 -> false positive. B is missed. **TP 1, FP 1, FN 1.**
- Brute-force optimal assignment: P1 -> B (0.538), P2 -> A (1.0). **TP 2.**

Result per tool, `(matched, FP, FN)`:

| tool | conf-descending input | conf-*ascending* input (tests the sort) |
|---|---|---|
| `analyze_failures.analyze_image` | (1, 1, 1) | (1, 1, 1) |
| `analysis/size_sensitivity.main` | (1, –, 1) | (1, –, 1) |
| `scripts/walkthrough.main` | (1, 1, 1) | (1, 1, 1) |
| `scripts/failure_cases.evaluate` | (1, 1, 1) | (1, 1, 1) |

All four give the greedy answer and all four sort by confidence first (feeding the predictions in
reverse order changes nothing). `size_sensitivity` does not count FPs by design (recall-only).

**Known limitation (documented, not a bug):** greedy-by-confidence can lose a match that a
Hungarian assignment would find, as above. This is the same family of procedure the Ultralytics
validator uses, and the effect is bounded to dense overlapping clusters. Every published detection
rate and miss count is a *lower* bound on what optimal matching would report; the direction is the
same for both models, so 640-vs-1024 deltas are not biased by it.

## 3. Label parsers — CONFIRMED CORRECT (two exact, one rounded by design)

Same line through `draw_boxes.yolo_line_to_corners`, `analyze_failures.yolo_labels_to_xyxy`,
`analysis/size_sensitivity.load_gt_boxes`; label file also contained a blank and a
whitespace-only line (all three skip them).

| line @ W x H | exact float xyxy | analyze_failures | size_sensitivity | draw_boxes |
|---|---|---|---|---|
| `3 0.333 0.5 0.25 0.5` @ 100x50 | 20.8, 12.5, 45.8, 37.5 | identical | identical | **21, 12, 46, 38** |
| `0 0.5 0.5 0.5 0.5` @ 100x100 | 25, 25, 75, 75 | identical | identical | 25, 25, 75, 75 |
| `0 0.98 0.5 0.1 0.2` @ 100x100 (out of frame) | 93, 40, **103**, 60 | 103 kept | 103 kept | 103 kept (clamped later, for drawing only) |
| `1 0.1234 0.4567 0.0037 0.0078` @ 1360x765 | 165.308, 346.392, 170.340, 352.359 | identical | identical | 165, 346, 170, 352 |

`np.array_equal(analyze_failures, size_sensitivity)` is **True** for every case: the two
measurement parsers are bit-identical (same expression order), so no measurement tool sees a
different box than another. `draw_boxes` rounds to integers for OpenCV (max error 0.5 px; note
Python's `round(12.5) == 12`, banker's rounding) — it feeds only visualisation, never a metric.

**EDGE CASE — class-id validation.** `draw_boxes` rejects class 10 with a clear error;
`analyze_failures.yolo_labels_to_xyxy` accepts it and returns `cls=[10]`, which would raise
`IndexError` later at `CLASS_NAMES[10]`. It fails loudly rather than silently and VisDrone's
converter emits only 0-9, so no effect today. Missing label file -> empty `(0, 4)` array in both
measurement parsers (`size_sensitivity` then skips the image; `analyze_failures` counts all
predictions on it as FP — the intended semantics for each).

## 4. Size bucketing — CONFIRMED CORRECT; two EDGE CASEs

**Boundary.** `bucket_index` uses half-open intervals `[lo, hi)`:
0.0 -> `<8px`, 7.999 -> `<8px`, **8.0 -> `8-16px`**, 15.999 -> `8-16px`, 16.0 -> `16-32px`,
64.0 -> `>64px`, 1e9 -> `>64px`. The labels say the same thing (`<8px` is strict).
`scripts/failure_cases.tiny_mask_for` uses `< 8` and agrees: sizes `[7.999, 8.0]` -> `[True, False]`.
A 17.00-px square in a 1360x765 image scales to exactly 8.000 at 640 and lands in `8-16px`.

**Letterbox scale.** The tool uses `640 / max(W, H)`. Ultralytics resizes the long side to `imgsz`
(`data/base.py:276`, `r = imgsz / max(h0, w0)`) and `LetterBox` uses `min(new/h, new/w)`
(`data/augment.py:1745`) — the same number for a square target. Checked for all three val image
shapes (1360x765 -> 0.4706, 1920x1080 -> 0.3333, 960x540 -> 0.6667): equal to machine precision.

**EDGE CASE — clipped size vs unclipped matching.** `gt_sizes_at_input_scale` clips a box to the
frame before taking its area, but every matching loop (all four tools) uses the *unclipped* GT
box. Synthetic: a 120x100 box at `[1300,700,1420,800]` in 1360x765 has clipped size 29.4
(`16-32px`) vs unclipped 51.6 (`32-64px`), and a perfect prediction of its visible part scores IoU
0.325 against the unclipped box -> counted as a miss. When it matters: only when more than half
the box is outside the frame. On val: **324** of 38,759 boxes extend past the frame, **0** have
less than half their area visible, so no val box is unreachable and the published rates are not
affected. This is also exactly what the Ultralytics validator does: `detect/val.py:143-148` feeds
`batch["bboxes"]` into matching unclipped (neither `LetterBox` nor `Format` clips for val), so the
tools agree with the official mAP protocol.

**EDGE CASE — Ultralytics' label sanity check.** `data/utils.py:349-350` discards an entire
image's labels if any raw value is > 1.01 or < -0.01. The 324 out-of-frame val boxes pass because
the check is on `xc, yc, w, h`, not on `xc + w/2`. Verified from the caches:
`train.cache` results `(2000, 0, 0, 0, 2000)`, `val.cache` `(548, 0, 0, 0, 548)` — zero corrupt.
Ultralytics and the analysis scripts therefore see the same 548 images and 38,759 boxes.

## 5. `make_subset.py` — CONFIRMED CORRECT; one EDGE CASE

- Two seeded draws in one process: identical.
- Re-drawing `random.seed(42); random.sample(sorted(names), 2000)` from today's
  `datasets/VisDrone/images/train` and comparing with the on-disk `VisDrone-2k/images/train`:
  **identical, 2000/2000** (Python 3.14.3).
- Orphans in the produced subset: train 2000/2000, val 548/548, test 1610/1610 images/labels;
  images-without-label = 0 and labels-without-image = 0 in every split. Labels-without-image cannot
  happen by construction (the loop iterates images). Images-without-label is possible in principle
  (`copy_split` copies the label only `if label.exists()`), but the source has a label for every image.

**EDGE CASE.** `iterdir()` is not filtered by suffix, so a stray non-image file in
`datasets/VisDrone/images/train` would be eligible for sampling and would shift every later pick.
None exists today (`[]`). Also, Python only guarantees `random()` sequences across versions; the
`sample` algorithm is not formally frozen, so "same seed = same subset" is a same-Python-line
guarantee. The subset itself is the artefact, and it matches.

## 6. `compare_runs.py` `ap_class_index` pairing — CONFIRMED CORRECT

Fake metrics with `ap_class_index = [0, 3, 9]` and `ap50 = [0.1, 0.7, 0.4]` (classes 1, 2, 4-8
absent) -> `{'pedestrian': 0.1, 'car': 0.7, 'motor': 0.4}`; names did not shift. `format_table`
with a class present in run B only prints `motor  -  0.400  -`. In Ultralytics,
`ap_per_class` (`utils/metrics.py:838-887`) builds `unique_classes` with `np.unique` and fills
`ap` row-by-row in that order, so `ap50[i]` belongs to `ap_class_index[i]` — the zip is right.
For the published table the question is moot anyway: all ten classes are present in val
(`val.cache` classes `[0..9]`), so `ap_class_index == [0..9]`.

## 7. Confidence threshold parity — CONFIRMED CORRECT; one EDGE CASE (cosmetic)

- `CONF_THRESHOLD = 0.25` in all five tools; `IOU_THRESHOLD = 0.5` in all four matching tools.
- Captured `predict()` kwargs on a fake model:
  `failure_cases.evaluate` -> `{'imgsz': 1024, 'conf': 0.25, 'device': 'cpu', 'verbose': False}`;
  `predict_compare.annotate_pred` -> `{'conf': 0.25, 'verbose': False}`;
  `analyze_failures.analyze_image` -> `{'conf': 0.25, 'verbose': False}`.
  The ones that omit `imgsz` inherit it from the checkpoint; verified
  `YOLO(best.pt).overrides['imgsz']` is 640 for `baseline_640_2k` and 1024 for `highres_1024_2k`,
  so explicit and inherited paths hit the same operating point. NMS `iou` and `max_det` are not set
  anywhere -> Ultralytics defaults (0.7 / 300) in all tools.
- Displayed vs matched confidence: `result.plot()` reads `result.boxes.conf`; `failure_cases.evaluate`
  sorts a NumPy *copy* and leaves the tensor untouched (`torch.equal(before, after)` is True), so
  the label drawn on a box is the confidence the matcher used.

**EDGE CASE (cosmetic).** `predict_compare.py:49` passes `font_size=0.35` and
`failure_cases.py:59` passes `font_size=10` to `result.plot()`; in cv2 mode Ultralytics ignores
`font_size` (only used when `pil=True`, `utils/plotting.py:322`). Nothing numeric depends on it.

## 8. Cross-tool invariant — CONFIRMED CORRECT; one EDGE CASE affecting the 4th decimal

`analyze_failures.py` and `analysis/size_sensitivity.py` on `runs/detect/baseline_640_2k/weights/best.pt`,
full val (548 images), CPU, GPU hidden (`CUDA_VISIBLE_DEVICES=-1` plus `device='cpu'` on every predict):

```
analyze_failures : detected 14,996  misclassified 2,996  missed 20,767  false_positive 7,744
                   detected + misclassified = 17,992
size_sensitivity : class-agnostic detected  = 17,992      <- invariant holds exactly
```

Per bucket (fresh CPU run / published GPU CSV):
`<8px` 1634/1635 · `8-16px` 7109/7113 · `16-32px` 6595/6595 · `32-64px` 2413/2413 · `>64px` 241/241
· all **17,992 / 17,997**. GT counts identical in every bucket.

**EDGE CASE — device numerics.** The fresh CPU run finds 5 fewer boxes than the published GPU run
(all in the two smallest buckets; rates move by <= 0.0003, e.g. `<8px` 0.1361 vs 0.1362). This is
float32-CPU vs GPU kernel differences on boxes sitting at the 0.25 confidence edge, not a logic
difference — the invariant holds within each run. Published CSV values are therefore
reproducible to three decimals, not four; nothing in `EXPERIMENTS.md` quotes a fourth decimal.

**Side finding (affects a published sentence).** This is the first saved `analyze_failures` run on
`baseline_640_2k`: FP = 7,744, misclassified = 2,996. `EXPERIMENTS.md` (1024 row and "Open
questions") says "FP 5.7k -> 8.8k", "FP rose 54%", and "pedestrian->people 30 -> 263"; the 5.7k
and 30 are the **smoke** model's numbers (`analysis/failures_smoke.txt`). Against the real 640
baseline the FP rise is roughly 7.7k -> 8.8k (~14 %, taking the 8.8k at face value; the 1024 run's
`analyze_failures` output is not saved either). Severity: documentation, medium — the "new
bottleneck" narrative is overstated by comparing to a 5-epoch model. Already logged as S5 in
`analysis/repo_review.md`; the fresh numbers above are the evidence.

---

## Method notes

- Harness A (items 1-7) used fake `Results` objects (tensors for `boxes.xyxy/conf/cls`,
  `orig_shape`, `orig_img`) so the tools' real loops ran without a model.
- Harness B (item 8) was launched once with `CUDA_VISIBLE_DEVICES=""`; on Windows an empty value
  deletes the variable instead of hiding the GPU, so that first launch touched the training GPU for
  about a minute before it was killed. The rerun used `-1` (verified `torch.cuda.is_available()`
  is False) and additionally pinned `device='cpu'` on every predict call. Worth remembering for any
  future "run this on CPU" script on this machine: `scripts/failure_cases.py` does it the safe way
  (`device='cpu'` argument), not via the environment.
