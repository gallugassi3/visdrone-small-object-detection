# README fact-check and completeness review

Written 2026-08-26 against the README as of this date (post-rewrite). Report only; nothing was
changed. Sources checked: `analysis/compare_baseline_640_2k_vs_highres_1024_2k.txt` (compare
file), `analysis/size_sensitivity_{baseline_640_2k,highres_1024_2k}.csv`,
`analysis/failures_smoke.txt`, `analysis/flipud_notes.md`, `analysis/logic_review.md` §8,
`runs/detect/*/results.csv` and `confusion_matrix_normalized.png`, a fresh scan of
`datasets/VisDrone/labels/*`, `git ls-files`, `git remote`, the Ultralytics source, and a CPU
re-inference of both checkpoints on the val image behind `assets/compare_park.png`.

**Bottom line:** 4 numbers are wrong, 3 claims are contradicted by the repo's own artifacts,
5 numbers have no tracked source, and the Quick start does not work on a fresh clone because of
where Ultralytics downloads the dataset. Everything else checks out.

---

## 1. FACTS

### Wrong (differs from its source)

| README line | README says | Source says | Note |
|---|---|---|---|
| 52 | Recall 0.299 → 0.383, +8.4pt | compare file: **0.300 → 0.393, +9.3pt** (results.csv best epoch: 0.300 / 0.392) | matches neither source |
| 53 | Precision 0.381 → 0.469, +8.8pt | compare file: **0.380 → 0.490, +11.0pt** (results.csv: 0.380 / 0.495) | matches neither source |
| 90 | "Extreme density (**462** objects in one frame)" | val max GT per image is **317** (`0000295_02400_d_0000033`); densest case in `failure_cases_1024.md` has 251 GT | 462 appears nowhere in the repo; train max is 902 |
| 76-78 | caption: "The 640 model misses the distant pedestrians **entirely** and calls the cargo tricycle a **bicycle**; the 1024 model recovers the distant people and labels the tricycle correctly" | CPU re-run on `0000021_00500_d_0000002.jpg` (17 GT): 640 finds **4 of 7** pedestrians (conf 0.67/0.55/0.54/0.43); 640 calls the tricycle **car 0.50**, not bicycle; what 640 calls "bicycle" is a GT **motor** on the path, which the 1024 model also mislabels (tricycle 0.35 on CPU, bicycle 0.3x in the GPU-rendered asset); the 1024 model's tricycle 0.38 is correct | 1 of 3 caption claims holds |

### Contradicted by the repo's own artifacts

| README line | Claim | Problem |
|---|---|---|
| 36-37 | "**The baseline's** dominant failure … 62.6% missed vs 6.3% misclassified, a 10:1 ratio" | These are the **5-epoch smoke** run's numbers (`failures_smoke.txt`). For `baseline_640_2k` (the model the metrics table calls the baseline) the measured values are 53.6% missed vs 7.7% misclassified (`logic_review.md` §8), a 7:1 ratio. The point survives; the label does not. |
| 84-97 | "three failure cases … analysis in `failure_cases_1024.md`": (1) extreme density, (2) lookalike twins, (3) compound objects decompose | The linked file's three cases are (1) sub-8px objects, (2) awning-tricycle 0-for-18, (3) vehicle-family wrong classes. None is "density"; "awning-tricycle detected as **car + pedestrian**" is not in the file (it records awning to motor ×2, to car ×4, to van ×1, and 22 of 32 missed). |
| 66 | "Where size is measured directly, the ranking is **perfectly monotonic**" | Absolute gains per bucket are +14.8 / **+15.0** / +7.8 / +3.3 / 0.0 pt; the `8-16px` bucket gained more than `<8px`. Only the *relative* gain (×2.09 / 1.32 / 1.10 / 1.04 / 1.00) is monotonic. Say "relative gain". |

### No tracked source (cannot be verified from the repo)

| README line | Number | What exists |
|---|---|---|
| 54, 67 | Inference 1.6 ms / 6.9 ms, ×4.3, "~145 FPS" | No file records speed; `compare_runs.py` prints it but does not save it. Ratio (4.31) and 1000/6.9 = 145 are arithmetically consistent. |
| 92-93 | motor/bicycle 134:134; pedestrian→people **30** → 263 | No `analyze_failures` output for the 1024 run is saved. The "30" is the smoke run's value (`failures_smoke.txt` line 29), not the 640 baseline's. |
| 93-94 | "False positives rose 14% vs the baseline" | Baseline FP = 7,744 is now measured (`logic_review.md` §8); the 8.8k for 1024 is unsaved. 7,744 → 8,800 = +13.6%, consistent *if* 8.8k holds. |
| 127 | download "~2.3GB" | Ultralytics' `VisDrone.yaml` says "~2 GB"; extracted data on disk is 1.9 GB. |
| 129-130 | "~2h" / "~4.5h" | Consistent with `EXPERIMENTS.md` epoch times only if "1:25" / "3:20" mean min:sec per epoch (80 × 1:25 = 1.9 h; 80 × 3:20 = 4.4 h). The unit is not stated anywhere. |

### Verified correct

| README line | Claim | Source / computation |
|---|---|---|
| 6, 13, 50 | mAP50 0.258 → 0.370, +43% | compare file; 0.370 / 0.258 = 1.434 |
| 51, 62 | mAP50-95 0.142 → 0.216, +52% | compare file; 1.521 |
| 35 | 70% of objects under 16px | (12,005 + 15,103) / 38,759 = 69.9% |
| 21-22 | `<8px` = 31% of objects; "doubles"; `>64px` "doesn't move" | 12,005 / 38,759 = 31.0%; 0.1362 → 0.2844 = ×2.09; 0.9269 → 0.9269 |
| 33 | 548 val images, 38,759 boxes | CSVs and label scan |
| 39-40 | "80 epochs lifted mid/large classes 13-19pt, pedestrians 4pt" | `confusion_matrix_normalized.png`, background row, smoke to 2k: bus 0.63→0.48 (15), tricycle 0.85→0.66 (19), truck 0.65→0.52 (13), van 0.45→0.33 (12), motor 0.82→0.69 (13), awning 0.78→0.64 (14); pedestrian 0.73→0.69 (4). *Caveat:* the two runs also differ in train set (full 6,471 vs 2k subset), so "80 epochs" is not the only variable. |
| 64 | motor +0.166, pedestrian +0.161 | compare file |
| 100 | flipud study: ~49% near-nadir | `flipud_notes.md`: 49 of 100 |
| 110 | 8,629 images, ~457K boxes | 6,471 + 548 + 1,610; 343,205 + 38,759 + 75,102 = 457,066 |
| 113 | median 42 objects/image, max 902 | train labels: median 42, max 902 |
| 114-115 | car 42%; awning-tricycle < 1% (45:1); four rarest = 7% of data, 40% of metric | train: 42.2%; 0.95%; ratio 44.6; 7.1%; 4/10 classes |
| 116 | 2,597 out-of-frame train boxes; 4 duplicates | label scan: 2,597 / 4 |
| 3 | Python 3.14 badge | venv is 3.14.3 |
| 158-159 | "9 blockers found and fixed"; "zero bugs" | `repo_review.md` B1-B9, all addressed; `logic_review.md` |
| 152 | "`results.csv` at the best epoch agrees" | ep79 mAP50 0.36996 |

---

## 2. FUNCTIONALITY

### Images and links: all resolve

All four images are tracked (`assets/size_sensitivity_comparison.png`, `assets/per_class_delta.png`,
`assets/compare_park.png`, `assets/compare_dense.png`). All internal link targets exist
(`assets/`, `analysis/failure_cases_1024.md`, `analysis/flipud_notes.md`, `EXPERIMENTS.md`,
`analysis/logic_review.md`, `analysis/repo_review.md`). The clone URL matches `git remote -v`.
Badges and the VisDrone link are external and fine.

### Quick start: two lines do not work as written

**Line 124, `python -m venv .venv && .venv/Scripts/activate` (bash block).** The activate
script's own header reads "This file must be used with `source bin/activate` from bash. You
cannot run it directly." In Git Bash it executes in a subshell and changes nothing; in PowerShell
or cmd the extensionless file is not runnable, and `&&` is a parse error in Windows
PowerShell 5.1. Working forms: `source .venv/Scripts/activate` (Git Bash) or
`.\.venv\Scripts\Activate.ps1` (PowerShell, the form `CLAUDE.md` uses).

**Lines 127-128, `python download_data.py` then `python make_subset.py`: broken on a fresh clone.**
`download_data.py` calls `check_det_dataset("VisDrone.yaml")`, which downloads into Ultralytics'
`datasets_dir` setting. Its default (`ultralytics/utils/__init__.py:1393-1398`) is
`<git-root>.parent / "datasets"`; for this repo that is `C:\Projects\datasets`, *outside* the clone.
`make_subset.py` then reads the hard-coded `datasets/VisDrone` inside the repo and fails. It
works on this machine only because `%APPDATA%\Ultralytics\settings.json` was edited to
`"datasets_dir": "C:\\Projects\\visdrone-project\\datasets"`. The Quick start needs either
`yolo settings datasets_dir=<repo>/datasets` before the download, or `make_subset.py` reading
`DATASETS_DIR` from Ultralytics settings.

### Quick start: lines that do work

- `pip install -r requirements.txt`: parses (38 pins); the torch note inside is correct.
- `python check_env.py`: runs (requires a CUDA GPU; crashes without one).
- `python train.py` / `python train_1024.py`: produce `runs/detect/baseline_640_2k` and
  `runs/detect/highres_1024_2k`, exactly the names `compare_runs.py` is given on line 131.
  Edge: `exist_ok` is not set, so a *second* run becomes `baseline_640_2k2` and `compare_runs`
  would silently evaluate the old directory.
- `python scripts/compare_runs.py baseline_640_2k highres_1024_2k`: plain script, chdirs to
  the project root itself; no `-m` needed.

### Repository guide

`scripts/walkthrough.py` (line 142) must be run as `python -m scripts.walkthrough`; the guide
does not say so, and `python scripts/walkthrough.py` fails with `ModuleNotFoundError: draw_boxes`.
The same applies to `scripts/failure_cases.py`, which is not listed at all.

---

## 3. COMPLETENESS (judgment)

1. **No LICENSE file.** `git ls-files` has none. It is the first thing an engineer checks before
   reading code they might reuse, and GitHub will show "no license" on the repo page.
2. **The failure-case figures are never shown, and the README's three cases are not the ones in
   the linked file.** `assets/failure_case_{1,2,3}.png` and `scripts/failure_cases.py` are the
   most honest content in the repo (the 1024 model going 0-for-18 on awning-tricycles is a
   better story than an unsourced "462 objects"), yet neither appears in the README or the
   Repository guide.
3. **"Baseline" means two different models.** The 62.6% / 6.3% and 13-19pt figures come from the
   5-epoch smoke run; the metrics table's baseline is the 80-epoch `baseline_640_2k`. A reader
   cannot reconcile "62.6% missed" with a 0.300 recall without knowing that.
4. **Provenance of the qualitative panels.** Nothing says which val images the two comparison
   panels are (`0000021_00500_d_0000002` for the park), that they are 0.5× downscales of a
   960×540 source, or that `compare_dense.png` has no caption at all. A reviewer who wants to
   re-run `predict_compare.py` on the same frame has to guess.
5. **Two protocols, never distinguished.** "Detection rate" (analysis tools: conf 0.25, IoU 0.5,
   greedy, class-agnostic) and mAP (Ultralytics val: conf 0.001, COCO-style AP) sit side by side
   with no note that they are different measurements, and nothing says the test split was never
   touched (all numbers are val). `analysis/imgsz_eval_notes.md` §3 explains this; one sentence
   and a link would close it.
