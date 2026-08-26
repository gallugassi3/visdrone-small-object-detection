# Pre-publication repository review

Written 2026-08-26 against commit `84a3a74`. Every tracked file (`git ls-files`, 36 files) was read.
Claims in the docs were checked against the artifacts that back them: `runs/detect/*/results.csv`
and `args.yaml`, `analysis/compare_baseline_640_2k_vs_highres_1024_2k.txt`, the size-sensitivity
CSVs, `analysis/failures_smoke.txt`, and a fresh scan of `datasets/VisDrone/labels/*` for the
dataset statistics. Every `Usage:` line was checked against how the file resolves imports and
paths today; all 14 Python files byte-compile. Report only; no code was changed.

## Usage-line verification

| File | Usage line | Runs as written? | Note |
|---|---|---|---|
| `train.py` | `python train.py` | yes | |
| `train_1024.py` | `python train_1024.py` | yes | |
| `train_scale.py` | `python train_scale.py` | yes | |
| `make_subset.py` | `python make_subset.py` | yes | needs `datasets/VisDrone` from `download_data.py` first; order documented nowhere |
| `draw_boxes.py` | `python draw_boxes.py [image_path]` | yes | default image is in `datasets/VisDrone` (full set), not `VisDrone-2k` |
| `analyze_failures.py` | `python analyze_failures.py <weights>` | yes | root-level import of `draw_boxes`, works from root |
| `predict_compare.py` | `python predict_compare.py <w> [w2]` | yes | all four `DEMO_IMAGES` exist in val |
| `analysis/size_sensitivity.py` | `python analysis/size_sensitivity.py [weights]` | yes | no root imports; default weights point at the smoke run (S3) |
| `scripts/compare_runs.py` | `python scripts/compare_runs.py A B` | yes | chdirs to root itself, no root imports |
| `scripts/walkthrough.py` | `python -m scripts.walkthrough [image]` | yes | `-m` is required (imports `draw_boxes`, `analyze_failures`); docstring has a doubled `Usage:` (S2) |
| `scripts/failure_cases.py` | `python -m scripts.failure_cases` | yes (ran 2026-08-26) | `-m` required; `DEVICE = "cpu"` is passed to `model.predict(device=DEVICE)`; the CPU note matches the code |
| `scripts/size_chart.py` | `python scripts/size_chart.py` | yes (ran 2026-08-26) | |
| `check_env.py`, `download_data.py` | (no usage line) | yes | `CLAUDE.md` documents `check_env.py`; nothing documents `download_data.py` |
| `CLAUDE.md` commands | activate, `check_env`, `train`, `draw_boxes` | yes | |

No secrets, API keys, e-mail addresses, or `C:\Users\...` paths appear in any tracked file.
No `TODO`/`FIXME` markers exist. Hebrew text exists in exactly one place (S1).

---

## BLOCKERS

**B1. `EXPERIMENTS.md` line 9: "Early-stopped at ep72 (best ep47), converged"**
`runs/detect/highres_1024_2k/results.csv` has 80 rows; mAP50 peaks at epoch 79 (0.370) and the
final epoch is 0.368. The run was not early-stopped and the best epoch was not 47 (epoch 47 reads
0.348). The stated basis for "longer same-config training would add nothing" is wrong, even though
the flat 70-80 tail (0.369 -> 0.368) still supports the conclusion.
*Fix:* replace with "ran the full 80 epochs; best epoch 79 (0.370), flat from ep70 (+0.001); converged".

**B2. `EXPERIMENTS.md` line 9 and lines 17-19: "epoch-table numbers differ (0.322)" / "read lower (0.322 vs 0.369)"**
Nothing in `results.csv` reads 0.322 at the end of training; the best epoch's mAP50 is 0.36996,
identical to the official `compare_runs.py` figure (0.370). The "validation protocol matters" lesson
in Standing decisions is built on a number that no tracked artifact reproduces.
*Fix:* delete the 0.322 claim (or, if it came from a mid-run console line, say which epoch) and
restate the standing decision as "official metrics come from `compare_runs.py`; results.csv agrees at best epoch".

**B3. `EXPERIMENTS.md` line 9: "tricycle +0.142"**
`analysis/compare_baseline_640_2k_vs_highres_1024_2k.txt` line 18 gives tricycle 0.152 -> 0.228 =
+0.076. The size-ordered ranking sentence ("gains ranked inversely by object size") leans on this value.
*Fix:* change to +0.076 and re-check the ranking claim (people +0.113 now outranks tricycle).

**B4. `EXPERIMENTS.md` line 9: "identical to baseline except imgsz=1024, batch=6, workers=2"**
`runs/detect/highres_1024_2k/args.yaml` records `workers: 4`, and `train_1024.py` line 22 sets
`workers=4`. Only `train_scale.py` uses `workers=2`.
*Fix:* drop "workers=2" from the highres row.

**B5. `EXPERIMENTS.md` line 9: 0.369 vs 0.370, "bicycle 0.046->0.125", "motor +0.165"**
The official table (`compare_*.txt`) says mAP50 0.370, bicycle 0.047 -> 0.124, motor +0.166. The
run log names that file as the source of truth, then quotes different roundings. Small, but a
reader who checks will find every third number off by one in the last digit.
*Fix:* copy the numbers verbatim from `compare_*.txt` (0.370, +0.112 mAP50, bicycle 0.047 -> 0.124, motor +0.166).

**B6. `README.md` lines 8-9 and 46: "resolution experiment in progress", "Results: in progress, landing here"**
The experiment finished on 2026-08-25; `EXPERIMENTS.md` calls it "confirmed decisively" and the
chart, comparison table, and failure cases all exist. The public landing page contradicts the log.
*Fix:* replace the status banner and the Results placeholder with the headline (mAP50 0.258 -> 0.370, +43 %),
embed `assets/size_sensitivity_comparison.png`, and link `analysis/failure_cases_1024.md`.

**B7. `README.md` line 38: "each object gets 1.6x more pixels"**
1024/640 = 1.6 is the linear scale factor; pixel count grows by 1.6² = 2.56x.
*Fix:* "1.6x more pixels on each side (2.56x more pixels)".

**B8. `README.md` line 50: "~471K boxes"**
The converted labels for the 8,629 images hold 457,066 boxes (train 343,205 / val 38,759 /
test 75,102). 471K is not the count of the boxes this repo trains on; if it is the raw annotation
count including the ignored-region/others classes the converter drops, it needs to say so.
*Fix:* "~457K boxes after conversion (ignored-region and 'others' annotations dropped)".

**B9. `requirements.txt`: UTF-16 encoding and a CUDA-suffixed torch pin**
The file starts with the bytes `FF FE` (UTF-16 LE BOM). GitHub renders it as garbage/binary and
`cat`/diff tools show `t o r c h`. Separately, lines 34-35 pin `torch==2.13.0+cu130` and
`torchvision==0.28.0+cu130`; these local-version wheels are not on PyPI, so a plain
`pip install -r requirements.txt` fails on a clean machine unless the PyTorch index URL is supplied.
*Fix:* re-save as UTF-8, add `--extra-index-url https://download.pytorch.org/whl/cu130` at the top
(or move torch/torchvision to a documented separate install step).

---

## SHOULD-FIX

**S1. `check_env.py` line 12: Hebrew comment** (`# בדיקה אמיתית: ...`).
Only non-English text in the repo; `CLAUDE.md` mandates English.
*Fix:* `# Real check: run a matmul on the GPU, not just detect it`.

**S2. `scripts/walkthrough.py` lines 13-15: docstring reads `Usage:\n    Usage:\n    python -m ...`**
Doubled label left over from the `-m` fix in `48346d6`.
*Fix:* delete one `Usage:` line.

**S3. `analysis/size_sensitivity.py` line 23: `DEFAULT_WEIGHTS = "runs/detect/baseline_640_smoke/weights/best.pt"`**
Default points at the throwaway smoke run; the two CSVs that matter came from the 2k runs. Line 5 of
the same docstring also says buckets use "pixel size at the model's input scale", which the second
paragraph and line 32 (`MODEL_INPUT_SIZE = 640  # ... always ... 640`) contradict for the 1024 model.
*Fix:* make the weights argument required (or default to `baseline_640_2k`) and reword line 5 to "at the 640 reference scale".

**S4. `analysis/size_sensitivity.csv`: orphaned smoke output with a different schema and different GT counts**
Header is `size_bucket,gt_count,detected_count,detection_rate`; the two later CSVs (and the script's
current `OUTPUT_CSV_TEMPLATE`) use `bucket,gt_count,detected,rate` and would name this file
`size_sensitivity_baseline_640_smoke.csv`. Its GT bucket counts also differ from the later files
(12,004 / 15,105 / 8,718 vs 12,005 / 15,103 / 8,719) because the box-clipping in
`gt_sizes_at_input_scale` was added after it was generated. The README table (lines 21-27) is built
from this stale version.
*Fix:* regenerate it with the current script (or rename it and note the schema/clip change in the README table caption).

**S5. `EXPERIMENTS.md` line 9 and lines 30-31: "FP 5.7k->8.8k", "pedestrian->people 30->263", "motor/bicycle 134:134"**
The 5,749 FP and 30 pedestrian->people figures are the smoke run's (`analysis/failures_smoke.txt`
lines 12 and 29), not `baseline_640_2k`'s, so the "new bottleneck" deltas compare 1024 against a
5-epoch model. The 1024-run and baseline-run `analyze_failures.py` outputs that would settle this
are not in the repo.
*Fix:* run `analyze_failures.py` on both 2k checkpoints, save `analysis/failures_{run}.txt`, and restate the deltas against the true baseline.

**S6. `README.md` lines 19-27: the only size-sensitivity table in the README is the smoke model's**
Readers see 7.8 % for `<8px` while `EXPERIMENTS.md` discusses 13.6 % -> 28.4 %. Line 6 calls the smoke
model "the baseline", the name every other file reserves for `baseline_640_2k`. The chart that shows
the real comparison (`assets/size_sensitivity_comparison.png`) is tracked but referenced nowhere.
*Fix:* replace the table with the two-run chart (or a two-column table from the 2k CSVs) and say "5-epoch smoke model" wherever that number is kept.

**S7. `README.md` lines 66-76: Repository guide is stale**
Missing: `train_scale.py`, `make_subset.py`, `download_data.py`, `check_env.py`, `scripts/compare_runs.py`
(the source of the official numbers), `scripts/walkthrough.py`, `scripts/failure_cases.py`,
`scripts/size_chart.py`, `assets/`, `visdrone2k.yaml`. Also line 70 says `train.py, train_1024.py`
are "the two experiment configurations"; there are now three.
*Fix:* regenerate the table from `git ls-files`, grouped as data pipeline / training / analysis / visualisation.

**S8. `README.md`: no setup or reproduction section**
There is no "clone, create venv, install, `download_data.py`, `make_subset.py`, `train.py`" sequence
anywhere public-facing (`CLAUDE.md` has the activation line only). The data pipeline order is
implied by the code, never stated.
*Fix:* add a five-line Quick start before "The problem".

**S9. `README.md` lines 78-83: "Sections below fill in after the resolution experiment: failure cases the 1024 model still misses..."**
That analysis now exists (`analysis/failure_cases_1024.md`, `assets/failure_case_*.png`) and is not linked.
*Fix:* replace the placeholder with a three-bullet summary of the failure cases and the link.

**S10. `CLAUDE.md` Structure section (lines 24-29): stale file list**
Lists four scripts; omits `scripts/`, `analysis/`, `assets/`, `EXPERIMENTS.md`, `make_subset.py`,
`predict_compare.py`, `analyze_failures.py`, the two extra train scripts, and the `-m` requirement.
*Fix:* mirror the README repo guide once S7 is done, and add "scripts that import root modules run as `python -m scripts.<name>`".

**S11. `train_scale.py` line 3 vs line 24: "Identical to highres_1024_2k except scale" but `workers=2` (highres used 4)**
Workers do not affect results, but the docstring's single-variable claim is the whole point of the run.
*Fix:* either set `workers=4` or add "(workers=2 for host load; does not affect training)".

**S12. Explicit vs inherited `imgsz`: inconsistent between scripts**
`scripts/compare_runs.py` argues (docstring, lines 3-8) and `analysis/imgsz_eval_notes.md` §3
recommends passing `imgsz` explicitly. `analysis/size_sensitivity.py` line 101,
`analyze_failures.py` line 57, `predict_compare.py` line 48, and `scripts/walkthrough.py` line 62 all
call `predict()` with no `imgsz`, relying on checkpoint inheritance. It works today, but the
project's own notes call this fragile.
*Fix:* pass `imgsz=` in those four calls (read from `args.yaml` as `compare_runs.py` does); note only, no refactor.

**S13. Duplicated logic across scripts (note only, as requested)**
- `iou_matrix`: `analyze_failures.py:43` and `analysis/size_sensitivity.py:54` are byte-identical.
- YOLO label parsing: `draw_boxes.yolo_line_to_corners` (37), `analyze_failures.yolo_labels_to_xyxy` (28),
  `analysis/size_sensitivity.load_gt_boxes` (40): three parsers of the same five-column line.
- Greedy IoU matching loop: `analyze_failures.py:70-86`, `analysis/size_sensitivity.py:116-121`,
  `scripts/walkthrough.py:82-100`, `scripts/failure_cases.py:181-190`: four copies.
- Size-at-640: `analysis/size_sensitivity.py:66` and `scripts/failure_cases.py:99` (same formula, different names).
- `dashed_rect`: `scripts/walkthrough.py:37` and `scripts/failure_cases.py:202`.
- `CLASS_NAMES` in `draw_boxes.py:13` duplicates `visdrone2k.yaml` `names`; `CONF_THRESHOLD`/`IOU_THRESHOLD` are declared in five files.
*Fix (later):* a small `visdrone/` package (`labels.py`, `matching.py`, `viz.py`) that the scripts import.

**S14. Scripts live in three places**
Analysis tools are split between the root (`analyze_failures.py`, `predict_compare.py`, `draw_boxes.py`),
`analysis/` (`size_sensitivity.py`, the only `.py` among data files), and `scripts/`. The `-m`
requirement applies only to the `scripts/` files that import root modules, which is not guessable.
*Fix:* move `analysis/size_sensitivity.py` and the root tools into `scripts/` (or state the rule in the README).

**S15. `scripts/failure_cases.py` lines 7-8, 37: CPU is hard-coded with a situational justification**
`DEVICE = "cpu"` is honoured (verified), but the docstring reason "because the GPU is reserved for
training" describes one afternoon, not the tool. A public reader with a free GPU gets slow inference and no switch.
*Fix:* add `--device` (default `cpu`, with the comment "12 images; CPU is fine and never competes with a training run").

**S16. `predict_compare.py` lines 4-5 and `assets/compare_dense.png`, `compare_park.png`: provenance gap**
The docstring says panels "feed the README", but the README embeds nothing, and the two tracked
panels were renamed by hand from `comparisons/<image>_compare.png`, so which val images and which
weights they show is not recorded anywhere.
*Fix:* add a one-line caption per asset (image id, models) in the README or an `assets/README.md`.

**S17. `analysis/imgsz_eval_notes.md` line 71: `runs/detect/<1024 run>/weights/best.pt` placeholder**
The run has existed since 2026-08-25.
*Fix:* `highres_1024_2k`.

---

## NICE-TO-HAVE

**N1. `result.plot(font_size=...)` is a no-op in cv2 mode**: `predict_compare.py:49` passes
`font_size=0.35`, `scripts/failure_cases.py:59` passes `PLOT_FONT_SIZE = 10`. In
`ultralytics/utils/plotting.py:322-335`, `font_size` is used only when `pil=True`; the cv2 path
scales text from `line_width / 3`. Both arguments look intentional and do nothing.
*Fix:* drop the argument (or pass `pil=True` if the size matters).

**N2. Mixed line endings, no `.gitattributes`**: 22 files are CRLF, 7 newer files
(`CLAUDE.md`, `analysis/*.md`, `scripts/failure_cases.py`, `scripts/size_chart.py`) are LF.
*Fix:* add `.gitattributes` with `* text=auto` and renormalise once.

**N3. `check_env.py`**: crashes with an unhelpful traceback on a CPU-only machine (lines 7-8, 13),
no `main()` guard, and it is the one script without a docstring.
*Fix:* guard the CUDA lines with `if torch.cuda.is_available():` and add a two-line docstring.

**N4. `download_data.py`**: no docstring, no usage line, and it downloads ~2 GB via the built-in
`VisDrone.yaml` with no message about where it goes or how long it takes.
*Fix:* add a docstring stating the target path (`datasets/VisDrone`) and that `make_subset.py` runs next.

**N5. Missing type hints** (project rule: hints on all signatures): `scripts/walkthrough.py:37`
`dashed_rect(img, p1, p2, color, dash=8)` and `:49` `label(img, text: str, x: int, y: int, color)`;
`scripts/failure_cases.py:140` `top_per_sequence(items: list, key, limit: int) -> list` and `:78`
`result: object` (should be `ultralytics.engine.results.Results`).
*Fix:* annotate with `np.ndarray`, `tuple[int, int]`, `Callable[[...], int]`, and `Results`.

**N6. `scripts/failure_cases.py` lines 249, 255, 260, 295**: four lines over 100 characters (the
`"  missed by class: " + ", ".join(...)` expressions and two print/format lines).
*Fix:* build the sorted list on its own line first.

**N7. `EXPERIMENTS.md` "Epoch time" column**: `~8.5 min`, `~1:25`, `~3:20` mix units; `1:25` is
ambiguous (min:sec per epoch, or h:mm for the run?).
*Fix:* one unit, e.g. "s/epoch", and a separate "total" if wanted.

**N8. `EXPERIMENTS.md`**: the in-flight `lowscale_1024_2k` run (6 epochs so far) has no row or
"in progress" line, while its script is tracked.
*Fix:* add a placeholder row so the log and the code agree on which experiments exist.

**N9. `train.py` is named generically while its siblings are `train_1024.py` / `train_scale.py`**,
and the smoke configuration it once held (full train set, batch=8, optimizer=auto) survives only in
git history (`a249c65`).
*Fix:* rename to `train_640.py` and mention the smoke commit in the `EXPERIMENTS.md` smoke row.

**N10. `draw_boxes.py:24` default image lives in `datasets/VisDrone` (full set)** while every other
script targets `VisDrone-2k`; fine once both exist, surprising otherwise.
*Fix:* point the default at a `VisDrone-2k` val image.

**N11. `assets/` weighs 5.6 MB** (three failure-case PNGs are 4.2 MB, `compare_dense.png` 1.2 MB) for
a repo whose code is ~60 KB. Acceptable, but every future regeneration adds the same again to history.
*Fix:* save the failure cases as JPEG quality 90, or downscale to 1024 px wide.

**N12. `analysis/failure_cases_1024.md`**: the overlay legend is defined in the note but not on the
images themselves; a reader who opens `assets/failure_case_1.png` from the file browser sees
unexplained orange and magenta boxes.
*Fix:* burn a three-line legend strip into the top-left of each PNG in `render()`.

---

## Executive summary

1. The numbers the public will check first are partly wrong: `EXPERIMENTS.md` says the 1024 run early-stopped at epoch 72 with best epoch 47 and that the console read 0.322; `results.csv` shows a full 80 epochs, best epoch 79 at 0.370, and no 0.322 anywhere. tricycle "+0.142" is +0.076, and "workers=2" is workers=4.
2. `README.md` still says the resolution experiment is "in progress" with results "landing here", while the log, chart, comparison table, and failure cases all exist and are unreferenced; two dataset facts (2.56x pixels, 457K boxes) are also off.
3. `requirements.txt` is UTF-16 and pins `+cu130` wheels that PyPI does not serve, so the one-line install a visitor will try does not work; there is no setup section to tell them otherwise.
4. Every usage line in the repo runs as written today (`python -m` where needed); the remaining doc debt is stale defaults, one Hebrew comment, a doubled `Usage:`, and README/CLAUDE.md file lists that predate half the scripts.
5. Nothing sensitive is tracked. The code duplication (four matching loops, three label parsers) is real but cosmetic for a learning repo; note it in the README as "known", fix it after the docs.
