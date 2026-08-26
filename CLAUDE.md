# Project context for Claude Code

Small-object detection on VisDrone with YOLO11n. The project is a controlled resolution
experiment (640 vs 1024, +43% mAP50) plus a scale-augmentation follow-up. Owner: Gal.

## Working rules

- **Report-only by default.** Analysis and review tasks write findings to `analysis/`;
  never modify code or docs unless the task explicitly says so. Never commit; git is
  handled manually by the owner.
- **English only** in all files, comments, and reports. Comments explain *why*, not *what*.
- **Numbers come from source files, never from memory or chat**: results.csv, args.yaml,
  and `analysis/compare_*.txt` (the official comparison; its delta column is computed from
  unrounded values). This rule caught three drift bugs already.
- **The GPU may be busy training.** Any inference for analysis runs with `device='cpu'`
  unless told otherwise. Note: on Windows, `CUDA_VISIBLE_DEVICES=""` deletes the variable
  instead of hiding the GPU; use `device` arguments.
- Style: type hints, constants over magic numbers, fail-loud guards on parsed file formats.

## Repository map

- Root: experiment configs (`train*.py`), data tooling (`download_data.py`, `make_subset.py`),
  standalone analysis (`analyze_failures.py`, `draw_boxes.py`, `predict_compare.py`, `check_env.py`)
- `scripts/`: run as `python -m scripts.<name>` when they import from root
  (compare_runs, walkthrough, size_chart, delta_chart, failure_cases)
- `analysis/`: measurement outputs (CSVs, compare files), methodology notes, reviews
- `assets/`: README figures (committed); `runs/`, `datasets/`, `comparisons/` are gitignored
- `EXPERIMENTS.md`: the run log; one row per experiment, conclusions drive the next one

## Known technical debt (deliberate)

- The IoU-matching loop and label parsers are duplicated across four tools, verified
  identical by adversarial tests (`analysis/logic_review.md`); not refactored before
  publication to avoid re-validating all tools. Greedy matching is a documented lower
  bound consistent with COCO-style protocols.