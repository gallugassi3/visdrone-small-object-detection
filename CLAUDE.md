# visdrone-small-object-detection

## Purpose
Learning project for computer-vision interview prep (YOLO11n on VisDrone).
Default mode is **explain and review**: analyze code, results, and concepts.
Do not write or add features unless explicitly asked.

## Commands
- Activate env (required first, per terminal): `.\.venv\Scripts\Activate.ps1`
- Verify env: `python check_env.py`
- Train: `python train.py`
- Visualize ground-truth labels: `python draw_boxes.py [image_path]`

## Code style
- English comments only.
- Comments explain *why*, not *what*.
- Type hints on all function signatures.
- Named constants instead of magic numbers.

## Git
Git is handled manually by the user. **Never commit, push, stage, or otherwise
modify git state.**

## Structure
- `train.py`, `download_data.py`, `draw_boxes.py`, `check_env.py` — project scripts
- `datasets/` — VisDrone data (gitignored artifact)
- `runs/` — Ultralytics training outputs (gitignored artifact)
- `weights/`, `*.pt` — model weights (gitignored)
- Hardware: RTX 3070 Laptop 8GB VRAM, Windows — batch size and imgsz are VRAM-constrained