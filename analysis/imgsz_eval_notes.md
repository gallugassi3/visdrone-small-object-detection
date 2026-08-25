# Validation resolution: how Ultralytics resizes val images and whether `imgsz` is inherited

Source: code inspection of the installed package (`ultralytics 8.4.128`, `.venv/Lib/site-packages/ultralytics/`)
plus an empirical dataset build on `datasets/VisDrone-2k/images/val`. Written 2026-08-25.

## 1. Does `model.val()` default to the training `imgsz`? Yes.

- `YOLO("best.pt")` loads `ckpt["train_args"]` and keeps only
  `{"imgsz", "data", "task", "single_cls"}` as `self.overrides`
  (`engine/model.py:251`, `_reset_ckpt_args` at `engine/model.py:1088`).
- `Model.val()` builds its args as `{**self.overrides, "rect": True, **kwargs, "mode": "val"}`
  (`engine/model.py:608-609`). Explicit kwargs win; otherwise `imgsz` comes from the checkpoint.
- `best.pt` retains `train_args` — `strip_optimizer` rewrites it but keeps all default-config keys
  (`utils/torch_utils.py:833-837`).
- Confirmed: `YOLO("runs/detect/baseline_640_smoke/weights/best.pt").overrides` →
  `{'task': 'detect', 'data': '...VisDrone.yaml', 'imgsz': 640, 'single_cls': False, ...}`.
- The CLI (`yolo val model=x.pt`) goes through the same `Model` class → same behaviour.
- Validation during training and the end-of-training `final_eval` use the trainer's own args
  (`imgsz` = training value) with `rect=True` (`models/yolo/detect/train.py:76`,
  `engine/trainer.py:941`). So the mAP printed at the end of a run is already "val at own resolution".

**Caveats**
- `data` is also inherited. The smoke checkpoint stores the *absolute path of the built-in*
  `VisDrone.yaml` (full dataset), not `visdrone2k.yaml`. Checkpoints trained via `train.py`
  store the string `"visdrone2k.yaml"`, which resolves relative to the cwd.
  → Always pass `data="visdrone2k.yaml"` explicitly and run from the project root.
- Everything else (`batch`, `conf`, `iou`, `device`, ...) reverts to defaults; val uses
  `conf=0.001`, NMS `iou=0.7`, `max_det=300`.

## 2. How val images are resized at `imgsz=1024` (rect letterbox)

Pipeline for a `.pt` model (`rect` is forced True by `Model.val()`; the validator only
disables it for exported non-dynamic formats, `engine/validator.py:232-233`):

1. **Aspect-preserving resize of the long side to `imgsz`** — `data/base.py:load_image`, `rect_mode=True`:
   `r = imgsz / max(H, W)`; if `r != 1` resize to `(min(ceil(W·r), imgsz), min(ceil(H·r), imgsz))`
   with `cv2.INTER_LINEAR`. **Note: this upscales images whose long side is below `imgsz`** —
   there is no scale-up guard at this stage.
2. **Per-batch rectangular target shape** — `data/base.py:set_rectangle` (386-409). Images are
   sorted by aspect ratio; each batch gets
   `batch_shape = ceil(ar_shape · imgsz / stride + pad) · stride` with `pad = 0.5` for val
   (`data/build.py:249`) and `stride = 32`. The `+0.5` adds up to half a stride of extra border.
3. **Letterbox pad to that shape** — `LetterBox(new_shape=(imgsz, imgsz), scaleup=False)`
   (`data/dataset.py:318`), which pops `rect_shape` from the label and uses it instead of the square
   (`data/augment.py`, `__call__`). Because step 1 already resized, `r` clamps to 1.0 → this step is
   **pure padding**, centered, gray 114. No stretching (`scale_fill=False`), no `auto` minimal-rect here.
4. Predictions are mapped back to original-image coordinates via `ratio_pad` and scored against the
   original-resolution labels — so mAP@640 and mAP@1024 are evaluated on identical GT.

Training (`rect=False`) instead builds square `imgsz × imgsz` mosaics, but step 1 is the same
`load_image` call, so the *object scale* seen at val (`imgsz / max(H, W)`) equals the training
scale before `scale=0.5` jitter. Val is not "out of distribution" in scale.

### Concrete numbers, `VisDrone-2k` val (548 images, all 16:9)

| original W×H (count)  | @640 after step 1 | @640 tensor | @1024 after step 1 | @1024 tensor | object scale @640 / @1024 |
|-----------------------|-------------------|-------------|--------------------|--------------|---------------------------|
| 1360×765 (408)        | 640×360           | 672×384     | 1024×576           | 1056×608     | 0.471 / 0.753             |
| 1920×1080 (19)        | 640×360           | 672×384     | 1024×576           | 1056×608     | 0.333 / 0.533             |
| 960×540 (121)         | 640×360 (down)    | 672×384     | 1024×576 (**up 1.07×**) | 1056×608 | 0.667 / 1.067          |

Empirically: one batch shape per resolution (`(384, 672)` at 640, `(608, 1056)` at 1024), 35 batches
at batch=16; sample `ratio_pad` = `((0.333, 0.333), (16, 12))` at 640 and `((0.533, 0.533), (16, 16))`
at 1024. The val tensor is **not** `imgsz × imgsz`: it is one stride wider than `imgsz` (from `pad=0.5`)
and much shorter, so 1024-val costs ~0.6× a square 1024 forward pass.

## 3. Checklist for the 640-vs-1024 comparison

```python
YOLO("runs/detect/baseline_640_2k/weights/best.pt").val(data="visdrone2k.yaml", imgsz=640)
YOLO("runs/detect/<1024 run>/weights/best.pt").val(data="visdrone2k.yaml", imgsz=1024)
```

- Passing `imgsz` explicitly is redundant but makes the protocol visible in the code.
- Pass `data=` explicitly (see caveat above) and run from the project root.
- Both runs' end-of-training mAP already equals `val()` at own resolution — the numbers should match
  if the same `best.pt` and `data` are used.
- Do not compare `val()` mAP against `analysis/size_sensitivity.py` rates: val uses `conf=0.001`,
  class-aware matching, and COCO-style AP; the script uses `conf=0.25`, class-agnostic, recall only.
- Cross-resolution check (640-model at 1024 or vice versa) is a separate experiment; it changes
  object scale relative to training and must be labeled as such.
