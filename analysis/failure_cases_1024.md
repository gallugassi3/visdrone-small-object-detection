# Where the 1024 model still fails

Written 2026-08-26. Model: `runs/detect/highres_1024_2k/weights/best.pt` at imgsz 1024,
conf 0.25, greedy IoU>=0.5 matching (class checked after the match, as in `analyze_failures.py`).
Inference on CPU (GPU was training). Produced by `python -m scripts.failure_cases`.

Selection: all 548 val label files were ranked, without inference, by number of `<8px` GT boxes
(size = sqrt(area) at the 640 scale, same definition as the size-sensitivity tables), by bicycle
count, and by awning-tricycle count. The top 4 per list, one image per video sequence, went
through the model (12 images); the three below are the strongest example of each failure story
and come from three different sequences.

Reading the images (`assets/failure_case_*.png`): solid boxes with a class and confidence are
the model's predictions (`result.plot()`). **Orange dashed** = GT box nothing matched (miss).
**Magenta box with `GT:<class>`** = matched, but with the wrong class.

## 1. `failure_case_1.png` — small objects: the remaining bottleneck

`0000295_01000_d_0000026.jpg` (1360x765). 251 GT boxes, 170 of them `<8px` at the 640 scale
(that is, under ~17 px on the original image, ~13 px on the 1024 network input).

- 108 predictions, **175 misses, 147 of which are `<8px`**. 3 misclassified, 32 false positives.
- Misses by class: car 68, pedestrian 62, bicycle 21, people 11, motor 6, van 4, bus 3.
- The orange dashes pile up past the overpass, where traffic and pedestrians recede toward the
  vanishing point, and along the far left sidewalk. The model is confident on everything in the
  near half of the frame (cars at 0.85-0.91) and blind past the overpass.
- The shared-bike rack on the right sidewalk holds most of the 24 bicycles; the model picks out 2
  and misses 21. Packed, overlapping bicycles look like texture at this scale.

This is the `<8px` bucket from the size-sensitivity table (28.4 % detection rate) in one picture.
Going 640 -> 1024 doubled that rate, but the objects here are still 1-2 cells wide at stride 8,
so more resolution, a P2 head, or tiling is the lever, not more epochs.

## 2. `failure_case_2.png` — awning-tricycle: zero for eighteen

`0000215_00909_d_0000258.jpg` (1360x765). 93 GT boxes, only 16 tiny, but 18 awning-tricycles
parked in the lot at the top of the frame.

- 85 predictions, 30 misses (7 tiny), 8 misclassified, 22 false positives.
- **Awning-tricycle: 18 GT, 15 missed, 3 misclassified (2 -> motor, 1 -> car), 0 correct.**
  The objects are not small (they are larger than the pedestrians the model does find next to
  them); the class is simply not learned. Cars in the same lot are found at 0.8+.
- Also visible: `van -> car` twice, the most common confusion in the whole val set.

awning-tricycle has the lowest per-class mAP50 of the run, 0.116 (0.058 at 640). It looks like a
motor/tricycle with a canopy; at 0.25 conf the model either suppresses it or calls it the
nearest common class.

## 3. `failure_case_3.png` — wrong classes on the vehicle family

`0000244_01500_d_0000004.jpg` (960x540, the smallest source image of the three). 96 GT boxes,
14 awning-tricycles, and a row of parked tricycles bottom-left.

- 68 predictions, 44 misses, **17 misclassified** (the highest of the 12 shortlisted), 16 false
  positives.
- Confusions: awning-tricycle -> car 3, tricycle -> car 3, truck -> car 2, van -> truck 2,
  van -> car 2, bus -> truck 2, awning-tricycle -> van 1, car -> van 1, car -> awning-tricycle 1.
  Every confusion stays inside the vehicle family; localisation is right, the label is wrong.
- Misses are dominated by the parked tricycles (19 of 44) in the bottom-left cluster, where the
  dashed boxes overlap each other; awning-tricycle goes 3 correct / 4 wrong class / 7 missed.
- Down the street, `car` is the default answer: a truck and several three-wheelers carry a
  `car 0.5-0.7` label with a magenta `GT:` tag underneath.

Localisation succeeds here; the failure is fine-grained classification between vehicle types
that share a viewpoint and silhouette from above. That is a different fix from case 1 (class
balance, `cls` loss weight, or more data for the 3-wheeler classes), not resolution.

## Caveats

- Matching is class-agnostic greedy by confidence, so a box that overlaps two GT objects is
  credited to the higher-IoU one; "missed" counts in dense clusters are slightly pessimistic.
- 0.25 conf is the same threshold as every other analysis here; some misses would appear as
  low-confidence predictions at 0.1.
- The `0000295` sequence supplied 4 of the top-5 tiny-object images. Only one is shown, but the
  full-val `<8px` rate is driven by a handful of such dense far-view sequences.
- No training scripts were modified.
