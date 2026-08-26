# Is `flipud=0.5` defensible for VisDrone-2k?

Written 2026-08-25. Method: 100 images sampled from `datasets/VisDrone-2k/images/train`
(2000 images, `random.seed(42)`), tiled into contact sheets and classified by eye.
Per-image labels: `analysis/flipud_sample_labels.csv`.

## Classification

| view                | definition                                                        | count |
|---------------------|-------------------------------------------------------------------|------:|
| near-nadir          | top-down; no horizon; facades at most lean radially from center   |    49 |
| steep oblique       | no horizon or sky, but facades and a clear far-at-top perspective |    30 |
| oblique w/ horizon  | sky, skyline, or hills visible                                    |    21 |

Under the two-class rule from the earlier discussion (oblique = horizon or facades visible):
nadir 49 : oblique 51. The nadir/steep-oblique boundary is subjective; ±10 images either way
would not change the conclusion.

Caveats
- The 100 images come from 48 sequences; viewpoint is mostly constant within a sequence
  (12 sequences had mixed labels, typically nadir vs steep-oblique). The effective sample is
  closer to 48 than 100.
- A sky-fraction heuristic (bright/low-saturation or bluish pixels in the top 15 %) found
  17 of the 21 horizon images but had 21 false alarms on night scenes and bright pavement,
  not reliable enough to automate the split without a better feature.

## Why this matters for a vertical flip

A vertical flip is a valid augmentation only if the flipped image is a plausible test image.

- **Near-nadir (49 %)**: valid. The drone heading is arbitrary, so a top-down scene flipped
  vertically is just a different heading. Radially leaning facades are flip-symmetric.
- **Steep oblique (30 %)**: invalid. After a flip, the perspective gradient is inverted
  (objects grow toward the top, facades point the wrong way, pedestrians are upside-down).
  No real frame looks like this.
- **Oblique with horizon (21 %)**: invalid, sky at the bottom.

So `flipud=0.5` would produce an impossible image for roughly half of all training samples
(mosaic tiles are flipped independently, so most mosaics would contain at least one).
The cost is not just wasted capacity: the `pedestrian` vs `people` distinction rests partly on
posture, and both are drawn upright in every oblique frame; flipping half of them removes a
cue the model would otherwise learn. Pedestrian/people are 31 % of all boxes.

## Conclusion

**Not defensible as a default at 0.5 for this dataset.** The reasoning that makes `flipud`
harmless, "drone footage has no canonical up", holds for only about half of VisDrone-2k.
This is the same reason `degrees` (rotation) stays at 0.

If it is tested at all, treat it as an ablation, not a baseline setting:
- use a low probability (0.1-0.2) rather than 0.5, so the impossible-image rate stays small;
- judge it on per-class AP for `pedestrian` and `people`, the most orientation-sensitive
  classes, not on overall mAP alone;
- the principled version, flipping only nadir sequences, would need a per-sequence viewpoint
  label, which Ultralytics' augmentation pipeline does not support without custom code.

No training scripts were modified.
