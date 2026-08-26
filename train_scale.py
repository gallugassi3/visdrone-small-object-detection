"""Scale-augmentation experiment: does scale=0.5 waste the resolution gain?

Identical to highres_1024_2k except scale augmentation reduced 0.5 -> 0.2.
Hypothesis: random downscaling by up to 2x pushes already-tiny objects below
the assignment threshold, so on tiny-object data the default is destructive;
gentler scaling should preserve more of the <8px bucket during training.
A null result is also informative: it would mean mosaic/scale interplay
already compensates.

Usage:
    python train_scale.py
"""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")

    model.train(
        data="visdrone2k.yaml",
        epochs=80,
        imgsz=1024,
        batch=6,
        workers=2,
        device=0,
        optimizer="SGD",        # pinned for cross-experiment comparability
        lr0=0.01,
        patience=25,
        scale=0.2,              # the single changed variable (default 0.5)
        name="lowscale_1024_2k",
    )


if __name__ == "__main__":  # required on Windows: DataLoader workers spawn processes
    main()