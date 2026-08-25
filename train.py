"""Train YOLO11n on the VisDrone 2k subset.

Optimizer and LR are pinned: with optimizer="auto", Ultralytics silently
switches between AdamW and SGD based on total iteration count, which would
make runs of different lengths incomparable.

Usage:
    python train.py
"""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")  # COCO-pretrained weights (transfer learning)

    model.train(
        data="visdrone2k.yaml",
        epochs=80,
        imgsz=640,
        batch=16,               # smoke run peaked at ~4.4/8GB with batch=8
        workers=4,
        device=0,
        optimizer="SGD",        # pinned for cross-experiment comparability
        lr0=0.01,
        patience=25,            # stop early if val mAP stalls
        name="baseline_640_2k",
    )


if __name__ == "__main__":  # required on Windows: DataLoader workers spawn processes
    main()