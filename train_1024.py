"""Resolution experiment: YOLO11n on the VisDrone 2k subset at imgsz=1024.

Identical to baseline_640_2k except imgsz (and batch, reduced to fit VRAM).
Tests the central hypothesis: the dominant failure mode at 640 is
non-detection of sub-stride objects, so higher input resolution should
lift the recall ceiling for small classes (pedestrian, people, motor).

Usage:
    python train_1024.py
"""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")  # same COCO-pretrained starting point as baseline

    model.train(
        data="visdrone2k.yaml",
        epochs=80,
        imgsz=1024,
        batch=6,                # 1024px activations ~2.5x larger than 640; conservative for 8GB
        workers=4,
        device=0,
        optimizer="SGD",        # pinned for cross-experiment comparability
        lr0=0.01,
        patience=25,
        name="highres_1024_2k",
    )


if __name__ == "__main__":  # required on Windows: DataLoader workers spawn processes
    main()