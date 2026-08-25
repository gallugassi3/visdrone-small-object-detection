"""Train YOLO11n on the VisDrone dataset.

Usage:
    python train.py
"""
from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")  # COCO-pretrained weights (transfer learning)

    model.train(
        data="VisDrone.yaml",
        epochs=5,               # smoke run: validate the pipeline, measure epoch time
        imgsz=640,
        batch=8,                # conservative for 8GB VRAM on a laptop GPU
        workers=4,
        device=0,
        name="baseline_640_smoke",
    )


if __name__ == "__main__":  # required on Windows: DataLoader workers spawn processes
    main()