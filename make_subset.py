"""Create a reproducible random subset of the VisDrone train split.

Keeps val and test untouched: they stay full-size so metrics remain
comparable across experiments. Only training time is being traded.

Usage:
    python make_subset.py
"""
import random
import shutil
from pathlib import Path

SOURCE = Path("datasets/VisDrone")
TARGET = Path("datasets/VisDrone-2k")
TRAIN_SUBSET_SIZE = 2000
SEED = 42  # fixed for reproducibility


def copy_split(split: str, image_names: list[str]) -> None:
    for sub in ("images", "labels"):
        (TARGET / sub / split).mkdir(parents=True, exist_ok=True)
    for name in image_names:
        stem = Path(name).stem
        shutil.copy2(SOURCE / "images" / split / name,
                     TARGET / "images" / split / name)
        label = SOURCE / "labels" / split / f"{stem}.txt"
        if label.exists():
            shutil.copy2(label, TARGET / "labels" / split / f"{stem}.txt")


def main() -> None:
    train_images = sorted(p.name for p in (SOURCE / "images" / "train").iterdir())
    random.seed(SEED)
    subset = random.sample(train_images, TRAIN_SUBSET_SIZE)

    print(f"Sampling {TRAIN_SUBSET_SIZE}/{len(train_images)} train images (seed={SEED})")
    copy_split("train", subset)

    for split in ("val", "test"):
        names = sorted(p.name for p in (SOURCE / "images" / split).iterdir())
        print(f"Copying full {split}: {len(names)} images")
        copy_split(split, names)

    print(f"Done: {TARGET}")


if __name__ == "__main__":
    main()