"""
setup_dataset.py
=================
Organises the Person Face Sketches dataset into the
trainA / trainB / testA / testB layout that CycleGAN expects.

Detects and handles multiple layouts automatically, including
the exact Kaggle layout you have:

    data/
      train/
        photos/      <- trainA
        sketches/    <- trainB
      test/
        photos/      <- testA
      val/
        photos/      <- also merged into testA
        sketches/    <- also merged into testB

Usage
-----
  python setup_dataset.py --src ./data
  python setup_dataset.py --src ./data --dst ./dataset
"""

import argparse
import os
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def copy_images(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            # Avoid filename collisions when merging multiple source folders
            target = dst / p.name
            if target.exists():
                target = dst / f"{p.stem}_{p.parent.name}{p.suffix}"
            shutil.copy(str(p), str(target))
            copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./data",
                        help="Root folder of your downloaded dataset (default: ./data)")
    parser.add_argument("--dst", default="./dataset",
                        help="Output folder for organised dataset (default: ./dataset)")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] Source folder not found: {src}")
        print(f"        Make sure your dataset is at: {src.resolve()}")
        return

    print(f"\n{'='*55}")
    print(f"  CycleGAN Dataset Setup")
    print(f"  Source : {src.resolve()}")
    print(f"  Output : {dst.resolve()}")
    print(f"{'='*55}\n")

    # Detect layout
    # Layout A: data/train/photos  data/train/sketches  (your layout)
    train_photos   = src / "train"  / "photos"
    train_sketches = src / "train"  / "sketches"
    test_photos    = src / "test"   / "photos"
    test_sketches  = src / "test"   / "sketches"
    val_photos     = src / "val"    / "photos"
    val_sketches   = src / "val"    / "sketches"

    if train_photos.exists() and train_sketches.exists():
        print("[setup] Detected layout: data/train/photos + data/train/sketches")
        print("[setup] Also merging val/ into test sets if present.\n")

        # trainA  ←  train/photos
        print(f"  Copying trainA  ←  {train_photos}")
        n = copy_images(train_photos, dst / "trainA")
        print(f"           {n:,} images copied")

        # trainB  ←  train/sketches
        print(f"  Copying trainB  ←  {train_sketches}")
        n = copy_images(train_sketches, dst / "trainB")
        print(f"           {n:,} images copied")

        # testA   ←  test/photos  [+ val/photos]
        print(f"  Copying testA   ←  {test_photos}", end="")
        n = copy_images(test_photos, dst / "testA") if test_photos.exists() else 0
        if val_photos.exists():
            print(f"  +  {val_photos}", end="")
            n += copy_images(val_photos, dst / "testA")
        print(f"\n           {n:,} images copied")

        # testB   ←  test/sketches  [+ val/sketches]
        print(f"  Copying testB   ←  {test_sketches}", end="")
        n = copy_images(test_sketches, dst / "testB") if test_sketches.exists() else 0
        if val_sketches.exists():
            print(f"  +  {val_sketches}", end="")
            n += copy_images(val_sketches, dst / "testB")
        print(f"\n           {n:,} images copied")

    # Layout B: already trainA/trainB
    elif (src / "trainA").exists() and (src / "trainB").exists():
        print("[setup] Dataset already in trainA/trainB layout.")
        if src.resolve() != dst.resolve():
            print(f"[setup] Copying to {dst} …")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))
        print("[setup] Nothing to reorganise.")

    # Layout C: flat photos/ + sketches/
    elif (src / "photos").exists() and (src / "sketches").exists():
        import random
        print("[setup] Detected layout: photos/ + sketches/ (flat)")
        TEST_RATIO = 0.10
        random.seed(42)

        def split_copy(folder, train_dst, test_dst):
            imgs = sorted([p for p in folder.rglob("*")
                           if p.suffix.lower() in IMG_EXT])
            random.shuffle(imgs)
            n_test = max(1, int(len(imgs) * TEST_RATIO))
            for p in imgs[n_test:]: shutil.copy(str(p), train_dst)
            for p in imgs[:n_test]: shutil.copy(str(p), test_dst)
            return len(imgs) - n_test, n_test

        (dst / "trainA").mkdir(parents=True, exist_ok=True)
        (dst / "trainB").mkdir(parents=True, exist_ok=True)
        (dst / "testA").mkdir(parents=True, exist_ok=True)
        (dst / "testB").mkdir(parents=True, exist_ok=True)

        tr, te = split_copy(src/"photos",   dst/"trainA", dst/"testA")
        print(f"  trainA: {tr:,}   testA: {te}")
        tr, te = split_copy(src/"sketches", dst/"trainB", dst/"testB")
        print(f"  trainB: {tr:,}   testB: {te}")

    else:
        print("[ERROR] Could not detect dataset layout.")
        print(f"        Folders found inside {src}:")
        for d in sorted(src.iterdir()):
            if d.is_dir():
                print(f"          {d.name}/")
        print("\n        Expected one of:")
        print("          data/train/photos  +  data/train/sketches")
        print("          data/photos        +  data/sketches")
        print("          data/trainA        +  data/trainB")
        return

    # Final summary
    print(f"\n{'='*55}")
    print(f"  Dataset ready at: {dst.resolve()}")
    print(f"{'='*55}")
    for split in ["trainA", "trainB", "testA", "testB"]:
        folder = dst / split
        if folder.exists():
            n = count_images(folder)
            print(f"  {split:8s}  {n:>6,} images")

    print(f"\n  Ready to train:")
    print(f"  python local_train.py")
    print(f"  -- or --")
    print(f"  python train.py --data_root {dst}\n")


if __name__ == "__main__":
    main()