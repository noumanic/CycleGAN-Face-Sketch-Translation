"""
predict.py — CLI Inference Tool
================================
Translate a single image or a whole folder without running Flask.

Usage
-----
  # Single image (auto-detect direction)
  python predict.py --input face.jpg --checkpoint checkpoints/latest.pth

  # Force direction
  python predict.py --input sketch.png --direction sketch2face

  # Batch folder
  python predict.py --input ./testA --output ./results --direction face2sketch

  # Side-by-side comparison grid
  python predict.py --input face.jpg --compare
"""

import argparse
import os
from pathlib import Path

from PIL import Image
import numpy as np

from inference import load_generators, translate_image, detect_domain, pil_to_bytes

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_comparison(original: Image.Image, translated: Image.Image,
                    direction: str, out_path: str):
    """Save a side-by-side original | translated image."""
    w, h = 256, 256
    original   = original.resize((w, h), Image.BICUBIC)
    translated = translated.resize((w, h), Image.BICUBIC)

    # Labels
    from PIL import ImageDraw, ImageFont
    grid = Image.new("RGB", (w * 2 + 10, h + 30), (245, 240, 232))
    grid.paste(original,   (0,        30))
    grid.paste(translated, (w + 10,   30))

    draw = ImageDraw.Draw(grid)
    lbl_l = "Input"
    lbl_r = "→ Sketch" if "face2sketch" in direction else "→ Face"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    draw.text((5,    5), lbl_l, fill=(40, 30, 20), font=font)
    draw.text((w+15, 5), lbl_r, fill=(40, 30, 20), font=font)
    grid.save(out_path)


def process_single(img_path: str, out_path: str, direction: str,
                   compare: bool) -> str:
    img = Image.open(img_path).convert("RGB")
    detected = detect_domain(img)
    result, actual_dir = translate_image(img, direction)

    if compare:
        save_comparison(img, result, actual_dir, out_path)
    else:
        result.save(out_path)

    return actual_dir, detected


def main():
    p = argparse.ArgumentParser(description="CycleGAN CLI inference")
    p.add_argument("--input",      required=True,
                   help="Input image file or folder")
    p.add_argument("--output",     default=None,
                   help="Output file or folder (default: ./predictions)")
    p.add_argument("--checkpoint", default="./checkpoints/latest.pth",
                   help="Path to trained checkpoint")
    p.add_argument("--direction",  default="auto",
                   choices=["auto", "face2sketch", "sketch2face"],
                   help="Translation direction (default: auto-detect)")
    p.add_argument("--compare",    action="store_true",
                   help="Save side-by-side input/output comparison")
    p.add_argument("--device",     default=None,
                   help="Device override: cuda | cpu (default: auto)")
    args = p.parse_args()

    # ---- Load model ----
    if not os.path.isfile(args.checkpoint):
        print(f"[predict] ERROR: Checkpoint not found: {args.checkpoint}")
        print("          Train first:  python local_train.py")
        return

    load_generators(args.checkpoint, device=args.device)

    src = Path(args.input)

    # ---- Single image ----
    if src.is_file():
        if args.output:
            out_path = args.output
        else:
            suffix = "_compare.png" if args.compare else "_translated.png"
            out_path = str(src.with_name(src.stem + suffix))

        direction, detected = process_single(
            str(src), out_path, args.direction, args.compare
        )
        print(f"[predict] {src.name}")
        print(f"          Detected domain : {detected}")
        print(f"          Direction       : {direction}")
        print(f"          Saved to        : {out_path}")
        return

    # ---- Batch folder ----
    if src.is_dir():
        out_dir = Path(args.output or "./predictions")
        out_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in sorted(src.iterdir())
                  if p.suffix.lower() in IMG_EXT]

        if not images:
            print(f"[predict] No images found in {src}")
            return

        print(f"[predict] Processing {len(images)} images → {out_dir}")
        ok = fail = 0

        for img_path in images:
            try:
                suffix = "_compare.png" if args.compare else "_translated.png"
                out_path = str(out_dir / (img_path.stem + suffix))
                direction, detected = process_single(
                    str(img_path), out_path, args.direction, args.compare
                )
                print(f"  ✓  {img_path.name:40s}  [{detected} → {direction}]")
                ok += 1
            except Exception as exc:
                print(f"  ✗  {img_path.name:40s}  ERROR: {exc}")
                fail += 1

        print(f"\n[predict] Done — {ok} succeeded, {fail} failed")
        print(f"          Output: {out_dir}")
        return

    print(f"[predict] ERROR: {src} is not a file or directory.")


if __name__ == "__main__":
    main()
