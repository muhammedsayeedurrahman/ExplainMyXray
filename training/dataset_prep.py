"""
Dataset Preprocessing Pipeline for ExplainMyXray.

Prepares BIMCV PadChest chest X-ray images for MedGemma fine-tuning:
  1. 16-bit to 8-bit conversion with percentile normalisation
  2. Auto-crop of dark scanner edges
  3. Square padding
  4. Resize to 512x512
  5. CLAHE contrast enhancement
  6. Mild sharpening
  7. Conversion to RGB PNG

Usage:
    python training/dataset_prep.py \
        --csv  /path/to/padchest_labels.csv \
        --images /path/to/raw/images \
        --output /path/to/processed
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageEnhance
from tqdm import tqdm

IMAGE_SIZE = 512


def _convert_16bit_to_8bit(img):
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 - p1 < 1:
        p1, p99 = arr.min(), arr.max()
    if p99 - p1 < 1:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)


def _auto_crop_dark_edges(gray, threshold_ratio=0.05):
    h, w = gray.shape
    if h < 100 or w < 100:
        return gray
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    center_mean = center.mean()
    if center_mean < 10:
        return gray
    threshold = center_mean * threshold_ratio
    top, bottom, left, right = 0, h, 0, w
    for row in range(h // 6):
        if gray[row, w // 4 : 3 * w // 4].mean() < threshold:
            top = row + 1
        else:
            break
    for row in range(h - 1, h - h // 6, -1):
        if gray[row, w // 4 : 3 * w // 4].mean() < threshold:
            bottom = row
        else:
            break
    for col in range(w // 6):
        if gray[h // 4 : 3 * h // 4, col].mean() < threshold:
            left = col + 1
        else:
            break
    for col in range(w - 1, w - w // 6, -1):
        if gray[h // 4 : 3 * h // 4, col].mean() < threshold:
            right = col
        else:
            break
    if (bottom - top) < h * 0.6 or (right - left) < w * 0.6:
        return gray
    return gray[top:bottom, left:right]


def _pad_to_square(gray, pad_value=0):
    h, w = gray.shape
    if h == w:
        return gray
    t = max(h, w)
    padded = np.full((t, t), pad_value, dtype=gray.dtype)
    padded[(t - h) // 2 : (t - h) // 2 + h, (t - w) // 2 : (t - w) // 2 + w] = gray
    return padded


def _process_one(args):
    img_name, img_dir, images_root, out_path = args
    if os.path.exists(out_path):
        return "exist"
    in_path = os.path.join(images_root, str(img_dir), img_name)
    if not os.path.exists(in_path):
        return "missing"
    try:
        img = Image.open(in_path)
        if img.mode in ("I;16", "I", "I;16B", "I;16L"):
            gray = _convert_16bit_to_8bit(img)
        elif img.mode == "L":
            gray = np.array(img, dtype=np.uint8)
        elif img.mode in ("RGB", "RGBA"):
            gray = np.array(img.convert("L"), dtype=np.uint8)
        else:
            gray = _convert_16bit_to_8bit(img)
        gray = _auto_crop_dark_edges(gray)
        gray = _pad_to_square(gray)
        img_pil = Image.fromarray(gray, mode="L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        gray = np.array(img_pil, dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img_pil = Image.fromarray(gray, mode="L")
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.2)
        final_img = img_pil.convert("RGB")
        final_img.save(out_path, format="PNG", optimize=False, compress_level=0)
        return "done"
    except Exception:
        return "error"


def main():
    parser = argparse.ArgumentParser(description="Preprocess PadChest X-rays for training")
    parser.add_argument("--csv", required=True, help="Path to PadChest labels CSV")
    parser.add_argument("--images", required=True, help="Root directory of raw images")
    parser.add_argument("--output", required=True, help="Output directory for processed images")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.output, exist_ok=True)

    tasks = []
    for _, row in df.iterrows():
        out_path = os.path.join(args.output, row["ImageID"])
        tasks.append((row["ImageID"], row["ImageDir"], args.images, out_path))

    print(f"Processing {len(tasks)} images with {args.workers} workers...")
    results = {"done": 0, "exist": 0, "missing": 0, "error": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for res in tqdm(executor.map(_process_one, tasks), total=len(tasks)):
            results[res] += 1

    print(f"\nDone: {results['done']}  Existed: {results['exist']}  "
          f"Missing: {results['missing']}  Errors: {results['error']}")


if __name__ == "__main__":
    main()
