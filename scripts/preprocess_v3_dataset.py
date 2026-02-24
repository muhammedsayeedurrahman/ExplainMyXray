"""
PadChest Image Preprocessing — Original Script.

NOTE: This is the original development script preserved for reference.
For the clean, CLI-driven version, see training/dataset_prep.py.

Preprocesses PadChest X-ray images: 16-bit to 8-bit conversion,
auto-crop dark edges, pad to square, resize, CLAHE, and sharpen.
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# CONFIG — Update these paths for your environment
INPUT_CSV = os.environ.get("PADCHEST_CSV", "padchest_labels.csv")
IMAGES_ROOT = os.environ.get("PADCHEST_IMAGES", "images")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/processed_v3")
IMAGE_SIZE = 512

# Preprocessing Logic
def _convert_16bit_to_8bit(img):
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 3: arr = arr[:, :, 0]
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 - p1 < 1: p1, p99 = arr.min(), arr.max()
    if p99 - p1 < 1: return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)

def _auto_crop_dark_edges(gray, threshold_ratio=0.05):
    h, w = gray.shape
    if h < 100 or w < 100: return gray
    center = gray[h//4:3*h//4, w//4:3*w//4]
    center_mean = center.mean()
    if center_mean < 10: return gray
    threshold = center_mean * threshold_ratio
    top, bottom, left, right = 0, h, 0, w
    for row in range(h // 6):
        if gray[row, w//4:3*w//4].mean() < threshold: top = row + 1
        else: break
    for row in range(h - 1, h - h // 6, -1):
        if gray[row, w//4:3*w//4].mean() < threshold: bottom = row
        else: break
    for col in range(w // 6):
        if gray[h//4:3*h//4, col].mean() < threshold: left = col + 1
        else: break
    for col in range(w - 1, w - w // 6, -1):
        if gray[h//4:3*h//4, col].mean() < threshold: right = col
        else: break
    if (bottom - top) < h * 0.6 or (right - left) < w * 0.6: return gray
    return gray[top:bottom, left:right]

def _pad_to_square(gray, pad_value=0):
    h, w = gray.shape
    if h == w: return gray
    t = max(h, w)
    padded = np.full((t, t), pad_value, dtype=gray.dtype)
    padded[(t-h)//2:(t-h)//2+h, (t-w)//2:(t-w)//2+w] = gray
    return padded

def preprocess_image_file(args):
    img_name, img_dir, out_path = args
    if os.path.exists(out_path): return "exist" # Skip if done

    in_path = os.path.join(IMAGES_ROOT, str(img_dir), img_name)
    if not os.path.exists(in_path): return "missing"

    try:
        img = Image.open(in_path)
        # 1. Convert to 8-bit
        if img.mode in ('I;16', 'I', 'I;16B', 'I;16L'):
            gray = _convert_16bit_to_8bit(img)
        elif img.mode == 'L': gray = np.array(img, dtype=np.uint8)
        elif img.mode in ('RGB', 'RGBA'): gray = np.array(img.convert('L'), dtype=np.uint8)
        else: gray = _convert_16bit_to_8bit(img)

        # 2. Crop
        gray = _auto_crop_dark_edges(gray)

        # 3. Pad
        gray = _pad_to_square(gray)

        # 4. Resize
        img_pil = Image.fromarray(gray, mode='L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        gray = np.array(img_pil, dtype=np.uint8)

        # 5. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 6. Sharpen
        img_pil = Image.fromarray(gray, mode='L')
        img_pil = ImageEnhance.Sharpness(img_pil).enhance(1.2)

        # 7. Convert to RGB and Save
        final_img = img_pil.convert('RGB')
        final_img.save(out_path, format="PNG", optimize=False, compress_level=0)
        return "done"
    except Exception:
        return "error"

def main():
    print(f"Reading CSV: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Preparing tasks for {len(df)} images...")
    tasks = []
    for _, row in df.iterrows():
        img_name = row['ImageID']
        img_dir = row['ImageDir']
        out_path = os.path.join(OUTPUT_DIR, img_name)
        tasks.append((img_name, img_dir, out_path))

    workers = int(os.environ.get("NUM_WORKERS", "8"))
    print(f"Starting processing with {workers} workers...")

    results = {"done":0, "exist":0, "missing":0, "error":0}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in tqdm(executor.map(preprocess_image_file, tasks), total=len(tasks)):
            results[res] += 1

    print("\nProcessing Complete!")
    print(f"  Processed: {results['done']}")
    print(f"  Already Existed: {results['exist']}")
    print(f"  Missing: {results['missing']}")
    print(f"  Errors: {results['error']}")
    print(f"Output Directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
