"""
Phase 1 Training Pipeline — Original Script (26 cells).

NOTE: This is the original development script preserved for reference.
It runs all training cells sequentially: data loading, preprocessing,
model loading, LoRA config, training, evaluation, and visualisation.

For the clean, CLI-driven version, see training/finetune_phase1.py.

Configuration:
  Set the following environment variables or edit the Config class in cell 6:
    - PADCHEST_CSV: path to PadChest CSV file
    - PADCHEST_IMAGES: path to preprocessed image directory
    - OUTPUT_DIR: output directory for checkpoints
    - HF_TOKEN: HuggingFace token for model access
"""

import sys, os, io, time, contextlib

# Non-interactive matplotlib
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = os.environ.get("CELL_OUTPUT_DIR", "output/v3-cell-outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_cell(cell_num, label, code_fn):
    """Run a cell function, capture its stdout to a file, and print to console."""
    out_file = os.path.join(OUTPUT_DIR, f"cell_{cell_num:02d}_{label}.txt")
    buf = io.StringIO()
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  CELL {cell_num}: {label}")
    print(f"{'='*60}")
    try:
        with contextlib.redirect_stdout(buf):
            code_fn()
        output = buf.getvalue()
        elapsed = time.time() - t0
        output += f"\n[Completed in {elapsed:.1f}s]\n"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(output)
        print(f"  -> Saved: {out_file}")
    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"ERROR in Cell {cell_num}: {e}\n[Failed after {elapsed:.1f}s]"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(buf.getvalue() + "\n" + err_msg)
        print(buf.getvalue())
        print(err_msg)
        raise

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
def cell_01():
    import subprocess
    packages = [
        "transformers>=4.52.0", "trl>=0.17.0", "peft>=0.15.0",
        "accelerate>=1.5.0", "bitsandbytes>=0.44.0", "datasets>=3.5.0",
        "evaluate", "scikit-learn", "Pillow>=10.0", "gdown",
        "opencv-python", "tensorboard",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import torch, transformers, trl, peft
    print(f'torch={torch.__version__}, transformers={transformers.__version__}, '
          f'trl={trl.__version__}, peft={peft.__version__}')
    print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
    print("Dependencies installed.")

run_cell(1, "install_dependencies", cell_01)

# ============================================================
# CELL 2: Auto-Detect Google Drive
# ============================================================
GDRIVE_PADCHEST = None  # Not using Google Drive — local dataset

def cell_02():
    global GDRIVE_PADCHEST
    GDRIVE_PADCHEST = None
    print("Google Drive for Desktop not found (using local dataset).")
    print("Local paths will be set in Config (Cell 6).")

run_cell(2, "detect_gdrive", cell_02)

# ============================================================
# CELL 3: Imports
# ============================================================
def cell_03():
    global os, ast, random, warnings, gc, time, Path, Any, Optional, Tuple
    global Counter, dataclass, torch, np, pd, plt, mpatches, Rectangle
    global Image, ImageFilter, ImageEnhance, ImageOps, cv2
    global AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, pipeline_fn
    global LoraConfig, SFTConfig, SFTTrainer, Dataset, DatasetDict

    import os, ast, random, warnings, gc, time as time_mod
    time = time_mod
    from pathlib import Path
    from typing import Any, Optional, Tuple
    from collections import Counter as Counter_cls
    Counter = Counter_cls
    from dataclasses import dataclass as dc
    dataclass = dc
    import torch as torch_mod
    torch = torch_mod
    import numpy as np_mod
    np = np_mod
    import pandas as pd_mod
    pd = pd_mod
    import matplotlib.pyplot as plt_mod
    plt = plt_mod
    import matplotlib.patches as mp
    mpatches = mp
    from matplotlib.patches import Rectangle as Rect
    Rectangle = Rect
    from PIL import Image as Img, ImageFilter as IF, ImageEnhance as IE, ImageOps as IO
    Image = Img; ImageFilter = IF; ImageEnhance = IE; ImageOps = IO
    import cv2 as cv
    cv2 = cv
    from transformers import (AutoProcessor as AP, AutoModelForImageTextToText as AM,
                              BitsAndBytesConfig as BB, pipeline as PL)
    AutoProcessor = AP; AutoModelForImageTextToText = AM; BitsAndBytesConfig = BB; pipeline_fn = PL
    from peft import LoraConfig as LC
    LoraConfig = LC
    from trl import SFTConfig as SC, SFTTrainer as ST
    SFTConfig = SC; SFTTrainer = ST
    from datasets import Dataset as DS, DatasetDict as DSD
    Dataset = DS; DatasetDict = DSD
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")

run_cell(3, "imports", cell_03)

# ============================================================
# CELL 4: HuggingFace Authentication
# ============================================================
def cell_04():
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
        if os.path.exists(token_path):
            with open(token_path) as f:
                hf_token = f.read().strip()
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace.")
    else:
        print("WARNING: No HF token found. Model download may fail.")

run_cell(4, "hf_auth", cell_04)

# ============================================================
# CELL 5: GPU Configuration
# ============================================================
USE_BF16 = False
COMPUTE_DTYPE = None
SEED = 42

def cell_05():
    global USE_BF16, COMPUTE_DTYPE, SEED
    import random as r, numpy as n
    if not torch.cuda.is_available(): raise RuntimeError('No GPU!')
    cc = torch.cuda.get_device_capability(0)
    USE_BF16 = cc[0] >= 8
    COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CC: {cc[0]}.{cc[1]}, Precision: {"bf16" if USE_BF16 else "fp16"}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    SEED = 42
    r.seed(SEED); n.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

run_cell(5, "gpu_config", cell_05)

# ============================================================
# CELL 6: Master Configuration
# ============================================================
cfg = None

def cell_06():
    global cfg
    from dataclasses import dataclass
    @dataclass
    class Config:
        model_id: str = "google/medgemma-4b-it"
        use_full_padchest: bool = True
        # Paths — set via environment variables or edit defaults below
        padchest_csv: str = os.environ.get("PADCHEST_CSV", "padchest_labels.csv")
        padchest_images: str = os.environ.get("PADCHEST_IMAGES", "processed_images")
        output_dir: str = os.environ.get("OUTPUT_DIR", "output/v3-output")
        # QLoRA
        lora_r: int = 64
        lora_alpha: int = 128
        lora_dropout: float = 0.05
        load_in_4bit: bool = True
        bnb_4bit_quant_type: str = "nf4"
        bnb_4bit_use_double_quant: bool = True
        # Training
        num_train_epochs: int = 8
        per_device_train_batch_size: int = 1
        per_device_eval_batch_size: int = 1
        gradient_accumulation_steps: int = 16
        learning_rate: float = 5e-5
        warmup_ratio: float = 0.1
        max_grad_norm: float = 0.3
        lr_scheduler_type: str = "cosine_with_restarts"
        logging_steps: int = 10
        eval_steps: int = 50
        save_steps: int = 100
        max_seq_length: int = 768
        label_smoothing_factor: float = 0.05
        # Image Preprocessing
        image_size: int = 512
        apply_clahe: bool = True
        clahe_clip_limit: float = 2.0
        clahe_grid_size: int = 8
        auto_crop_edges: bool = True
        edge_crop_threshold: float = 0.05
        pad_to_square: bool = True
        pad_color: int = 0
        # Splits
        train_ratio: float = 0.90
        val_ratio: float = 0.05
        test_ratio: float = 0.05
        max_samples: int = 0
        use_curriculum: bool = True
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f'CSV: {cfg.padchest_csv} (exists: {os.path.isfile(cfg.padchest_csv)})')
    print(f'Images: {cfg.padchest_images} (exists: {os.path.isdir(cfg.padchest_images)})')
    print(f'Output: {cfg.output_dir}')
    print(f'LoRA r={cfg.lora_r}, LR={cfg.learning_rate}, Epochs={cfg.num_train_epochs}')
    print(f'Effective batch: {cfg.per_device_train_batch_size} x {cfg.gradient_accumulation_steps} = {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}')

run_cell(6, "config", cell_06)

# Remaining cells (7-26) follow the same pattern but are omitted for brevity.
# They handle: preprocessing pipeline, CSV parsing, dataset conversion,
# curriculum learning, model loading, LoRA config, training, evaluation,
# and visualisation. See training/finetune_phase1.py for the clean version.

print("\n" + "=" * 60)
print("  NOTE: Cells 7-26 omitted in cleaned script.")
print("  Use training/finetune_phase1.py for reproducible training.")
print("=" * 60)
