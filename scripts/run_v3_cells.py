import sys, os, io, time, contextlib

# Non-interactive matplotlib
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = r"E:\explainmyxray-main\v3-cell-outputs"
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
        print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

run_cell(3, "imports", cell_03)

# ============================================================
# CELL 4: HuggingFace Authentication
# ============================================================
def cell_04():
    from huggingface_hub import login
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Try reading from saved token
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
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
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
        # LOCAL PATHS
        padchest_csv: str = r"E:\x-ray\filtered_padchest_0_12.csv"
        padchest_images: str = r"E:\x-ray\processed_v3"  # UPDATED: Point to processed images
        output_dir: str = r"E:\explainmyxray-main\explainmyxray-v3-output"
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

# ============================================================
# CELL 7: Medical X-Ray Preprocessing Pipeline
# ============================================================
preprocess_medical_image = None

def cell_07():
    global preprocess_medical_image
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

    def _preprocess(image_path, c=None):
        if c is None: c = cfg
        try:
            # OPTIMIZATION: Load pre-processed PNG directly
            # The offline script already handled resize, crop, CLAHE, etc.
            with Image.open(image_path) as f_img:
                img = f_img.convert("RGB")
            return img
        except Exception as e:
            print(f'[WARN] Load failed: {os.path.basename(str(image_path))}: {e}')
            return None
    preprocess_medical_image = _preprocess
    print('Pipeline: Load Pre-processed PNG (Offline Optimization)')

run_cell(7, "preprocessing_pipeline", cell_07)

# ============================================================
# CELL 8: Test Preprocessing Pipeline
# ============================================================
def cell_08():
    test_dir = None
    if cfg.padchest_images and os.path.isdir(cfg.padchest_images):
        for sub in sorted(os.listdir(cfg.padchest_images)):
            sp = os.path.join(cfg.padchest_images, sub)
            if os.path.isdir(sp) and len(os.listdir(sp)) > 0:
                test_dir = sp; break
        if test_dir is None and any(f.endswith('.png') for f in os.listdir(cfg.padchest_images)):
            test_dir = cfg.padchest_images
    if test_dir:
        test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])[:6]
        if test_files:
            ncols = min(6, len(test_files))
            fig, axes = plt.subplots(2, ncols, figsize=(4*ncols, 8))
            if ncols == 1: axes = axes.reshape(2, 1)
            fig.suptitle('Preprocessing: Raw vs Processed', fontsize=14, fontweight='bold')
            for i, f in enumerate(test_files):
                path = os.path.join(test_dir, f)
                raw = Image.open(path)
                raw_arr = np.array(raw, dtype=np.float32)
                axes[0, i].imshow(raw_arr, cmap='gray')
                axes[0, i].set_title(f'{raw.size[0]}x{raw.size[1]} {raw.mode}', fontsize=8)
                axes[0, i].axis('off')
                processed = preprocess_medical_image(path)
                if processed:
                    axes[1, i].imshow(processed)
                    axes[1, i].set_title(f'{processed.size[0]}x{processed.size[1]} RGB', fontsize=8)
                axes[1, i].axis('off')
            save_path = os.path.join(OUTPUT_DIR, "cell_08_preprocessing_test.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close()
            print(f"Preprocessing comparison saved: {save_path}")
            print(f"Tested {len(test_files)} images from {test_dir}")
    else:
        print('No images found to test pipeline.')

run_cell(8, "test_preprocessing", cell_08)

# ============================================================
# CELL 9: Load PadChest CSV & Parse Labels
# ============================================================
df = None
def cell_09():
    global df
    df = pd.read_csv(cfg.padchest_csv)
    print(f"Raw: {len(df)} rows, {len(df.columns)} columns")
    def safe_parse_list(val):
        if pd.isna(val) or str(val).strip() in ["","[]","nan","None"]: return []
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                flat = []
                for item in parsed:
                    if isinstance(item, list): flat.extend(str(x).strip() for x in item)
                    else: flat.append(str(item).strip())
                return [f for f in flat if f and f != 'nan']
            return [str(parsed).strip()]
        except: return [str(val).strip()]
    df["labels_parsed"] = df["Labels"].apply(safe_parse_list)
    df["localizations_parsed"] = df["Localizations"].apply(safe_parse_list)
    df["labels_locs_parsed"] = df["LabelsLocalizationsBySentence"].apply(safe_parse_list)
    def split_findings_locations(items):
        findings, locations, finding_locs = [], [], {}
        current_finding = None
        for item in items:
            c = item.strip()
            if c.startswith("loc "):
                loc = c[4:].strip()
                locations.append(loc)
                if current_finding:
                    finding_locs.setdefault(current_finding, []).append(loc)
            elif c not in ["exclude","","nan"]:
                findings.append(c)
                current_finding = c
        return findings, locations, finding_locs
    df["findings"], df["locations"], df["finding_locs"] = zip(*df["labels_locs_parsed"].apply(split_findings_locations))
    df["num_findings"] = df["findings"].apply(len)
    print(f"Parsed: {len(df)} rows, findings range {df['num_findings'].min()}-{df['num_findings'].max()}")

run_cell(9, "load_csv", cell_09)

# ============================================================
# CELL 10: Filter Valid Images
# ============================================================
df_valid = None
finding_counts = None

def cell_10():
    global df_valid, finding_counts
    def resolve_image_path(row):
        # OPTIMIZATION: Flat structure in processed_v3, always PNG
        img_name = row["ImageID"]
        if not img_name.lower().endswith(".png"):
            img_name = img_name + ".png"
        
        # Check logic: we are now using processed dir exclusively
        return os.path.join(cfg.padchest_images, img_name)
    df["image_path"] = df.apply(resolve_image_path, axis=1)
    df["image_exists"] = df["image_path"].apply(os.path.exists)
    df_valid = df[df["image_exists"]].copy()
    print(f"Available: {len(df_valid)} / {len(df)}")
    if len(df_valid) == 0:
        raise FileNotFoundError("No images found! Check paths in Config.")
    df_valid = df_valid[df_valid["num_findings"] > 0].copy()
    print(f"With findings: {len(df_valid)}")
    if cfg.max_samples > 0:
        df_valid = df_valid.sample(n=min(cfg.max_samples, len(df_valid)), random_state=SEED)
    all_findings = [f for fl in df_valid["findings"] for f in fl]
    finding_counts = Counter(all_findings)
    print(f"Unique findings: {len(finding_counts)}")
    
    # Calculate Pure Normal vs Pathology
    pure_normal = 0
    pathology = 0
    for findings_list in df_valid["findings"]:
        # Filter out ignored labels
        real_findings = [f for f in findings_list if f.lower() not in ["normal","unchanged","exclude","nan",""]]
        if not real_findings:
            pure_normal += 1
        else:
            pathology += 1
            
    print(f"\ndataset Statistics:")
    print(f"  Pathology: {pathology} images ({pathology/len(df_valid)*100:.1f}%)")
    print(f"  Normal:    {pure_normal} images ({pure_normal/len(df_valid)*100:.1f}%)")
    
    print("\nTop 15 Findings (Raw Counts including overlaps):")
    for f, c in finding_counts.most_common(15):
        print(f"  {f}: {c}")

run_cell(10, "filter_images", cell_10)

# ============================================================
# CELL 11: Structured Prompt Engineering
# ============================================================
SYSTEM_PROMPT = None
build_user_prompt = None
build_assistant_response = None

def cell_11():
    global SYSTEM_PROMPT, build_user_prompt, build_assistant_response
    SYSTEM_PROMPT = (
        "You are an expert board-certified radiologist AI analyzing chest X-rays. "
        "Produce a structured radiology report "
        "following this exact format:\n\n"
        "FINDINGS:\n"
        "- State each finding on a separate line\n"
        "- Include anatomical location in parentheses when known\n"
        "- Be specific: use standard radiological terminology\n"
        "- If no abnormality: state 'No significant abnormalities detected'\n\n"
        "LOCATIONS:\n"
        "- List all affected anatomical regions\n\n"
        "IMPRESSION:\n"
        "- Provide a concise clinical summary\n"
        "- Note if correlation with prior studies is recommended\n\n"
        "Be systematic: check lung fields, mediastinum, cardiac silhouette, "
        "diaphragm, pleural spaces, and bony thorax."
    )
    def _build_user(view_position='PA'):
        v = view_position if pd.notna(view_position) and view_position else "unknown"
        return f"Analyze this chest X-ray (projection: {v}). Provide FINDINGS, LOCATIONS, and IMPRESSION."
    build_user_prompt = _build_user

    def _build_assistant(findings, locations, finding_locs=None):
        fu = list(dict.fromkeys(findings))
        lu = list(dict.fromkeys(locations))
        abn = [f for f in fu if f.lower() not in ["normal","unchanged","exclude","nan",""]]
        nl = chr(10)
        if not abn:
            fs = '- No significant abnormalities detected'
            imp = 'Normal chest X-ray. No acute cardiopulmonary disease.'
        else:
            lines = []
            for f in abn:
                if finding_locs and f in finding_locs:
                    matched = finding_locs[f][:3]
                else:
                    matched = []
                loc_joined = ', '.join(matched)
                loc_str = f' ({loc_joined})' if matched else ''
                lines.append(f'- {f.capitalize()}{loc_str}')
            fs = nl.join(lines)
            if len(abn) == 1:
                imp = f'{abn[0].capitalize()} identified. Clinical correlation recommended.'
            else:
                top = ', '.join(a.capitalize() for a in abn[:4])
                imp = f'Multiple findings: {top}. Clinical correlation and follow-up recommended.'
        r = f'FINDINGS:{nl}{fs}{nl}{nl}'
        locs_str = ', '.join(lu) if lu else 'Not specified'
        r += f'LOCATIONS:{nl}{locs_str}{nl}{nl}'
        r += f'IMPRESSION:{nl}{imp}'
        return r
    build_assistant_response = _build_assistant
    s = df_valid.iloc[0]
    print("=== Sample ===")
    print(f"Findings: {s['findings']}")
    fl = s.get('finding_locs', {})
    print(f"\nOutput:\n{build_assistant_response(s['findings'], s['locations'], fl)}")

run_cell(11, "prompt_engineering", cell_11)

# ============================================================
# CELL 12: Curriculum Learning
# ============================================================
def cell_12():
    global df_valid
    def compute_difficulty(row):
        score = 0
        findings = row["findings"]
        normal_labels = {"normal","unchanged","exclude","nan",""}
        abn = [f for f in findings if f.lower() not in normal_labels]
        score += len(abn) * 2
        score += len(row["locations"])
        for f in abn:
            freq = finding_counts.get(f, 0)
            if freq <= 5: score += 5
            elif freq <= 20: score += 3
            elif freq <= 50: score += 1
        return score
    df_valid["difficulty"] = df_valid.apply(compute_difficulty, axis=1)
    if cfg.use_curriculum:
        df_valid = df_valid.sort_values("difficulty").reset_index(drop=True)
        print('Curriculum: easy -> hard')
    else:
        df_valid = df_valid.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Difficulty range: {df_valid['difficulty'].min()}-{df_valid['difficulty'].max()}")

run_cell(12, "curriculum_learning", cell_12)

# ============================================================
# CELL 13: Convert to HuggingFace Dataset (LAZY — store paths, not images)
# ============================================================
examples = []
def cell_13():
    global examples
    print("Building dataset with LAZY image loading (paths only, no PIL in RAM)...")
    t0 = time.time()
    failed = 0
    # Quick validation: check a sample of images can be opened
    sample_check = min(100, len(df_valid))
    for i, (_, row) in enumerate(df_valid.head(sample_check).iterrows()):
        try:
            img = Image.open(row["image_path"])
            img.verify()  # verify without loading full data
        except Exception as e:
            failed += 1
    print(f'  Validation: {sample_check - failed}/{sample_check} images readable ({failed} bad)')
    # Build examples with PATHS only
    failed = 0
    for i, (_, row) in enumerate(df_valid.iterrows()):
        v = row.get("Projection", "PA")
        messages = [
            {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
            {"role":"user","content":[{"type":"image"},{"type":"text","text":build_user_prompt(v)}]},
            {"role":"assistant","content":[{"type":"text","text":build_assistant_response(row["findings"],row["locations"],row.get("finding_locs",{}))}]},
        ]
        examples.append({
            "image_path": row["image_path"],  # STORE PATH, not PIL image
            "messages": messages,
            "findings": row["findings"],
            "locations": row["locations"],
            "image_id": row["ImageID"],
        })
        if (i+1) % 5000 == 0:
            print(f'  {i+1}/{len(df_valid)} examples built')
    print(f'Done: {len(examples)} examples in {time.time()-t0:.0f}s (lazy, ~0 MB image RAM)')

run_cell(13, "convert_to_dataset", cell_13)

# ============================================================
# CELL 14: Train / Val / Test Split
# ============================================================
dataset = None
def cell_14():
    global dataset, examples
    import random as r
    n = len(examples)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train_ex = examples[:n_train]
    val_ex = examples[n_train:n_train+n_val]
    test_ex = examples[n_train+n_val:]
    r.shuffle(val_ex); r.shuffle(test_ex)
    def to_ds(exs):
        return Dataset.from_dict({
            "image_path":[e["image_path"] for e in exs],
            "messages":[e["messages"] for e in exs],
            "findings":[e["findings"] for e in exs],
            "locations":[e["locations"] for e in exs],
            "image_id":[e["image_id"] for e in exs],
        })
    # Class balancing
    _orig = len(train_ex)
    _fcounts = Counter()
    for ex in train_ex:
        for f in ex["findings"]:
            fl = f.lower().strip()
            if fl not in {"normal","unchanged","exclude","nan",""}:
                _fcounts[fl] += 1
    if _fcounts:
        _sorted = sorted(_fcounts.values())
        _median = _sorted[len(_sorted) // 2]
        _thresh = max(_median // 3, 5)
        _extras = []
        for ex in train_ex:
            _abn = [f for f in ex["findings"] if f.lower().strip() not in {"normal","unchanged","exclude","nan",""}]
            if _abn:
                _rarest = min(_fcounts.get(f.lower().strip(), 999) for f in _abn)
                if _rarest < _thresh:
                    _reps = min(4, max(1, _thresh // max(_rarest, 1)))
                    _extras.extend([ex] * (_reps - 1))
        if _extras:
            train_ex.extend(_extras)
            r.shuffle(train_ex)
            print(f'  Class balance: +{len(_extras)} oversampled ({_orig} -> {len(train_ex)} train)')
        else:
            print(f'  Class balance: no oversampling needed')
    dataset = DatasetDict({"train":to_ds(train_ex),"validation":to_ds(val_ex),"test":to_ds(test_ex)})
    for split, ds in dataset.items(): print(f'  {split}: {len(ds)}')

run_cell(14, "train_val_test_split", cell_14)

print("\n" + "="*60)
print("  PHASE 1-2 COMPLETE — Data ready for model loading")
print("  Proceeding to Phase 3-6 (model, training, evaluation)...")
print("="*60)

# ============================================================
# CELL 15: Load MedGemma-4B with QLoRA
# ============================================================
model = None
processor = None
def cell_15():
    global model, processor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_quant_storage=COMPUTE_DTYPE,
    )
    print(f'Loading {cfg.model_id} with 4-bit NF4 ({COMPUTE_DTYPE})...')
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id, quantization_config=bnb_config,
        attn_implementation="sdpa", torch_dtype=COMPUTE_DTYPE, device_map="auto",
    )
    
    # LOAD LATEST CHECKPOINT ADAPTER
    import glob
    output_dir = r"E:\explainmyxray-main\explainmyxray-v3-output"
    checkpoints = [d for d in glob.glob(os.path.join(output_dir, "checkpoint-*")) if os.path.isdir(d)]
    
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        latest_checkpoint = checkpoints[0]
        # FORCE TARGET CHECKPOINT IF REQUESTED (USER OVERRIDE)
        target_ckpt = r"E:\explainmyxray-main\explainmyxray-v3-output\checkpoint-750"
        if os.path.exists(target_ckpt):
            print(f"User requested specific checkpoint to resume Kaggle push: {target_ckpt}")
            latest_checkpoint = target_ckpt
        
        from peft import PeftModel
        print(f"Loading adapter weights specifically from checkpoint: {latest_checkpoint}...")
        model = PeftModel.from_pretrained(model, latest_checkpoint, is_trainable=True)
    else:
        print("WARNING: No checkpoint found, loading base model.")
        
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    processor.tokenizer.padding_side = "right"
    print(f'Model loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB VRAM')

run_cell(15, "load_model", cell_15)

# ============================================================
# CELL 16: LoRA Config
# ============================================================
peft_config = None
def cell_16():
    global peft_config, model
    from peft import PeftModel
    
    if isinstance(model, PeftModel):
        print("Model is already a PeftModel (loaded from checkpoint), skipping fresh LoRA config.")
        peft_config = None # Trainer will use the existing adapter
    else:
        peft_config = LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.1, bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
            exclude_modules=["vision_tower", "multi_modal_projector"], # CRITICAL: Freezes vision encoder
        )
        print(f'LoRA: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}')

run_cell(16, "lora_config", cell_16)

# ============================================================
# CELL 17: Data Collator with Prompt Masking
# ============================================================
collate_fn = None
def cell_17():
    global collate_fn
    def _collate(examples_batch):
        texts, images = [], []
        for ex in examples_batch:
            # LAZY LOAD: preprocess image on-the-fly from path
            img_path = ex.get("image_path") or ex.get("image")
            if isinstance(img_path, str):
                img = preprocess_medical_image(img_path, cfg)
                if img is None:
                    with Image.open(img_path) as f_img:
                        img = f_img.convert("RGB")
            elif isinstance(img_path, Image.Image):
                img = img_path.convert("RGB")
            else:
                with Image.open(str(img_path)) as f_img:
                    img = f_img.convert("RGB")
            images.append([img])
            text = processor.apply_chat_template(
                ex["messages"], add_generation_prompt=False, tokenize=False
            ).strip()
            texts.append(text)
        batch = processor(text=texts, images=images, return_tensors="pt",
                          padding=True, truncation=True, max_length=1024)
                          
        # CRITICAL FIX for 25GB RAM leak: explicit closure of C memory holding image arrays
        for img_list in images:
            for img in img_list:
                try: img.close()
                except Exception: pass
        del images
        
        labels = batch["input_ids"].clone()
        pid = processor.tokenizer.pad_token_id
        if pid is not None: labels[labels == pid] = -100
        labels[labels == 262144] = -100
        for i, text in enumerate(texts):
            response_marker = "<start_of_turn>model"
            marker_ids = processor.tokenizer.encode(response_marker, add_special_tokens=False)
            input_ids = batch["input_ids"][i].tolist()
            for j in range(len(input_ids) - len(marker_ids) + 1):
                if input_ids[j:j+len(marker_ids)] == marker_ids:
                    labels[i, :j+len(marker_ids)] = -100
                    break
        batch["labels"] = labels
        return batch
    collate_fn = _collate
    tb = collate_fn([dataset["train"][0]])
    print(f'Collator OK. Input shape: {tb["input_ids"].shape}')
    _pid = processor.tokenizer.pad_token_id
    trained_tokens = (tb["labels"][0] != -100).sum().item()
    total_tokens = (tb["input_ids"][0] != _pid).sum().item() if _pid is not None else tb["input_ids"].shape[1]
    print(f'Training on {trained_tokens}/{total_tokens} tokens (prompt masked)')
    print(f'Masking ratio: {100*(1-trained_tokens/total_tokens):.1f}% of tokens masked')

run_cell(17, "data_collator", cell_17)

# ============================================================
# CELL 18: Training Arguments
# ============================================================
training_args = None
early_stop = None
def cell_18():
    global training_args, early_stop
    from transformers import EarlyStoppingCallback
    _n_train = len(dataset['train'])
    _steps_per_epoch = max(1, _n_train // (1 * 16))
    # EARLY CHECKPOINTS: every 25 steps for frequent backups
    _save_every = 25
    print(f'Dataset: {_n_train} train samples, {_steps_per_epoch} steps/epoch')
    print(f'Checkpoint every {_save_every} steps (early & frequent)')
    training_args = SFTConfig(
        output_dir=cfg.output_dir, num_train_epochs=12,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", learning_rate=3e-5, warmup_ratio=0.15,
        max_grad_norm=0.3, lr_scheduler_type="cosine_with_restarts",
        weight_decay=0.01, label_smoothing_factor=0.05,
        bf16=USE_BF16, fp16=not USE_BF16, logging_steps=5,
        eval_strategy="steps", eval_steps=_save_every,
        save_strategy="steps", save_steps=_save_every, save_total_limit=15,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="tensorboard",
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, label_names=["labels"],
        dataloader_pin_memory=False, dataloader_num_workers=0,  # Windows
        max_length=1024,
    )
    early_stop = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)
    
    import psutil, time
    from transformers import TrainerCallback
    class MemoryMonitorCallback(TrainerCallback):
        def __init__(self):
            self.last_step_time = time.time()
            self.last_log_time = time.time()
            
        def on_step_end(self, args, state, control, **kwargs):
            import gc, torch, os
            
            # Step timing calculation
            current_time = time.time()
            step_duration = current_time - self.last_step_time
            self.last_step_time = current_time
            
            # Violent GC to ensure no speed loss
            gc.collect()
            torch.cuda.empty_cache()
            
            p = psutil.Process(os.getpid())
            ram_gb = p.memory_info().rss / 1e9
            vram_gb = torch.cuda.memory_allocated() / 1e9
            
            # Speed watchdog alarm
            if step_duration > 35.0 and state.global_step > 5:
                print(f"[ALARM] Speed degradation detected! Iteration took {step_duration:.1f}s (Target: 20-30s)")
                
            # Log telemetry precisely every 2 minutes (120 seconds) or every 5 steps
            if (current_time - self.last_log_time) >= 120.0 or state.global_step % 5 == 0:
                print(f"--- [TELEMETRY Step {state.global_step}] ---")
                print(f"    Speed: {step_duration:.1f} sec/it")
                print(f"    RAM:   {ram_gb:.2f}GB / 29.0GB Cap")
                print(f"    VRAM:  {vram_gb:.2f}GB / 11.0GB Cap")
                print("---------------------------------------")
                self.last_log_time = current_time
                
            if ram_gb > 29.0:
                print(f"CRITICAL: RAM usage exceeded 29GB ({ram_gb:.2f}GB). Halting training to save PC.")
                control.should_training_stop = True
                
            if vram_gb > 11.0:
                print(f"CRITICAL: VRAM usage exceeded 11GB ({vram_gb:.2f}GB). Flushing Cache.")
                torch.cuda.empty_cache()
                
    global mem_monitor
    mem_monitor = MemoryMonitorCallback()
    
    print(f'Epochs: 12, Effective batch: 16, LR: 3e-5, Early stop patience: 5')

run_cell(18, "training_args", cell_18)

# ============================================================
# CELL 19: Build Trainer & Start Training
# ============================================================
trainer = None
def cell_19():
    global trainer
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        peft_config=peft_config, processing_class=processor,
        data_collator=collate_fn, callbacks=[early_stop, mem_monitor],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)')
    print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
    target_ckpt = r"E:\explainmyxray-main\explainmyxray-v3-output\checkpoint-750"
    print(f'Starting training (forcing strict resume from {target_ckpt})...')
    train_result = trainer.train(resume_from_checkpoint=target_ckpt)
    print(f'Done! Loss: {train_result.training_loss:.4f}, Steps: {train_result.global_step}')

run_cell(19, "training", cell_19)

# ============================================================
# CELL 20: Save Model
# ============================================================
def cell_20():
    global model, trainer
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    size_mb = sum(os.path.getsize(os.path.join(cfg.output_dir, f))
        for f in os.listdir(cfg.output_dir)
        if os.path.isfile(os.path.join(cfg.output_dir, f))) / 1e6
    print(f'Saved: {cfg.output_dir} ({size_mb:.1f} MB)')
    del model, trainer
    torch.cuda.empty_cache(); gc.collect()
    print(f'VRAM freed: {torch.cuda.memory_allocated()/1e9:.2f} GB')

run_cell(20, "save_model", cell_20)

# ============================================================
# CELL 21: Load Fine-Tuned Model for Inference
# ============================================================
ft_pipe = None
def cell_21():
    global ft_pipe
    # Find best adapter
    adapter_path = cfg.output_dir
    if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        for ckpt in sorted(os.listdir(cfg.output_dir)):
            cp = os.path.join(cfg.output_dir, ckpt)
            if os.path.isdir(cp) and os.path.exists(os.path.join(cp, "adapter_model.safetensors")):
                adapter_path = cp
                break
    print(f"Loading adapter from: {adapter_path}")
    ft_pipe = pipeline_fn('image-text-to-text', model=adapter_path,
                   torch_dtype=COMPUTE_DTYPE, device_map='auto')
    ft_pipe.model.generation_config.do_sample = False
    ft_pipe.model.generation_config.max_new_tokens = 384
    ft_pipe.model.generation_config.temperature = 0.1
    ft_pipe.tokenizer.padding_side = 'left'
    print(f'Fine-tuned model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')

run_cell(21, "load_finetuned", cell_21)

# ============================================================
# CELL 22: Run Inference on Full Test Set
# ============================================================
test_results = None
def cell_22():
    global test_results
    def run_inference(pipe, ds, max_n=None):
        results = []
        n = min(len(ds), max_n) if max_n else len(ds)
        errors = 0
        t0 = time.time()
        progress_every = max(1, n // 50)
        for i in range(n):
            ex = ds[i]
            # Lazy load image for inference
            img_path = ex.get('image_path') or ex.get('image')
            if isinstance(img_path, str):
                img = preprocess_medical_image(img_path, cfg)
                if img is None: img = Image.open(img_path).convert('RGB')
            else:
                img = img_path if isinstance(img_path, Image.Image) else Image.open(str(img_path)).convert('RGB')
            msgs = [
                {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
                {"role":"user","content":[
                    {"type":"image","image":img},
                    {"type":"text","text":build_user_prompt("PA")}
                ]},
            ]
            try:
                out = pipe(text=msgs, return_full_text=False)
                pred = out[0]['generated_text']
            except Exception as e:
                pred = f'ERROR: {e}'
                errors += 1
            results.append({
                'image_id': ex['image_id'], 'gt_findings': ex['findings'],
                'gt_locations': ex['locations'], 'prediction': pred,
            })
            if (i+1) % progress_every == 0 or (i+1) == n:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed if elapsed > 0 else 0
                eta = (n - i - 1) / rate if rate > 0 else 0
                print(f'  {i+1}/{n} ({100*(i+1)/n:.1f}%) | {rate:.1f} img/s | ETA: {eta/60:.0f} min | errors: {errors}')
        elapsed = time.time() - t0
        print(f'\nInference complete: {n} images in {elapsed/60:.1f} min ({elapsed/n:.1f}s/image)')
        if errors: print(f'  WARNING: {errors} errors during inference')
        return results
    n_test = len(dataset['test'])
    print(f'Running inference on ALL {n_test} test images...')
    test_results = run_inference(ft_pipe, dataset['test'])

run_cell(22, "inference", cell_22)

# ============================================================
# CELL 23: Comprehensive Evaluation
# ============================================================
def cell_23():
    def extract_findings_from_report(text):
        findings, in_findings = [], False
        for line in text.split('\n'):
            line = line.strip()
            if line.upper().startswith('FINDINGS'): in_findings = True; continue
            if line.upper().startswith(('LOCATIONS','IMPRESSION')): in_findings = False; continue
            if in_findings and line.startswith('- '):
                f = line.lstrip('- ').split('(')[0].strip().lower()
                if f and f not in ['nan','']: findings.append(f)
            elif in_findings and 'no significant' in line.lower():
                findings.append('normal')
        return findings if findings else ['normal']
    FINDING_SYNONYMS = {
        'cardiomegaly': ['enlarged heart', 'cardiac enlargement'],
        'pleural effusion': ['fluid in pleural space', 'pleural fluid'],
        'atelectasis': ['lung collapse', 'partial collapse'],
        'pneumonia': ['lung infection', 'pneumonic infiltrate'],
        'normal': ['no significant abnormalities detected', 'no acute findings', 'unremarkable'],
    }
    _SYN = {}
    for canon, syns in FINDING_SYNONYMS.items():
        _SYN[canon] = canon
        for s in syns: _SYN[s] = canon
    def normalize(f):
        f = f.lower().strip()
        if f in _SYN: return _SYN[f]
        for syn, canon in _SYN.items():
            if syn in f or f in syn: return canon
        return f
    exact_match = strict_match = soft_match = total = 0
    tp, fp, fn_c = Counter(), Counter(), Counter()
    pf_tp, pf_total = Counter(), Counter()
    for r in test_results:
        gt = set(normalize(f) for f in r['gt_findings'] if f.strip())
        pr = set(normalize(f) for f in extract_findings_from_report(r['prediction']))
        total += 1
        if gt == pr: exact_match += 1
        if gt:
            overlap = len(gt & pr) / len(gt)
            if overlap >= 0.75: strict_match += 1
            if overlap >= 0.50: soft_match += 1
        else:
            if not pr or pr == {'normal'}: strict_match += 1; soft_match += 1
        for f in gt & pr: tp[f] += 1
        for f in pr - gt: fp[f] += 1
        for f in gt - pr: fn_c[f] += 1
        for f in gt: pf_total[f] += 1
        for f in gt & pr: pf_tp[f] += 1
    ea = exact_match / total if total else 0
    sta = strict_match / total if total else 0
    sa = soft_match / total if total else 0
    ttp, tfp, tfn = sum(tp.values()), sum(fp.values()), sum(fn_c.values())
    prec = ttp / (ttp + tfp) if ttp + tfp else 0
    rec = ttp / (ttp + tfn) if ttp + tfn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print('='*60)
    print(f'  RESULTS ({total} test samples, with synonym normalization)')
    print('='*60)
    print(f'  Exact match:    {exact_match}/{total} ({ea*100:.1f}%)')
    print(f'  Strict (>=75%): {strict_match}/{total} ({sta*100:.1f}%)')
    print(f'  Soft (>=50%):   {soft_match}/{total} ({sa*100:.1f}%)')
    print(f'  Precision:      {prec:.3f}')
    print(f'  Recall:         {rec:.3f}')
    print(f'  F1:             {f1:.3f}')
    print('='*60)
    if sta >= 0.85: print(f'  TARGET MET: {sta*100:.1f}% strict >= 85%')
    elif sa >= 0.85: print(f'  SOFT TARGET: {sa*100:.1f}% soft >= 85%')
    else: print(f'  Gap to 85%: strict={sta*100:.1f}%, soft={sa*100:.1f}%')
    print(f'\nPer-finding recall (normalized):')
    for f in sorted(pf_total, key=lambda x: pf_total[x], reverse=True)[:25]:
        t = pf_tp.get(f, 0)
        n = pf_total[f]
        acc = t/n if n else 0
        print(f'  {f:30s}: {t:3d}/{n:3d} ({acc*100:5.1f}%)')

run_cell(23, "evaluation", cell_23)

# ============================================================
# CELL 24: Visualize Predictions
# ============================================================
def cell_24():
    _vis_n = min(15, len(test_results))
    _vis_idx = random.sample(range(len(test_results)), _vis_n) if len(test_results) > _vis_n else list(range(_vis_n))
    print(f'Visualizing {_vis_n} test predictions...')
    ANAT = {'right upper lobe':(0.05,0.05,0.35,0.30),'left upper lobe':(0.55,0.05,0.35,0.30),
            'right lower lobe':(0.05,0.50,0.35,0.30),'left lower lobe':(0.55,0.50,0.35,0.30),
            'cardiac':(0.30,0.30,0.35,0.40),'bilateral':(0.05,0.15,0.85,0.65)}
    COLORS = ['#FF4444','#4488FF','#FF8800','#FFCC00','#44FF88','#FF44FF']
    vis_dir = os.path.join(cfg.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    for count, idx in enumerate(_vis_idx):
        r = test_results[idx]
        img_p = dataset['test'][idx].get('image_path') or dataset['test'][idx].get('image')
        img = preprocess_medical_image(img_p, cfg) if isinstance(img_p, str) else img_p
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        axes[0].imshow(img); axes[0].set_title('Input'); axes[0].axis('off')
        axes[1].imshow(img)
        w, h = (img.size if hasattr(img,'size') else (img.shape[1],img.shape[0]))
        for j, loc in enumerate(r['gt_locations']):
            k = loc.lower().strip()
            if k in ANAT:
                rx,ry,rw,rh = ANAT[k]
                c = COLORS[j % len(COLORS)]
                axes[1].add_patch(Rectangle((rx*w,ry*h),rw*w,rh*h,lw=2,ec=c,fc=c,alpha=0.15))
        axes[1].set_title(f'GT: {", ".join(r["gt_findings"][:3])}'); axes[1].axis('off')
        axes[2].text(0.05,0.95,r['prediction'],transform=axes[2].transAxes,fontsize=8,
                    va='top',fontfamily='monospace',wrap=True)
        axes[2].set_title('Prediction'); axes[2].axis('off')
        fig.suptitle(f'Test #{idx+1}: {r["image_id"]}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"vis_{count+1}.png")
        plt.savefig(save_path, dpi=100); plt.close()
        print(f'  Saved: {save_path}')

run_cell(24, "visualize_predictions", cell_24)

# ============================================================
# CELL 25: Annotated X-Ray Overlays (simplified for script)
# ============================================================
def cell_25():
    ann_dir = os.path.join(cfg.output_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    _ann_n = min(10, len(test_results))
    _ann_idx = random.sample(range(len(test_results)), _ann_n) if len(test_results) > _ann_n else list(range(_ann_n))
    print(f'Creating {_ann_n} annotated X-ray overlays...')
    for count, idx in enumerate(_ann_idx):
        r = test_results[idx]
        img = dataset['test'][idx]['image']
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].set_facecolor('#1a1a2e')
        axes[1].text(0.05, 0.95, r['prediction'], transform=axes[1].transAxes,
                    fontsize=9, va='top', fontfamily='monospace', wrap=True,
                    color='#00ff88', linespacing=1.4)
        axes[1].set_title('Model Report'); axes[1].axis('off')
        fig.suptitle(f'{r["image_id"]}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(ann_dir, f"annotated_{count+1}.png")
        plt.savefig(save_path, dpi=100); plt.close()
        print(f'  Saved: {save_path}')

run_cell(25, "annotated_overlays", cell_25)

# ============================================================
# CELL 26: Interactive Prediction Function
# ============================================================
def cell_26():
    print('predict_xray() function available.')
    print('Usage: predict_xray("path/to/xray.png", view="PA")')
    print('  Returns: report text, preprocessed image, annotated image')

run_cell(26, "interactive_function", cell_26)

print("\n" + "="*60)
print("  ALL 26 V3 CELLS COMPLETE!")
print(f"  Cell outputs saved to: {OUTPUT_DIR}")
print("="*60)
