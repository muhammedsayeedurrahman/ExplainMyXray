"""
Phase 1: Diagnostic Report Fine-Tuning for ExplainMyXray.

Fine-tunes MedGemma-4B using QLoRA on the PadChest dataset to generate
structured radiology reports (FINDINGS / LOCATIONS / IMPRESSION).

Usage:
    python training/finetune_phase1.py \
        --csv  /path/to/padchest_labels.csv \
        --images /path/to/processed/images \
        --output /path/to/output
"""

import argparse
import ast
import os
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_ID = "google/medgemma-4b-it"
SEED = 42

SYSTEM_PROMPT = (
    "You are an expert board-certified radiologist AI analyzing chest X-rays. "
    "Produce a structured radiology report following this exact format:\n\n"
    "FINDINGS:\n- State each finding on a separate line\n"
    "- Include anatomical location in parentheses when known\n\n"
    "LOCATIONS:\n- List all affected anatomical regions\n\n"
    "IMPRESSION:\n- Provide a concise clinical summary\n\n"
    "Be systematic: check lung fields, mediastinum, cardiac silhouette, "
    "diaphragm, pleural spaces, and bony thorax."
)


def safe_parse_list(val):
    if pd.isna(val) or str(val).strip() in ("", "[]", "nan", "None"):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            flat = []
            for item in parsed:
                if isinstance(item, list):
                    flat.extend(str(x).strip() for x in item)
                else:
                    flat.append(str(item).strip())
            return [f for f in flat if f and f != "nan"]
        return [str(parsed).strip()]
    except Exception:
        return [str(val).strip()]


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
        elif c not in ("exclude", "", "nan"):
            findings.append(c)
            current_finding = c
    return findings, locations, finding_locs


def build_assistant_response(findings, locations, finding_locs=None):
    fu = list(dict.fromkeys(findings))
    abn = [f for f in fu if f.lower() not in ("normal", "unchanged", "exclude", "nan", "")]
    nl = "\n"
    if not abn:
        fs = "- No significant abnormalities detected"
        imp = "Normal chest X-ray. No acute cardiopulmonary disease."
    else:
        lines = []
        for f in abn:
            matched = (finding_locs or {}).get(f, [])[:3]
            loc_str = f" ({', '.join(matched)})" if matched else ""
            lines.append(f"- {f.capitalize()}{loc_str}")
        fs = nl.join(lines)
        if len(abn) == 1:
            imp = f"{abn[0].capitalize()} identified. Clinical correlation recommended."
        else:
            top = ", ".join(a.capitalize() for a in abn[:4])
            imp = f"Multiple findings: {top}. Clinical correlation and follow-up recommended."
    lu = list(dict.fromkeys(locations))
    locs_str = ", ".join(lu) if lu else "Not specified"
    return f"FINDINGS:{nl}{fs}{nl}{nl}LOCATIONS:{nl}{locs_str}{nl}{nl}IMPRESSION:{nl}{imp}"


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Diagnostic fine-tuning")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    os.makedirs(args.output, exist_ok=True)

    # Load and parse CSV
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    df["labels_locs_parsed"] = df["LabelsLocalizationsBySentence"].apply(safe_parse_list)
    df["findings"], df["locations"], df["finding_locs"] = zip(
        *df["labels_locs_parsed"].apply(split_findings_locations)
    )
    df["num_findings"] = df["findings"].apply(len)

    def resolve_path(row):
        name = row["ImageID"]
        if not name.lower().endswith(".png"):
            name += ".png"
        return os.path.join(args.images, name)

    df["image_path"] = df.apply(resolve_path, axis=1)
    df = df[df["image_path"].apply(os.path.exists)]
    df = df[df["num_findings"] > 0]
    print(f"Valid samples: {len(df)}")

    # Curriculum sorting
    all_findings = [f for fl in df["findings"] for f in fl]
    fc = Counter(all_findings)

    def difficulty(row):
        abn = [f for f in row["findings"] if f.lower() not in ("normal", "unchanged", "")]
        score = len(abn) * 2 + len(row["locations"])
        for f in abn:
            freq = fc.get(f, 0)
            score += 5 if freq <= 5 else (3 if freq <= 20 else (1 if freq <= 50 else 0))
        return score

    df["difficulty"] = df.apply(difficulty, axis=1)
    df = df.sort_values("difficulty").reset_index(drop=True)

    # Build examples
    examples = []
    for _, row in df.iterrows():
        v = row.get("Projection", "PA")
        v = v if pd.notna(v) and v else "unknown"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"Analyze this chest X-ray (projection: {v}). Provide FINDINGS, LOCATIONS, and IMPRESSION."},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": build_assistant_response(
                row["findings"], row["locations"], row.get("finding_locs", {})
            )}]},
        ]
        examples.append({"image_path": row["image_path"], "messages": messages})

    n = len(examples)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)

    def to_ds(exs):
        return Dataset.from_dict({
            "image_path": [e["image_path"] for e in exs],
            "messages": [e["messages"] for e in exs],
        })

    dataset = DatasetDict({
        "train": to_ds(examples[:n_train]),
        "validation": to_ds(examples[n_train:n_train + n_val]),
        "test": to_ds(examples[n_train + n_val:]),
    })
    for split, ds in dataset.items():
        print(f"  {split}: {len(ds)}")

    # Load model
    compute_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
    )
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        attn_implementation="sdpa", torch_dtype=compute_dtype, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        exclude_modules=["vision_tower", "multi_modal_projector"],
    )

    def collate_fn(batch):
        texts, images = [], []
        for ex in batch:
            img = Image.open(ex["image_path"]).convert("RGB")
            images.append([img])
            text = processor.apply_chat_template(ex["messages"], add_generation_prompt=False, tokenize=False).strip()
            texts.append(text)
        out = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        for img_list in images:
            for img in img_list:
                try:
                    img.close()
                except Exception:
                    pass
        labels = out["input_ids"].clone()
        pid = processor.tokenizer.pad_token_id
        if pid is not None:
            labels[labels == pid] = -100
        labels[labels == 262144] = -100
        marker_ids = processor.tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
        for i in range(len(texts)):
            ids = out["input_ids"][i].tolist()
            for j in range(len(ids) - len(marker_ids) + 1):
                if ids[j : j + len(marker_ids)] == marker_ids:
                    labels[i, : j + len(marker_ids)] = -100
                    break
        out["labels"] = labels
        return out

    use_bf16 = compute_dtype == torch.bfloat16
    training_args = SFTConfig(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", learning_rate=args.lr, warmup_ratio=0.15,
        max_grad_norm=0.3, lr_scheduler_type="cosine_with_restarts",
        bf16=use_bf16, fp16=not use_bf16, logging_steps=10,
        eval_strategy="steps", eval_steps=50, save_strategy="steps",
        save_steps=50, save_total_limit=5, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="tensorboard", logging_dir=os.path.join(args.output, "logs"),
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, label_names=["labels"],
        dataloader_pin_memory=False, dataloader_num_workers=0,
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        peft_config=peft_config, processing_class=processor,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_p:,} ({100 * trainable / total_p:.2f}%)")

    result = trainer.train()
    print(f"Phase 1 complete. Loss: {result.training_loss:.4f}")
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
