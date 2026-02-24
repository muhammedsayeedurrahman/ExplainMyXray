"""
Phase 2: Spatial Annotation Fine-Tuning for ExplainMyXray.

Trains on bounding-box annotated data using PaliGemma's native <loc> tokens.
The multi_modal_projector is unfrozen so visual features learn geometry.

Usage:
    python training/finetune_phase2.py \
        --csv /path/to/indiana_spatial_data.csv \
        --output /path/to/spatial-output \
        --phase1_ckpt /path/to/phase1/checkpoint
"""

import argparse
import gc
import os
import time

import psutil
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/medgemma-4b-it"


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Spatial fine-tuning")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase1_ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading spatial dataset: {args.csv}")
    dataset = load_dataset("csv", data_files=args.csv, split="train")

    print(f"Loading {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto",
    )

    if args.phase1_ckpt and os.path.exists(args.phase1_ckpt):
        print(f"Loading Phase 1 adapter from {args.phase1_ckpt}...")
        model = PeftModel.from_pretrained(model, args.phase1_ckpt, is_trainable=True)

    lora_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
        modules_to_save=["multi_modal_projector"],
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_p:,} ({100 * trainable / total_p:.2f}%)")

    def format_data(examples):
        texts, paths = [], []
        for prompt, completion, img_path in zip(
            examples["prompt"], examples["completion"], examples["image_path"]
        ):
            conv = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": completion}]},
            ]
            texts.append(processor.apply_chat_template(conv, add_generation_prompt=False))
            paths.append(img_path)
        return {"text": texts, "image_path": paths}

    dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        images = []
        for ex in examples:
            img = load_image(ex["image_path"])
            if img is None:
                img = Image.new("RGB", (512, 512), color="black")
            images.append([img])
        batch = processor(text=texts, images=images, padding=True,
                          truncation=True, max_length=512, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for i in range(len(texts)):
            user_prompt = texts[i].split("model\n")[0] + "model\n"
            prompt_ids = processor.tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
            labels[i, : len(prompt_ids)] = -100
            labels[i, batch["attention_mask"][i] == 0] = -100
        batch["labels"] = labels
        return batch

    class MemoryMonitor(TrainerCallback):
        def __init__(self):
            self.last = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            gc.collect()
            torch.cuda.empty_cache()
            if state.global_step % 5 == 0:
                ram = psutil.Process(os.getpid()).memory_info().rss / 1e9
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"[Step {state.global_step}] RAM: {ram:.2f}GB | VRAM: {vram:.2f}GB | {time.time() - self.last:.1f}s/step")
            self.last = time.time()

    training_args = SFTConfig(
        output_dir=args.output, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", learning_rate=args.lr,
        logging_steps=5, save_strategy="epoch", bf16=True,
        dataloader_num_workers=0, remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        processing_class=processor, data_collator=collate_fn,
        callbacks=[MemoryMonitor()],
    )

    result = trainer.train()
    print(f"Phase 2 complete. Loss: {result.training_loss:.4f}")
    trainer.save_model(args.output)
    processor.save_pretrained(args.output)
    print(f"Saved spatial adapter to {args.output}")


if __name__ == "__main__":
    main()
