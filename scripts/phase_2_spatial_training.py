"""
Phase 2: Spatial Annotation Training (Bounding Boxes) — Original Script.

NOTE: This is the original development script preserved for reference.
For the clean, CLI-driven version, see training/finetune_phase2.py.

This script trains the model on indiana_spatial_data.csv containing
precise <locY1><locX1><locY2><locX2> PaliGemma tokens. The
multi_modal_projector is unfrozen so visual features can map to
geometric constraints.
"""

import os
import gc
import torch
torch.cuda.empty_cache()
import psutil
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from PIL import Image

# ==============================================================================
# CONFIGURATION — Update these paths for your environment
# ==============================================================================
MODEL_ID = "google/medgemma-4b-it"
CSV_PATH = os.environ.get("SPATIAL_CSV", "indiana_spatial_data.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/phase2-spatial")
# Optional: Phase 1 Checkpoint to start from (set to None to train from base)
PHASE_1_CKPT = os.environ.get("PHASE1_CKPT", None)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load Dataset
print("Loading spatial dataset...")
dataset = load_dataset("csv", data_files=CSV_PATH, split="train")

def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

# 3. Load Processor and Model
print("Loading Base Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)

# If we want to build upon Phase 1's vocabulary, we load its adapter first:
if PHASE_1_CKPT and os.path.exists(PHASE_1_CKPT):
    print(f"Loading Phase 1 adapter weights from {PHASE_1_CKPT}...")
    model = PeftModel.from_pretrained(model, PHASE_1_CKPT, is_trainable=True)

# 4. LoRA Configuration (Crucial: Unfreeze Projector)
print("Configuring LoRA and Unfreezing Projector...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    # We explicitly exclude the heavy vision tower, but ALLOW the projector
    modules_to_save=["multi_modal_projector"]
)

model = get_peft_model(model, lora_config)

# Verify trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_p = sum(p.numel() for p in model.parameters())
print(f'Trainable Parameters: {trainable:,} / {total_p:,} ({100*trainable/total_p:.2f}%)')

# 5. Data formatting
def format_data(examples):
    texts = []
    image_paths = []

    for prompt, labels_text, img_path in zip(examples['prompt'], examples['completion'], examples['image_path']):
        # Format conversation exactly as PaliGemma/MedGemma expects
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": labels_text}]}
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        texts.append(text_prompt)
        image_paths.append(img_path)

    return {"text": texts, "image_path": image_paths}

dataset = dataset.map(format_data, batched=True, remove_columns=dataset.column_names)

def collate_fn(examples):
    texts = [ex["text"] for ex in examples]
    images = []
    for ex in examples:
        img = load_image(ex["image_path"])
        if img is None:
            img = Image.new('RGB', (512, 512), color='black') # Fallback to prevent crash
        images.append([img])

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Mask user prompt in labels so we only calculate loss on the Assistant's bounding box output
    labels = batch["input_ids"].clone()
    for i in range(len(texts)):
        user_prompt = texts[i].split("model\n")[0] + "model\n"
        prompt_ids = processor.tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
        labels[i, :len(prompt_ids)] = -100
        # Also mask padding
        labels[i, batch["attention_mask"][i] == 0] = -100

    batch["labels"] = labels
    return batch

# 6. Training Configuration & Memory Monitor
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=30, # Small dataset = high epochs
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    learning_rate=5e-5, # Slightly higher LR for projector initialization
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True}
)

from transformers import TrainerCallback
import time

class MemoryMonitorCallback(TrainerCallback):
    def __init__(self):
        self.last_step_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        step_duration = current_time - self.last_step_time
        self.last_step_time = current_time

        gc.collect()
        torch.cuda.empty_cache()

        p = psutil.Process(os.getpid())
        ram_gb = p.memory_info().rss / 1e9
        vram_gb = torch.cuda.memory_allocated() / 1e9

        if state.global_step % 5 == 0:
            print(f"[Phase 2 - Step {state.global_step}] RAM: {ram_gb:.2f}GB | VRAM: {vram_gb:.2f}GB | Speed: {step_duration:.1f}s")

        if ram_gb > 29.0 or vram_gb > 11.0:
            print("CRITICAL: Memory cap exceeded. Flushing...")
            torch.cuda.empty_cache()
            if ram_gb > 29.0: control.should_training_stop = True

# 7. Start Training
print("Starting Phase 2 Spatial Training...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=processor,
    data_collator=collate_fn,
    callbacks=[MemoryMonitorCallback()]
)

train_result = trainer.train()
print(f"Phase 2 Complete! Loss: {train_result.training_loss:.4f}")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Saved Spatial Projector Adapter.")
