---
library_name: peft
base_model: google/medgemma-4b-it
tags:
  - medgemma
  - medical-ai
  - chest-xray
  - radiology
  - qlora
  - peft
  - spatial-localization
  - paligemma
license: mit
datasets:
  - PadChest
  - Indiana-CXR
language:
  - en
pipeline_tag: image-text-to-text
---

# ExplainMyXray — MedGemma-4B QLoRA Adapter

**AI-powered chest X-ray interpretation with disease localisation.**

> Kaggle MedGemma Impact Challenge Submission

## Model Description

This is a QLoRA adapter for [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) that adds:

1. **Structured Radiology Reports** — FINDINGS, LOCATIONS, and IMPRESSION sections
2. **Spatial Disease Localisation** — Bounding boxes via PaliGemma `<loc>` tokens

### Training

| Phase | Dataset | Samples | Description |
|-------|---------|---------|-------------|
| Phase 1 | PadChest | 34,614 | Diagnostic text generation |
| Phase 2 | Indiana CXR | ~200 | Spatial `<loc>` token training with unfrozen `multi_modal_projector` |

### Architecture

- **Base Model:** google/medgemma-4b-it (PaliGemma: SigLIP + Gemma 3)
- **Method:** 4-bit QLoRA (NF4, double quantization)
- **LoRA:** r=32, alpha=64, targets: q_proj, v_proj
- **Special:** `multi_modal_projector` unfrozen in Phase 2 for geometric reasoning

### Results

| Metric | Score |
|--------|-------|
| Token Accuracy | 84.53% |
| Zero-Shot Spatial Generalisation | 100% (3 test cases) |

## Usage

```python
from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration
from peft import PeftModel
import torch
from PIL import Image

# Load base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/medgemma-4b-it",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "muhammedsayeedurrahman/ExplainMyXray-MedGemma-QLoRA")
model.eval()

# Load processor
processor = AutoProcessor.from_pretrained("muhammedsayeedurrahman/ExplainMyXray-MedGemma-QLoRA")

# Run inference
image = Image.open("chest_xray.png").convert("RGB")
conversation = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Locate abnormalities."}]}
]
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
print(processor.decode(generated_ids, skip_special_tokens=True))
```

## Links

- **GitHub:** [ExplainMyXray](https://github.com/muhammedsayeedurrahman/ExplainMyXray)
- **Base Model:** [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

## License

MIT — For educational and research purposes only. Not intended for clinical diagnostic use.
