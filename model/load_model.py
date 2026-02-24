"""
Model loading utilities for ExplainMyXray.

Loads the fine-tuned MedGemma-4B QLoRA adapter and processor
for chest X-ray diagnostic inference with spatial localization.
"""

import os
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

# Default adapter path: model/adapters/ relative to this file
_DEFAULT_ADAPTER_DIR = Path(__file__).resolve().parent / "adapters"

# Base model on HuggingFace Hub
BASE_MODEL_ID = "google/medgemma-4b-it"

MODEL_VERSION = "MedGemma-4B v3.0 QLoRA"
SYSTEM_VERSION = "3.1.0"


def get_adapter_path() -> str:
    """Resolve the adapter weights directory.

    Priority:
      1. ``MODEL_PATH`` environment variable
      2. ``model/adapters/`` inside the repository
      3. Raise FileNotFoundError
    """
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).is_dir():
        return env_path

    if _DEFAULT_ADAPTER_DIR.is_dir() and (
        _DEFAULT_ADAPTER_DIR / "adapter_config.json"
    ).exists():
        return str(_DEFAULT_ADAPTER_DIR)

    raise FileNotFoundError(
        "Adapter weights not found. Set the MODEL_PATH environment variable "
        "or place adapter files in model/adapters/."
    )


def load_processor(adapter_path: str | None = None) -> AutoProcessor:
    """Load the processor / tokenizer from the adapter directory."""
    path = adapter_path or get_adapter_path()
    processor = AutoProcessor.from_pretrained(path)
    return processor


def load_model(adapter_path: str | None = None, device_map: str = "auto"):
    """Load the quantized MedGemma-4B base model with the QLoRA adapter.

    Returns:
        model: The merged PeftModel ready for inference.
        processor: The matched AutoProcessor.
    """
    path = adapter_path or get_adapter_path()

    processor = load_processor(path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    from transformers import PaliGemmaForConditionalGeneration

    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    model = PeftModel.from_pretrained(base_model, path)
    model.eval()

    return model, processor


def get_device_name() -> str:
    """Return a human-readable device string."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"
