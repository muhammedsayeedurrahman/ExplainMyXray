"""
Inference utilities for ExplainMyXray.

Handles:
  - PaliGemma ``<loc>`` token parsing into bounding-box coordinates
  - Single-image diagnostic inference
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "Region"
    confidence: Optional[float] = None


@dataclass
class AnalysisResult:
    """Full result of a single X-ray analysis."""
    explanation: str
    bboxes: List[BoundingBox] = field(default_factory=list)
    raw_tokens: str = ""
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    device: str = "N/A"


# ---------------------------------------------------------------------------
# Loc-token parser
# ---------------------------------------------------------------------------
_LOC_PATTERN = re.compile(
    r"(<loc\d{4}>)(<loc\d{4}>)(<loc\d{4}>)(<loc\d{4}>)"
)


def parse_loc_tokens(
    text: str,
    image_size: Tuple[int, int] = (512, 512),
) -> Tuple[str, List[BoundingBox], str]:
    """Extract PaliGemma ``<loc>`` tokens and convert to pixel coordinates.

    PaliGemma encodes bounding boxes as four consecutive ``<locNNNN>`` tokens
    where each integer is in [0, 1024] representing a normalised coordinate.
    The token order is ``<locY1><locX1><locY2><locX2>``.

    Args:
        text: Raw model output containing ``<loc>`` tokens.
        image_size: ``(width, height)`` of the source image.

    Returns:
        A tuple of ``(clean_text, bboxes, raw_tokens_string)``.
    """
    matches = _LOC_PATTERN.findall(text)
    width, height = image_size

    bboxes: List[BoundingBox] = []
    for i, match in enumerate(matches):
        try:
            y1_tok = int(match[0][4:8])
            x1_tok = int(match[1][4:8])
            y2_tok = int(match[2][4:8])
            x2_tok = int(match[3][4:8])

            y1 = (y1_tok / 1024.0) * height
            x1 = (x1_tok / 1024.0) * width
            y2 = (y2_tok / 1024.0) * height
            x2 = (x2_tok / 1024.0) * width

            bboxes.append(BoundingBox(
                xmin=min(x1, x2), ymin=min(y1, y2),
                xmax=max(x1, x2), ymax=max(y1, y2),
                label=f"Region {i + 1}",
            ))
        except Exception:
            continue

    clean_text = _LOC_PATTERN.sub("", text).strip()
    raw_tokens = " ".join("".join(m) for m in matches)
    return clean_text, bboxes, raw_tokens


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str = "Locate abnormalities.",
    max_new_tokens: int = 128,
) -> AnalysisResult:
    """Run a single inference pass on a chest X-ray image.

    Args:
        model: Loaded PaliGemma / MedGemma model.
        processor: Matching AutoProcessor.
        image: PIL RGB image.
        prompt: Text prompt for the model.
        max_new_tokens: Generation length limit.

    Returns:
        An ``AnalysisResult`` with explanation, bounding boxes, and metadata.
    """
    start = time.time()
    img_w, img_h = image.size

    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs = processor(
        text=text_prompt, images=image, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    clean_text, bboxes, raw_tokens = parse_loc_tokens(
        output_text, image_size=image.size
    )
    elapsed_ms = (time.time() - start) * 1000

    return AnalysisResult(
        explanation=clean_text or "No significant abnormalities detected.",
        bboxes=bboxes,
        raw_tokens=raw_tokens,
        processing_time_ms=round(elapsed_ms, 1),
        image_width=img_w,
        image_height=img_h,
        device=device_name,
    )
