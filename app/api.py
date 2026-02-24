import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import torch
from PIL import Image

# Add project root to path so we can import model package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model.load_model import MODEL_VERSION, SYSTEM_VERSION, get_device_name
from model.inference import parse_loc_tokens as _parse_loc_tokens

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
model = None
processor = None


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: Optional[str] = "Region"
    confidence: Optional[float] = None


class ExplanationResponse(BaseModel):
    explanation: str
    bboxes: List[BoundingBox]
    raw_tokens: str
    status: str = "success"
    model_version: str = MODEL_VERSION
    processing_time_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0
    device: str = "N/A"


# ---------------------------------------------------------------------------
# Lifespan -- model loading
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor

    try:
        from model.load_model import load_model
        print("Loading MedGemma-4B with QLoRA adapter (base + adapter)...")
        model, processor = load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("API will run in DEMO MODE — returning sample output clearly labelled as such.")
        model = None

    yield

    model = None
    processor = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ExplainMyXray API",
    description="Chest X-ray diagnostic reporting with spatial localization",
    version=SYSTEM_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_ready": model is not None,
        "model_version": MODEL_VERSION,
        "system_version": SYSTEM_VERSION,
        "device": get_device_name(),
        "mode": "inference" if model is not None else "demo",
    }


@app.post("/explain", response_model=ExplanationResponse)
async def explain_xray(file: UploadFile):
    """Generates text diagnosis and bounding boxes for the given X-ray."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    start_time = time.time()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size

    if model is None:
        # ---------------------------------------------------------------
        # DEMO MODE: Model weights not available on this machine.
        # Returns a clearly-labelled sample response so the UI can be
        # tested end-to-end without GPU / model download.
        # ---------------------------------------------------------------
        elapsed = (time.time() - start_time) * 1000
        return ExplanationResponse(
            explanation=(
                "[DEMO MODE — model not loaded] "
                "The cardiac silhouette appears enlarged with a cardiothoracic "
                "ratio exceeding 0.5, suggesting cardiomegaly. No acute pulmonary "
                "infiltrates identified. Costophrenic angles are clear bilaterally. "
                "The mediastinal contour is within normal limits. No pleural effusion "
                "or pneumothorax is observed."
            ),
            bboxes=[BoundingBox(xmin=150, ymin=150, xmax=350, ymax=350, label="Cardiac Silhouette (demo)")],
            raw_tokens="<loc0250><loc0250><loc0750><loc0750>",
            status="demo",
            model_version=MODEL_VERSION,
            processing_time_ms=round(elapsed, 1),
            image_width=img_w,
            image_height=img_h,
            device=get_device_name(),
        )

    try:
        prompt = "Locate abnormalities."
        conversation = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
        ]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

        clean_text, bbox_objs, raw_tokens = _parse_loc_tokens(output_text, image_size=image.size)
        elapsed = (time.time() - start_time) * 1000

        # Convert dataclass bboxes to Pydantic models
        bboxes = [
            BoundingBox(xmin=b.xmin, ymin=b.ymin, xmax=b.xmax, ymax=b.ymax, label=b.label)
            for b in bbox_objs
        ]

        return ExplanationResponse(
            explanation=clean_text if clean_text else "No significant abnormalities detected.",
            bboxes=bboxes,
            raw_tokens=raw_tokens,
            status="success",
            model_version=MODEL_VERSION,
            processing_time_ms=round(elapsed, 1),
            image_width=img_w,
            image_height=img_h,
            device=get_device_name(),
        )

    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
