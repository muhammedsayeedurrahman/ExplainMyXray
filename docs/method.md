# ExplainMyXray — Method

## Problem Statement

Traditional chest X-ray AI systems fall into two separate categories:
1. **Image classifiers** — output a label ("pneumonia") with no spatial evidence
2. **Disjointed pipelines** — combine a detector (YOLO) with a separate text generator

Both approaches suffer from the **"Black Box" trust deficit**: clinicians cannot verify *where* the AI is looking or *why* it made a decision.

## Our Approach

ExplainMyXray uses a **single unified VLM** (MedGemma-4B) that simultaneously generates:
- **Structured diagnostic text** (FINDINGS, LOCATIONS, IMPRESSION)
- **Spatial bounding boxes** via native `<loc>` token generation

This eliminates the need for separate detection and generation models.

## Two-Phase Sequential Fine-Tuning

### Phase 1: Diagnostic Vocabulary (Text Generation)

**Goal**: Teach MedGemma to produce structured radiology reports.

**Dataset**: BIMCV PadChest (34,614 processed chest X-rays)
- 174 unique radiological findings
- 104 anatomical locations
- Multi-label annotations with finding-location pairs

**Preprocessing Pipeline**:
1. 16-bit to 8-bit conversion (percentile normalisation)
2. Auto-crop of dark scanner edges
3. Square padding
4. Resize to 512×512
5. CLAHE contrast enhancement (clip=2.0, grid=8×8)
6. Mild sharpening (factor=1.2)

**Training Details**:
- QLoRA: r=64, alpha=128, dropout=0.05
- Targets: `q_proj`, `v_proj` (language decoder only)
- Frozen: `vision_tower`, `multi_modal_projector`
- Optimizer: Paged AdamW 8-bit
- LR: 5e-5 with cosine-with-restarts schedule
- Warmup: 15%
- Effective batch size: 16 (1 × 16 gradient accumulation)
- Early stopping: patience=5 on eval loss
- Prompt masking: only compute loss on assistant response tokens

**Curriculum Learning**:
- Samples sorted by difficulty score
- Difficulty = 2×(num_abnormal_findings) + (num_locations) + rarity_bonus
- Model sees easy cases (normal, single finding) first

### Phase 2: Spatial Geometry (Bounding Boxes)

**Goal**: Teach MedGemma to output precise spatial coordinates.

**Dataset**: Indiana University CXR with bounding-box annotations
- Coordinates converted to PaliGemma `<loc>` token format
- Format: `<locY1><locX1><locY2><locX2>` (normalised 0–1024)

**Key Innovation**: Unfreezing the `multi_modal_projector`
- The projector maps visual embeddings into language space
- By making it trainable, the model learns to encode spatial geometry
- The vision tower remains frozen to preserve pre-trained representations

**Training Details**:
- QLoRA: r=32, alpha=64
- `modules_to_save`: `["multi_modal_projector"]`
- 30 epochs (small dataset requires more passes)
- Gradient accumulation: 8

## Evaluation Methodology

**Metrics**:
- **Exact Match**: Predicted findings exactly equal ground truth
- **Strict Match (≥75%)**: At least 75% finding overlap
- **Soft Match (≥50%)**: At least 50% finding overlap
- **Micro Precision / Recall / F1**: Per-finding aggregated
- **Per-finding Recall**: Breakdown by condition

**Synonym Normalisation**: Findings are mapped to canonical forms before comparison (e.g., "enlarged heart" → "cardiomegaly").

## Results Summary

| Metric | Score |
|--------|-------|
| Token Accuracy | 84.53% |
| Spatial Generalisation | 100% zero-shot on unseen pathologies |

The model demonstrates **zero-shot spatial generalisation**: when presented with diseased X-rays it was never spatially trained on, it correctly draws bounding boxes around pathological regions.
