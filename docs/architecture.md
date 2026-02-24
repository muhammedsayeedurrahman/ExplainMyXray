# ExplainMyXray — Architecture

## System Overview

ExplainMyXray is a unified Vision-Language Model (VLM) system that generates structured radiology reports **and** spatial disease localisation from a single model forward pass.

```
                    ┌──────────────────────────────────┐
                    │         Chest X-Ray Image         │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │    Medical SigLIP Vision Encoder   │
                    │    (Frozen — 400M parameters)      │
                    └──────────────┬───────────────────┘
                                   │ Visual embeddings
                    ┌──────────────▼───────────────────┐
                    │   Multi-Modal Projector (Unfrozen) │
                    │   Maps visual → language space     │
                    └──────────────┬───────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │              Gemma 3 Language Decoder            │
          │         (QLoRA — r=32/64, alpha=64/128)         │
          │                                                  │
          │  Generates:                                      │
          │    • Structured text (FINDINGS / IMPRESSION)     │
          │    • <loc> spatial tokens (bounding boxes)       │
          └──────────────────┬─────────────────────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │         Post-Processing          │
              │  • Parse <loc> → pixel coords    │
              │  • Render bounding box overlays  │
              │  • Format structured report      │
              └─────────────────────────────────┘
```

## Base Model: MedGemma-4B

| Component | Details |
|-----------|---------|
| **Architecture** | PaliGemma (SigLIP encoder + Gemma 3 decoder) |
| **Parameters** | 4 billion |
| **Vision Encoder** | Medical SigLIP — pre-trained on medical images |
| **Language Model** | Gemma 3 2B decoder |
| **Native Capabilities** | `<loc>` geometric token generation |
| **HuggingFace ID** | `google/medgemma-4b-it` |

## Fine-Tuning Strategy

### Phase 1 — Diagnostic Report Generation

- **Method**: QLoRA (4-bit NF4, double quantization)
- **LoRA Config**: r=64, alpha=128, targets: `q_proj`, `v_proj`
- **Frozen**: Vision tower + multi-modal projector
- **Dataset**: 34,614 PadChest X-rays with structured labels
- **Task**: Generate FINDINGS / LOCATIONS / IMPRESSION reports
- **Training**: Curriculum learning (easy → hard), cosine LR with restarts

### Phase 2 — Spatial Localisation

- **Method**: QLoRA with **unfrozen multi-modal projector**
- **LoRA Config**: r=32, alpha=64, targets: `q_proj`, `v_proj`
- **Unfrozen**: `multi_modal_projector` (via `modules_to_save`)
- **Dataset**: Indiana University CXR with bounding-box annotations
- **Task**: Generate `<locY1><locX1><locY2><locX2>` spatial tokens
- **Key Innovation**: Teaching geometry through the projector layer

## Deployment Architecture

```
┌──────────────┐       HTTP        ┌──────────────────┐
│   Streamlit   │  ◄────────────►  │    FastAPI        │
│   Frontend    │   /explain       │    Backend        │
│   (Port 8501) │   /health        │    (Port 8000)    │
└──────────────┘                   └────────┬─────────┘
                                            │
                                   ┌────────▼─────────┐
                                   │  MedGemma-4B      │
                                   │  + QLoRA Adapter   │
                                   │  (4-bit quantized) │
                                   └──────────────────┘
```

## Quantization

| Parameter | Value |
|-----------|-------|
| Quantization | 4-bit NF4 |
| Compute dtype | bfloat16 |
| Double quantization | Yes |
| VRAM usage | ~2.5 GB (model) + ~3 GB (inference) |
| Minimum GPU | 8 GB VRAM |
