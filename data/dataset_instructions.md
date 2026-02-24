# Dataset Setup

## Primary Dataset: BIMCV PadChest

ExplainMyXray is trained on the [BIMCV PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) dataset.

| Property | Value |
|----------|-------|
| Images | 160,000+ chest X-rays |
| Findings | 174 unique radiological findings |
| Locations | 104 anatomical locations |
| Format | PNG (various bit depths) |
| Labels | Multi-label, physician-annotated |

### Download

PadChest is publicly available for research purposes:

1. Visit the [BIMCV PadChest page](https://bimcv.cipf.es/bimcv-projects/padchest/)
2. Request access and download the dataset
3. Extract images into a folder structure:

```
padchest/
├── PADCHEST_chest_x_ray_images_labels_160K.csv
└── images/
    ├── 0/
    ├── 1/
    ...
    └── 37/
```

### Preprocessing

Run the preprocessing pipeline to normalise images:

```bash
python training/dataset_prep.py \
    --csv /path/to/PADCHEST_labels.csv \
    --images /path/to/images \
    --output /path/to/processed
```

This converts all images to 512×512 RGB PNGs with CLAHE enhancement.

## Spatial Annotation Dataset: Indiana University CXR

Phase 2 spatial training uses bounding-box annotations derived from the Indiana University Chest X-ray Collection.

The spatial annotations CSV (`indiana_spatial_data.csv`) contains:
- `prompt`: Text instruction for the model
- `completion`: Expected output with `<loc>` tokens
- `image_path`: Path to the source X-ray image

### Format

Each row maps an image + prompt to a completion containing PaliGemma `<loc>` tokens:

```
<locY1><locX1><locY2><locX2> finding_description
```

Where each `<locNNNN>` value is in [0, 1024] representing normalised coordinates.

## Data Not Included in This Repository

Due to size and licensing, the raw datasets are **not** included. You must download them separately:

- PadChest: ~50 GB of images
- Indiana CXR: Available via [OpenI](https://openi.nlm.nih.gov/)

The fine-tuned model adapter weights **are** included in `model/adapters/`, so you can run inference without the training data.
