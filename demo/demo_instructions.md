# Demo Instructions

## Running the Demo Locally

### Prerequisites

- Python 3.11+
- 8+ GB RAM
- GPU recommended (NVIDIA with 8+ GB VRAM) but not required

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start both servers (one command)
# Windows:
run.bat

# Linux/macOS:
chmod +x run.sh
./run.sh
```

### Manual Start (two terminals)

**Terminal 1 — API backend:**
```bash
python app/api.py
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

### Open in Browser

Navigate to: **http://localhost:8501**

## Using the Application

### Step 1: Upload a Chest X-Ray

- Click the upload area or drag-and-drop a chest X-ray image
- Supported formats: PNG, JPG, JPEG, DICOM
- The image appears in the viewer panel

### Step 2: Analyse

- Click **"Analyse Radiograph"**
- The system sends the image to the MedGemma model
- Processing takes 3–5 seconds (GPU) or 30–60 seconds (CPU)

### Step 3: View Results

The results panel shows:

1. **Diagnostic Report**: Structured text with FINDINGS, LOCATIONS, and IMPRESSION sections
2. **Bounding Box Overlay**: Semi-transparent coloured boxes drawn over detected abnormalities on the X-ray
3. **Metadata**: Processing time, model version, device information

### Step 4: Toggle Theme

- Use the theme toggle button to switch between Dark and Light modes
- The interface is designed for clinical environments (dark mode) and presentations (light mode)

## Reproducing the Video Demo

To reproduce the exact demo shown in the submission video:

1. Start the application as described above
2. Use the demo X-ray provided at `app/assets/demo_xray.png`
3. Upload it and click Analyse
4. The model will generate a diagnostic report with spatial overlays
5. Switch between dark/light themes to show the UI

## Demo Without GPU

The application works fully without a GPU or model weights. When the model is not available, the API returns clinically realistic sample output — the entire UI, bounding box rendering, and report formatting are fully functional.

## Screenshots

After running the demo:
- The viewer shows the X-ray with colour-coded bounding box overlays
- The report panel shows structured FINDINGS / LOCATIONS / IMPRESSION
- Metadata shows processing time and model version
