# 🔍 CapCheck: Real vs. AI-Generated Image & Video Detector

> **An explainable digital media forensics system designed to detect whether an image or video is Authentic (Camera-Captured) or AI-Generated (Midjourney, Stable Diffusion, DALL-E, Sora, Runway, Deepfakes) with transparent explanations for every calculation.**

---

## 🌟 Key Features

- **Multi-Modal Inspection**: Analyzes both **Images** (JPEG, PNG, WebP) and **Videos** (MP4, WebM, MOV).
- **5 Explainable Forensic Pillars**:
  1. **Metadata & Provenance Forensics**: Inspects camera hardware EXIF records vs. AI generation prompt chunks (`parameters`, `Software`).
  2. **Error Level Analysis (ELA)**: Recompresses images to detect 8x8 DCT compression discrepancies and spliced face-swap boundaries.
  3. **2D Fast Fourier Transform (FFT)**: Converts images to frequency space to detect periodic checkerboard grid artifacts left by neural upsamplers.
  4. **Noise Residual & Texture Forensics**: Analyzes physical silicon sensor grain (PRNU) vs. synthetic "plastic" over-smoothing.
  5. **Video Temporal Consistency**: Measures inter-frame Structural Similarity (SSIM) and temporal jitter to catch generative flickering and morphing.
- **"Explain Everything" Philosophy**: Every metric, score, and visual map is explained in plain English, designed for college viva, interviews, and presentations.
- **Interactive Web Dashboard**: Side-by-side visualizers, video player with frame anomaly timeline, and printable forensic audit report.
- **Command-Line Interface (CLI)**: Direct terminal analysis with colored output and JSON export.
- **Pre-Loaded Test Media**: Includes realistic sample images and video clips for instant 1-click testing.

---

## 🚀 Quick Start Guide

### 1. Run the Web Dashboard

To launch the web interface with one command:

```bash
./run.sh
```

Or manually:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run Flask application
python3 app.py
```

Then open your browser at **[http://127.0.0.1:5001](http://127.0.0.1:5001)**.

---

### 2. Run via Command Line (CLI)

You can analyze any image or video directly from the terminal:

```bash
# Inspect authentic camera photo
./venv/bin/python3 cli.py samples/real_sample.jpg

# Inspect AI-generated image
./venv/bin/python3 cli.py samples/ai_synthetic_sample.png

# Inspect deepfake splice photo
./venv/bin/python3 cli.py samples/deepfake_splice_sample.jpg

# Inspect authentic video
./venv/bin/python3 cli.py samples/real_sample.mp4

# Inspect AI video
./venv/bin/python3 cli.py samples/ai_video_sample.mp4
```

To export raw JSON data:
```bash
./venv/bin/python3 cli.py samples/real_sample.jpg --json
```

---

### 3. Run Automated Tests

To verify all forensic engines:

```bash
./venv/bin/python3 -m unittest tests/test_forensics.py
```

---

## 📁 Project Structure

```
Joshnaa/
├── app.py                      # Flask web server & REST API
├── cli.py                      # Command-line inspection tool
├── generate_samples.py         # Script to create authentic & synthetic test media
├── run.sh                      # One-click launcher script
├── requirements.txt            # Python dependencies
├── PROJECT_EXPLANATION.md      # Comprehensive Viva Voce & Presentation Defense Guide
├── README.md                   # Project overview & documentation
│
├── forensics/                  # Core Explainable Forensic Engines
│   ├── __init__.py
│   ├── metadata_analyzer.py    # Pillar 1: Camera EXIF & AI prompt chunk detection
│   ├── ela_analyzer.py         # Pillar 2: Error Level Analysis & compression heatmaps
│   ├── fft_analyzer.py         # Pillar 3: 2D FFT spectral checkerboard detection
│   ├── noise_analyzer.py       # Pillar 4: Sensor noise residual (PRNU) & texture sharpness
│   ├── video_analyzer.py       # Pillar 5: Video frame sampling, SSIM & temporal jitter
│   └── ensemble_scorer.py      # Weighted ensemble verdict & explainability generator
│
├── samples/                    # Bundled Test Media
│   ├── real_sample.jpg         # Authentic photo with Canon EXIF & sensor noise
│   ├── ai_synthetic_sample.png # AI synthetic image with prompt metadata & FFT artifacts
│   ├── deepfake_splice_sample.jpg # Spliced image with ELA compression disparity
│   ├── real_sample.mp4         # Authentic video with smooth optical flow
│   └── ai_video_sample.mp4     # AI video with temporal jitter and frame morphing
│
├── static/                     # Frontend Assets
│   ├── css/style.css           # Modern dark-mode responsive dashboard theme
│   └── js/app.js               # Interactive canvas, comparison tabs & chart rendering
│
├── templates/
│   └── index.html              # Interactive web dashboard template
│
└── tests/
    └── test_forensics.py       # Unit test suite
```

---

## 🎓 Viva Voce & Presentation Guide

For complete details on:
- How to explain every algorithm in 2 minutes
- The physics of cameras vs. neural network decoders
- Mathematical formulas for 2D FFT, ELA, and SSIM
- Top 10 Viva Voce questions & model answers
- 10-Slide presentation outline for PowerPoint

👉 **Read [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)**.
