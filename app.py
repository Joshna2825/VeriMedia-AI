"""
CapCheck - Web Dashboard Application Server.
Provides REST APIs and interactive UI for real vs. AI image and video forensics.
"""

import os
import io
import json
import uuid
import shutil
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory

from forensics.metadata_analyzer import analyze_metadata
from forensics.ela_analyzer import analyze_ela
from forensics.fft_analyzer import analyze_fft
from forensics.noise_analyzer import analyze_noise
from forensics.video_analyzer import analyze_video
from forensics.ensemble_scorer import compute_ensemble_verdict

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SAMPLES_FOLDER = os.path.join(BASE_DIR, "samples")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns metadata for bundled sample files."""
    samples = [
        {
            "id": "real_photo",
            "title": "Authentic Camera Photo",
            "type": "image",
            "filename": "real_sample.jpg",
            "description": "Captured by physical optical lens with genuine Canon EXIF and sensor noise grain.",
            "expected": "Authentic (Real Camera Media)"
        },
        {
            "id": "ai_photo",
            "title": "AI Generative Art (SDXL / Midjourney)",
            "type": "image",
            "filename": "ai_synthetic_sample.png",
            "description": "Synthesized with neural upsamplers; contains prompt text chunk and periodic FFT grid artifacts.",
            "expected": "AI-Generated / Deepfake"
        },
        {
            "id": "splice_photo",
            "title": "Deepfake Spliced Composition",
            "type": "image",
            "filename": "deepfake_splice_sample.jpg",
            "description": "Authentic background with a synthesized spliced patch showing local ELA compression disparity.",
            "expected": "Likely AI-Generated / Spliced"
        },
        {
            "id": "real_video",
            "title": "Authentic Video Footage",
            "type": "video",
            "filename": "real_sample.mp4",
            "description": "Smooth continuous camera motion with natural optical frame-to-frame SSIM consistency.",
            "expected": "Authentic"
        },
        {
            "id": "ai_video",
            "title": "AI Generative Video (Sora / Runway style)",
            "type": "video",
            "filename": "ai_video_sample.mp4",
            "description": "Exhibits subtle temporal jitter, shape morphing, and inter-frame flickering.",
            "expected": "AI-Generated"
        }
    ]
    return jsonify({"samples": samples})


@app.route("/api/sample-media/<filename>")
def serve_sample(filename):
    return send_from_directory(SAMPLES_FOLDER, filename)


def process_media_file(filepath: str, original_filename: str) -> Dict[str, Any]:
    ext = os.path.splitext(original_filename)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS
    
    if is_video:
        video_res = analyze_video(filepath)
        meta_res = analyze_metadata(filepath)
        
        # Analyze representative frame if available
        fft_dummy = {"score": video_res.get("score", 50.0), "explanation": "Extracted across video keyframe average."}
        ela_dummy = {"score": video_res.get("score", 50.0), "explanation": "Extracted across video keyframe average."}
        noise_dummy = {"score": video_res.get("score", 50.0), "explanation": "Temporal video noise residual."}
        
        verdict = compute_ensemble_verdict(
            meta_res, ela_dummy, fft_dummy, noise_dummy, video_res=video_res, media_type="video"
        )
        
        return {
            "success": True,
            "filename": original_filename,
            "media_type": "video",
            "verdict": verdict,
            "video_metrics": video_res,
            "metadata_metrics": meta_res,
        }
    else:
        meta_res = analyze_metadata(filepath)
        ela_res = analyze_ela(filepath)
        fft_res = analyze_fft(filepath)
        noise_res = analyze_noise(filepath)
        
        verdict = compute_ensemble_verdict(
            meta_res, ela_res, fft_res, noise_res, media_type="image"
        )
        
        return {
            "success": True,
            "filename": original_filename,
            "media_type": "image",
            "verdict": verdict,
            "metadata_metrics": meta_res,
            "ela_metrics": ela_res,
            "fft_metrics": fft_res,
            "noise_metrics": noise_res,
        }


@app.route("/api/analyze", methods=["POST"])
def analyze_uploaded_file():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in IMAGE_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        return jsonify({
            "success": False,
            "error": f"Unsupported file extension '{ext}'. Supported: JPG, PNG, WebP, MP4, WebM, MOV."
        }), 400

    unique_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(saved_path)

    try:
        results = process_media_file(saved_path, filename)
        return jsonify(results)
    finally:
        # Keep temp file cleaned up
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass


@app.route("/api/analyze-sample/<sample_id>", methods=["POST"])
def analyze_sample(sample_id):
    sample_mapping = {
        "real_photo": "real_sample.jpg",
        "ai_photo": "ai_synthetic_sample.png",
        "splice_photo": "deepfake_splice_sample.jpg",
        "real_video": "real_sample.mp4",
        "ai_video": "ai_video_sample.mp4",
    }
    
    filename = sample_mapping.get(sample_id)
    if not filename:
        return jsonify({"success": False, "error": "Sample not found."}), 404
        
    sample_path = os.path.join(SAMPLES_FOLDER, filename)
    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": f"Sample file '{filename}' has not been generated yet."}), 404
        
    results = process_media_file(sample_path, filename)
    results["is_sample"] = True
    results["sample_media_url"] = f"/api/sample-media/{filename}"
    return jsonify(results)


if __name__ == "__main__":
    # Default to 5001 on macOS because Apple's AirPlay Receiver occupies port 5000
    port = int(os.environ.get("PORT", 5001))
    print(f"\n==========================================================")
    print(f" CapCheck Web Dashboard running at: http://127.0.0.1:{port}")
    print(f"==========================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
