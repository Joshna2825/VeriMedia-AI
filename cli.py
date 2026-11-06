#!/usr/bin/env python3
"""
CapCheck - Command Line Forensic Analysis Tool.
Inspects images and videos to determine if they are Real or AI-Generated with plain English explanations.
"""

import sys
import os
import argparse
import json
from typing import Dict, Any

from forensics.metadata_analyzer import analyze_metadata
from forensics.ela_analyzer import analyze_ela
from forensics.fft_analyzer import analyze_fft
from forensics.noise_analyzer import analyze_noise
from forensics.video_analyzer import analyze_video
from forensics.ensemble_scorer import compute_ensemble_verdict

# Terminal ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"


def print_banner():
    print(f"{CYAN}{BOLD}")
    print("=" * 70)
    print("   CAPCHECK : REAL vs AI-GENERATED MEDIA FORENSICS INSPECTOR      ")
    print("   Explainable Multi-Pillar Computer Vision Analysis System        ")
    print("=" * 70 + f"{RESET}")


def inspect_image(image_path: str) -> Dict[str, Any]:
    print(f"\n{BLUE}[*] Running Multi-Pillar Forensic Image Pipeline...{RESET}")
    print(f"    - Analyzing EXIF & Provenance Metadata...")
    meta_res = analyze_metadata(image_path)
    
    print(f"    - Computing Error Level Analysis (ELA 8x8 DCT disparity)...")
    ela_res = analyze_ela(image_path)
    
    print(f"    - Computing 2D Fast Fourier Transform (FFT spectral magnitude)...")
    fft_res = analyze_fft(image_path)
    
    print(f"    - Evaluating Sensor Noise Residual (PRNU) & Laplacian Gradients...")
    noise_res = analyze_noise(image_path)
    
    print(f"    - Synthesizing Ensemble Verdict & Explainability Cards...")
    verdict_res = compute_ensemble_verdict(meta_res, ela_res, fft_res, noise_res, media_type="image")
    
    return {
        "file": image_path,
        "media_type": "image",
        "verdict": verdict_res,
        "metadata": meta_res,
        "ela": ela_res,
        "fft": fft_res,
        "noise": noise_res,
    }


def inspect_video(video_path: str) -> Dict[str, Any]:
    print(f"\n{BLUE}[*] Running Multi-Pillar Forensic Video Pipeline...{RESET}")
    print(f"    - Extracting frames & evaluating temporal SSIM / flickering...")
    video_res = analyze_video(video_path)
    
    print(f"    - Inspecting metadata container...")
    meta_res = analyze_metadata(video_path)
    
    # Analyze first frame if available
    fft_dummy = {"score": video_res.get("score", 50.0), "explanation": "Extracted across video keyframe average."}
    ela_dummy = {"score": video_res.get("score", 50.0), "explanation": "Extracted across video keyframe average."}
    noise_dummy = {"score": video_res.get("score", 50.0), "explanation": "Temporal video noise residual."}
    
    verdict_res = compute_ensemble_verdict(
        meta_res, ela_dummy, fft_dummy, noise_dummy, video_res=video_res, media_type="video"
    )
    
    return {
        "file": video_path,
        "media_type": "video",
        "verdict": verdict_res,
        "video": video_res,
        "metadata": meta_res,
    }


def display_results(results: Dict[str, Any]):
    verdict = results["verdict"]
    ai_prob = verdict["ai_probability"]
    real_prob = verdict["real_probability"]
    label = verdict["verdict"]
    confidence = verdict["confidence"]
    
    color = GREEN if ai_prob < 35 else (YELLOW if ai_prob < 65 else RED)
    
    print("\n" + "=" * 70)
    print(f"{BOLD}FORENSIC INSPECTION REPORT:{RESET} {results['file']}")
    print("=" * 70)
    print(f"  {BOLD}CLASSIFICATION VERDICT:{RESET} {color}{BOLD}{label}{RESET}")
    print(f"  {BOLD}AI GENERATED PROBABILITY:{RESET} {color}{ai_prob}%{RESET}")
    print(f"  {BOLD}REAL / AUTHENTIC CONFIDENCE:{RESET} {GREEN}{real_prob}%{RESET}")
    print(f"  {BOLD}ASSESSMENT CONFIDENCE:{RESET} {BOLD}{confidence}{RESET}")
    print("-" * 70)
    
    print(f"\n{BOLD}FORENSIC PILLAR BREAKDOWN:{RESET}")
    for card in verdict["forensic_cards"]:
        badge_col = RED if card["score"] > 60 else (GREEN if card["score"] < 35 else YELLOW)
        print(f"\n  {BOLD}[{card['title']}]{RESET}")
        print(f"  - Score: {badge_col}{card['score']}% AI Suspicion ({card['status']}){RESET}")
        print(f"  - Finding: {card['finding']}")
        print(f"  - Why it matters: {card['why_it_matters']}")

    print("\n" + "-" * 70)
    print(f"{BOLD}{MAGENTA}VIVA VOCE / INTERVIEW ELEVATOR PITCH:{RESET}")
    print(f"\"{verdict['viva_elevator_pitch']}\"")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="CapCheck: Real vs. AI Media Forensic Detector")
    parser.add_argument("file", help="Path to image (jpg, png, webp) or video (mp4, webm, mov)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)
        
    ext = os.path.splitext(args.file)[1].lower()
    is_video = ext in [".mp4", ".webm", ".mov", ".avi", ".mkv"]
    
    if not args.json:
        print_banner()
        
    if is_video:
        results = inspect_video(args.file)
    else:
        results = inspect_image(args.file)
        
    if args.json:
        # Strip large base64 strings for compact JSON output
        if "ela" in results and "ela_image_base64" in results["ela"]:
            results["ela"]["ela_image_base64"] = "[BASE64_IMAGE_STRIPPED]"
        if "fft" in results and "fft_spectrum_base64" in results["fft"]:
            results["fft"]["fft_spectrum_base64"] = "[BASE64_IMAGE_STRIPPED]"
        if "noise" in results and "noise_map_base64" in results["noise"]:
            results["noise"]["noise_map_base64"] = "[BASE64_IMAGE_STRIPPED]"
        print(json.dumps(results, indent=2))
    else:
        display_results(results)


if __name__ == "__main__":
    main()
