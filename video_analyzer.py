"""
Video Temporal Forensics Analyzer.
Detects frame-to-frame temporal inconsistency, morphing, and generative flickering in videos.
"""

from typing import Dict, Any, List
import io
import os
import base64
import numpy as np
import cv2
from PIL import Image

from .fft_analyzer import analyze_fft
from .ela_analyzer import analyze_ela


def compute_frame_ssim(gray1: np.ndarray, gray2: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) between two consecutive grayscale frames.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = gray1.astype(np.float64)
    img2 = gray2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def analyze_video(video_path: str, max_frames_to_sample: int = 24) -> Dict[str, Any]:
    """
    Analyzes a video file for generative temporal anomalies, frame flickering, and deepfake artifacts.
    
    Parameters:
      video_path: path to video file (.mp4, .webm, .mov, etc.)
      max_frames_to_sample: number of evenly spaced frames to evaluate
      
    Returns a dict containing:
      - score: 0 to 100 overall video AI probability
      - avg_ssim: average frame-to-frame structural similarity
      - ssim_jitter: standard deviation of SSIM (measure of temporal flickering)
      - frame_anomaly_timeline: list of per-frame timestamp and anomaly scores for graphing
      - suspicious_frames: keyframe snapshots with highest anomalies
      - explanation: plain English summary for presentation
    """
    if not os.path.exists(video_path):
        return {
            "score": 50.0,
            "error": "Video file not found.",
            "explanation": "Could not access the specified video file."
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "score": 50.0,
            "error": "Failed to decode video stream.",
            "explanation": "Could not open video stream. Please ensure the file is a valid MP4/WebM video."
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    if total_frames <= 0:
        # Fallback reading sequentially
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret or len(frames_list) >= 120:
                break
            frames_list.append(frame)
        cap.release()
        total_frames = len(frames_list)
        fps = 24.0
    else:
        frames_list = None

    if total_frames < 2:
        cap.release()
        return {
            "score": 50.0,
            "error": "Video too short.",
            "explanation": "Video has fewer than 2 frames and cannot be analyzed for temporal consistency."
        }

    # Determine frame indices to sample
    sample_count = min(max_frames_to_sample, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    
    sampled_frames = []
    timestamps = []

    if frames_list is not None:
        for idx in frame_indices:
            if idx < len(frames_list):
                sampled_frames.append(frames_list[idx])
                timestamps.append(idx / fps)
    else:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                sampled_frames.append(frame)
                timestamps.append(idx / fps)
        cap.release()

    if len(sampled_frames) < 2:
        return {
            "score": 50.0,
            "error": "Failed to extract sufficient frames.",
            "explanation": "Unable to extract enough frames for temporal analysis."
        }

    # Step 1: Compute Frame-to-Frame Temporal Consistency (SSIM & Optical Jitter)
    ssim_values = []
    timeline = []
    keyframe_ai_scores = []
    
    # Analyze a few representative keyframes (e.g. 3 frames: start, mid, end) for FFT/ELA to save time
    temp_dir = os.path.dirname(video_path)
    cached_kf_scores = {}
    
    # Select up to 4 keyframe indices to deeply inspect with FFT/ELA
    indices_to_deep_inspect = set(np.linspace(0, len(sampled_frames) - 1, min(4, len(sampled_frames)), dtype=int))
    
    for i in range(len(sampled_frames)):
        cur_frame = sampled_frames[i]
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray, (256, 256))
        
        if i in indices_to_deep_inspect:
            temp_frame_path = os.path.join(temp_dir, f"_temp_kf_{i}.jpg")
            cv2.imwrite(temp_frame_path, cur_frame)
            try:
                kf_fft = analyze_fft(temp_frame_path, target_size=256)
                kf_ela = analyze_ela(temp_frame_path)
                kf_score = (kf_fft.get("score", 50.0) * 0.55) + (kf_ela.get("score", 50.0) * 0.45)
            finally:
                if os.path.exists(temp_frame_path):
                    try:
                        os.remove(temp_frame_path)
                    except OSError:
                        pass
            cached_kf_scores[i] = kf_score
        else:
            # Interpolate or use nearest inspected keyframe score
            nearest = min(indices_to_deep_inspect, key=lambda idx: abs(idx - i))
            kf_score = cached_kf_scores.get(nearest, 50.0)
            
        keyframe_ai_scores.append(kf_score)

        if i > 0:
            prev_frame = sampled_frames[i - 1]
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.resize(prev_gray, (256, 256))
            
            ssim_val = compute_frame_ssim(prev_gray, cur_gray)
            ssim_values.append(ssim_val)
        else:
            ssim_val = 1.0

        time_str = f"{int(timestamps[i] // 60):02d}:{timestamps[i] % 60:04.1f}"
        
        timeline.append({
            "frame_idx": int(frame_indices[i]),
            "timestamp": time_str,
            "time_seconds": round(timestamps[i], 2),
            "ssim": round(ssim_val, 3),
            "keyframe_ai_score": round(kf_score, 1)
        })

    # Step 2: Temporal metrics
    avg_ssim = float(np.mean(ssim_values)) if ssim_values else 0.85
    ssim_jitter = float(np.std(ssim_values)) if ssim_values else 0.05
    avg_kf_score = float(np.mean(keyframe_ai_scores)) if keyframe_ai_scores else 50.0

    # Step 3: Video scoring
    # AI video models (Runway, Sora, Deepfakes) exhibit:
    # 1. High keyframe spectral scores (checkerboard patterns inside frames)
    # 2. Inconsistent temporal progression (high ssim_jitter: sudden spikes or drops)
    # 3. Unnatural micro-flickering in static scenes
    
    # Calculate temporal flickering score (0-100)
    # Natural videos with camera shake have mild jitter (0.005 - 0.025)
    # Generative AI video or deepfake face morphs have jitter > 0.03
    if ssim_jitter > 0.06:
        temporal_score = 90.0
    elif ssim_jitter > 0.03:
        temporal_score = 65.0 + (ssim_jitter - 0.03) * 800.0
    elif ssim_jitter > 0.015:
        temporal_score = 40.0 + (ssim_jitter - 0.015) * 1600.0
    else:
        # Smooth physical optical motion
        temporal_score = 15.0 + (ssim_jitter * 1000.0)

    temporal_score = max(5.0, min(95.0, temporal_score))
    
    # Video score is combination of keyframe spatial forensics and temporal motion consistency
    overall_video_score = (avg_kf_score * 0.60) + (temporal_score * 0.40)
    overall_video_score = max(5.0, min(95.0, overall_video_score))

    # Step 4: Extract visual thumbnails of the most suspicious and cleanest keyframe
    sorted_kf_indices = np.argsort(keyframe_ai_scores)
    highest_idx = int(sorted_kf_indices[-1])
    lowest_idx = int(sorted_kf_indices[0])

    def frame_to_b64(bgr_frame):
        # Resize thumbnail to 320px width
        h, w = bgr_frame.shape[:2]
        thumb_w = 320
        thumb_h = int(h * (thumb_w / w))
        thumb = cv2.resize(bgr_frame, (thumb_w, thumb_h))
        _, enc = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(enc).decode('utf-8')}"

    highest_anomaly_frame = {
        "timestamp": timeline[highest_idx]["timestamp"],
        "score": keyframe_ai_scores[highest_idx],
        "image_base64": frame_to_b64(sampled_frames[highest_idx])
    }
    
    lowest_anomaly_frame = {
        "timestamp": timeline[lowest_idx]["timestamp"],
        "score": keyframe_ai_scores[lowest_idx],
        "image_base64": frame_to_b64(sampled_frames[lowest_idx])
    }

    explanation = (
        f"Video Temporal Analysis evaluated {len(sampled_frames)} sampled keyframes across "
        f"{duration_sec:.1f} seconds of footage. "
        f"Average inter-frame SSIM consistency: {avg_ssim:.3f} (jitter σ: {ssim_jitter:.3f}). "
        f"Average keyframe forensic anomaly: {avg_kf_score:.1f}%. "
        f"Physical camera videos maintain physically continuous optical flow with natural camera shake. "
        f"AI video generators (Sora, Runway, Pika, Deepfakes) introduce micro-flickering, texture morphing, "
        f"and subtle frame-to-frame latent drift."
    )

    return {
        "score": round(overall_video_score, 1),
        "total_frames": total_frames,
        "duration_seconds": round(duration_sec, 2),
        "fps": round(fps, 1),
        "avg_ssim": round(avg_ssim, 3),
        "ssim_jitter": round(ssim_jitter, 3),
        "timeline": timeline,
        "highest_anomaly_frame": highest_anomaly_frame,
        "lowest_anomaly_frame": lowest_anomaly_frame,
        "explanation": explanation,
        "details": f"{total_frames} frames ({duration_sec:.1f}s), Avg SSIM: {avg_ssim:.2f}, Jitter σ: {ssim_jitter:.3f}"
    }
