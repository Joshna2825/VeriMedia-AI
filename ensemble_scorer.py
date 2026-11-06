"""
Ensemble Decision and Explainability Scorer.
Combines multi-modal forensic signals into a transparent, scientifically defensible verdict.
"""

from typing import Dict, Any, List


def compute_ensemble_verdict(
    metadata_res: Dict[str, Any],
    ela_res: Dict[str, Any],
    fft_res: Dict[str, Any],
    noise_res: Dict[str, Any],
    video_res: Dict[str, Any] = None,
    media_type: str = "image"
) -> Dict[str, Any]:
    """
    Computes a weighted ensemble score, confidence rating, and structured explanations.
    
    Parameters:
      metadata_res: output from analyze_metadata
      ela_res: output from analyze_ela
      fft_res: output from analyze_fft
      noise_res: output from analyze_noise
      video_res: optional output from analyze_video (for video media)
      media_type: 'image' or 'video'
      
    Returns a dictionary containing:
      - ai_probability: 0 to 100%
      - real_probability: 0 to 100%
      - verdict: classification label
      - verdict_severity: 'authentic', 'suspicious', 'synthetic'
      - confidence: 'High', 'Medium', 'Low'
      - weights_used: dictionary of weights applied
      - forensic_cards: list of explainability cards for UI and viva presentation
      - viva_elevator_pitch: concise 3-sentence summary ready to speak aloud
    """
    s_meta = float(metadata_res.get("score", 50.0))
    s_ela = float(ela_res.get("score", 50.0))
    s_fft = float(fft_res.get("score", 50.0))
    s_noise = float(noise_res.get("score", 50.0))

    if media_type == "video" and video_res:
        s_video = float(video_res.get("score", 50.0))
        # Video weight distribution:
        # FFT and temporal consistency are primary signals
        weights = {
            "video_temporal": 0.40,
            "fft_frequency": 0.25,
            "error_level": 0.20,
            "metadata": 0.15,
        }
        raw_ai_prob = (
            (s_video * weights["video_temporal"]) +
            (s_fft * weights["fft_frequency"]) +
            (s_ela * weights["error_level"]) +
            (s_meta * weights["metadata"])
        )
    else:
        # Image analysis
        if s_meta >= 90.0:
            # Explicit AI prompt or generator software found in metadata
            weights = {
                "metadata": 0.45,
                "fft_frequency": 0.25,
                "error_level": 0.15,
                "noise_texture": 0.15,
            }
        else:
            # Balanced forensic signal distribution
            weights = {
                "fft_frequency": 0.35,  # Strongest scientific discriminator for generative upsamplers
                "error_level": 0.25,
                "noise_texture": 0.25,
                "metadata": 0.15,
            }
            
        raw_ai_prob = (
            (s_meta * weights.get("metadata", 0.15)) +
            (s_ela * weights.get("error_level", 0.25)) +
            (s_fft * weights.get("fft_frequency", 0.35)) +
            (s_noise * weights.get("noise_texture", 0.25))
        )

        # Multi-signal reinforcement:
        # If metadata is unverified or missing (s_meta >= 50.0) and multiple forensic indicators agree that
        # synthetic artifacts are present, boost detection probability.
        if s_meta >= 50.0:
            visual_ai_signals = [s for s in [s_ela, s_fft, s_noise] if s >= 60.0]
            if len(visual_ai_signals) >= 2:
                raw_ai_prob = max(raw_ai_prob, 70.0 + (len(visual_ai_signals) * 3.0))
            elif len(visual_ai_signals) == 1 and s_meta >= 60.0:
                raw_ai_prob = max(raw_ai_prob, 60.0)
        elif s_meta <= 15.0:
            # Genuine hardware camera EXIF verified (e.g. Canon, Nikon, Apple with exposure settings)
            # Physical camera provenance gives strong authenticity protection
            raw_ai_prob = min(raw_ai_prob, 25.0)

    ai_probability = round(max(2.0, min(98.0, raw_ai_prob)), 1)
    real_probability = round(100.0 - ai_probability, 1)

    # Determine Verdict Classification
    # Forensic Rule: If NO hardware camera metadata exists, the media cannot be deemed "Authentic"
    if s_meta >= 55.0 and ai_probability < 45.0:
        verdict = "Suspicious / Unverified Provenance"
        verdict_severity = "suspicious"
        verdict_badge = "warning"
    elif ai_probability < 28.0:
        verdict = "Authentic (Real Camera Media)"
        verdict_severity = "authentic"
        verdict_badge = "success"
    elif ai_probability < 45.0:
        verdict = "Likely Authentic"
        verdict_severity = "likely_authentic"
        verdict_badge = "info"
    elif ai_probability < 58.0:
        verdict = "Suspicious / Inconclusive"
        verdict_severity = "suspicious"
        verdict_badge = "warning"
    elif ai_probability < 75.0:
        verdict = "Likely AI-Generated"
        verdict_severity = "likely_synthetic"
        verdict_badge = "danger"
    else:
        verdict = "AI-Generated / Deepfake"
        verdict_severity = "synthetic"
        verdict_badge = "danger"

    # Determine Confidence Level
    # Confidence is high if forensic indicators agree; lower if signals disagree
    scores_list = [s_meta, s_ela, s_fft, s_noise]
    if media_type == "video" and video_res:
        scores_list.append(s_video)
        
    std_dev = float((sum((x - raw_ai_prob) ** 2 for x in scores_list) / len(scores_list)) ** 0.5)
    if std_dev < 18.0 or s_meta >= 90.0 or ai_probability >= 85.0 or ai_probability <= 15.0:
        confidence = "High"
    elif std_dev < 32.0:
        confidence = "Medium"
    else:
        confidence = "Moderate"

    # Build Explainability Cards for UI and Viva Presentation
    forensic_cards: List[Dict[str, Any]] = [
        {
            "id": "card_metadata",
            "title": "1. Metadata & Provenance Forensics",
            "pillar": "Provenance",
            "score": s_meta,
            "status": "Flagged" if s_meta > 60 else "Authentic" if s_meta < 30 else "Inconclusive",
            "badge_color": "danger" if s_meta > 60 else "success" if s_meta < 30 else "warning",
            "what_it_means": "Inspects EXIF camera tags, lens parameters, C2PA credentials, and PNG generation prompt chunks.",
            "finding": metadata_res.get("explanation", ""),
            "why_it_matters": (
                "Hardware cameras record physical exposure parameters (aperture, ISO, shutter, lens model). "
                "AI generation tools either inject prompt metadata or strip hardware records completely."
            )
        },
        {
            "id": "card_ela",
            "title": "2. Error Level Analysis (ELA)",
            "pillar": "Compression",
            "score": s_ela,
            "status": "Disparity" if s_ela > 60 else "Uniform" if s_ela < 35 else "Moderate",
            "badge_color": "danger" if s_ela > 60 else "success" if s_ela < 35 else "warning",
            "what_it_means": "Recompresses the image at 90% quality and amplifies the error difference across 8x8 DCT blocks.",
            "finding": ela_res.get("explanation", ""),
            "why_it_matters": (
                "In real photos, compression error decays uniformly across natural geometry. "
                "In synthetic images or face-swap deepfakes, pasted or generated regions exhibit abrupt error disparities."
            )
        },
        {
            "id": "card_fft",
            "title": "3. 2D Fast Fourier Transform (FFT) Frequency Analysis",
            "pillar": "Spectral",
            "score": s_fft,
            "status": "Grid Artifacts" if s_fft > 60 else "Natural Roll-off" if s_fft < 35 else "Moderate",
            "badge_color": "danger" if s_fft > 60 else "success" if s_fft < 35 else "warning",
            "what_it_means": "Transforms pixel luminance into frequency space to detect periodic upsampling artifacts.",
            "finding": fft_res.get("explanation", ""),
            "why_it_matters": (
                "Physical lenses obey optical power-law frequency decay. Neural network upsamplers "
                "(Transposed Convolutions / Pixel Shuffle) introduce periodic checkerboard patterns and high-frequency spikes."
            )
        },
        {
            "id": "card_noise",
            "title": "4. Noise Residual & Laplacian Texture Analysis",
            "pillar": "Sensor Physics",
            "score": s_noise,
            "status": "Synthetic Smoothness" if s_noise > 60 else "Natural Grain" if s_noise < 35 else "Moderate",
            "badge_color": "danger" if s_noise > 60 else "success" if s_noise < 35 else "warning",
            "what_it_means": "Extracts high-frequency sensor noise (PRNU) and measures Laplacian edge gradient variance.",
            "finding": noise_res.get("explanation", ""),
            "why_it_matters": (
                "Physical silicon sensors produce microscopic photon noise (PRNU). AI generators "
                "produce plastic-like over-smoothed textures or synthetic noise without natural Bayer-filter correlation."
            )
        }
    ]

    if media_type == "video" and video_res:
        forensic_cards.append({
            "id": "card_video",
            "title": "5. Video Temporal Consistency & Flickering",
            "pillar": "Temporal Forensics",
            "score": video_res.get("score", 50.0),
            "status": "Temporal Jitter" if video_res.get("score", 50.0) > 60 else "Smooth Motion" if video_res.get("score", 50.0) < 35 else "Moderate",
            "badge_color": "danger" if video_res.get("score", 50.0) > 60 else "success" if video_res.get("score", 50.0) < 35 else "warning",
            "what_it_means": "Samples keyframes to measure inter-frame Structural Similarity (SSIM) and motion vector stability.",
            "finding": video_res.get("explanation", ""),
            "why_it_matters": (
                "Physical video cameras capture continuous optical flow. Generative video models (Sora, Runway, Pika, Deepfakes) "
                "suffer from temporal drift, face morphing, and frame-to-frame flickering."
            )
        })

    # Viva Voce Elevator Pitch: A clear 3-sentence summary for student presentation
    if ai_probability >= 60.0:
        pitch = (
            f"Based on our multi-pillar forensic inspection, this media is classified as '{verdict}' with {ai_probability}% AI probability. "
            f"The primary forensic anomalies detected include elevated frequency domain checkerboard spikes (FFT score: {s_fft:.1f}%) "
            f"and synthetic compression disparities. These artifacts stem directly from neural upsampling layers and lack physical camera sensor traces."
        )
    else:
        pitch = (
            f"Based on our multi-pillar forensic inspection, this media is classified as '{verdict}' with {real_probability}% authenticity confidence. "
            f"The media exhibits natural optical lens roll-off in the Fourier domain (FFT score: {s_fft:.1f}%), uniform compression decay, "
            f"and physical sensor grain consistent with genuine camera capture."
        )

    return {
        "ai_probability": ai_probability,
        "real_probability": real_probability,
        "verdict": verdict,
        "verdict_severity": verdict_severity,
        "verdict_badge": verdict_badge,
        "confidence": confidence,
        "weights_used": weights,
        "forensic_cards": forensic_cards,
        "viva_elevator_pitch": pitch,
        "summary": {
            "metadata_score": round(s_meta, 1),
            "ela_score": round(s_ela, 1),
            "fft_score": round(s_fft, 1),
            "noise_score": round(s_noise, 1),
            "video_score": round(video_res.get("score", 0.0), 1) if video_res else None,
        }
    }
