"""
Noise Residual and Texture Forensics Analyzer.
Evaluates physical camera sensor noise (PRNU), synthetic smoothing, and Laplacian edge sharpness.
"""

from typing import Dict, Any
import io
import base64
import numpy as np
from PIL import Image
import cv2


def analyze_noise(image_path: str) -> Dict[str, Any]:
    """
    Analyzes physical sensor noise residuals and texture gradients.
    
    Parameters:
      image_path: path to the target image file.
      
    Returns a dict containing:
      - score: 0 to 100 (0 = authentic physical sensor grain, 100 = synthetic smoothing or artificial noise)
      - noise_std: standard deviation of the high-frequency residual
      - laplacian_var: variance of the Laplacian edge gradient
      - channel_correlation: correlation between RGB noise residuals
      - noise_map_base64: visual amplified representation of sensor noise
      - explanation: plain English summary
      - details: technical metrics
    """
    try:
        with Image.open(image_path) as img:
            rgb_img = img.convert("RGB")
            # Limit dimension for efficient noise calculation
            max_dim = 1024
            if max(rgb_img.size) > max_dim:
                rgb_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            arr = np.array(rgb_img, dtype=np.float32)

        # 1. Compute Noise Residual by subtracting median-filtered image
        # Real camera sensors have microscopic silicon grain (PRNU - Photo-Response Non-Uniformity)
        gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        residual = gray.astype(np.float32) - denoised.astype(np.float32)
        
        noise_std = float(np.std(residual))
        noise_mean = float(np.mean(np.abs(residual)))
        
        # 2. Laplacian Edge Sharpness Variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        
        # 3. Inter-channel Noise Correlation
        # Physical camera sensors with Bayer filters demonstrate high R-G-B noise coupling
        r_res = arr[:, :, 0] - cv2.medianBlur(arr[:, :, 0].astype(np.uint8), 3)
        g_res = arr[:, :, 1] - cv2.medianBlur(arr[:, :, 1].astype(np.uint8), 3)
        b_res = arr[:, :, 2] - cv2.medianBlur(arr[:, :, 2].astype(np.uint8), 3)
        
        rg_corr = float(np.corrcoef(r_res.flatten(), g_res.flatten())[0, 1])
        gb_corr = float(np.corrcoef(g_res.flatten(), b_res.flatten())[0, 1])
        avg_channel_corr = (rg_corr + gb_corr) / 2.0
        
        # 4. Patch Variance Dispersion (Local texture disparity)
        # Evaluates airbrushed skin vs hyper-sharp clothing/jewelry disparity
        patch_size = 16
        ph, pw = gray.shape[0] // patch_size, gray.shape[1] // patch_size
        if ph > 2 and pw > 2:
            patches = gray[:ph * patch_size, :pw * patch_size].reshape(ph, patch_size, pw, patch_size).swapaxes(1, 2)
            patch_vars = patches.var(axis=(2, 3))
            p_mean = float(np.mean(patch_vars))
            p_std = float(np.std(patch_vars))
            patch_dispersion = p_std / (p_mean + 1e-6)
        else:
            patch_dispersion = 0.5

        # 5. Color Saturation Variance (Midjourney/Diffusion artistic saturation swings)
        hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
        sat_std = float(np.std(hsv[:, :, 1]))

        # 6. Generate Visual Noise Residual Map
        # Amplify residual and center around gray 128
        amplified_noise = np.clip(128.0 + (residual * 15.0), 0, 255).astype(np.uint8)
        _, enc_buf = cv2.imencode(".png", amplified_noise)
        b64_str = base64.b64encode(enc_buf).decode("utf-8")
        noise_map_base64 = f"data:image/png;base64,{b64_str}"

        # 7. Scoring Logic:
        # Real camera images typically have:
        # - Moderate noise standard deviation (1.8 <= noise_std <= 8.5) representing natural physical sensor grain
        # - Balanced Laplacian variance (80 <= laplacian_var <= 900)
        # - Natural channel correlation (0.45 <= avg_channel_corr <= 0.85)
        #
        # AI images typically have:
        # - Extremely low noise_std (< 1.2) = "plastic" ultra-smooth AI skin/surfaces
        # - OR unnaturally uniform synthetic noise (noise_std > 11.0 with weak channel correlation)
        # - Hyper-sharp Laplacian variance (> 1200) with airbrushed patches (patch_dispersion > 0.95)
        # - Exaggerated color saturation swings (sat_std > 48.0)
        
        score = 25.0
        findings = []

        if noise_std < 1.2:
            score += 45.0
            findings.append("Unnaturally low sensor noise (synthetic plastic/airbrushed smoothing)")
        elif noise_std > 11.0:
            score += 35.0
            findings.append("Atypical high noise variance lacking optical sensor structure")
            
        if laplacian_var < 50.0:
            score += 20.0
            findings.append("Abnormally low edge gradient variance (overly blurred synthesized micro-textures)")
        elif laplacian_var > 1200.0:
            score += 25.0
            findings.append("Hyper-sharp artificial edge micro-gradients")

        if patch_dispersion > 0.95 and laplacian_var > 200.0:
            score += 25.0
            findings.append("High local texture disparity (smooth airbrushed skin paired with hyper-detailed textures)")

        if sat_std > 48.0:
            score += 15.0
            findings.append("High color saturation variance (characteristic of generative AI color grading)")

        if avg_channel_corr < 0.35:
            score += 15.0
            findings.append("Low inter-channel noise coupling (inconsistent with Bayer filter sensor physics)")

        score = max(5.0, min(95.0, score))
        
        explanation = (
            f"Noise Residual Analysis examined microscopic pixel-level sensor grain. "
            f"Residual noise standard deviation: {noise_std:.2f}, Laplacian edge sharpness: {laplacian_var:.1f}. "
            f"Physical camera sensors produce natural Poisson photon noise (PRNU), "
            f"whereas generative AI models frequently display synthetic over-smoothing ('plastic texture') "
            f"or synthetic noise that lacks natural hardware Bayer filter correlation."
        )

        return {
            "score": round(score, 1),
            "noise_std": round(noise_std, 3),
            "laplacian_var": round(laplacian_var, 2),
            "channel_correlation": round(avg_channel_corr, 3),
            "noise_map_base64": noise_map_base64,
            "explanation": explanation,
            "details": f"Sensor Noise σ: {noise_std:.2f}, Edge Sharpness: {laplacian_var:.1f}, Channel Coupling: {avg_channel_corr:.2f}"
        }

    except Exception as e:
        return {
            "score": 50.0,
            "noise_std": 0.0,
            "laplacian_var": 0.0,
            "channel_correlation": 0.0,
            "noise_map_base64": "",
            "explanation": f"Noise residual analysis error: {str(e)}",
            "details": "Could not extract sensor noise residuals."
        }
