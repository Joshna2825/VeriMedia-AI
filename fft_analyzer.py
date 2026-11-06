"""
2D Fast Fourier Transform (FFT) Frequency Domain Analyzer.
Detects periodic checkerboard artifacts and spectral anomalies characteristic of neural upsamplers.
"""

from typing import Dict, Any, Tuple
import io
import base64
import numpy as np
from PIL import Image
import cv2


def analyze_fft(image_path: str, target_size: int = 512) -> Dict[str, Any]:
    """
    Performs 2D Fast Fourier Transform frequency analysis on the image.
    
    Parameters:
      image_path: path to the target image file.
      target_size: dimension to resize for standardized spectral frequency analysis.
      
    Returns a dict containing:
      - score: 0 to 100 (0 = natural optical radial decay, 100 = artificial periodic grid artifacts)
      - high_freq_ratio: ratio of high-frequency power to mid-frequency power
      - radial_slope: rate of spectral energy decay
      - spectral_peaks_count: number of anomalous frequency peaks
      - fft_spectrum_base64: data URI of the colorized 2D power spectrum
      - explanation: plain English summary
      - details: technical metrics
    """
    try:
        # Step 1: Load and convert to grayscale
        with Image.open(image_path) as img:
            img_gray = img.convert("L")
            # Resize to a square power-of-two size for standardized FFT bins
            img_resized = img_gray.resize((target_size, target_size), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32)

        # Step 2: Apply 2D Fast Fourier Transform
        # Apply Hanning window to reduce spectral leakage from image boundary edges
        window_y = np.hanning(target_size)
        window_x = np.hanning(target_size)
        window_2d = np.outer(window_y, window_x)
        windowed_arr = arr * window_2d

        fft2 = np.fft.fft2(windowed_arr)
        fft_shift = np.fft.fftshift(fft2)
        magnitude = np.abs(fft_shift)
        
        # Log magnitude spectrum: S(u, v) = log(1 + |F(u, v)|)
        log_mag = np.log(1.0 + magnitude)
        
        # Normalize log magnitude to 0-255 for visualization
        mag_min = float(np.min(log_mag))
        mag_max = float(np.max(log_mag))
        if mag_max > mag_min:
            norm_mag = ((log_mag - mag_min) / (mag_max - mag_min) * 255.0).astype(np.uint8)
        else:
            norm_mag = np.zeros((target_size, target_size), dtype=np.uint8)

        # Apply a colormap (COLORMAP_INFERNO or COLORMAP_TURBO) for clear visualization
        color_spectrum = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        
        # Encode color spectrum image to Base64 PNG
        _, enc_buf = cv2.imencode(".png", color_spectrum)
        b64_str = base64.b64encode(enc_buf).decode("utf-8")
        fft_spectrum_base64 = f"data:image/png;base64,{b64_str}"

        # Step 3: Compute Azimuthal Radial Average Profile
        center_y, center_x = target_size // 2, target_size // 2
        y, x = np.ogrid[:target_size, :target_size]
        r = np.hypot(x - center_x, y - center_y).astype(np.int32)
        
        max_r = min(center_x, center_y)
        # Radial bins
        radial_profile = np.zeros(max_r, dtype=np.float32)
        for radius in range(max_r):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = float(np.mean(log_mag[mask]))
                
        # Low, Mid, and High Frequency energy segments
        low_band = radial_profile[int(max_r * 0.05):int(max_r * 0.25)]
        mid_band = radial_profile[int(max_r * 0.25):int(max_r * 0.60)]
        high_band = radial_profile[int(max_r * 0.60):int(max_r * 0.95)]
        
        mean_low = float(np.mean(low_band)) if len(low_band) > 0 else 1.0
        mean_mid = float(np.mean(mid_band)) if len(mid_band) > 0 else 1.0
        mean_high = float(np.mean(high_band)) if len(high_band) > 0 else 1.0
        
        # Energy ratios
        high_to_mid_ratio = mean_high / (mean_mid + 1e-6)
        
        # Check for unnatural peaks in high frequencies (spikes characteristic of grid checkerboard artifacts)
        diff_profile = np.diff(radial_profile[int(max_r * 0.3):int(max_r * 0.95)])
        positive_surges = np.sum(diff_profile > 0.08)
        
        # Check cross-axis symmetry (generative decoders often leave horizontal/vertical crosses)
        h_cross = np.mean(log_mag[center_y - 2:center_y + 3, :])
        v_cross = np.mean(log_mag[:, center_x - 2:center_x + 3])
        diag_bg = (np.mean(log_mag) + 1e-6)
        cross_ratio = ((h_cross + v_cross) / 2.0) / diag_bg

        # Radial spectral power-law decay slope alpha (log(frequency) vs log(power))
        freq_indices = np.arange(int(max_r * 0.1), int(max_r * 0.85))
        freq_vals = radial_profile[int(max_r * 0.1):int(max_r * 0.85)]
        valid = freq_vals > 0
        if np.sum(valid) > 10:
            log_f = np.log(freq_indices[valid])
            log_p = np.log(freq_vals[valid])
            spectral_slope, _ = np.polyfit(log_f, log_p, 1)
        else:
            spectral_slope = -1.2

        # Step 4: Scoring Heuristics
        # Natural optical lenses have smooth 1/f^alpha decay (slope typically -0.5 to -1.8).
        # AI images show either:
        # 1. Unnatural high-frequency retention (high_to_mid_ratio > 0.85)
        # 2. Prominent grid spikes (positive_surges >= 4)
        # 3. Unnatural cross-axis frequency patterns (cross_ratio > 1.25)
        # 4. Precipitous spectral dropoff (spectral_slope < -2.1, characteristic of diffusion denoising)
        
        score = 20.0  # baseline for natural optical decay
        findings = []

        if positive_surges >= 8:
            score += 45.0
            findings.append("Periodic frequency spikes detected (neural upsampling checkerboard artifact)")
        elif positive_surges >= 3:
            score += 25.0
            findings.append("Mild spectral frequency oscillations detected")
            
        if spectral_slope < -2.1:
            score += 35.0
            findings.append("Precipitous high-frequency spectral dropoff (characteristic of iterative diffusion denoising)")
        elif spectral_slope > -0.6 and high_to_mid_ratio > 0.85:
            score += 25.0
            findings.append("Abnormally elevated high-frequency energy ratio")
        elif high_to_mid_ratio < 0.65:
            score += 20.0
            findings.append("Premature spectral dropoff (synthetic skin and surface smoothing)")

        if cross_ratio > 1.28:
            score += 15.0
            findings.append("Pronounced axial frequency cross (transposed convolution grid artifact)")

        score = max(5.0, min(95.0, score))
        
        if score > 60.0:
            summary = "Periodic grid patterns and synthetic spectral energy distribution detected."
        else:
            summary = "Natural radial frequency distribution consistent with physical optical lens roll-off."

        explanation = (
            f"2D Fast Fourier Transform (FFT) analyzed the frequency spectrum of the image. "
            f"Findings: {summary} "
            f"High-to-mid frequency ratio: {high_to_mid_ratio:.3f}, spectral surges: {int(positive_surges)}. "
            f"Physical camera lenses naturally scatter high-frequency light (1/f² power law), "
            f"whereas generative AI decoders (transposed convolutions / latent upsamplers) leave "
            f"invisible periodic grid checkerboard patterns in frequency space."
        )

        return {
            "score": round(score, 1),
            "high_freq_ratio": round(high_to_mid_ratio, 3),
            "spectral_peaks_count": int(positive_surges),
            "cross_ratio": round(cross_ratio, 3),
            "fft_spectrum_base64": fft_spectrum_base64,
            "explanation": explanation,
            "details": f"High/Mid Ratio: {high_to_mid_ratio:.2f}, Cross-Ratio: {cross_ratio:.2f}, Spikes: {int(positive_surges)}"
        }

    except Exception as e:
        return {
            "score": 50.0,
            "high_freq_ratio": 0.0,
            "spectral_peaks_count": 0,
            "cross_ratio": 0.0,
            "fft_spectrum_base64": "",
            "explanation": f"FFT frequency analysis error: {str(e)}",
            "details": "Could not compute 2D Fourier transform."
        }
