"""
Error Level Analysis (ELA) Analyzer.
Detects compression discrepancies and manipulation artifacts across image regions.
"""

from typing import Dict, Any, Tuple
import io
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def analyze_ela(image_path: str, quality: int = 90, scale: int = 15) -> Dict[str, Any]:
    """
    Performs Error Level Analysis on the given image.
    
    Parameters:
      image_path: path to the target image file.
      quality: compression quality for the reference JPEG (default: 90).
      scale: visual amplification multiplier for error visualization (default: 15).
      
    Returns a dict with:
      - score: 0 to 100 (0 = uniform natural compression, 100 = high compression disparity/synthetic)
      - mean_error: average pixel discrepancy
      - max_error: maximum pixel discrepancy
      - std_error: error distribution variance
      - ela_image_base64: data URI string for direct web rendering
      - explanation: plain English summary
      - details: technical metrics
    """
    try:
        with Image.open(image_path) as original_img:
            # Convert to RGB to ensure consistent color space
            rgb_orig = original_img.convert("RGB")
            
            # Limit resolution for fast, responsive processing
            max_dim = 1200
            if max(rgb_orig.size) > max_dim:
                rgb_orig.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            
            # Step 1: Save to temporary in-memory buffer at specified JPEG quality
            buffer = io.BytesIO()
            rgb_orig.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            
            # Step 2: Open re-compressed image
            with Image.open(buffer) as resaved_img:
                # Step 3: Compute pixel-wise difference
                diff = ImageChops.difference(rgb_orig, resaved_img)
                
                # Convert difference to numpy array for metric calculation
                diff_arr = np.array(diff, dtype=np.float32)
                
                # Calculate channel-averaged error
                gray_diff = np.mean(diff_arr, axis=2)
                mean_error = float(np.mean(gray_diff))
                max_error = float(np.max(gray_diff))
                std_error = float(np.std(gray_diff))
                
                # Block-level variance (8x8 blocks mimicking JPEG DCT blocks)
                h, w = gray_diff.shape
                pad_h = (8 - (h % 8)) % 8
                pad_w = (8 - (w % 8)) % 8
                padded = np.pad(gray_diff, ((0, pad_h), (0, pad_w)), mode='reflect')
                
                # Reshape into 8x8 blocks and compute block means
                bh, bw = padded.shape[0] // 8, padded.shape[1] // 8
                blocks = padded.reshape(bh, 8, bw, 8).swapaxes(1, 2)
                block_means = blocks.mean(axis=(2, 3))
                block_variance = float(np.var(block_means))
                
                # Step 4: Generate amplified visual heatmap for UI inspection
                extrema = diff.getextrema()
                max_diff = max([ex[1] for ex in extrema])
                if max_diff == 0:
                    max_diff = 1
                
                # Amplify difference so human eyes can inspect it clearly
                scale_factor = 255.0 / max_diff if max_diff < 50 else scale
                enhancer = ImageEnhance.Brightness(diff)
                amplified_diff = enhancer.enhance(min(scale_factor, 25.0))
                
                # Convert amplified ELA image to Base64 PNG
                out_buffer = io.BytesIO()
                amplified_diff.save(out_buffer, format="PNG")
                b64_str = base64.b64encode(out_buffer.getvalue()).decode("utf-8")
                ela_image_base64 = f"data:image/png;base64,{b64_str}"

        # Step 5: Heuristic scoring for AI vs Real
        # Real optical images with complex human scenes/clothing have block variance typically > 3.5.
        # AI images or deepfakes frequently show:
        # a) Unnaturally low block variance (< 3.0) and low mean error (< 2.5):
        #    Diffusion generation synthesizes smooth gradient fields with uniform latent representation,
        #    creating an unnaturally flat compression response across the image.
        # b) Very high block variance (> 16.0): Splice boundaries or localized face replacement.
        # c) Moderate balanced variance (3.5 - 14.0): Typical camera optical scene behavior.
        
        if block_variance < 1.8 and mean_error < 2.5:
            # Strong synthetic ultra-clean artifact (typical of diffusion and AI portrait outputs)
            score = 78.0
            anomaly_type = "Unnaturally flat compression error decay (diffusion surface smoothness)"
        elif block_variance < 3.2 and mean_error < 2.5:
            # Low block variance on detailed scenes indicates synthetic smoothing
            score = 68.0
            anomaly_type = "Elevated compression flatness (synthetic or social re-compression)"
        elif block_variance > 16.0:
            # Splice or sharp generation boundary
            score = min(92.0, 55.0 + (block_variance * 1.5))
            anomaly_type = "High local compression discrepancy (inconsistent error levels / spliced)"
        elif block_variance > 11.0:
            score = 60.0
            anomaly_type = "Elevated compression boundary disparity"
        else:
            # Consistent natural optical compression with normal physical texture variance
            score = max(12.0, 15.0 + (block_variance * 1.8))
            anomaly_type = "Uniform compression error distribution (natural optical behavior)"

        explanation = (
            f"Error Level Analysis evaluated how the image responds to re-compression. "
            f"Result: {anomaly_type}. "
            f"The image exhibited an average error level of {mean_error:.2f} (std: {std_error:.2f}) "
            f"and an 8x8 DCT block variance of {block_variance:.2f}. "
            f"Authentic camera photos typically demonstrate uniform, gradual error decay across textures, "
            f"whereas AI generations or deepfakes show sharp compression boundaries or synthetic over-smoothing."
        )

        return {
            "score": round(score, 1),
            "mean_error": round(mean_error, 3),
            "max_error": round(max_error, 3),
            "std_error": round(std_error, 3),
            "block_variance": round(block_variance, 3),
            "ela_image_base64": ela_image_base64,
            "explanation": explanation,
            "details": f"Mean Error: {mean_error:.2f}, Block Variance: {block_variance:.2f}"
        }

    except Exception as e:
        return {
            "score": 50.0,
            "mean_error": 0.0,
            "max_error": 0.0,
            "std_error": 0.0,
            "block_variance": 0.0,
            "ela_image_base64": "",
            "explanation": f"Error during ELA computation: {str(e)}",
            "details": "Could not compute compression error difference."
        }
