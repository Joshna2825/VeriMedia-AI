"""
Generate sample media assets (authentic and AI-generated) for instant testing and demonstration.
"""

import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import cv2


def generate_samples(output_dir: str = "samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Authentic Camera Photo (with physical noise and Canon EXIF)
    # -------------------------------------------------------------
    real_path = os.path.join(output_dir, "real_sample.jpg")
    width, height = 800, 600
    
    # Create natural scenery gradient (sky to landscape)
    real_img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        ratio = y / height
        # Sky blue blending to earth/vegetation tone
        r = 100 * (1 - ratio) + 40 * ratio
        g = 150 * (1 - ratio) + 120 * ratio
        b = 220 * (1 - ratio) + 50 * ratio
        real_img[y, :] = [r, g, b]
        
    # Add natural soft shapes (mountain / hills)
    for x in range(width):
        hill1 = int(350 + 60 * np.sin(x * 0.01) + 20 * np.cos(x * 0.025))
        if hill1 < height:
            real_img[hill1:, x] = real_img[hill1:, x] * 0.6 + np.array([30, 80, 40]) * 0.4
            
    # Add simulated physical camera sensor noise (Poisson photon noise + PRNU)
    prnu = np.random.normal(0, 3.5, (height, width, 3))
    real_img = np.clip(real_img + prnu, 0, 255).astype(np.uint8)
    
    pil_real = Image.fromarray(real_img)
    
    # Add genuine camera EXIF metadata
    exif = pil_real.getexif()
    # 271 = Make, 272 = Model, 305 = Software, 33434 = ExposureTime, 33437 = FNumber, 34855 = ISOSpeedRatings
    exif[271] = "Canon"
    exif[272] = "Canon EOS 5D Mark IV"
    exif[305] = "Digital Photo Professional"
    exif[33434] = 0.004  # 1/250 sec
    exif[33437] = 2.8    # f/2.8
    exif[34855] = 200    # ISO 200
    exif[42036] = "EF 50mm f/1.4 USM"
    
    pil_real.save(real_path, "JPEG", quality=95, exif=exif)
    print(f"Generated authentic camera sample: {real_path}")

    # -------------------------------------------------------------
    # 2. AI-Generated Synthetic Image (with periodic upsampling artifacts & prompt text chunk)
    # -------------------------------------------------------------
    ai_path = os.path.join(output_dir, "ai_synthetic_sample.png")
    
    # Create synthetic futuristic portrait/orb with smooth gradients
    ai_img = np.zeros((height, width, 3), dtype=np.float32)
    cy, cx = height // 2, width // 2
    y_idx, x_idx = np.ogrid[:height, :width]
    dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    
    # Glowing neon orb with hyper-smooth plastic gradient
    mask = dist < 220
    ai_img[mask, 0] = np.clip(255 - dist[mask] * 0.8, 0, 255)
    ai_img[mask, 1] = np.clip(100 + dist[mask] * 0.5, 0, 255)
    ai_img[mask, 2] = np.clip(240 - dist[mask] * 0.3, 0, 255)
    
    # Inject periodic high-frequency checkerboard grid artifact
    # Characteristic of Transposed Convolutions / Pixel Shuffle in neural decoders
    u = np.arange(width)
    v = np.arange(height)
    grid_pattern = np.outer(np.sin(2 * np.pi * v / 8.0), np.sin(2 * np.pi * u / 8.0))
    grid_pattern_3d = np.repeat(grid_pattern[:, :, np.newaxis], 3, axis=2) * 9.0
    
    ai_img = np.clip(ai_img + grid_pattern_3d, 0, 255).astype(np.uint8)
    pil_ai = Image.fromarray(ai_img)
    
    # Inject PNG text chunk with typical Stable Diffusion / Midjourney prompt metadata
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text(
        "parameters",
        "masterpiece, best quality, hyperrealistic glowing neon cyber orb, volumetric studio lighting, 8k octane render\n"
        "Negative prompt: low quality, blurry, deformed, noisy\n"
        "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7.5, Seed: 2981029318, Size: 800x600, Model: StableDiffusionXL_v1.0"
    )
    png_info.add_text("Software", "Automatic1111 WebUI / Stable Diffusion")
    
    pil_ai.save(ai_path, "PNG", pnginfo=png_info)
    print(f"Generated AI synthetic sample: {ai_path}")

    # -------------------------------------------------------------
    # 3. Deepfake Splice Sample (authentic background with spliced synthetic face patch)
    # -------------------------------------------------------------
    splice_path = os.path.join(output_dir, "deepfake_splice_sample.jpg")
    splice_img = np.array(pil_real).copy()
    
    # Splice in a circular region with distinct compression disparity and smoothing
    patch_cy, patch_cx = height // 2, width // 2
    patch_r = 110
    patch_mask = ((y_idx - patch_cy)**2 + (x_idx - patch_cx)**2) < patch_r**2
    
    # Over-smooth the patch and introduce boundary discontinuity
    smoothed_patch = cv2.bilateralFilter(splice_img, 15, 80, 80)
    splice_img[patch_mask] = smoothed_patch[patch_mask]
    
    # Add subtle checkerboard artifact inside spliced region
    splice_img_f = splice_img.astype(np.float32)
    splice_img_f[patch_mask] += (grid_pattern_3d[patch_mask] * 1.5)
    splice_img = np.clip(splice_img_f, 0, 255).astype(np.uint8)
    
    pil_splice = Image.fromarray(splice_img)
    pil_splice.save(splice_path, "JPEG", quality=85)
    print(f"Generated deepfake splice sample: {splice_path}")

    # -------------------------------------------------------------
    # 4. Authentic Sample Video (smooth physical camera motion)
    # -------------------------------------------------------------
    real_video_path = os.path.join(output_dir, "real_sample.mp4")
    vw, vh = 640, 480
    fps = 24
    num_frames = 48  # 2 seconds
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out_real = cv2.VideoWriter(real_video_path, fourcc, fps, (vw, vh))
    for f in range(num_frames):
        frame = np.zeros((vh, vw, 3), dtype=np.uint8)
        # Background gradient
        frame[:, :] = [40, 60, 80]
        # Smoothly moving natural circle (consistent optical flow)
        pos_x = int(150 + (f / num_frames) * 340)
        pos_y = int(240 + 30 * np.sin(f * 0.15))
        cv2.circle(frame, (pos_x, pos_y), 60, (0, 180, 220), -1)
        # Add consistent camera sensor grain
        grain = np.random.normal(0, 4.0, (vh, vw, 3)).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        out_real.write(frame)
    out_real.release()
    print(f"Generated authentic video sample: {real_video_path}")

    # -------------------------------------------------------------
    # 5. AI Generative Video Sample (temporal flickering & morphing)
    # -------------------------------------------------------------
    ai_video_path = os.path.join(output_dir, "ai_video_sample.mp4")
    out_ai = cv2.VideoWriter(ai_video_path, fourcc, fps, (vw, vh))
    
    for f in range(num_frames):
        frame = np.zeros((vh, vw, 3), dtype=np.uint8)
        frame[:, :] = [30, 30, 45]
        
        # Sudden morphing and latent jumps every 3-4 frames
        is_glitch_frame = (f % 4 == 0)
        jitter_r = int(60 + (35 if is_glitch_frame else 0) + 15 * np.sin(f * 0.9) + np.random.randint(-10, 11))
        pos_x = int(150 + (f / num_frames) * 340 + np.random.randint(-15, 16))
        pos_y = int(240 + 20 * np.sin(f * 0.3) + (25 if is_glitch_frame else 0) + np.random.randint(-12, 13))
        
        # Color flickering between frames (typical of AI video decoders)
        color_b = int(np.clip(180 + np.random.randint(-50, 50), 50, 255))
        color_g = int(np.clip(100 + np.random.randint(-40, 40), 30, 220))
        color_r = int(np.clip(220 + np.random.randint(-60, 60), 80, 255))
        
        cv2.circle(frame, (pos_x, pos_y), max(10, jitter_r), (color_b, color_g, color_r), -1)
        
        # Inject high-frequency periodic grid onto AI frames
        u_v = np.arange(vw)
        v_v = np.arange(vh)
        ai_grid = np.outer(np.sin(2 * np.pi * v_v / 8.0), np.sin(2 * np.pi * u_v / 8.0))
        grid_3d = np.repeat(ai_grid[:, :, np.newaxis], 3, axis=2) * 16.0
        frame_f = np.clip(frame.astype(np.float32) + grid_3d, 0, 255).astype(np.uint8)
        
        out_ai.write(frame_f)
    out_ai.release()
    print(f"Generated AI video sample: {ai_video_path}")


if __name__ == "__main__":
    generate_samples()
