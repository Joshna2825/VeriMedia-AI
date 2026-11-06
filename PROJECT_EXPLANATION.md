# 🎓 CapCheck: Project Explanation & Viva Defense Guide

> **Project Title:** CapCheck: Real vs. AI-Generated Image & Video Detector with Multi-Pillar Explainability  
> **Domain:** Computer Vision, Digital Media Forensics, Signal Processing, Artificial Intelligence  
> **Key Objective:** Detect whether an image or video is **Real (Camera-Captured)** or **AI-Generated (Synthetic / Deepfake)** while making **every single calculation simple and transparent to explain** in presentations, viva voce examinations, and security audits.

---

## 1. Executive Summary: What Problem Does This Solve?

Generative AI models—such as **Midjourney, Stable Diffusion (SDXL/Flux), DALL-E 3** for images, and **Sora, Runway Gen-3, Pika, Kling, and Deepfake face-swaps** for video—can now generate photorealistic synthetic media. 

Traditional neural network detectors act as **black boxes**:
- When a neural network classifies an image as "AI: 95%", neither the student nor the professor can inspect *why* that prediction was made.
- If the deep learning model encounters a new AI generator it was never trained on, it often fails completely (out-of-distribution failure).

**Our Approach:**
Instead of a black box, **CapCheck** uses **Explainable Digital Media Forensics**. It inspects the physical and mathematical laws of how light is captured by a real optical camera versus how neural networks synthesize pixels from latent vectors.

```
                    ┌───────────────────────────────┐
                    │  Uploaded Media (Image/Video) │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────┬───────────┴───┬───────────────┬───────────────┐
        ▼               ▼               ▼               ▼               ▼
   [Pillar 1]      [Pillar 2]      [Pillar 3]      [Pillar 4]      [Pillar 5]
    Metadata &     Error Level       2D FFT         Sensor Noise     Video Temporal
    Provenance      Analysis       Frequency        Residual        Consistency
    (EXIF/Tags)      (ELA)        (Checkerboard)    (PRNU Grain)    (SSIM/Jitter)
        │               │               │               │               │
        └───────────────┼───────────────┼───────────────┼───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Ensemble Decision Engine     │
        │  Weighted Probability (0-100%)│
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │  Transparent Explainability   │
        │  Interactive Visual Maps + UI │
        └───────────────────────────────┘
```

---

## 2. Core Scientific Concept: Real Camera vs. AI Synthesis

To explain this project simply to anyone, remember this single distinction:

| Characteristic | Real Camera Media | AI-Generated Media |
| :--- | :--- | :--- |
| **How it is created** | Physical photons pass through glass lenses onto a silicon sensor. | Mathematical matrix multiplications in a neural network. |
| **Sensor Grain** | Natural Poisson photon noise & Photo-Response Non-Uniformity (PRNU). | Over-smoothed "plastic" skin or uniform synthetic noise. |
| **Frequency Spectrum** | Natural optical blur with smooth $1/f^2$ energy roll-off. | **Periodic checkerboard grid artifacts** from upsamplers. |
| **Compression (JPEG)** | Uniform error degradation across all 8x8 DCT blocks. | Disparate compression error rates across boundaries or faces. |
| **Video Motion** | Continuous physical optical flow and camera momentum. | **Micro-flickering, face morphing, and temporal drift**. |
| **Metadata** | Hardware records: Camera make, shutter, ISO, lens model. | Stripped completely, or contains prompt text chunks. |

---

## 3. The 5 Forensic Pillars Explained Simply

### 🔍 Pillar 1: Metadata & Provenance Forensics
* **What it does:** Scans image headers and container structures.
* **How it works:**
  1. Real cameras record EXIF hardware tags: `Make` (Canon, Apple, Sony), `Model`, `FNumber`, `ExposureTime`, `ISOSpeedRatings`, and `LensModel`.
  2. AI generators (Stable Diffusion, ComfyUI, Automatic1111) often embed generation workflows or prompt metadata in PNG text chunks (`parameters: masterpiece, 8k portrait...`).
  3. Other AI generators output clean, stripped images lacking any hardware metadata.
* **Why it matters:** Hardware exposure parameters are strong indicators of genuine camera capture.

---

### 💥 Pillar 2: Error Level Analysis (ELA)
* **What it does:** Detects compression discrepancies and manipulated or spliced regions.
* **The Mathematics:**
  1. JPEG compression divides an image into $8 \times 8$ blocks and applies the Discrete Cosine Transform (DCT).
  2. Each subsequent save at quality level $Q = 90\%$ introduces an error difference $\Delta$:
     $$\Delta(x, y) = |I_{\text{original}}(x, y) - I_{\text{recompressed}}(x, y)|$$
  3. We amplify this error difference by a scale factor $\alpha = 15\times$ to produce a visible heatmap.
* **How to explain it:** In authentic photos, the entire image reached compression equilibrium at the same time, producing uniform error levels. If an AI generator synthesizes an image or splices a face onto another body, the synthesized region has a vastly different compression error rate, which lights up brightly on the ELA heatmap.

---

### ⚡ Pillar 3: 2D Fast Fourier Transform (FFT) Frequency Analysis
* **What it does:** Detects invisible periodic grid and checkerboard artifacts left by generative neural upsamplers.
* **The Mathematics:**
  1. Computes the 2D Discrete Fourier Transform:
     $$\mathcal{F}(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
  2. Centers the low-frequency DC component and calculates the log magnitude power spectrum:
     $$S(u, v) = \log(1 + |\mathcal{F}(u, v)|)$$
* **Why Generative AI creates Frequency Artifacts:**
  Generative neural networks (StyleGAN, Stable Diffusion, DALL-E) upsample low-resolution latent representations ($64\times 64$) into full resolution ($512\times 512$ or $1024\times 1024$) using **Transposed Convolutions** or **Pixel Shuffle**. 
  Because convolutional kernels overlap periodically across the grid, they leave microscopic periodic checkerboard patterns. In the spatial domain, human eyes cannot see them; but in the **Fourier frequency domain**, they show up as distinct, bright symmetrical starburst spikes and axial crosses!
* **How to explain it:** Real lenses naturally scatter light smoothly. AI generators use mathematical grid upsamplers that leave periodic mathematical fingerprints in frequency space.

---

### 📡 Pillar 4: Noise Residual & Laplacian Texture Analysis
* **What it does:** Inspects microscopic silicon sensor grain and surface sharpness.
* **How it works:**
  1. Isolates high-frequency noise residual by subtracting a median-filtered version of the image:
     $$N(x, y) = I(x, y) - \text{MedianFilter}(I(x, y))$$
  2. Computes the standard deviation $\sigma_N$ of this noise residual.
  3. Calculates Laplacian edge gradient variance:
     $$\text{Var}(\nabla^2 I) = \text{Var}\left(\frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}\right)$$
  4. Evaluates Bayer color filter coupling across Red, Green, and Blue noise channels.
* **How to explain it:** Physical silicon photodiodes produce natural photon noise (PRNU). AI images frequently suffer from "plastic skin" (unnaturally low noise $\sigma_N < 1.2$) or artificial Gaussian noise that lacks physical Bayer filter channel coupling.

---

### 🎬 Pillar 5: Video Temporal Consistency Analysis
* **What it does:** Detects frame-to-frame flickering, morphing, and latent drift in videos.
* **The Mathematics:**
  1. Extracts evenly spaced keyframes across the video stream.
  2. Computes the Structural Similarity Index (SSIM) between consecutive frames:
     $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
  3. Measures temporal SSIM jitter: $\sigma_{\text{SSIM}} = \text{std}(\text{SSIM}_1, \text{SSIM}_2, \dots, \text{SSIM}_n)$.
* **How to explain it:** Authentic videos obey continuous optical flow and camera inertia. AI video generators (Sora, Runway, Pika, Deepfakes) synthesize frames in discrete latent batches, causing subtle micro-flickering, lighting jumps, and texture morphing (such as drifting background details or distorted facial features).

---

## 4. The Ensemble Scoring Formula

All forensic pillar scores ($S_i \in [0, 100]$) are synthesized by a weighted decision engine:

### For Images:
$$\text{AI Probability} = w_{\text{meta}} S_{\text{meta}} + w_{\text{ela}} S_{\text{ela}} + w_{\text{fft}} S_{\text{fft}} + w_{\text{noise}} S_{\text{noise}}$$
* Default weights: $w_{\text{fft}} = 0.35$ (strongest invariant), $w_{\text{ela}} = 0.25$, $w_{\text{noise}} = 0.25$, $w_{\text{meta}} = 0.15$.
* If explicit AI prompt text or software signatures are found in metadata, $w_{\text{meta}}$ scales to $0.45$.

### For Videos:
$$\text{AI Probability} = 0.40 \cdot S_{\text{temporal}} + 0.25 \cdot S_{\text{fft}} + 0.20 \cdot S_{\text{ela}} + 0.15 \cdot S_{\text{meta}}$$

### Classification Scale:
- **0% – 28%**: Authentic (Real Camera Media)
- **28% – 45%**: Likely Authentic
- **45% – 60%**: Suspicious / Inconclusive
- **60% – 78%**: Likely AI-Generated
- **78% – 100%**: AI-Generated / Deepfake

---

## 5. Top 10 Viva Voce Questions & Model Answers

### Q1: What makes your project unique compared to existing deepfake detectors?
> **Answer:** Most existing detectors use black-box deep learning (e.g. CNNs or Vision Transformers). While they might predict a label, they cannot explain *why*. Our project uses **multi-pillar explainable forensics** based on physical sensor optics, Error Level Analysis (ELA), 2D Fourier frequency domain analysis, and video temporal SSIM. Every score is mathematically auditable and accompanied by plain-English explanations and visual heatmaps.

### Q2: Why does an AI-generated image display checkerboard artifacts in the Fourier domain?
> **Answer:** AI models generate high-resolution images from low-dimensional latent space using upsampling operations like Transposed Convolutions or Pixel Shuffle. Because convolutional kernels overlap periodically during pixel decoding, they introduce microscopic spatial periodicities. When transformed via 2D Fast Fourier Transform (FFT), these periodicities appear as distinct high-frequency spikes and symmetrical cross patterns.

### Q3: What is Error Level Analysis (ELA) and how does it detect manipulation?
> **Answer:** ELA resaves an image at a standardized JPEG quality (90%) and calculates the absolute difference between the original and recompressed pixels. In authentic photos, all 8x8 DCT blocks reach a uniform compression error equilibrium. When an image is synthesized or a face is spliced in, the spliced region has a different error rate, which shows up as a bright anomaly on the ELA heatmap.

### Q4: How does your system detect AI-generated videos versus authentic videos?
> **Answer:** We evaluate inter-frame temporal consistency using the Structural Similarity Index (SSIM). Real video cameras obey physical laws of continuous motion. AI video generators (such as Sora, Runway, or Deepfakes) synthesize frames in latent batches, leading to frame-to-frame micro-flickering, texture morphing, and elevated SSIM jitter standard deviation ($\sigma > 0.03$).

### Q5: What happens if an image has had its EXIF metadata stripped (e.g., downloaded from WhatsApp or Twitter)?
> **Answer:** Social media platforms routinely compress images and strip EXIF tags. If metadata is missing, our system assigns metadata a neutral score (50%) with only 15% overall weight, and relies on the remaining visual, frequency (FFT), compression (ELA), and sensor noise (PRNU) pillars to reach an accurate verdict.

### Q6: What is PRNU (Photo-Response Non-Uniformity) in camera forensics?
> **Answer:** PRNU is a microscopic physical imperfection inherent to silicon semiconductor sensor arrays. Every physical camera sensor imprints a unique high-frequency noise fingerprint onto every photo it captures. Generative AI models lack a physical silicon sensor, so they either produce unnaturally smooth surfaces ("plastic textures") or synthetic noise lacking Bayer filter correlation.

### Q7: Can this system detect spliced deepfakes as well as fully synthesized images?
> **Answer:** Yes! Fully synthesized images (Midjourney, DALL-E) are primarily flagged by FFT periodic grid spikes and prompt metadata, whereas localized spliced deepfakes (face swaps) are primarily flagged by localized Error Level Analysis (ELA) block variance disparities along the boundary of the face.

### Q8: What is the computational complexity of the 2D FFT?
> **Answer:** For an $N \times N$ image, the Fast Fourier Transform computes the 2D discrete Fourier transform in $\mathcal{O}(N^2 \log N)$ time, compared to $\mathcal{O}(N^4)$ for a naive discrete Fourier transform. This allows our system to perform full spectral analysis in under 50 milliseconds per image.

### Q9: What are the current limitations of the system?
> **Answer:** Heavy post-processing—such as extreme multiple re-compressions (e.g. screenshot of a compressed thumbnail) or heavy Gaussian blurring—can degrade high-frequency signals. However, because our framework uses 5 complementary pillars, even if one pillar is degraded, the others maintain detection efficacy.

### Q10: How can this system be expanded in the future?
> **Answer:** Future enhancements include:
> 1. Integration of C2PA digital cryptographic watermarking standards.
> 2. Audio-visual multimodal synchronization (detecting lip-sync phoneme discrepancies in video deepfakes).
> 3. Real-time browser extension for live media verification on social platforms.

---

## 6. Ready-to-Use 2-Minute Presentation Elevator Pitch

> *"Respected Examiners, I am pleased to present **CapCheck**, an explainable digital media forensics system designed to detect real versus AI-generated images and videos.*
>
> *With the rapid rise of Generative AI tools like Midjourney, DALL-E, Sora, and Deepfakes, distinguishing authentic media from synthetic media is critical for fighting misinformation, fraud, and identity theft. While black-box deep learning models predict labels without explanation, CapCheck is built on **explainable signal forensics**.*
>
> *Our system evaluates 5 distinct forensic pillars:*
> 1. *Metadata & Provenance to check camera exposure tags versus AI prompt markers.*
> 2. *Error Level Analysis (ELA) to identify 8x8 DCT compression disparities.*
> 3. *2D Fast Fourier Transform (FFT) to uncover invisible periodic checkerboard artifacts left by neural upsamplers.*
> 4. *Sensor Noise Residuals to detect physical silicon sensor grain versus synthetic over-smoothing.*
> 5. *Video Temporal Consistency to track frame-to-frame SSIM and generative flickering.*
>
> *These signals are synthesized into an ensemble probability score with an interactive visual dashboard, keyframe timeline graphs, and clear, human-auditable explanations for every metric. Thank you, and I look forward to your questions."*

---

## 7. 10-Slide Presentation Outline (For PowerPoint / Slides)

1. **Slide 1: Title & Team** — CapCheck: Real vs. AI-Generated Media Forensics & Explainability Inspector.
2. **Slide 2: Problem Statement** — Proliferation of Generative AI (Diffusion, Sora, Deepfakes) and the limitation of black-box detectors.
3. **Slide 3: Proposed Solution** — Multi-pillar explainable forensic framework for both images and videos.
4. **Slide 4: System Architecture** — Flowchart showing preprocessor, 5 forensic engines, ensemble scorer, and interactive dashboard.
5. **Slide 5: Pillar 1 & 2** — EXIF Provenance & Error Level Analysis (ELA) with visual heatmaps.
6. **Slide 6: Pillar 3** — 2D Fast Fourier Transform (FFT) & Neural upsampler checkerboard detection.
7. **Slide 7: Pillar 4 & 5** — Sensor Noise (PRNU) & Video Temporal SSIM Consistency / Flickering.
8. **Slide 8: Experimental Results** — Evaluation across real photos, AI synthetic images, spliced deepfakes, and AI videos.
9. **Slide 9: Advantages & Impact** — Transparency, auditable evidence, zero training requirement, robust out-of-distribution performance.
10. **Slide 10: Conclusion & Future Scope** — Summary, audio-visual lip sync extension, C2PA integration, Q&A.
