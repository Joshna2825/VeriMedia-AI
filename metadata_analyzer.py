"""
Metadata and Provenance Forensics Analyzer.
Inspects EXIF, PNG text chunks, and C2PA markers for camera hardware or AI generator signatures.
"""

from typing import Dict, Any, List
import os
from PIL import Image, ExifTags


KNOWN_AI_KEYWORDS = [
    "stable diffusion",
    "stablediffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "flux.1",
    "novelai",
    "comfyui",
    "automatic1111",
    "invokeai",
    "adobe firefly",
    "firefly",
    "civitai",
    "dreamstudio",
    "sora",
    "runway",
    "pika",
    "gen-2",
    "gen-3",
    "kling",
    "deepfake",
    "faceswap",
    "roop",
    "reactor",
]

CAMERA_HARDWARE_TAGS = [
    "Make",
    "Model",
    "LensModel",
    "FNumber",
    "ExposureTime",
    "ISOSpeedRatings",
    "FocalLength",
    "ShutterSpeedValue",
    "ApertureValue",
]


def analyze_metadata(image_path: str) -> Dict[str, Any]:
    """
    Analyzes an image's metadata for camera hardware traces or AI generator signatures.
    
    Returns a dict containing:
      - score: 0 to 100 (0 = highly authentic camera metadata, 100 = explicit AI signature)
      - status: 'ai_detected', 'suspicious_stripped', 'likely_camera'
      - hardware_tags_found: list of detected camera tags
      - ai_signatures_found: list of detected AI traces
      - raw_metadata: key-value pairs of useful tags
      - explanation: plain English summary for students / viva
    """
    ai_signatures: List[str] = []
    hardware_tags: List[str] = []
    raw_metadata: Dict[str, str] = {}
    
    # Check if media is a video format
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".mp4", ".webm", ".mov", ".avi", ".mkv"]:
        return {
            "score": 40.0,
            "status": "video_container",
            "hardware_tags_found": [],
            "ai_signatures_found": [],
            "raw_metadata": {"Container": ext.upper()},
            "explanation": (
                "Video stream container inspected. Standard camera EXIF photo tags do not reside in raw video streams. "
                "Forensic evaluation proceeds primarily via temporal SSIM and keyframe frequency analysis."
            ),
            "details": f"Video format {ext.upper()} container analyzed."
        }

    try:
        with Image.open(image_path) as img:
            format_name = img.format or "UNKNOWN"
            info = img.info or {}
            
            # 1. Inspect PNG info / text chunks (common in AI generations like Stable Diffusion, ComfyUI)
            for k, v in info.items():
                if isinstance(v, (str, bytes)):
                    val_str = str(v)
                    k_str = str(k)
                    # Check for prompt parameters
                    if k_str.lower() in ["parameters", "prompt", "workflow", "prompt_text"]:
                        ai_signatures.append(f"PNG text chunk '{k_str}' containing generation parameters")
                        raw_metadata[k_str] = val_str[:200] + ("..." if len(val_str) > 200 else "")
                    
                    # Check for AI keywords inside info values
                    lowered = val_str.lower()
                    for kw in KNOWN_AI_KEYWORDS:
                        if kw in lowered:
                            ai_signatures.append(f"AI generator reference '{kw}' in metadata field '{k_str}'")
                            raw_metadata[k_str] = val_str[:150]
            
            # 2. Inspect EXIF metadata (typical of physical cameras)
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    val_str = str(value)
                    
                    if tag_name in CAMERA_HARDWARE_TAGS:
                        hardware_tags.append(tag_name)
                        raw_metadata[tag_name] = val_str
                    
                    # Check Software, UserComment, ImageDescription for AI indicators
                    if tag_name in ["Software", "UserComment", "ImageDescription", "Artist"]:
                        raw_metadata[tag_name] = val_str[:150]
                        lowered = val_str.lower()
                        for kw in KNOWN_AI_KEYWORDS:
                            if kw in lowered:
                                ai_signatures.append(f"AI software marker '{kw}' in EXIF '{tag_name}'")
                                
                    # Check C2PA / Content Credentials markers
                    if "c2pa" in tag_name.lower() or "c2pa" in val_str.lower():
                        if "synthetic" in val_str.lower() or "trainedalgorithmicmedia" in val_str.lower():
                            ai_signatures.append("C2PA Content Credentials declare media as AI-synthesized")

    except Exception as e:
        return {
            "score": 50,
            "status": "error",
            "hardware_tags_found": [],
            "ai_signatures_found": [],
            "raw_metadata": {},
            "explanation": f"Unable to read image metadata: {str(e)}",
            "details": "Corrupted or non-standard image container."
        }
        
    # Deduplicate findings
    ai_signatures = list(dict.fromkeys(ai_signatures))
    hardware_tags = list(dict.fromkeys(hardware_tags))
    
    # 3. Determine score and plain-English explanation
    if len(ai_signatures) > 0:
        score = 98.0
        status = "ai_detected"
        explanation = (
            f"AI generator signatures were discovered embedded in the image metadata ({len(ai_signatures)} markers found). "
            f"The image contains explicit prompts, generation workflows, or AI software tags (such as {ai_signatures[0]})."
        )
    elif len(hardware_tags) >= 3:
        score = 10.0
        status = "likely_camera"
        tags_str = ", ".join(hardware_tags[:4])
        explanation = (
            f"Genuine optical camera metadata was identified ({tags_str}). "
            f"Physical cameras record hardware exposure parameters (aperture, shutter speed, ISO, lens make), "
            f"which strongly correlates with authentic optical capture."
        )
    elif len(hardware_tags) > 0:
        score = 25.0
        status = "partial_camera"
        explanation = (
            f"Partial camera metadata was found ({', '.join(hardware_tags)}). "
            f"While some camera tags exist, standard exposure records may have been partially compressed or modified by editing software."
        )
    else:
        # No metadata at all (very common in modern AI web outputs, and social media re-compressed images)
        score = 65.0
        status = "suspicious_stripped"
        explanation = (
            "All physical camera and hardware EXIF tags are missing. "
            "Generative AI platforms typically output clean, stripped images lacking sensor provenance. "
            "Note: Social media platforms also strip metadata, requiring visual and frequency forensics."
        )
        
    return {
        "score": score,
        "status": status,
        "hardware_tags_found": hardware_tags,
        "ai_signatures_found": ai_signatures,
        "raw_metadata": raw_metadata,
        "explanation": explanation,
        "details": f"{len(hardware_tags)} camera tags detected, {len(ai_signatures)} AI tags detected."
    }
