"""
Automated Forensic Pipeline Tests for CapCheck.
"""

import os
import unittest
from forensics.metadata_analyzer import analyze_metadata
from forensics.ela_analyzer import analyze_ela
from forensics.fft_analyzer import analyze_fft
from forensics.noise_analyzer import analyze_noise
from forensics.video_analyzer import analyze_video
from forensics.ensemble_scorer import compute_ensemble_verdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")


class TestForensicPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.real_img = os.path.join(SAMPLES_DIR, "real_sample.jpg")
        cls.ai_img = os.path.join(SAMPLES_DIR, "ai_synthetic_sample.png")
        cls.real_vid = os.path.join(SAMPLES_DIR, "real_sample.mp4")
        cls.ai_vid = os.path.join(SAMPLES_DIR, "ai_video_sample.mp4")

    def test_metadata_real_vs_ai(self):
        real_meta = analyze_metadata(self.real_img)
        self.assertLessEqual(real_meta["score"], 30.0)
        self.assertGreater(len(real_meta["hardware_tags_found"]), 0)

        ai_meta = analyze_metadata(self.ai_img)
        self.assertGreaterEqual(ai_meta["score"], 90.0)
        self.assertGreater(len(ai_meta["ai_signatures_found"]), 0)

    def test_ela_analyzer(self):
        ela_res = analyze_ela(self.real_img)
        self.assertIn("score", ela_res)
        self.assertIn("ela_image_base64", ela_res)
        self.assertTrue(ela_res["ela_image_base64"].startswith("data:image/png;base64,"))
        self.assertGreater(ela_res["mean_error"], 0.0)

    def test_fft_analyzer(self):
        fft_res = analyze_fft(self.ai_img)
        self.assertIn("score", fft_res)
        self.assertIn("fft_spectrum_base64", fft_res)
        self.assertTrue(fft_res["fft_spectrum_base64"].startswith("data:image/png;base64,"))
        self.assertGreater(fft_res["spectral_peaks_count"], 0)

    def test_noise_analyzer(self):
        noise_res = analyze_noise(self.real_img)
        self.assertIn("score", noise_res)
        self.assertIn("noise_std", noise_res)
        self.assertIn("noise_map_base64", noise_res)
        self.assertTrue(noise_res["noise_map_base64"].startswith("data:image/png;base64,"))

    def test_video_analyzer(self):
        vid_res = analyze_video(self.real_vid, max_frames_to_sample=12)
        self.assertIn("score", vid_res)
        self.assertIn("avg_ssim", vid_res)
        self.assertIn("timeline", vid_res)
        self.assertGreater(len(vid_res["timeline"]), 1)
        self.assertIn("highest_anomaly_frame", vid_res)

    def test_ensemble_scorer(self):
        meta_res = analyze_metadata(self.real_img)
        ela_res = analyze_ela(self.real_img)
        fft_res = analyze_fft(self.real_img)
        noise_res = analyze_noise(self.real_img)

        verdict = compute_ensemble_verdict(meta_res, ela_res, fft_res, noise_res)
        self.assertIn("ai_probability", verdict)
        self.assertIn("real_probability", verdict)
        self.assertIn("verdict", verdict)
        self.assertIn("forensic_cards", verdict)
        self.assertGreater(len(verdict["forensic_cards"]), 3)
        self.assertLessEqual(verdict["ai_probability"], 40.0)


if __name__ == "__main__":
    unittest.main()
