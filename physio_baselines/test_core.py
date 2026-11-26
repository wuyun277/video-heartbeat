"""Checks that protect interpretation of the experiment, not model accuracy."""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from .data import ROOT, interpolate_labels, load_config
from .reporting import BEGIN, END, summarize, update_documents
from .runner import evaluate
from .signals import extract, metrics
from .video import inject, lossless_writer


class CoreChecks(unittest.TestCase):
    def test_subject_average_does_not_weight_longer_clips_more(self):
        base = {"variant": "B0", "target": "x", "reference_type": "target_ppg", "extractor": "POS", "roi": "pooled", "status": "ok", "r_zero_signed": 1., "r_zero_abs": 1., "snr_db": 0.}
        rows = [dict(base, subject="a", hr_error_bpm=10.)] + [dict(base, subject="b", hr_error_bpm=0.) for _ in range(9)]
        self.assertEqual(summarize(rows)[0]["mae_bpm_subject_mean"], 5.)

    def test_both_documents_update_and_preserve_manual_judgment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "runs" / "sample"
            run.mkdir(parents=True)
            progress = root / "当前进展与下一步.md"
            progress.write_text(BEGIN + "\nold\n" + END + "\nManual research note", encoding="utf-8")
            with patch("physio_baselines.reporting.ROOT", root):
                update_documents(run, "run", "completed", "first")
                update_documents(run, "run", "failed", "second")
            self.assertIn("Manual research note", progress.read_text(encoding="utf-8"))
            self.assertIn("failed", progress.read_text(encoding="utf-8"))
            log = (root / "实验记录.md").read_text(encoding="utf-8")
            self.assertIn("completed", log)
            self.assertIn("failed", log)

    def test_missing_windows_are_counted(self):
        cfg = load_config(ROOT / "config.json")
        cfg["extractors"] = ["GREEN"]
        count, fps = 600, 30.
        times = np.arange(count) / fps
        truth = np.sin(2 * np.pi * 1.2 * times)
        rows, _ = evaluate(cfg, "subject8", "B0", "none", np.full((count, 5, 3), np.nan), None,
                           np.zeros(count, bool), times, fps, [("target", "target_ppg", truth)], np.ones(count) * 72, None)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r["status"] == "invalid" for r in rows))
        self.assertTrue(all(r["coverage"] == 0 for r in summarize(rows)))

    def test_known_pulse_recovered_by_classical_readers(self):
        fps = 30.
        times = np.arange(600) / fps
        pulse = np.sin(2 * np.pi * 1.2 * times)
        rng = np.random.default_rng(1)
        rgb = 120 * np.exp(0.01 * pulse[:, None] * np.array([0.6, 1., 0.3])) + rng.normal(0, 0.001, (600, 3))
        for reader in ["GREEN", "POS", "CHROM"]:
            with self.subTest(reader=reader):
                result = metrics(extract(rgb, fps, reader, [0.7, 3.]), pulse, fps, [0.7, 3.])
                self.assertLess(abs(result["hr_error_bpm"]), 0.6)
                self.assertGreater(result["r_zero_abs"], 0.95)

    def test_alignment_diagnostic_does_not_hide_zero_lag_error(self):
        fps = 30.
        times = np.arange(600) / fps
        target = np.sin(2 * np.pi * 1.5 * times)
        delayed = np.sin(2 * np.pi * 1.5 * (times - 0.15))
        result = metrics(delayed, target, fps, [0.7, 3.])
        self.assertLess(result["r_zero_abs"], 0.3)
        self.assertGreater(result["r_lag_abs"], 0.98)
        self.assertNotEqual(result["best_lag_seconds"], 0)

    def test_mask_and_rgb_channel_order(self):
        source = np.full((32, 32, 3), 100, np.uint8)
        mask = np.zeros((32, 32), np.float32)
        mask[8:24, 8:24] = 1
        edited = inject(source, mask, 2., 0.01, [0., 1., 0.])
        np.testing.assert_array_equal(edited[mask == 0], source[mask == 0])
        np.testing.assert_array_equal(edited[:, :, [0, 2]], source[:, :, [0, 2]])
        self.assertGreater(edited[16, 16, 1], source[16, 16, 1])

    def test_timestamp_duplicates_and_extrapolation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.txt"
            path.write_text("0 2 4 6\n60 62 64 66\n5 5 6 7\n", encoding="utf-8")
            ppg, hr = interpolate_labels(path, np.array([0., 1., 2.]))
            np.testing.assert_allclose(ppg, [1., 4., 6.])
            np.testing.assert_allclose(hr, [61., 64., 66.])
            with self.assertRaises(ValueError):
                interpolate_labels(path, np.array([0., 3.]))

    def test_reserved_subjects_cannot_enter_development(self):
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        cfg["subjects"] = ["subject14"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_ffv1_is_pixel_exact(self):
        rng = np.random.default_rng(4)
        frames = rng.integers(0, 256, (4, 32, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.avi"
            writer = lossless_writer(path, 30., frames.shape[1:])
            for frame in frames:
                writer.write(frame)
            writer.release()
            cap = cv2.VideoCapture(str(path))
            try:
                for expected in frames:
                    ok, actual = cap.read()
                    self.assertTrue(ok)
                    np.testing.assert_array_equal(actual, expected)
                self.assertFalse(cap.read()[0])
            finally:
                cap.release()


if __name__ == "__main__":
    unittest.main()
