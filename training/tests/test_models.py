"""Synthetic checks only: no dataset preparation, epochs or optimizer updates.

Run: python -m training.check_models
"""
import copy
from pathlib import Path
import unittest

import torch

from training.models import build_model, forward_batch
from training.engine.checkpoints import load_model_weights
from training.settings import DEFAULT_CONFIG, cache_signature, model_contract, read_config
from training.engine.checkpoints import training_contract
from training.engine.heart_rate import spectral_bpm
from training.models.efficientphys_transformer import TokenTemporalShift, PatchMerging


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_profiles_and_existing_c_cache(self):
        c = read_config(DEFAULT_CONFIG, "efficientphys_c")
        t = read_config(DEFAULT_CONFIG, "efficientphys_t")
        self.assertEqual(c["train"]["epochs"], 5)
        self.assertEqual(t["train"]["learning_rate"], 0.0001)
        self.assertNotEqual(cache_signature(c), cache_signature(t))
        old_path = Path("training/runs/20261005_165134_a24edf/config.json")
        if old_path.exists():
            old = read_config(old_path)
            self.assertEqual(cache_signature(c), cache_signature(old))
            self.assertEqual(model_contract(c), model_contract(old))
            self.assertEqual(training_contract(c), training_contract(old))
            weights = old_path.parent / "best_weights.pth"
            if weights.exists():
                load_model_weights(build_model(c), weights)

    def test_temporal_shift_does_not_cross_groups(self):
        x = torch.arange(20 * 3, dtype=torch.float32).reshape(20, 1, 1, 3)
        shifted = TokenTemporalShift(10)(x)
        self.assertEqual(shifted[9, 0, 0, 0].item(), 0)
        self.assertEqual(shifted[10, 0, 0, 1].item(), 0)
        self.assertTrue(torch.equal(shifted[..., 2], x[..., 2]))

    def test_odd_patch_merging(self):
        output = PatchMerging(12)(torch.randn(10, 9, 9, 12))
        self.assertEqual(tuple(output.shape), (10, 5, 5, 24))

    def test_c_forward(self):
        cfg = read_config(DEFAULT_CONFIG, "efficientphys_c")
        model = build_model(cfg).eval()
        with torch.no_grad():
            output = forward_batch(model, torch.randn(2, 10, 3, 72, 72))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_transformer_variants_and_gradients(self):
        base = read_config(DEFAULT_CONFIG, "efficientphys_t")
        for variant, heads in (("t1", [3, 6, 12, 24]), ("t2", [3, 6])):
            with self.subTest(variant=variant):
                cfg = copy.deepcopy(base)
                cfg["model"].update(variant=variant, num_heads=heads)
                model = build_model(cfg).train()
                # Exercise activation checkpointing and odd spatial grids.
                frames = torch.randn(1, 10, 3, 72, 72)
                output = forward_batch(model, frames)
                self.assertEqual(tuple(output.shape), (1, 10))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(model.patch_embed.weight.grad)
                self.assertTrue(torch.isfinite(model.patch_embed.weight.grad).all())
                self.assertTrue(torch.isfinite(model.head.weight.grad).all())
                # Verify separate clips produce the same eval outputs as individual calls.
                model.eval()
                clips = torch.randn(2, 10, 3, 72, 72)
                with torch.no_grad():
                    batched = forward_batch(model, clips)
                    separate = torch.cat([forward_batch(model, clips[i:i + 1]) for i in range(2)])
                self.assertTrue(torch.allclose(batched, separate, atol=1e-6))
                del model

    def test_waveform_hr_without_integrating_twice(self):
        import numpy as np
        wave = np.sin(2 * np.pi * 1.2 * np.arange(450) / 30)
        bpm = spectral_bpm(wave, 30, 0.75, 2.5, integrate=False)
        self.assertAlmostEqual(bpm, 72, delta=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
