"""Evaluate held-out subjects using the existing local window protocol."""
from pathlib import Path
import csv
import logging
import time
import uuid
import numpy as np
import torch
from training.models import build_model, forward_batch
from training.data.loaders import make_loader
from training.data.preprocessing import load_metadata
from training.settings import cache_signature, model_contract, write_json
from .runtime import select_device
from .checkpoints import load_model_weights
from .heart_rate import spectral_bpm

LOGGER = logging.getLogger(__name__)

def test(cfg, args):
    device = select_device(cfg)
    model = build_model(cfg).to(device)
    checkpoint = load_model_weights(model, args.checkpoint)
    if "config" in checkpoint:
        trained = checkpoint["config"]
        if model_contract(trained) != model_contract(cfg) or cache_signature(trained) != cache_signature(cfg):
            raise ValueError("Test model/preprocessing does not match checkpoint")
    model.eval()
    loader = make_loader(cfg, "test")
    grouped = {}
    squared_error, elements = 0.0, 0
    with torch.inference_mode():
        # 单片段前向：避免测试被试之间或不连续片段之间的帧差分。
        for frames, labels, names, starts in loader:
            for i, name in enumerate(names):
                predicted = forward_batch(model, frames[i:i + 1].to(device)).cpu().numpy()[0]
                truth = labels[i].numpy()
                if not np.isfinite(predicted).all():
                    raise RuntimeError("Non-finite test prediction")
                grouped.setdefault(name, []).append((int(starts[i]), predicted))
                squared_error += float(np.square(predicted - truth).sum())
                elements += len(truth)
    output = Path(args.checkpoint).resolve().parent / ("test_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    output.mkdir()
    rows = []
    if cfg["test"]["window_seconds"] < 4:
        raise ValueError("Use a test window of at least 4 seconds")
    for name, parts in grouped.items():
        parts.sort(key=lambda item: item[0])
        prediction = np.concatenate([item[1] for item in parts])
        folder, meta = load_metadata(cfg, name)
        fs = meta["fps"]
        if cfg["test"]["max_bpm"] / 60 >= fs / 2:
            raise ValueError(f"{name}: heart-rate range exceeds actual sampling Nyquist limit")
        window = round(cfg["test"]["window_seconds"] * fs)
        reference = np.load(folder / "reference_hr.npy")[:len(prediction)]
        label_type = cfg["preprocess"].get("label_type", "diff_normalized")
        prediction_key = "prediction_diff" if label_type == "diff_normalized" else "prediction_waveform"
        np.savez_compressed(output / f"{name}_prediction.npz", **{prediction_key: prediction},
                            reference_hr=reference, fps=fs, label_type=label_type)
        for start in range(0, len(prediction) - window + 1, window):
            bpm = spectral_bpm(prediction[start:start + window], fs, cfg["test"]["min_bpm"] / 60, cfg["test"]["max_bpm"] / 60,
                               integrate=label_type == "diff_normalized")
            target = float(np.mean(reference[start:start + window]))
            rows.append({"subject": name, "start_seconds": start / fs, "predicted_bpm": bpm,
                         "reference_bpm": target, "error_bpm": None if bpm is None else bpm - target})
    with (output / "heart_rate.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "start_seconds", "predicted_bpm", "reference_bpm", "error_bpm"])
        writer.writeheader()
        writer.writerows(rows)
    errors = np.array([row["error_bpm"] for row in rows if row["error_bpm"] is not None])
    write_json(output / "metrics.json", {"waveform_mse": squared_error / elements,
        "total_windows": len(rows), "valid_windows": len(errors),
        "mae_bpm": float(np.abs(errors).mean()) if len(errors) else None,
        "rmse_bpm": float(np.sqrt(np.square(errors).mean())) if len(errors) else None,
        "model": cfg["model"], "label_type": cfg["preprocess"].get("label_type", "diff_normalized"),
        "protocol": "nonoverlapping windows vs mean device HR; not the paper/toolbox benchmark protocol"})
    LOGGER.info("Test saved to %s | valid HR windows %d/%d", output, len(errors), len(rows))
