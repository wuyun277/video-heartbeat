"""Formula-based readers and an explicitly local evaluation protocol."""
from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(values, fps, band):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 32 or np.std(values) < 1e-10:
        raise ValueError("missing, short or constant waveform")
    return signal.sosfiltfilt(signal.butter(3, band, btype="bandpass", fs=fps, output="sos"), signal.detrend(values))


def standardize(values):
    spread = np.std(values)
    if spread < 1e-10 or not np.isfinite(values).all():
        raise ValueError("Cannot normalize constant or invalid target")
    return (values - np.mean(values)) / spread


def extract(rgb, fps, method, band):
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all() or np.any(rgb.mean(axis=0) < 1):
        raise ValueError("missing ROI pixels or invalid RGB")
    if method == "GREEN":
        return bandpass(rgb[:, 1], fps, band)
    size = int(np.ceil(1.6 * fps))
    if method == "CHROM":
        size += size % 2
    if len(rgb) < size:
        raise ValueError("Not enough frames for 1.6-second chrominance window")
    total, weights = np.zeros(len(rgb)), np.zeros(len(rgb))
    hop = 1 if method == "POS" else size // 2
    starts = list(range(0, len(rgb) - size + 1, hop))
    if starts[-1] != len(rgb) - size:
        starts.append(len(rgb) - size)
    for start in starts:
        block = rgb[start:start + size]
        red, green, blue = (block / block.mean(axis=0)).T
        if method == "POS":
            a, b = green - blue, green + blue - 2 * red
            pulse = a + np.std(a) / max(np.std(b), 1e-12) * b
            window = np.ones(size)
        elif method == "CHROM":
            a = bandpass(3 * red - 2 * green, fps, band)
            b = bandpass(1.5 * red + green - 1.5 * blue, fps, band)
            pulse = a - np.std(a) / max(np.std(b), 1e-12) * b
            window = np.hanning(size)
        else:
            raise ValueError("Unknown classical extractor " + method)
        total[start:start + size] += (pulse - pulse.mean()) * window
        weights[start:start + size] += window
    return bandpass(total / np.maximum(weights, 1e-12), fps, band)


def correlation(a, b):
    if len(a) < 8 or min(np.std(a), np.std(b)) < 1e-10:
        raise ValueError("flat filtered waveform")
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1, 1))


def spectrum(values, fps, band):
    nfft = max(2048, 4 * 2 ** int(np.ceil(np.log2(len(values)))))
    frequencies, power = signal.periodogram(values, fs=fps, window="hann", nfft=nfft)
    inside = (frequencies >= band[0]) & (frequencies <= band[1])
    index = np.flatnonzero(inside)[np.argmax(power[inside])]
    return frequencies, power, inside, float(frequencies[index] * 60)


def metrics(prediction, reference, fps, band, trim_seconds=0.8):
    trim = int(round(trim_seconds * fps))
    pred = bandpass(prediction, fps, band)
    truth = bandpass(reference, fps, band)
    if trim:
        pred, truth = pred[trim:-trim], truth[trim:-trim]
    r = correlation(pred, truth)
    f, power, inside, predicted_bpm = spectrum(pred, fps, band)
    _, _, _, reference_bpm = spectrum(truth, fps, band)
    fundamental = reference_bpm / 60
    near = inside & ((np.abs(f - fundamental) <= 0.1) | (np.abs(f - 2 * fundamental) <= 0.1))
    snr = float(10 * np.log10(max(power[near].sum(), 1e-20) / max(power[inside & ~near].sum(), 1e-20)))
    best_r, best_lag = abs(r), 0
    # A bounded, non-circular diagnostic. Zero-lag metrics remain primary.
    for lag in range(-int(round(0.5 * fps)), int(round(0.5 * fps)) + 1):
        if lag == 0:
            continue
        a, b = (pred[lag:], truth[:-lag]) if lag > 0 else (pred[:lag], truth[-lag:])
        value = abs(correlation(a, b))
        if value > best_r:
            best_r, best_lag = value, lag
    return {"predicted_bpm": predicted_bpm, "reference_spectral_bpm": reference_bpm,
            "hr_error_bpm": predicted_bpm - reference_bpm,
            "r_zero_signed": r, "r_zero_abs": abs(r), "r_lag_abs": best_r,
            "best_lag_seconds": best_lag / fps, "snr_db": snr}


class NeuralReader:
    """Reuse the existing PURE architecture, integrate difference predictions."""
    def __init__(self, cfg):
        import torch
        from efficientphys import EfficientPhys
        self.torch = torch
        self.cfg = cfg
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
        torch.manual_seed(0)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = EfficientPhys(frame_depth=10, image_size=cfg["image_size"])
        weights = torch.load(cfg["weights"], map_location="cpu", weights_only=True)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in weights.items()}, strict=True)
        self.model.to(self.device).eval()

    def predict(self, crops):
        torch = self.torch
        values = np.asarray(crops, dtype=np.float32)
        if values.std() < 1e-6:
            raise ValueError("Flat face crops")
        values = (values - values.mean()) / values.std()
        pieces = []
        step = self.cfg["chunk_frames"]
        with torch.inference_mode():
            for start in range(0, len(values), step):
                chunk = values[start:start + step]
                count = len(chunk)
                padded = int(np.ceil(count / 10)) * 10
                # Match the model's frame-difference contract; final appended frame duplicates the last one.
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], padded + 1 - count, axis=0)])
                tensor = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).to(self.device)
                pieces.append(self.model(tensor).flatten().float().cpu().numpy()[:count])
        return np.cumsum(np.concatenate(pieces).astype(float))
