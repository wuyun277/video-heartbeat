"""EfficientPhys inference adapted from the official rPPG-Toolbox implementation.

Model architecture and preprocessing follow:
https://github.com/ubicomplab/rPPG-Toolbox
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import signal
import torch
from torch import nn


# 归一化空间注意力，使每张掩码的平均权重为 0.5。
class AttentionMask(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        spatial_sum = value.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        return value / spatial_sum * value.shape[2] * value.shape[3] * 0.5


# 沿时间方向移动部分通道，让二维卷积融合相邻帧信息。
class TemporalShift(nn.Module):
    def __init__(self, segments: int = 10, fold_div: int = 3):
        super().__init__()
        self.segments = segments
        self.fold_div = fold_div

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        nt, channels, height, width = value.shape
        if nt % self.segments:
            raise ValueError(f"EfficientPhys input length {nt} is not divisible by {self.segments}")
        value = value.view(nt // self.segments, self.segments, channels, height, width)
        fold = channels // self.fold_div
        shifted = torch.zeros_like(value)
        shifted[:, :-1, :fold] = value[:, 1:, :fold]
        shifted[:, 1:, fold:2 * fold] = value[:, :-1, fold:2 * fold]
        shifted[:, :, 2 * fold:] = value[:, :, 2 * fold:]
        return shifted.view(nt, channels, height, width)


# 网络层名称和张量形状与现有工具箱权重保持兼容。
class EfficientPhys(nn.Module):
    """Official EfficientPhys network shape used by rPPG-Toolbox checkpoints."""

    def __init__(self, frame_depth: int = 10, image_size: int = 72):
        super().__init__()
        if image_size not in (36, 72, 96):
            raise ValueError("Official EfficientPhys checkpoints support image sizes 36, 72 or 96")
        self.TSM_1 = TemporalShift(frame_depth)
        self.TSM_2 = TemporalShift(frame_depth)
        self.TSM_3 = TemporalShift(frame_depth)
        self.TSM_4 = TemporalShift(frame_depth)
        self.motion_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.motion_conv2 = nn.Conv2d(32, 32, 3)
        self.motion_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.motion_conv4 = nn.Conv2d(64, 64, 3)
        self.apperance_att_conv1 = nn.Conv2d(32, 1, 1)
        self.attn_mask_1 = AttentionMask()
        self.apperance_att_conv2 = nn.Conv2d(64, 1, 1)
        self.attn_mask_2 = AttentionMask()
        self.avg_pooling_1 = nn.AvgPool2d((2, 2))
        self.avg_pooling_2 = nn.AvgPool2d((2, 2))
        self.avg_pooling_3 = nn.AvgPool2d((2, 2))
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.5)
        dense_sizes = {36: 3136, 72: 16384, 96: 30976}
        self.final_dense_1 = nn.Linear(dense_sizes[image_size], 128)
        self.final_dense_2 = nn.Linear(128, 1)
        self.batch_norm = nn.BatchNorm2d(3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.batch_norm(torch.diff(inputs, dim=0))
        d1 = torch.tanh(self.motion_conv1(self.TSM_1(inputs)))
        d2 = torch.tanh(self.motion_conv2(self.TSM_2(d1)))
        gated1 = d2 * self.attn_mask_1(torch.sigmoid(self.apperance_att_conv1(d2)))
        d4 = self.TSM_3(self.dropout_1(self.avg_pooling_1(gated1)))
        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(self.TSM_4(d5)))
        gated2 = d6 * self.attn_mask_2(torch.sigmoid(self.apperance_att_conv2(d6)))
        d8 = self.dropout_3(self.avg_pooling_3(gated2))
        return self.final_dense_2(self.dropout_4(torch.tanh(self.final_dense_1(d8.flatten(1)))))


# 根据关键点放大人脸框，并将裁剪范围限制在图像边界内。
def face_box(frame: np.ndarray, points: np.ndarray, scale: float) -> tuple[int, int, int, int] | None:
    """Create one enlarged square face box; keep it fixed over an analysis segment."""
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    side = max(x1 - x0, y1 - y0) * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = int(round(cx + side / 2)), int(round(cy + side / 2))
    left, top = max(0, left), max(0, top)
    right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)
    if right - left < 24 or bottom - top < 24:
        return None
    return left, top, right, bottom


# 截取人脸并缩放至模型尺寸，将 BGR 像素转换为 RGB。
def face_crop(frame: np.ndarray, points: np.ndarray, size: int, scale: float,
              box: tuple[int, int, int, int] | None = None) -> np.ndarray | None:
    """Crop an OpenCV BGR frame and return RGB pixels for EfficientPhys."""
    box = box or face_box(frame, points, scale)
    if box is None:
        return None
    left, top, right, bottom = box
    left, top = max(0, left), max(0, top)
    right, bottom = min(frame.shape[1], right), min(frame.shape[0], bottom)
    if right - left < 24 or bottom - top < 24:
        return None
    # 工具箱使用 RGB 视频；仅转换模型裁剪图像，显示帧保持 BGR。
    crop = cv2.resize(frame[top:bottom, left:right], (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


# 封装权重加载、片段推理、频谱分析和质量判定。
class EfficientPhysEstimator:
    def __init__(self, model_path: Path, device: str, frame_depth: int = 10,
                 image_size: int = 72, min_spectral_score: float = 0.15, min_window: float = 8.0):
        if not np.isfinite(min_window) or min_window <= 0:
            raise ValueError("min_window must be finite and positive")
        self.min_window = min_window
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("EfficientPhys requested CUDA, but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.frame_depth = frame_depth
        self.image_size = image_size
        self.min_spectral_score = min_spectral_score
        self.model = EfficientPhys(frame_depth=frame_depth, image_size=image_size)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    # samples 保存时间戳和 RGB 人脸帧；low、high 为频段上下限（Hz）。
    def estimate(self, samples, low: float, high: float, previous_bpm: float | None = None):
        timestamps = np.asarray([item[0] for item in samples], dtype=np.float64)
        intervals = np.diff(timestamps)
        if len(intervals) < self.frame_depth or np.median(intervals) <= 0 or intervals.max() > 0.25:
            return None
        fs = 1.0 / float(np.median(intervals))
        if fs < max(12.0, 2 * high + 2):
            return None

        usable = (len(samples) - 1) // self.frame_depth * self.frame_depth
        if usable < max(self.frame_depth * 2, round(self.min_window * fs)):
            return None
        sos = signal.butter(3, [low, high], btype="bandpass", fs=fs, output="sos")
        padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if usable <= padlen:
            return None
        clip = np.asarray([item[1] for item in list(samples)[-(usable + 1):]], dtype=np.float32)
        deviation = float(clip.std())
        if deviation < 1e-6:
            return None
        # 官方 PURE 权重推理对整个片段统一计算均值和标准差进行标准化。
        clip = (clip - clip.mean()) / deviation
        tensor = torch.from_numpy(clip.transpose(0, 3, 1, 2)).to(self.device)
        with torch.inference_mode():
            predicted = self.model(tensor).flatten().float().cpu().numpy()

        if len(predicted) <= padlen:
            return None
        predicted = signal.detrend(predicted)
        filtered = signal.sosfiltfilt(sos, predicted)
        frequencies, power = signal.welch(
            filtered, fs=fs, nperseg=min(len(filtered), round(10 * fs)),
            nfft=max(2048, 2 ** int(np.ceil(np.log2(len(filtered))))),
        )
        band = (frequencies >= low) & (frequencies <= high)
        band_indices = np.flatnonzero(band)
        band_power = power[band]
        local = signal.find_peaks(np.r_[0.0, band_power, 0.0])[0] - 1
        peak_indices = [band_indices[index] for index in local]
        peak_indices.sort(key=lambda index: power[index], reverse=True)
        peak_index = peak_indices[0]

        def peak_mass(index: int) -> float:
            neighborhood = band & (np.abs(frequencies - frequencies[index]) <= 0.10)
            return float(power[neighborhood].sum())

        dominant_mass = peak_mass(peak_index)
        # 模型预测的二次谐波可能略强于脉搏基频。
        # 实测半频峰具备足够功率支持时，才优先将其作为基频候选。
        half = [index for index in peak_indices[1:]
                if abs(frequencies[index] * 2 - frequencies[peak_index]) * 60 <= 8
                and peak_mass(index) >= dominant_mass * 0.50]
        if half:
            peak_index = max(half, key=peak_mass)
        if previous_bpm is not None:
            nearby = [index for index in peak_indices
                      if abs(frequencies[index] * 60 - previous_bpm) <= 12
                      and peak_mass(index) >= dominant_mass * 0.40]
            if nearby:
                peak_index = min(nearby, key=lambda index: abs(frequencies[index] * 60 - previous_bpm))

        bpm = float(frequencies[peak_index] * 60)
        near = band & (np.abs(frequencies - frequencies[peak_index]) <= 0.10)
        quality = float(power[near].sum() / max(power[band].sum(), 1e-20))
        accepted = quality >= self.min_spectral_score
        return {
            "bpm": bpm,
            "quality": quality,
            "status": "GOOD" if accepted else "WEAK SIGNAL",
            "reason": "EfficientPhys pulse spectrum accepted" if accepted else "EfficientPhys spectrum is not concentrated enough",
            "fs": fs,
            "trace": filtered,
            "frequencies": frequencies,
            "spectrum": power,
            "region": 0,
            "support": [],
            "rates": np.full(3, np.nan),
            "primary_scores": np.zeros(3),
            "matched_rates": [None] * 3,
            "roi_scores": [0.0] * 3,
            "candidates": [[], [], []],
            "excluded": ["EfficientPhys uses a face crop"] * 3,
            "moving": np.zeros(3),
            "dark": np.zeros(3),
            "clipped": np.zeros(3),
        }
