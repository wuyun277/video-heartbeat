"""Load complete, non-overlapping UBFC clips from preprocessing caches."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import load_metadata

class UBFCChunks(Dataset):
    def __init__(self, cfg, split):
        self.length = cfg["train"]["chunk_length"]
        self.image_normalization = cfg["preprocess"].get("image_normalization", "standardized")
        self.subjects = []
        self.index = []
        for name in cfg["subjects"][split]:
            folder, meta = load_metadata(cfg, name)
            sid = len(self.subjects)
            self.subjects.append((folder, meta))
            self.index.extend((sid, start) for start in range(0, meta["frames"] - self.length + 1, self.length))
        if not self.index:
            raise ValueError(f"{split}: no full training chunks; run prepare first")

    def __len__(self):
        return len(self.index)

    # 按需读取缓存，用视频统计量标准化，返回时间×通道×高×宽张量。
    def __getitem__(self, index):
        sid, start = self.index[index]
        folder, meta = self.subjects[sid]
        end = start + self.length
        frames = np.load(folder / "frames.npy", mmap_mode="r")[start:end].astype(np.float32)
        if self.image_normalization == "standardized":
            frames = (frames - meta["mean"]) / meta["std"]
        labels = np.load(folder / "labels.npy", mmap_mode="r")[start:end].copy()
        return torch.from_numpy(frames.transpose(0, 3, 1, 2).copy()), torch.from_numpy(labels), meta["subject"], start
