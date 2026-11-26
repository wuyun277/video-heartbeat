"""Read UBFC labels, preprocess videos and manage signed caches."""
from __future__ import annotations

import json
import logging
from pathlib import Path
import uuid

import cv2
import numpy as np
from training.settings import cache_root, cache_signature, write_json

LOGGER = logging.getLogger(__name__)

def source_stamp(folder):
    return {name: {"size": (folder / name).stat().st_size,
                   "mtime_ns": (folder / name).stat().st_mtime_ns}
            for name in ("vid.avi", "ground_truth.txt")}

def read_ground_truth(path):
    rows = [np.fromstring(line, sep=" ", dtype=np.float64)
            for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != 3 or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{path}: expected 3 equal-length rows (PPG, HR, timestamps)")
    intervals = np.diff(rows[2])
    if any(not np.isfinite(row).all() for row in rows) or np.any(intervals < 0):
        raise ValueError(f"{path}: invalid values or backwards timestamps")
    if rows[2][-1] <= rows[2][0]:
        raise ValueError(f"{path}: timestamps have no positive duration")
    if np.any(intervals == 0):
        LOGGER.warning("%s: %d repeated timestamps; index alignment keeps all samples, timestamp alignment averages duplicates",
                       path, np.count_nonzero(intervals == 0))
    return rows

def fixed_face_box(frame, subject, cfg):
    p = cfg["preprocess"]
    manual = p["manual_face_boxes"].get(subject)
    if manual is not None:
        if len(manual) != 4:
            raise ValueError("Manual box must be [x, y, width, height]")
        x, y, w, h = map(int, manual)
    else:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if detector.empty():
            raise RuntimeError("OpenCV Haar cascade asset is unavailable")
        faces = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not len(faces):
            raise ValueError(f"{subject}: no face in first frame; set manual_face_boxes instead of using background")
        x, y, w, h = map(int, max(faces, key=lambda box: box[2]))
        scale = p["face_box_scale"]
        # 使用工具箱一致的放大方式，并限制左上角不超出图像边界。
        x, y = int(max(0, x - (scale - 1) * w / 2)), int(max(0, y - (scale - 1) * h / 2))
        w, h = int(w * scale), int(h * scale)
    right, bottom = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x < 0 or y < 0 or right - x < 24 or bottom - y < 24:
        raise ValueError(f"{subject}: invalid face box")
    return x, y, right, bottom

def load_metadata(cfg, subject):
    folder = cache_root(cfg) / subject
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if meta["signature"] != cache_signature(cfg) or meta["subject"] != subject:
        raise ValueError(f"{subject}: cache configuration mismatch")
    raw = Path(cfg["paths"]["data_dir"]) / subject
    if raw.exists() and meta["source"] != source_stamp(raw):
        raise ValueError(f"{subject}: raw data changed since preprocessing; use a new cache directory")
    frames = np.load(folder / "frames.npy", mmap_mode="r")
    if frames.shape != (meta["frames"], cfg["preprocess"]["image_size"], cfg["preprocess"]["image_size"], 3):
        raise ValueError(f"{subject}: invalid cached frame shape")
    if frames.dtype != np.uint8:
        raise ValueError(f"{subject}: expected uint8 frame cache")
    for name in ("labels.npy", "reference_hr.npy"):
        values = np.load(folder / name, mmap_mode="r")
        if values.shape != (meta["frames"],) or not np.isfinite(values).all():
            raise ValueError(f"{subject}: invalid cached {name}")
    return folder, meta

def prepare_subject(cfg, subject):
    root = cache_root(cfg)
    target = root / subject
    if target.exists():
        load_metadata(cfg, subject)
        LOGGER.info("%s: valid cache exists; skipping", subject)
        return
    raw = Path(cfg["paths"]["data_dir"]) / subject
    stamp = source_stamp(raw)
    ppg, hr, times = read_ground_truth(raw / "ground_truth.txt")
    p = cfg["preprocess"]
    cap = cv2.VideoCapture(str(raw / "vid.avi"))
    crops = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {raw / 'vid.avi'}")
        nominal = cap.get(cv2.CAP_PROP_FPS)
        declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(nominal) or nominal <= 0:
            raise ValueError(f"{subject}: invalid video FPS {nominal}")
        if abs(nominal - p["fps"]) / p["fps"] > 0.10:
            raise ValueError(f"{subject}: video FPS {nominal} differs by more than 10% from configured {p['fps']}")
        if abs(nominal - p["fps"]) > 0.5:
            LOGGER.warning("%s: metadata FPS %.3f differs from nominal %.1f; keeping original frames and recording actual FPS for testing",
                           subject, nominal, p["fps"])
        box = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if box is None:
                box = ((0, 0, frame.shape[1], frame.shape[0])
                       if p.get("crop_mode", "face") == "full_frame"
                       else fixed_face_box(frame, subject, cfg))
            x0, y0, x1, y1 = box
            crop = cv2.resize(frame[y0:y1, x0:x1], (p["image_size"], p["image_size"]), interpolation=cv2.INTER_AREA)
            crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if len(crops) % 300 == 0:
                LOGGER.info("%s: decoded %d frames", subject, len(crops))
    finally:
        cap.release()
    if len(crops) < cfg["train"]["chunk_length"]:
        raise ValueError(f"{subject}: too few decoded frames")
    if declared_count > 0 and declared_count != len(crops):
        raise ValueError(f"{subject}: decoded {len(crops)} of {declared_count} frames; check for truncated video")
    frames = np.stack(crops)
    del crops
    count = len(frames)
    duration = (count - 1) / nominal
    if abs((times[-1] - times[0]) - duration) > max(1.0, duration * 0.05):
        raise ValueError(f"{subject}: PPG duration and video duration differ significantly")
    if p["alignment"] == "index":
        # 官方 UBFC loader 按样本位置使用标签；长度不同才做等间距插值。
        axis = np.linspace(0, 1, len(ppg))
        grid = np.linspace(0, 1, count)
    else:
        axis = times - times[0]
        # 时间戳精度可能产生重复；插值前对相同时间的标签取平均。
        axis, starts, counts = np.unique(axis, return_index=True, return_counts=True)
        ppg = np.add.reduceat(ppg, starts) / counts
        hr = np.add.reduceat(hr, starts) / counts
        grid = np.arange(count) / nominal
        if grid[-1] > axis[-1] + 1 / nominal:
            raise ValueError("Timestamp alignment would extrapolate PPG beyond available labels")
    ppg = np.interp(grid, axis, ppg)
    hr = np.interp(grid, axis, hr)
    diff = np.diff(ppg)
    std = float(diff.std())
    mean, image_std = float(frames.mean()), float(frames.std())
    if std < 1e-8 or image_std < 1e-8:
        raise ValueError(f"{subject}: constant PPG or video")
    if p.get("label_type", "diff_normalized") == "standardized":
        pulse_std = float(ppg.std())
        if pulse_std < 1e-8:
            raise ValueError(f"{subject}: constant PPG")
        labels = ((ppg - ppg.mean()) / pulse_std).astype(np.float32)
    else:
        labels = np.r_[diff / std, 0].astype(np.float32)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".{subject}-{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    # uint8 保存图像；读取片段时使用整段视频的统计量标准化，避免 float 缓存膨胀。
    np.save(temporary / "frames.npy", frames)
    np.save(temporary / "labels.npy", labels)
    np.save(temporary / "reference_hr.npy", hr.astype(np.float32))
    cv2.imwrite(str(temporary / "crop_preview.jpg"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    write_json(temporary / "metadata.json", {
        "subject": subject, "signature": cache_signature(cfg), "source": stamp,
        "frames": count, "fps": nominal, "configured_fps": p["fps"], "mean": mean, "std": image_std,
        "face_box_xyxy": list(box), "alignment": p["alignment"], "color_order": "RGB"
    })
    temporary.rename(target)
    LOGGER.info("%s: cached %d frames, %.1fs; preview: %s", subject, count, duration, target / "crop_preview.jpg")
