"""Read-only UBFC inventory and timestamp-aware label interpolation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in ("data_dir", "runs_dir"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg["efficientphys"]["weights"] = str((path.parent / cfg["efficientphys"]["weights"]).resolve())
    if not cfg["subjects"] or len(set(cfg["subjects"])) != len(cfg["subjects"]):
        raise ValueError("subjects must be nonempty and unique")
    donors = [t["subject"] for t in cfg["targets"] if t["kind"] == "ppg"]
    overlap = set(cfg["subjects"] + donors) & set(cfg["reserved_subjects"])
    if overlap:
        raise ValueError("Reserved subjects cannot enter development experiments: " + str(sorted(overlap)))
    for subject in cfg["subjects"] + donors:
        if Path(subject).name != subject or subject in (".", ".."):
            raise ValueError("Subject must be a folder name")
    for name in ("clip_seconds", "window_seconds"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    if cfg["clip_start_seconds"] < 0 or cfg["clip_seconds"] < cfg["window_seconds"]:
        raise ValueError("Invalid clip interval")
    low, high = cfg["band_hz"]
    if not 0 < low < high or cfg["window_seconds"] <= 2 * cfg["edge_trim_seconds"] + 2:
        raise ValueError("Invalid frequency band or edge trim")
    if not cfg["strengths"] or any(not math.isfinite(v) or not 0 < v <= 0.05 for v in cfg["strengths"]):
        raise ValueError("Use explicit strengths in (0, 0.05]")
    weights = np.asarray(cfg["rgb_weights"], dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.max(np.abs(weights)) > 2:
        raise ValueError("Invalid RGB weights")
    if cfg["mask_blur_sigma"] < 0:
        raise ValueError("Invalid blur sigma")
    if len({t["name"] for t in cfg["targets"]}) != len(cfg["targets"]):
        raise ValueError("Target names must be unique")
    for target in cfg["targets"]:
        if not target["name"].replace("_", "").isalnum():
            raise ValueError("Use alphanumeric target names")
        if target["kind"] == "sine":
            if not low < target["bpm"] / 60 < high:
                raise ValueError("Sine target outside evaluation band")
        elif target["kind"] != "ppg":
            raise ValueError("Unknown target kind")
    if not set(cfg["extractors"]) <= {"GREEN", "POS", "CHROM", "EfficientPhys"}:
        raise ValueError("Unknown extractor")
    if cfg["efficientphys"]["output_type"] != "diff_normalized":
        raise ValueError("This PURE weight protocol expects diff_normalized output")
    chunk = cfg["efficientphys"]["chunk_frames"]
    if not isinstance(chunk, int) or chunk < 10 or chunk % 10:
        raise ValueError("EfficientPhys chunk_frames must be a positive multiple of 10")
    return cfg


def read_labels(path):
    rows = [np.fromstring(line, sep=" ") for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(rows) != 3 or len(rows[0]) < 2 or len({len(x) for x in rows}) != 1:
        raise ValueError("Expected three equal-length UBFC rows: PPG, HR, seconds")
    ppg, hr, times = np.asarray(rows)
    if not np.isfinite(rows).all() or np.any(np.diff(times) < 0) or times[-1] <= times[0]:
        raise ValueError("Non-finite labels or invalid label times")
    return ppg, hr, times - times[0]


def interpolate_labels(path, times):
    ppg, hr, axis = read_labels(path)
    unique, starts, counts = np.unique(axis, return_index=True, return_counts=True)
    if times[0] < unique[0] - 1e-6 or times[-1] > unique[-1] + 1e-6:
        raise ValueError("Label interpolation would extrapolate outside measured time range")
    return tuple(np.interp(times, unique, np.add.reduceat(x, starts) / counts) for x in (ppg, hr))


def inventory(cfg):
    root = Path(cfg["data_dir"])
    if not root.is_dir():
        raise ValueError("Dataset directory does not exist: " + str(root))
    records = []
    for folder in sorted(root.iterdir(), key=lambda p: (int(p.name[7:]) if p.name.startswith("subject") and p.name[7:].isdigit() else 99999, p.name)):
        if not folder.is_dir():
            continue
        row = {"subject": folder.name, "role": "development" if folder.name in cfg["subjects"] else "reserved" if folder.name in cfg["reserved_subjects"] else "available"}
        try:
            video, labels = folder / "vid.avi", folder / "ground_truth.txt"
            ppg, hr, times = read_labels(labels)
            cap = cv2.VideoCapture(str(video))
            try:
                if not cap.isOpened():
                    raise ValueError("Cannot open video")
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, frame = cap.read()
                if not ok or not np.isfinite(fps) or fps <= 2 * cfg["band_hz"][1] or count < 2:
                    raise ValueError("Invalid video metadata or first-frame decode")
                row.update(fps=float(fps), frames=count, width=frame.shape[1], height=frame.shape[0], video_duration_s=(count - 1) / fps)
            finally:
                cap.release()
            row.update(label_samples=len(ppg), label_duration_s=float(times[-1]), duplicate_label_times=int(np.sum(np.diff(times) == 0)), mean_device_hr=float(hr.mean()), video_bytes=video.stat().st_size, video_mtime_ns=video.stat().st_mtime_ns, labels_sha256=digest(labels), status="ok")
            row["duration_difference_s"] = row["video_duration_s"] - row["label_duration_s"]
        except Exception as exc:
            row.update(status="error", error=str(exc))
        records.append(row)
    missing = set(cfg["subjects"]) - {r["subject"] for r in records}
    if missing:
        raise ValueError("Missing selected subjects: " + str(sorted(missing)))
    return records


def load_clip(cfg, subject):
    path = Path(cfg["data_dir"]) / subject / "vid.avi"
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode " + str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 2 * cfg["band_hz"][1]:
            raise ValueError("Invalid video FPS")
        first = int(round(cfg["clip_start_seconds"] * fps))
        requested = int(round(cfg["clip_seconds"] * fps))
        if first and not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise ValueError("Video seek failed")
        if first and abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - first) > 0.5:
            raise ValueError("Video seek did not reach requested frame")
        frames = []
        for _ in range(requested):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Clip truncated before requested frame count")
            frames.append(frame)
    finally:
        cap.release()
    times = (np.arange(len(frames)) + first) / fps
    ppg, hr = interpolate_labels(path.with_name("ground_truth.txt"), times)
    return np.asarray(frames), times, fps, ppg, hr
