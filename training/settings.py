"""Shared data configuration and separate C/T training profiles."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
MODEL_NAMES = ("efficientphys_c", "efficientphys_t")


def model_contract(cfg):
    # Legacy checkpoints contain only frame_depth for C.
    model = copy.deepcopy(cfg["model"])
    model.setdefault("name", "efficientphys_c")
    return model


def read_config(path, model_name=None):
    path = Path(path).resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if "models" in source:
        selected = model_name or source.get("active_model", "efficientphys_c")
        if selected not in MODEL_NAMES or selected not in source["models"]:
            raise ValueError(f"Unknown model profile: {selected}")
        cfg = {k: copy.deepcopy(v) for k, v in source.items() if k not in ("models", "active_model")}
        profile = source["models"][selected]
        # Profiles may override shared data/processing settings independently.
        for key in ("paths", "subjects", "preprocess", "test"):
            cfg.setdefault(key, {}).update(copy.deepcopy(profile.get(key, {})))
        cfg["model"] = copy.deepcopy(profile["model"])
        cfg["model"]["name"] = selected
        cfg["train"] = copy.deepcopy(profile["train"])
    else:
        # Old single-model JSON files remain valid, including saved run configs.
        cfg = copy.deepcopy(source)
        selected = cfg["model"].get("name", "efficientphys_c")
        if model_name is not None and model_name != selected:
            raise ValueError("Requested model does not match single-model configuration")
    if selected not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {selected}")
    groups = cfg["subjects"]
    seen = set()
    for split in ("train", "valid", "test"):
        names = groups[split]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{split}: provide nonempty, unique subject names")
        if any(not re.fullmatch(r"subject\d+", name) for name in names):
            raise ValueError("Subject names must have the form subject8")
        if seen.intersection(names):
            raise ValueError("Subjects must not overlap across train/valid/test")
        seen.update(names)
    t, p = cfg["train"], cfg["preprocess"]
    for key in ("epochs", "batch_size", "chunk_length", "log_every"):
        if not isinstance(t[key], int) or t[key] <= 0:
            raise ValueError(f"train.{key} must be a positive integer")
    depth = cfg["model"]["frame_depth"]
    if not isinstance(depth, int) or depth < 2 or t["chunk_length"] % depth:
        raise ValueError("chunk_length must be divisible by frame_depth >= 2")
    if p["image_size"] not in (36, 72, 96):
        raise ValueError("image_size must be 36, 72 or 96")
    if selected == "efficientphys_t":
        m = cfg["model"]
        variant = m.get("variant", "t1")
        heads = m.get("num_heads", [3, 6, 12, 24] if variant == "t1" else [3, 6])
        if variant not in ("t1", "t2") or len(heads) != (4 if variant == "t1" else 2):
            raise ValueError("Invalid Transformer variant or head count")
        for key, default in (("patch_size", 4), ("embed_dim", 96), ("window_size", 7)):
            value = m.get(key, default)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"model.{key} must be a positive integer")
        if p["image_size"] % m.get("patch_size", 4):
            raise ValueError("image_size must be divisible by patch_size")
        for i, head in enumerate(heads):
            if not isinstance(head, int) or head <= 0 or m.get("embed_dim", 96) * 2 ** i % head:
                raise ValueError("Each stage dimension must be divisible by num_heads")
        if not 0 <= m.get("dropout", 0.0) < 1 or m.get("mlp_ratio", 4.0) <= 0:
            raise ValueError("Invalid Transformer dropout or mlp_ratio")
    for value in (p["fps"], p["face_box_scale"], t["learning_rate"], cfg["test"]["window_seconds"]):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("FPS, crop scale, learning rate and test window must be positive")
    if p["alignment"] not in ("index", "timestamp"):
        raise ValueError("alignment must be index or timestamp")
    if p.get("crop_mode", "face") not in ("face", "full_frame"):
        raise ValueError("crop_mode must be face or full_frame")
    if p.get("image_normalization", "standardized") not in ("standardized", "raw"):
        raise ValueError("image_normalization must be standardized or raw")
    if p.get("label_type", "diff_normalized") not in ("diff_normalized", "standardized"):
        raise ValueError("label_type must be diff_normalized or standardized")
    if t.get("loss", "mse") != "mse":
        raise ValueError("Only mse is supported by these training profiles")
    if t.get("scheduler", "onecycle") not in ("onecycle", "constant"):
        raise ValueError("scheduler must be onecycle or constant")
    if t["num_workers"] < 0 or not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("Invalid num_workers or weight_decay")
    if not 0 < cfg["test"]["min_bpm"] < cfg["test"]["max_bpm"] < p["fps"] * 30:
        raise ValueError("Invalid heart-rate range for configured FPS")
    for key, value in cfg["paths"].items():
        resolved = Path(value).expanduser()
        cfg["paths"][key] = str(resolved if resolved.is_absolute() else path.parent / resolved)
    return cfg


def cache_signature(cfg):
    # Preserve the existing C cache signature; T's extra processing keys isolate its cache.
    payload = {"version": 1, "preprocess": cfg["preprocess"], "color_order": "RGB"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cache_root(cfg):
    return Path(cfg["paths"]["cache_dir"]) / cache_signature(cfg)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
