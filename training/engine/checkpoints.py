"""Atomic saves, strict model loading and resume compatibility contracts."""
from pathlib import Path
import torch
from training.settings import cache_signature

def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)

def training_contract(cfg):
    # 断点恢复不能改变数据/网络/优化器/总 epoch：OneCycleLR 的计划必须一致。
    model_cfg = dict(cfg["model"])
    train_cfg = {k: v for k, v in cfg["train"].items() if k not in ("device", "num_workers", "log_every")}
    # C's former implicit defaults must not break old resume checkpoints.
    if model_cfg.get("name", "efficientphys_c") == "efficientphys_c":
        model_cfg.pop("name", None)
        if train_cfg.get("loss") == "mse":
            train_cfg.pop("loss")
        if train_cfg.get("scheduler") == "onecycle":
            train_cfg.pop("scheduler")
    return {"signature": cache_signature(cfg), "subjects": cfg["subjects"],
            "model": model_cfg, "train": train_cfg}

def load_model_weights(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    return payload
