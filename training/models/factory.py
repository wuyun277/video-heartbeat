"""Create C/T models and adapt clip batches to their input contracts."""
from __future__ import annotations

import torch
from .efficientphys_cnn import EfficientPhysCNN

def forward_batch(model, frames):
    """Match the toolbox trainer: flatten clips and duplicate the final frame.

    This intentionally preserves the toolbox's cross-clip boundary differences.
    Do not change it silently when comparing to the original training recipe.
    """
    n, length, channels, height, width = frames.shape
    # T runs each clip separately to avoid cross-subject temporal differences.
    from training.models.efficientphys_transformer import EfficientPhysTransformer
    if isinstance(model, EfficientPhysTransformer):
        return torch.stack([model(torch.cat((clip, clip[-1:]), dim=0)).squeeze(-1) for clip in frames])
    flattened = frames.reshape(n * length, channels, height, width)
    return model(torch.cat((flattened, flattened[-1:]), dim=0)).reshape(n, length)

def build_model(cfg):
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "efficientphys_c")
    if name == "efficientphys_c":
        return EfficientPhysCNN(model_cfg["frame_depth"], cfg["preprocess"]["image_size"])
    if name == "efficientphys_t":
        from training.models.efficientphys_transformer import EfficientPhysTransformer
        options = {key: value for key, value in model_cfg.items() if key != "name"}
        return EfficientPhysTransformer(image_size=cfg["preprocess"]["image_size"], **options)
    raise ValueError(f"Unknown model: {name}")
