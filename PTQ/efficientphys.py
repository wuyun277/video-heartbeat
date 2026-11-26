"""现有 EfficientPhys-C/T 项目的可选适配接口，仅在调用时加载训练模块。

只量化 nn.Linear，卷积与卷积式 patch embedding 保持原精度。
复用训练代码的 forward_batch，沿用末帧补齐及 Transformer 逐片段处理规则。
"""

from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from .calibration import CalibrationBatch
from .checkpoint import load_quantized, save_quantized
from .config import CalibrationConfig, PTQConfig
from .pipeline import QuantizationResult, quantize_model


def efficientphys_forward(model: nn.Module, batch: CalibrationBatch) -> Any:
    """把校准输入交给项目原有前向适配器，避免改变时序边界处理。"""
    from training.models.factory import forward_batch

    if len(batch.args) != 1 or batch.kwargs:
        raise ValueError("EfficientPhys expects one [batch,time,channels,height,width] Tensor")
    return forward_batch(model, batch.args[0])


def quantize_efficientphys(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    quantization_config: Optional[PTQConfig] = None,
    *,
    model_name: Optional[str] = None,
    device: str = "cpu",
    calibration_config: Optional[CalibrationConfig] = None,
    include: tuple[str, ...] = ("*",),
    exclude: tuple[str, ...] = (),
    overwrite: bool = False,
) -> QuantizationResult:
    """使用已有训练集缓存校准，并保存量化结果，不训练或评估网络。

    数据需提前预处理，输入权重应为项目 load_model_weights 支持的浮点检查点。
    """
    from training.data.loaders import make_loader
    from training.engine.checkpoints import load_model_weights
    from training.models.factory import build_model
    from training.settings import read_config

    source = Path(checkpoint_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("Output path must differ from the original checkpoint")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    cfg = read_config(config_path, model_name)
    cfg["train"]["device"] = device
    cfg["train"]["num_workers"] = 0  # 校准采用单进程加载，避免额外 worker 开销。
    model = build_model(cfg)
    load_model_weights(model, source)
    model.to(device).eval()
    calibration_config = calibration_config or CalibrationConfig()
    generator = torch.Generator().manual_seed(calibration_config.seed)
    loader = make_loader(cfg, "train", generator=generator)  # 校准不使用验证集或测试集。
    result = quantize_model(
        model, loader, quantization_config, calibration_config=calibration_config,
        prepare_batch=lambda batch: batch[0], forward_fn=efficientphys_forward,
        include=include, exclude=exclude, inplace=True,
    )
    result.model.eval()
    save_quantized(result, destination, overwrite=overwrite, metadata={
        "architecture": "efficientphys",
        "training_config": cfg,
        "source_checkpoint": str(source),
        "calibration_split": "train",
    })
    return result


def load_efficientphys_quantized(
    config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    *,
    model_name: Optional[str] = None,
    device: str = "cpu",
) -> nn.Module:
    """根据原训练配置重建模型，再加载低位权重。"""
    from training.models.factory import build_model
    from training.settings import read_config

    cfg = read_config(config_path, model_name)
    model = build_model(cfg).to(device)
    return load_quantized(model, checkpoint_path, inplace=True)
