"""保存打包权重及层配置，并加载到调用方提供的原始网络结构中。

只保存张量和普通元数据，读取时使用 weights_only=True，不序列化模型类。
使用本项目自定义格式，恢复时必须提供相同架构，包括未量化的模块。
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

from .linear import PackedLinear
from .pipeline import QuantizationResult, replace_module


FORMAT = "video_heartbeat.ptq"
FORMAT_VERSION = 1
_DTYPES = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """复制到 CPU 时保留相同张量的别名，避免 tied embedding/head 被重复保存。"""
    copied = {}
    state = {}
    for name, value in model.state_dict().items():
        key = (str(value.device), value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()), value.dtype)
        if key not in copied:
            copied[key] = value.detach().cpu()
        state[name] = copied[key]
    return state


def save_quantized(
    model: Union[nn.Module, QuantizationResult],
    path: Union[str, Path],
    *,
    metadata: Optional[dict[str, Any]] = None,
    overwrite: bool = False,
) -> Path:
    """默认禁止覆盖已有文件；传入 QuantizationResult 时一并保存量化配置。

    metadata 应仅含张量、基本类型及其容器，以便用 weights_only=True 读取。
    """
    details = dict(metadata or {})
    if isinstance(model, QuantizationResult):
        details["quantization"] = {
            "method": model.method,
            "config": model.config,
            "calibration_config": model.calibration_config,
            "skipped": model.skipped,
        }
        model = model.model
    replacements = {}
    for name, layer in model.named_modules():
        if isinstance(layer, PackedLinear):
            # 保存构造参数，恢复时先替换网络结构，再加载对应的打包张量。
            replacements[name] = {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "bits": layer.bits,
                "group_size": layer.group_size,
                "bias": layer.bias is not None,
                "dtype": str(layer._dtype_anchor.dtype),
                "method": layer.method,
            }
    if not replacements:
        raise ValueError("The model contains no packed layers")
    payload = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "layers": replacements,
        "state_dict": cpu_state_dict(model),
        "metadata": details,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # xb 采用独占创建，避免路径填错时静默覆盖原训练权重。
    with destination.open("wb" if overwrite else "xb") as stream:
        torch.save(payload, stream)
    return destination


def load_quantized(
    model: nn.Module,
    path: Union[str, Path],
    *,
    inplace: bool = False,
) -> nn.Module:
    """在传入模型所在设备恢复打包层，返回已切到 eval 的模型。

    用法::

        restored = load_quantized(build_model(config), "quantized.pt")

    未量化参数遵循 load_state_dict 的精度转换规则，
    因此请先用目标浮点精度构造原始网络。
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("format") != FORMAT:
        raise ValueError("Not a PTQ packed checkpoint")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported PTQ checkpoint version")
    if not isinstance(payload.get("layers"), Mapping) or not payload["layers"]:
        raise ValueError("Missing packed layer specifications")
    target = model if inplace else deepcopy(model)
    replacements = []
    # 先核对全部层的尺寸和偏置配置，再开始替换，减少中途失败造成的部分修改。
    for name, spec in payload["layers"].items():
        layer = target.get_submodule(name)
        if type(layer) is not nn.Linear:
            raise ValueError(f"{name!r}: expected the original nn.Linear architecture")
        if (layer.in_features, layer.out_features, layer.bias is not None) != (
            spec["in_features"], spec["out_features"], spec["bias"],
        ):
            raise ValueError(f"{name!r}: checkpoint and architecture dimensions differ")
        if spec["dtype"] not in _DTYPES:
            raise ValueError(f"Unsupported floating-point dtype: {spec['dtype']}")
        replacements.append((name, layer, spec))
    for name, layer, spec in replacements:
        packed = PackedLinear(
            spec["in_features"], spec["out_features"], spec["bits"], spec["group_size"],
            spec["bias"], _DTYPES[spec["dtype"]], layer.weight.device, spec["method"],
        )
        target = replace_module(target, name, packed)
    target.load_state_dict(payload["state_dict"], strict=True)
    return target.eval()
