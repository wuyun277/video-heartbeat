"""GPTQ、AWQ 共用的仿射量化：整数编码 q，对应浮点近似值 (q - zero) * scale。

权重形状为 [输出通道, 输入通道]，每行沿输入列连续分组，支持不足一组的尾部。
"""

import torch
from torch import Tensor

from .config import QuantizationConfig


def effective_group_size(columns: int, group_size: int) -> int:
    """把 -1 或大于列数的配置转换为实际组宽。"""
    return columns if group_size == -1 else min(columns, group_size)


def affine_parameters(weight: Tensor, bits: int, symmetric: bool) -> tuple[Tensor, Tensor]:
    """对一个二维权重组，逐行求缩放系数 scale 和整数零点 zero。"""
    # 把 0 包含在范围内，保证零权重可以精确表示。
    minimum = weight.amin(dim=1).clamp(max=0)
    maximum = weight.amax(dim=1).clamp(min=0)
    if symmetric:
        scale = torch.maximum(-minimum, maximum) / (2 ** (bits - 1) - 1)
    else:
        scale = (maximum - minimum) / (2**bits - 1)
    # 全零组采用 scale=1，避免除零；两种量化方式共用这一步。
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    if symmetric:
        # 用无符号整数保存编码，把有符号整数的原点平移到中间。
        zero = torch.full_like(minimum, 2 ** (bits - 1), dtype=torch.int16)
    else:
        zero = (-minimum / scale).round().clamp(0, 2**bits - 1).to(torch.int16)
    return scale, zero


def encode(weight: Tensor, scale: Tensor, zero: Tensor, bits: int) -> Tensor:
    """q = round(w / scale + zero)，再截断到 bits 位整数范围。"""
    return (weight / scale + zero).round().clamp(0, 2**bits - 1).to(torch.uint8)


def group_parameters(weight: Tensor, config: QuantizationConfig) -> tuple[Tensor, Tensor]:
    """返回 [输出通道, 组数] 形状的 scale 和 zero。"""
    size = effective_group_size(weight.shape[1], config.group_size)
    scales, zeros = [], []
    for start in range(0, weight.shape[1], size):
        scale, zero = affine_parameters(weight[:, start : start + size], config.bits, config.symmetric)
        scales.append(scale)
        zeros.append(zero)
    return torch.stack(scales, dim=1), torch.stack(zeros, dim=1)


def dequantize(codes: Tensor, scales: Tensor, zeros: Tensor, group_size: int) -> Tensor:
    """把每组的参数映射回各输入列，再恢复浮点权重。"""
    size = effective_group_size(codes.shape[1], group_size)
    groups = torch.arange(codes.shape[1], device=codes.device) // size
    return (codes.float() - zeros[:, groups].float()) * scales[:, groups].float()


def quantize_groups(weight: Tensor, config: QuantizationConfig) -> tuple[Tensor, Tensor, Tensor]:
    """按组求参数并编码，返回未做位打包的整数、scale、zero。"""
    scales, zeros = group_parameters(weight, config)
    size = effective_group_size(weight.shape[1], config.group_size)
    groups = torch.arange(weight.shape[1], device=weight.device) // size
    codes = encode(weight, scales[:, groups], zeros[:, groups], config.bits)
    return codes, scales, zeros


def fake_quantize(weight: Tensor, config: QuantizationConfig) -> Tensor:
    """先量化再反量化，用浮点张量模拟量化误差，供 AWQ 搜索使用。"""
    codes, scales, zeros = quantize_groups(weight, config)
    return dequantize(codes, scales, zeros, config.group_size)
