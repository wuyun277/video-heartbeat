"""模型级入口：逐层校准，把普通 nn.Linear 替换为低位存储层。

每次使用当前已部分量化的模型重放输入，只保留一层的统计量。
GPTQ 的 Hessian 内存仍为 O(输入通道数的平方)。此参考实现覆盖单设备模型
直接调用的 nn.Linear，不覆盖卷积、自定义子类、函数式或融合算子的线性计算。
"""

from collections import Counter
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from itertools import chain
from typing import Any, Optional

import torch
from torch import nn

from .awq import quantize_linear_awq
from .calibration import ActivationStats, CalibrationBatch, cache_batches
from .config import AWQConfig, CalibrationConfig, GPTQConfig, PTQConfig
from .gptq import quantize_linear_gptq
from .linear import PackedLinear


@dataclass
class LayerSummary:
    """单层的校准样本数和存储字节数，不代表模型精度或推理峰值内存。"""

    name: str
    in_features: int
    out_features: int
    calibration_tokens: int
    original_layer_bytes: int
    packed_layer_bytes: int


@dataclass
class QuantizationResult:
    """量化后的模型、实际配置，以及已处理和跳过的层。"""

    model: nn.Module
    method: str
    config: dict[str, Any]
    calibration_config: dict[str, Any]
    layers: list[LayerSummary] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


def replace_module(model: nn.Module, name: str, replacement: nn.Module) -> nn.Module:
    """按点分路径替换子模块；空路径表示直接替换根模型。"""
    if not name:
        return replacement
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name)  # get_submodule("") 返回模型自身。
    setattr(parent, child_name, replacement)
    return model


def _storage_key(parameter: torch.Tensor) -> tuple[str, int]:
    """用底层存储地址识别共享参数，包括不同 Parameter 对同一存储的视图。"""
    return str(parameter.device), parameter.untyped_storage().data_ptr()


def _model_device(model: nn.Module) -> torch.device:
    """要求全部参数和缓冲区位于同一真实设备，拒绝分片和 meta 模型。"""
    devices = {tensor.device for tensor in chain(model.parameters(), model.buffers())}
    if len(devices) != 1 or next(iter(devices)).type == "meta":
        raise ValueError("Move the complete model to one real device before quantization")
    return next(iter(devices))


def _default_forward(model: nn.Module, batch: CalibrationBatch) -> Any:
    return model(*batch.args, **batch.kwargs)


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    calibration_data: Iterable[Any],
    config: Optional[PTQConfig] = None,
    *,
    calibration_config: Optional[CalibrationConfig] = None,
    prepare_batch: Optional[Callable[[Any], Any]] = None,
    forward_fn: Optional[Callable[[nn.Module, CalibrationBatch], Any]] = None,
    include: tuple[str, ...] = ("*",),
    exclude: tuple[str, ...] = ("lm_head", "*.lm_head"),
    inplace: bool = False,
) -> QuantizationResult:
    """按配置选择 GPTQ 或 AWQ，逐层量化，不训练模型或计算评估指标。

    用法示例，导入模块时不会执行::

        result = quantize_model(
            model, loader, GPTQConfig(bits=4, group_size=128),
            prepare_batch=lambda batch: batch[0],
        )
        quantized_model = result.model

    字典输入作为关键字参数，Tensor 作为单个位置参数；复杂输入用 CalibrationBatch。
    include/exclude 对点分模块名做通配符匹配。共享参数、已有 hook 的层会跳过，
    校准时未调用的层也会记录到 result.skipped。

    默认复制模型。inplace=True 可省去复制，但出错时可能已有部分层被替换。
    校准期间临时切到 eval，结束后恢复各层原来的训练标志；实际推理请调用 eval()。
    输入应为尚未含有 PackedLinear 的浮点模型。
    """
    config = config or GPTQConfig()
    if not isinstance(config, (GPTQConfig, AWQConfig)):
        raise TypeError("config must be GPTQConfig or AWQConfig")
    if any(isinstance(module, PackedLinear) for module in model.modules()):
        raise ValueError("Provide a floating-point model, not an already packed model")
    device = _model_device(model)
    method = "gptq" if isinstance(config, GPTQConfig) else "awq"
    quantizer = quantize_linear_gptq if method == "gptq" else quantize_linear_awq
    calibration_config = calibration_config or CalibrationConfig()
    target = model if inplace else deepcopy(model)
    result = QuantizationResult(target, method, asdict(config), asdict(calibration_config))

    # 同时统计重复引用，避免替换某个别名后破坏 embedding/head 等共享关系。
    owners = Counter(_storage_key(parameter) for _, parameter in target.named_parameters(remove_duplicate=False))
    selected = []
    for name, layer in target.named_modules():
        if type(layer) is not nn.Linear:
            continue
        if not any(fnmatchcase(name, pattern) for pattern in include):
            result.skipped[name] = "not selected by include patterns"
        elif any(fnmatchcase(name, pattern) for pattern in exclude):
            result.skipped[name] = "excluded by name pattern"
        elif min(layer.in_features, layer.out_features) < 1:
            result.skipped[name] = "empty input or output dimension"
        elif any(owners[_storage_key(parameter)] > 1 for parameter in layer.parameters()):
            result.skipped[name] = "shared module or tied parameter storage"
        elif layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks:
            result.skipped[name] = "existing layer hooks would not survive replacement"
        elif torch.nn.utils.parametrize.is_parametrized(layer):
            result.skipped[name] = "parametrized layer"
        else:
            selected.append(name)
    if not selected:
        raise ValueError(f"No eligible nn.Linear layers; skipped={result.skipped}")

    batches = cache_batches(calibration_data, calibration_config, prepare_batch)
    run_forward = forward_fn or _default_forward
    training_flags = {name: module.training for name, module in target.named_modules()}
    target.eval()
    try:
        for name in selected:
            layer = target.get_submodule(name)
            stats = ActivationStats(layer.in_features, device, calibration_config, method == "gptq")

            # hook 只观察当前层，异常时也必须移除，避免后续前向继续积累统计。
            handle = layer.register_forward_pre_hook(stats.collect_inputs, with_kwargs=True)
            try:
                for batch in batches:
                    run_forward(target, batch.to(device))
            finally:
                handle.remove()
            if stats.count == 0:
                result.skipped[name] = "layer forward was not called by calibration data"
                del stats
                continue
            packed = quantizer(layer, stats, config)
            result.layers.append(LayerSummary(
                name, layer.in_features, layer.out_features, stats.count,
                sum(value.numel() * value.element_size() for value in layer.parameters()),
                sum(value.numel() * value.element_size() for value in packed.buffers()),
            ))
            target = replace_module(target, name, packed)
            result.model = target
            # 及时释放 Hessian 和原始层，避免逐层累积显存占用。
            del stats, layer
        if not result.layers:
            raise ValueError("Calibration did not execute any eligible nn.Linear forward")
    finally:
        # 直接恢复各模块的标志，保留原模型可能存在的混合 train/eval 状态。
        for name, module in target.named_modules():
            if name in training_flags:
                module.training = training_flags[name]
    return result


@torch.no_grad()
def dequantize_model(model: nn.Module, *, inplace: bool = False) -> nn.Module:
    """把打包层还原为普通 Linear，权重仍为量化后的近似值。"""
    target = model if inplace else deepcopy(model)
    for name, layer in list(target.named_modules()):
        if isinstance(layer, PackedLinear):
            restored = layer.to_float()
            restored.training = layer.training
            target = replace_module(target, name, restored)
    return target
