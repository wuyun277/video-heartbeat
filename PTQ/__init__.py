"""纯 PyTorch 的 GPTQ/AWQ 训练后权重量化实现，兼容 Python 3.9 类型标注。

公共接口：
    quantize_model(model, data, GPTQConfig(...)) -> QuantizationResult
    quantize_model(model, data, AWQConfig(...)) -> QuantizationResult
    save_quantized(result, path)
    load_quantized(original_architecture, path) -> nn.Module

支持普通 nn.Linear 的 2/4/8 位权重打包，推理时解包成浮点权重计算。
导入模块不执行量化、训练、测试或下载，不依赖自定义 CUDA 内核。
"""

from .awq import quantize_linear_awq
from .calibration import ActivationStats, CalibrationBatch
from .checkpoint import load_quantized, save_quantized
from .config import AWQConfig, CalibrationConfig, GPTQConfig, PTQConfig, QuantizationConfig
from .gptq import quantize_linear_gptq
from .linear import PackedLinear
from .pipeline import LayerSummary, QuantizationResult, dequantize_model, quantize_model

__all__ = [
    "AWQConfig", "ActivationStats", "CalibrationBatch", "CalibrationConfig",
    "GPTQConfig", "LayerSummary", "PackedLinear", "PTQConfig", "QuantizationConfig",
    "QuantizationResult", "dequantize_model", "load_quantized",
    "quantize_linear_awq", "quantize_linear_gptq", "quantize_model", "save_quantized",
]
