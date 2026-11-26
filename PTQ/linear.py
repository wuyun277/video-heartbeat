"""低位权重存储与浮点参考推理。

采用本项目自定义格式，与 AutoGPTQ/AutoAWQ 内核格式不兼容。
每次前向先恢复完整浮点权重，再调用 F.linear，因此打包能压缩文件，
但不能据此推断推理更快，或运行时峰值内存也会按位宽比例减少。
"""

from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .quantization import dequantize, effective_group_size


def pack_codes(codes: Tensor, bits: int) -> Tensor:
    """每字节打包 8/bits 个编码：4 位时，两个权重分别占低、高 4 位。"""
    if bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4 or 8")
    lanes = 8 // bits
    # 尾部补零凑齐一字节；先转为 int32，避免移位时溢出。
    grouped = F.pad(codes, (0, (-codes.shape[1]) % lanes)).int().reshape(codes.shape[0], -1, lanes)
    offsets = _bit_offsets(bits, codes.device)
    return (grouped << offsets).sum(dim=-1).to(torch.uint8).contiguous()


def unpack_codes(packed: Tensor, bits: int, columns: int) -> Tensor:
    """逐段右移并取掩码，展开编码后去掉补齐的尾列。"""
    offsets = _bit_offsets(bits, packed.device)
    values = (packed.to(torch.int32).unsqueeze(-1) >> offsets) & (2**bits - 1)
    return values.reshape(packed.shape[0], -1)[:, :columns].to(torch.uint8)


def _bit_offsets(bits: int, device: torch.device) -> Tensor:
    """打包和解包共用位偏移，例如 4 位编码的偏移为 [0, 4]。"""
    return torch.arange(8 // bits, device=device, dtype=torch.int32) * bits


class PackedLinear(nn.Module):
    """用于推理的 nn.Linear 替代层，仅量化权重，保留浮点偏置与激活。

    AWQ 保存 Q(W*s)，input_scale 保存通道缩放 s；反量化时恢复 Q(W*s)/s。
    GPTQ 不做通道缩放，对应的 input_scale 全为 1。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[Union[torch.device, str]] = None,
        method: str = "gptq",
    ) -> None:
        super().__init__()
        if bits not in (2, 4, 8) or min(in_features, out_features) < 1:
            raise ValueError("Invalid bit width or linear dimensions")
        if group_size != -1 and group_size < 1:
            raise ValueError("Invalid group size")
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size, self.method = bits, group_size, method
        size = effective_group_size(in_features, group_size)
        groups = (in_features + size - 1) // size
        packed_columns = (in_features + 8 // bits - 1) // (8 // bits)
        self.register_buffer("qweight", torch.zeros(out_features, packed_columns, dtype=torch.uint8, device=device))
        self.register_buffer("scales", torch.ones(out_features, groups, dtype=torch.float32, device=device))
        self.register_buffer("zeros", torch.zeros(out_features, groups, dtype=torch.int16, device=device))
        self.register_buffer("input_scale", torch.ones(in_features, dtype=torch.float32, device=device))
        self.register_buffer("bias", torch.zeros(out_features, dtype=dtype, device=device) if bias else None)
        # 零长度缓冲区记录浮点精度，无偏置层也能跟随 half()/bfloat16() 转换。
        self.register_buffer("_dtype_anchor", torch.empty(0, dtype=dtype, device=device))
        self.eval()

    @classmethod
    @torch.no_grad()
    def from_quantized(
        cls,
        source: nn.Linear,
        codes: Tensor,
        scales: Tensor,
        zeros: Tensor,
        *,
        bits: int,
        group_size: int,
        method: str,
        input_scale: Optional[Tensor] = None,
    ) -> "PackedLinear":
        """由量化算法的结果构造打包层，复制原层的尺寸、精度和偏置。"""
        result = cls(
            source.in_features, source.out_features, bits, group_size,
            source.bias is not None, source.weight.dtype, source.weight.device, method,
        )
        result.qweight.copy_(pack_codes(codes, bits))
        result.scales.copy_(scales)
        result.zeros.copy_(zeros)
        if input_scale is not None:
            result.input_scale.copy_(input_scale)
        if source.bias is not None:
            result.bias.copy_(source.bias.detach())
        return result

    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> Tensor:
        """先解码，再在浮点域中合并 AWQ 的逆缩放。"""
        codes = unpack_codes(self.qweight, self.bits, self.in_features)
        weight = dequantize(codes, self.scales, self.zeros, self.group_size)
        return (weight / self.input_scale.float()).to(dtype or self._dtype_anchor.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        # 当前为通用参考实现，矩阵乘法仍由浮点 F.linear 执行。
        weight = self.dequantize_weight(inputs.dtype)
        bias = self.bias.to(inputs.dtype) if self.bias is not None else None
        return F.linear(inputs, weight, bias)

    @torch.no_grad()
    def to_float(self) -> nn.Linear:
        """转换回普通 Linear；得到的是量化后的近似权重，无法恢复原始精度。"""
        result = nn.Linear(
            self.in_features, self.out_features, self.bias is not None,
            device=self.qweight.device, dtype=self._dtype_anchor.dtype,
        )
        result.weight.copy_(self.dequantize_weight())
        if self.bias is not None:
            result.bias.copy_(self.bias)
        return result.eval()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, method={self.method!r}"
        )
