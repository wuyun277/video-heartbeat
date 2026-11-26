"""量化与校准配置；只保存参数，不在导入时执行任何计算。"""

from dataclasses import dataclass
import math
from typing import Union


@dataclass(frozen=True)
class QuantizationConfig:
    """GPTQ、AWQ 共用的分组量化参数。"""

    bits: int = 4                  # 每个权重占用的位数，当前支持 2/4/8 位
    group_size: int = 128           # 每行按输入列分组，-1 表示整行共用参数
    symmetric: bool = False        # True：对称量化；False：带零点的非对称量化

    def __post_init__(self) -> None:
        if self.bits not in (2, 4, 8):
            raise ValueError("Packed weights support 2, 4 or 8 bits")
        if self.group_size != -1 and self.group_size < 1:
            raise ValueError("group_size must be positive, or -1 for a whole row")


@dataclass(frozen=True)
class GPTQConfig(QuantizationConfig):
    """GPTQ 用二阶信息补偿量化误差。"""

    block_size: int = 128           # 块内逐列更新，块间用矩阵乘法集中更新
    damping: float = 0.01           # Hessian 对角阻尼相对系数，提高分解稳定性
    act_order: bool = True          # 优先处理 Hessian 对角值较大的输入通道

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if not math.isfinite(self.damping) or self.damping <= 0:
            raise ValueError("damping must be finite and positive")


@dataclass(frozen=True)
class AWQConfig(QuantizationConfig):
    """AWQ 搜索通道缩放系数，并可进一步搜索权重裁剪范围。"""

    scale_steps: int = 20           # 通道缩放指数 alpha 的候选数
    clip_steps: int = 10            # 每组裁剪阈值的候选数
    max_clip_shrink: float = 0.5     # 裁剪阈值最多缩小的比例
    output_chunk_size: int = 128    # 按输出通道分块，限制搜索时的临时内存
    clip: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(self.scale_steps, self.clip_steps, self.output_chunk_size) < 1:
            raise ValueError("Search steps and output_chunk_size must be positive")
        if not 0 <= self.max_clip_shrink < 1:
            raise ValueError("max_clip_shrink must be in [0, 1)")


# Python 3.9 使用 Union；共用别名避免在各接口重复较长的类型表达式。
PTQConfig = Union[GPTQConfig, AWQConfig]


@dataclass(frozen=True)
class CalibrationConfig:
    """限制校准工作量；token 指展平后的一行激活，不限于文本 token。"""

    max_batches: int = 16           # 最多缓存多少个输入批次，供逐层重复前向
    max_tokens: int = 512           # AWQ 最多保留多少行激活样本
    tokens_per_batch: int = 4096    # 每次 hook 最多统计多少行激活
    seed: int = 0                   # 激活采样使用的局部随机种子

    def __post_init__(self) -> None:
        if min(self.max_batches, self.max_tokens, self.tokens_per_batch) < 1:
            raise ValueError("Calibration limits must be positive")
