"""缓存校准输入，并逐层收集激活均值、Gram 矩阵和有限数量的样本。"""

from dataclasses import dataclass, field
from itertools import islice
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Optional, Union

import torch
from torch import Tensor, nn

from .config import CalibrationConfig


@dataclass
class CalibrationBatch:
    """显式保存前向的位置参数和关键字参数，数据标签应由调用方先移除。"""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to(self, device: Union[torch.device, str]) -> "CalibrationBatch":
        """统一迁移整批嵌套输入，缓存和逐层重放共用此接口。"""
        return CalibrationBatch(move_tree(self.args, device), move_tree(self.kwargs, device))


def move_tree(value: Any, device: Union[torch.device, str]) -> Any:
    """递归处理 Tensor、字典、元组和列表，不改变整型输入的精度。"""
    if isinstance(value, Tensor):
        # CPU 上也复制一份，避免 DataLoader 复用缓冲区后修改已缓存的输入。
        result = value.detach().to(device)
        return result.clone() if result.device.type == "cpu" else result
    if isinstance(value, Mapping):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    return value


def cache_batches(
    data: Iterable[Any],
    config: CalibrationConfig,
    prepare_batch: Optional[Callable[[Any], Any]] = None,
) -> list[CalibrationBatch]:
    """字典作为关键字参数，元组/列表作为位置参数，其余值作为单个参数。

    若 DataLoader 返回 (inputs, labels)，请传 prepare_batch=lambda batch: batch[0]。
    这里不会猜测哪个字段是标签，也不会自动删除字段。
    """
    cached = []
    for item in islice(data, config.max_batches):
        item = prepare_batch(item) if prepare_batch else item
        if isinstance(item, CalibrationBatch):
            batch = item
        elif isinstance(item, Mapping):
            batch = CalibrationBatch(kwargs=dict(item))
        elif isinstance(item, (tuple, list)):
            batch = CalibrationBatch(args=tuple(item))
        else:
            batch = CalibrationBatch(args=(item,))
        cached.append(batch.to("cpu"))
    if not cached:
        raise ValueError("Calibration data is empty")
    return cached


class ActivationStats:
    """流式统计每列 mean(abs(x))，GPTQ 另收集 X.T @ X。

    每次 hook 至多统计 tokens_per_batch 行，再按独立随机优先级保留
    max_tokens 行，供 AWQ 搜索。局部生成器不改变调用方的全局随机状态。
    """

    def __init__(self, columns: int, device: torch.device, config: CalibrationConfig, hessian: bool) -> None:
        self.columns, self.device, self.config = columns, device, config
        self.count = 0
        self.abs_sum = torch.zeros(columns, dtype=torch.float32, device=device)
        self.gram = torch.zeros(columns, columns, dtype=torch.float32, device=device) if hessian else None
        self.samples = torch.empty(0, columns, dtype=torch.float32)
        self.priorities = torch.empty(0)
        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def collect_inputs(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """前向 hook：兼容 layer(x) 与 layer(input=x) 两种调用方式。"""
        inputs = args[0] if args else kwargs.get("input")
        if not isinstance(inputs, Tensor):
            raise TypeError("nn.Linear calibration requires a Tensor input")
        self.add(inputs)

    @torch.no_grad()
    def add(self, inputs: Tensor) -> None:
        """把 [..., 输入通道] 展平为 [样本行数, 输入通道] 后更新统计量。"""
        if not inputs.is_floating_point() or inputs.shape[-1] != self.columns:
            raise ValueError("Expected floating-point linear inputs with matching final dimension")
        values = inputs.detach().reshape(-1, self.columns)
        if values.shape[0] > self.config.tokens_per_batch:
            # 均匀抽取位置，避免只看到长序列开头的激活。
            indices = torch.linspace(
                0, values.shape[0] - 1, self.config.tokens_per_batch, device=values.device,
            ).long()
            values = values[indices]
        values = values.to(device=self.device, dtype=torch.float32)
        if values.numel() == 0:
            return
        if not torch.isfinite(values).all():
            raise ValueError("Calibration activations contain NaN or infinity")
        self.count += values.shape[0]
        self.abs_sum.add_(values.abs().sum(dim=0))
        if self.gram is not None:
            self.gram.addmm_(values.t(), values)
        # 给新旧样本各保留一个随机优先级，取最大的 K 个形成有界样本池。
        priorities = torch.rand(values.shape[0], generator=self.generator)
        priorities = torch.cat((self.priorities, priorities))
        samples = torch.cat((self.samples, values.cpu()))
        keep = priorities.topk(min(self.config.max_tokens, priorities.numel())).indices
        self.priorities, self.samples = priorities[keep], samples[keep]

    @property
    def mean_abs(self) -> Tensor:
        """AWQ 用每个输入通道的平均绝对激活衡量重要性。"""
        if not self.count:
            raise ValueError("No activations were observed")
        return self.abs_sum / self.count

    def hessian(self) -> Tensor:
        """GPTQ 的局部输出重建损失对应 H = 2 * X.T @ X / N。"""
        if self.gram is None or not self.count:
            raise ValueError("No Hessian statistics were collected")
        return self.gram * (2.0 / self.count)
