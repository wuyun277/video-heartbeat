
from typing import Optional

import torch
from torch import Tensor, nn

from .calibration import ActivationStats
from .config import AWQConfig
from .linear import PackedLinear
from .quantization import effective_group_size, fake_quantize, quantize_groups


#找到合适的缩放因子
@torch.no_grad()
def _search_scale(weight: Tensor, inputs: Tensor, mean_abs: Tensor, config: AWQConfig) -> Tensor:
    """在离散 alpha 网格上选择输出重建误差最小的通道缩放 s。

    参数：
        weight：[O, I] 的浮点权重；inputs：[N, I] 的校准样本。
        mean_abs：[I]，校准模块对已统计的激活计算的逐通道 mean(abs(x))；
            它可来自比 inputs 样本池更多的样本，二者分别用于构造候选和评分。
        config：控制候选数量、量化位宽、分组方式及输出分块大小。
    返回：
        [I] 的正缩放向量；最优仅指当前候选集合中的最优结果。

    目标为 mean((X @ (Q(W*s)/s - W).T)**2)。偏置在两个输出相减时抵消。
    这里衡量激活加权的输出误差，而不是单独比较权重的均方误差。
    """
    # 使用平均绝对激活构造候选缩放因子。
    activations = mean_abs.clamp_min(1e-4)#防止通道值为0
    best_scale = torch.ones_like(activations)
    best_loss = float("inf")
    for step in range(config.scale_steps):
        # 多个候选时覆盖 [0, 1]：alpha=0 对应所有通道缩放为 1，
        # alpha 越大，平均绝对激活较大的通道相对被放大得越明显。
        # 只有一个候选时取 alpha=0；max(1, ...) 避免分母为零。
        alpha = step / max(1, config.scale_steps - 1)
        scale = activations.pow(alpha)
        # 用极值的几何均值控制整体量级，不改变通道间的相对比例。
        # 归一化后最小和最大缩放系数的乘积约为 1，并非让所有元素均值为 1。
        scale /= (scale.amin() * scale.amax()).sqrt()
        # [1, I] 广播到所有输出行：同一个输入通道共用同一个 s。
        # fake_quantize 会根据当前候选重新求组参数，返回浮点近似权重；
        # 再除以 s，把它变回可直接作用于原始输入 X 的有效权重。
        effective = fake_quantize(weight * scale[None, :], config) / scale[None, :]
        difference = effective - weight
        squared_error = torch.zeros((), device=weight.device)
        # delta 为 [N, 当前输出块宽]。按输出通道累计平方和，
        # 避免一次分配 [N, O]；在浮点舍入允许的范围内，目标与整块计算相同。
        for start in range(0, weight.shape[0], config.output_chunk_size):
            delta = inputs @ difference[start : start + config.output_chunk_size].t()
            squared_error += delta.square().sum()
        # 除以 N*O 得到该层输出元素的 MSE，item() 转成 Python 标量用于比较。
        loss = (squared_error / (inputs.shape[0] * weight.shape[0])).item()
        if loss < best_loss:
            # clone 保存当前最优候选的独立副本，后续迭代只继续搜索。
            best_loss, best_scale = loss, scale.clone()
    # 若所有候选的损失均无效，best_loss 保持 inf，拒绝返回默认缩放值。
    if not torch.isfinite(best_scale).all() or best_loss == float("inf"):
        raise RuntimeError("AWQ scale search produced no finite candidate")
    return best_scale


#裁剪阈值
@torch.no_grad()
def _search_clip(weight: Tensor, inputs: Tensor, config: AWQConfig) -> Tensor:
    """对缩放后的权重逐输出行、逐输入组搜索对称裁剪阈值。

    weight 为 W*s，形状 [O, I]；inputs 为 X/s，形状 [N, I]。
    每个输出行的每个输入组分别选择阈值，返回相同形状的浮点裁剪权重，
    尚未做最终整数编码。裁剪区间 [-bound, bound] 是对称的，
    组内量化是否对称则由 config.symmetric 单独控制。

    评分为 mean((X_group @ (Q(W_clipped) - W_original).T)**2)，
    基准始终是本次传入的未裁剪权重，包含裁剪误差和量化误差两部分。
    各组独立优化，未计入不同输入组的输出误差之间的交叉项，属于局部搜索。
    """
    # 在副本上写回最终阈值，保证评分时的 original 不会被已选结果覆盖。
    clipped = weight.clone()
    size = effective_group_size(weight.shape[1], config.group_size)
    # 沿输入列分组；group_size=-1 表示整行一组，最后一组允许不足 size 列。
    for start in range(0, weight.shape[1], size):
        stop = min(start + size, weight.shape[1])
        group_inputs = inputs[:, start:stop]
        # 输出行分块只控制临时内存，各行仍独立选择自己的最佳阈值。
        for row in range(0, weight.shape[0], config.output_chunk_size):
            end = min(row + config.output_chunk_size, weight.shape[0])
            # original：[当前输出块宽, 当前组宽]；bound：[当前输出块宽, 1]。
            original = weight[row:end, start:stop]
            bound = original.abs().amax(dim=1, keepdim=True)
            best_bound = bound.clone()
            best_error = torch.full((end - row,), float("inf"), device=weight.device)
            for step in range(config.clip_steps):
                # step=0 不裁剪；多个候选时，最后一个把幅度缩小
                # max_clip_shrink 的比例。0.5 表示最多降至原幅度的 50%。
                shrink = config.max_clip_shrink * step / max(1, config.clip_steps - 1)
                candidate_bound = bound * (1 - shrink)
                candidate = original.clamp(-candidate_bound, candidate_bound)
                # 当前切片恰好是一组（包括尾组），fake_quantize 会按裁剪后的
                # 范围重新求每行量化参数，因此裁剪也会改变可用的量化步长。
                quantized = fake_quantize(candidate, config)
                delta = group_inputs @ (quantized - original).t()
                # delta：[N, 当前输出块宽]；只在样本轴取均值，保留每行的评分。
                error = delta.square().mean(dim=0)
                improved = error < best_error
                # 同一组内，不同输出行可以选择各自的最佳裁剪阈值。
                best_error = torch.where(improved, error, best_error)
                best_bound = torch.where(improved[:, None], candidate_bound, best_bound)
            clipped[row:end, start:stop] = original.clamp(-best_bound, best_bound)
    return clipped


@torch.no_grad()
def quantize_linear_awq(
    layer: nn.Linear, stats: ActivationStats, config: Optional[AWQConfig] = None,
) -> PackedLinear:
    """用 AWQ 将单个浮点 Linear 转成保存低位权重的 PackedLinear。

    stats 必须由该层输入收集，包含 mean_abs 和非空的 samples；不需要 Hessian。
    config=None 使用 AWQConfig 默认配置。返回新层，保留原层尺寸、偏置和设备，
    原层权重不被修改。搜索以 float32 进行，不建立自动求导图。

    返回层保存的是缩放域中的整数权重、量化参数和通道缩放 s。
    PackedLinear 反量化时将权重除以 s，使调用方仍可传入原始输入 X。
    """
    config = config or AWQConfig()
    # detach 脱离训练图，float 将搜索统一到 float32；后面的变换均生成新张量。
    weight = layer.weight.detach().float()
    if not torch.isfinite(weight).all():
        raise ValueError("Linear weights contain NaN or infinity")
    if not stats.count:
        raise ValueError("AWQ requires calibration activations")
    # 样本池在校准模块中保存在 CPU，此处移到权重设备上执行矩阵计算。
    inputs = stats.samples.to(weight.device)
    scale = _search_scale(weight, inputs, stats.mean_abs.to(weight.device), config)
    transformed = weight * scale[None, :]
    if config.clip:
        # 必须在相同的变换域中评分：W 乘 s，X 就要除以 s。
        # 此时每个候选的输出误差才对应原层的输出误差。
        transformed = _search_clip(transformed, inputs / scale[None, :], config)
    # codes：[O, I]，每个低位编码暂用一个 uint8 保存，尚未位打包。
    # scales / zeros：[O, ceil(I/实际组宽)]，分别为量化步长和整数零点。
    # 注意 scales 是组级量化步长，scale 是 [I] 的 AWQ 通道缩放，含义不同。
    codes, scales, zeros = quantize_groups(transformed, config)
    # from_quantized 负责真正的位打包并复制偏置；当前 PackedLinear.forward
    # 先恢复浮点权重再做 F.linear，压缩存储本身不代表有专用低位推理加速。
    return PackedLinear.from_quantized(
        layer, codes, scales, zeros, bits=config.bits,
        group_size=config.group_size, method="awq", input_scale=scale,
    )
