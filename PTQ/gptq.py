"""GPTQ：用校准输入的二阶信息逐列量化，并补偿尚未量化的权重。

统一维度：N 为校准样本行数，I 为输入通道数，O 为输出通道数。
输入 X 为 [N, I]，权重 W 为 [O, I]，线性层输出为 X @ W.T + bias。
对任一输出行，将其权重误差视为列向量 delta，局部重建损失为：
    L = ||X @ delta||^2 / N，H = 2 * X.T @ X / N，L = delta.T @ H @ delta / 2。
这是固定校准输入下的线性层输出损失；H 对该二次目标是精确 Hessian，
但它只代理整网任务损失，不能理解为直接计算了任务损失的 Hessian。

普通舍入会把当前列的误差留在输出中。GPTQ 利用输入通道之间的相关性，
调整后续仍可修改的列，尽量抵消这部分输出误差，再继续量化下一列。
所有输出行共享同一个 [I, I] Hessian，因此可以并行处理 O 行权重。

流程为：准备 H -> 处理无激活列 -> 固定组参数 -> 可选的激活排序
-> 求逆 Hessian 的 Cholesky 因子 -> 分块逐列量化与补偿 -> 位打包。
仅量化权重；无需反向传播，偏置和推理激活保持浮点。

"""

from typing import Optional

import torch
from torch import Tensor, nn

from .calibration import ActivationStats
from .config import GPTQConfig
from .linear import PackedLinear
from .quantization import effective_group_size, encode, group_parameters



def _inverse_cholesky(hessian: Tensor, damping: float) -> Tensor:
    """为逐列补偿构造 [I, I] 上三角因子 R。

    输入 hessian 为校准得到的对称半正定矩阵；先使用 H_damped = H + jitter*I，
    再求 R，使 R.T @ R = inverse(H_damped)。返回的是逆矩阵的 Cholesky 因子，
    不是 H 的 Cholesky 因子，也不是完整逆矩阵，后续补偿公式必须与之对应。

    damping 是相对于 H 对角均值的阻尼比例。阻尼改善样本不足、通道相关等
    情况下的病态矩阵，同时也改变了被优化的二次目标。
    """
    identity = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    magnitude = hessian.diag().mean().clamp_min(torch.finfo(hessian.dtype).eps)
    # 两次下界保护使全零/极小统计量也能得到正阻尼，避免零阻尼导致奇异分解。
    jitter = (magnitude * damping).clamp_min(torch.finfo(hessian.dtype).eps)
    for _ in range(6):
        # 默认得到下三角 L，满足 L @ L.T = H_damped。
        # cholesky_ex 以 info=0 表示成功，便于在失败时增加阻尼重试。
        factor, info = torch.linalg.cholesky_ex(hessian + identity * jitter)
        if info.item() == 0:
            # 利用 L 求出 H_damped 的逆，再对逆矩阵做上三角 Cholesky 分解。
            inverse = torch.cholesky_inverse(factor)
            upper, inverse_info = torch.linalg.cholesky_ex(inverse, upper=True)
            if inverse_info.item() == 0 and torch.isfinite(upper).all():
                return upper
        jitter *= 10  # 数值分解失败时逐步增加阻尼，不直接使用不稳定的逆矩阵。
    raise RuntimeError("GPTQ Hessian factorization failed; increase damping or improve calibration data")


#分块量化并补偿剩余列
@torch.no_grad()
def quantize_linear_gptq(
    layer: nn.Linear, stats: ActivationStats, config: Optional[GPTQConfig] = None,
) -> PackedLinear:
    """将一个浮点 Linear 量化为 PackedLinear，不修改原层权重。

    stats 必须针对该层输入收集 Hessian 统计；无需使用 AWQ 的校准样本池。
    config=None 使用 GPTQConfig 默认配置。计算使用 float32 并关闭梯度记录，
    返回层保存整数编码、逐行逐组参数及原层偏置。

    group_size 定义量化参数共享范围，block_size 定义误差更新的计算块；
    两者用途不同，不要求相等，也不要求块边界与组边界重合。
    """
    config = config or GPTQConfig()
    # GPTQ 会原地更新待量化权重，所以即使原层已经是 float32 也必须 clone。
    weight = layer.weight.detach().float().clone()
    if not torch.isfinite(weight).all():
        raise ValueError("Linear weights contain NaN or infinity")
    # hessian() 从校准阶段累计的 Gram 矩阵生成 H=2*X.T@X/N。
    # 不需要保留完整 X，但 [I, I] 的二阶统计仍需占用平方级内存。
    hessian = stats.hessian().to(weight.device)
    if hessian.shape != (layer.in_features, layer.in_features):
        raise ValueError("Hessian shape does not match this layer")

    # H[j,j]=2*mean(X[:,j]^2)。对角为零说明该输入列在已统计样本中始终为零，
    # 对应权重在这些样本上不影响输出；不代表它对所有未来输入都不重要。
    # 将其权重置零，并把 H 的对应对角置为 1
    dead = hessian.diag() == 0
    indices = torch.arange(layer.in_features, device=weight.device)
    hessian[indices[dead], indices[dead]] = 1
    weight[:, dead] = 0
    # 静态分组：先在原始列顺序下确定 [O, 组数] 的步长/零点，后续不重估。
    # 后面的误差补偿会改变权重，但每列仍使用它原先所属组的固定量化网格。
    scales, zeros = group_parameters(weight, config)
    group_size = effective_group_size(layer.in_features, config.group_size)
    # act_order 按输入的平均平方幅度降序处理，是启发式排序。
    # order[j] 表示“处理顺序中的第 j 列”对应的原始输入列编号。
    order = hessian.diag().argsort(descending=True) if config.act_order else indices
    # 权重列和 H 的两个轴必须同步重排，否则相关性和权重列会错位。
    # contiguous 让重排后的权重以连续布局参与接下来的块操作。
    weight = weight[:, order].contiguous()
    hessian = hessian[order][:, order]
    upper = _inverse_cholesky(hessian, config.damping)
    # 尽管按重排顺序计算，codes 始终按原始输入列位置写入，方便直接替换原层。
    # 此处每个编码暂占一个 uint8，最终由 PackedLinear 做 2/4/8 位打包。
    codes = torch.empty_like(weight, dtype=torch.uint8)

    # 块内逐列补偿，块外积累误差后一次性做矩阵更新。
    for start in range(0, layer.in_features, config.block_size):
        stop = min(start + config.block_size, layer.in_features)
        # working / errors：[O, 当前块宽]，最后一块允许短于 block_size。
        # weight 中该块已经接收了之前所有块的误差补偿。
        # working 单独接收本块内部的逐列更新，errors 保存各列归一化残差。
        working = weight[:, start:stop].clone()
        errors = torch.zeros_like(working)
        for offset in range(stop - start):
            # offset 是块内下标，column 是排序后全局下标，original_column 是原始下标。
            column = start + offset
            original_column = int(order[column].item())
            group = original_column // group_size  # 按原始列取静态组参数。
            scale, zero = scales[:, group], zeros[:, group]
            # original 为 [O]：当前待量化列在已接收补偿后的值，并非原层的初始权重。
            original = working[:, offset].clone()
            # 仿射编码 q=clamp(round(w/scale+zero), 0, 2**bits-1)。
            # 反量化值用于计算实际误差；补偿必须针对可表示的浮点近似值进行。
            code = encode(original, scale, zero, config.bits)
            reconstructed = (code.float() - zero.float()) * scale
            codes[:, original_column] = code
            # 令 R=upper，e_j=(w_j-w_hat_j)/R[j,j]，更新后续列：
            #     W[:, j:] -= e_j[:, None] * R[j, j:][None, :]。
            # R 的三角结构编码了逐列消元后的二阶信息；不能把分母直接换成 H[j,j]。
            error = (original - reconstructed) / upper[column, column]
            errors[:, offset] = error
            # [O,1] 与 [1,剩余块宽] 的外积同时补偿所有输出行。
            # 切片包含当前列，更新后它在数值上等于 reconstructed；
            # 该列编码已固定，以后的列只更新各自右侧，因此不会再次改动它。
            working[:, offset:].sub_(error[:, None] * upper[column, column:stop][None, :])
        if stop < layer.in_features:
            # 块内的每个 e_j 都作用于块外列，但块外列要到后续块才会被量化。
            # 因此可以把这些外积相加合并为一次矩阵乘法：
            # [O,块宽] @ [块宽,I-stop] -> [O,I-stop]，减少逐列更新的读写开销。
            weight[:, stop:].sub_(errors @ upper[start:stop, stop:])

    # codes 已按原始列顺序保存，scales/zeros 也属于原始静态分组，无需再逆置换。
    # 此处不传 input_scale，GPTQ 的通道缩放默认为 1；偏置由工厂方法复制。
    # PackedLinear 当前使用反量化后的浮点 F.linear 推理，不包含专用低位计算内核。
    return PackedLinear.from_quantized(
        layer, codes, scales, zeros, bits=config.bits,
        group_size=config.group_size, method="gptq",
    )
