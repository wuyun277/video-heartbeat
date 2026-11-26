"""命令行量化与检查点转换入口，在项目根目录用 python -m PTQ 调用。

以下仅为用法示例，导入模块时不会执行::

    python -m PTQ efficientphys --model efficientphys_t --method gptq \
        --checkpoint training/runs/efficientphys_t/best.pt \
        --output PTQ/output/efficientphys_t_gptq.pt

    python -m PTQ efficientphys --model efficientphys_t --method awq \
        --checkpoint training/runs/efficientphys_t/best.pt \
        --output PTQ/output/efficientphys_t_awq.pt

    python -m PTQ model --factory my_model:build_model \
        --checkpoint weights.pt --calibration calibration.pt --method awq \
        --output PTQ/output/model_awq.pt

通用模式的 calibration.pt 保存输入批次列表：Tensor 为单个位置参数，
元组/列表为多个位置参数，字典为关键字参数。单独保存的 Tensor 或字典
整体作为一个批次，数据标签应预先移除。
--factory 指定本地模型工厂函数，--factory-kwargs 提供 JSON 格式的关键字参数。
"""

import argparse
from collections.abc import Mapping
from importlib import import_module
import json
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .checkpoint import save_quantized
from .config import AWQConfig, CalibrationConfig, GPTQConfig, PTQConfig
from .efficientphys import quantize_efficientphys
from .pipeline import QuantizationResult, quantize_model


def _add_common(parser: argparse.ArgumentParser) -> None:
    """两个命令共用同一套量化和校准选项。"""
    parser.add_argument("--method", choices=("gptq", "awq"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tokens-per-batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", nargs="+", default=["*"])
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--block-size", type=int, default=128, help="GPTQ 分块大小")
    parser.add_argument("--damping", type=float, default=0.01, help="GPTQ 的 Hessian 相对阻尼")
    parser.add_argument("--act-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scale-steps", type=int, default=20, help="AWQ 缩放候选数")
    parser.add_argument("--clip-steps", type=int, default=10, help="AWQ 裁剪候选数")
    parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)


def _configs(args: argparse.Namespace) -> tuple[PTQConfig, CalibrationConfig]:
    """共同参数只收集一次，算法特有参数分别交给对应配置类。"""
    common = {"bits": args.bits, "group_size": args.group_size, "symmetric": args.symmetric}
    if args.method == "gptq":
        quantization = GPTQConfig(**common, block_size=args.block_size, damping=args.damping, act_order=args.act_order)
    else:
        quantization = AWQConfig(**common, scale_steps=args.scale_steps, clip_steps=args.clip_steps, clip=args.clip)
    calibration = CalibrationConfig(args.max_batches, args.max_tokens, args.tokens_per_batch, args.seed)
    return quantization, calibration


def _load_float_state(model: nn.Module, path: Path, key: Optional[str]) -> None:
    """兼容原始 state_dict 及包含 model/state_dict 字段的训练检查点。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if key is not None:
        state = payload[key]
    elif isinstance(payload, Mapping) and "model" in payload:
        state = payload["model"]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    model.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)


def _generic(
    args: argparse.Namespace, config: PTQConfig, calibration: CalibrationConfig,
) -> QuantizationResult:
    """加载本地工厂模型和输入批次，调用统一量化入口并保存。"""
    module_name, separator, function_name = args.factory.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("--factory must have the form package.module:function")
    options = json.loads(args.factory_kwargs)
    if not isinstance(options, dict):
        raise ValueError("--factory-kwargs must be a JSON object")
    model = getattr(import_module(module_name), function_name)(**options)
    if not isinstance(model, nn.Module):
        raise TypeError("The factory must return a torch.nn.Module")
    _load_float_state(model, args.checkpoint, args.state_key)
    model.to(args.device).eval()
    data = torch.load(args.calibration, map_location="cpu", weights_only=True)
    if isinstance(data, (torch.Tensor, Mapping)):
        data = [data]
    if not isinstance(data, (tuple, list)):
        raise TypeError("Calibration file must contain a batch list, Tensor or input dict")
    result = quantize_model(
        model, data, config, calibration_config=calibration, inplace=True,
        include=tuple(args.include), exclude=tuple(args.exclude),
    )
    result.model.eval()
    save_quantized(result, args.output, overwrite=args.overwrite, metadata={
        "factory": args.factory, "factory_kwargs": options,
        "source_checkpoint": str(args.checkpoint.resolve()),
    })
    return result


def main(argv: Optional[list[str]] = None) -> None:
    """选择项目模式或通用模式；不会自动执行测试或精度评估。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    project = commands.add_parser("efficientphys", help="使用已有 EfficientPhys 配置和训练集缓存")
    _add_common(project)
    project.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "training" / "config.json")
    project.add_argument("--model", choices=("efficientphys_c", "efficientphys_t"))
    generic = commands.add_parser("model", help="使用本地模型工厂和已保存的输入批次")
    _add_common(generic)
    generic.add_argument("--factory", required=True)
    generic.add_argument("--factory-kwargs", default="{}")
    generic.add_argument("--state-key", help="显式指定检查点中保存模型权重的字段")
    generic.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args(argv)
    # 在加载大模型前检查目标路径，避免做完校准才发现输出会覆盖输入。
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("--output must differ from --checkpoint")
    if args.command == "model" and args.calibration.resolve() == args.output.resolve():
        parser.error("--output must differ from --calibration")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists. Select another path or explicitly pass --overwrite")
    config, calibration = _configs(args)
    if args.command == "efficientphys":
        result = quantize_efficientphys(
            args.config, args.checkpoint, args.output, config,
            model_name=args.model, device=args.device, calibration_config=calibration,
            include=tuple(args.include), exclude=tuple(args.exclude), overwrite=args.overwrite,
        )
    else:
        result = _generic(args, config, calibration)
    print(f"Saved {args.method.upper()} checkpoint to {args.output.resolve()}")
    print(f"Packed {len(result.layers)} Linear layers; skipped {len(result.skipped)} layers")
    for name, reason in result.skipped.items():
        print(f"  skipped {name or '<root>'}: {reason}")


if __name__ == "__main__":
    main()
