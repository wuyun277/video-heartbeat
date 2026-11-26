# 视频心率估计与生理信号实验

本仓库收录人脸视频远程光电容积脉搏波（rPPG）估计、EfficientPhys 训练、视频生理信号注入基线及训练后量化的代码。

本仓库只包含源码、配置和必要说明。视频与图片、UBFC 数据集、模型权重、训练缓存、运行结果、实验记录和其他数据均需自行准备。

## 代码结构

- app.py、dashboard.py、fusion.py：摄像头或视频心率估计与界面。
- efficientphys.py：EfficientPhys 结构及推理。
- training/：UBFC 数据处理、EfficientPhys-C/T 训练和评估；见 [训练说明](training/README.md)。
- physio_baselines/：生理信号注入与评测的开发基线；见 [实验说明](physio_baselines/README.md)。
- PTQ/：GPTQ/AWQ 训练后量化实现。
- tools/：可选的数据准备和视频工具。

## 安装与运行

使用 Python 环境安装项目依赖：

    python -m pip install -r requirements.txt

config.json 默认使用 EfficientPhys，并指向未包含的 models/PURE_EfficientPhys.pth。如果尚未准备权重，可在 config.json 的 method 中将 name 改为 pos，然后运行：

    python app.py --camera 0

也可配置自己的视频并通过 python app.py --video <视频路径> 运行。训练所需依赖与数据目录配置见 training/README.md，基线实验的数据、权重和运行方式见 physio_baselines/README.md。

可用以下命令查看各模块参数：

    python -m training --help
    python -m PTQ --help
    python -m physio_baselines --help

## 实验边界与来源

本项目为研究和演示代码，输出未经临床验证。传统方法 POS/CHROM 用于从视频提取信号；注入基线用于将目标波形调制到视频中。

EfficientPhys 等部分实现参考了 rPPG-Toolbox；来源及适用的第三方许可见 [training/THIRD_PARTY.md](training/THIRD_PARTY.md) 和 [training/LICENSE.rPPG-Toolbox](training/LICENSE.rPPG-Toolbox)。