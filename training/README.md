> 此仓库仅提供代码与配置；数据、权重、运行结果及实验记录均需自行准备或生成。

# EfficientPhys-C / EfficientPhys-T 训练

训练模块独立于 Demo，使用当前 UBFC-rPPG 子集。C 参考 rPPG-Toolbox；T 根据 EfficientPhys 论文第 3.2 节重建，提供 T1 和 T2。当前数据不是原论文 AFRL＋合成数据或 PURE→UBFC 实验，不宣称复现论文精度。

## 模块结构

```text
training/
├── __main__.py              # 推荐命令入口：python -m training
├── train.py                 # 兼容原有 python -m training 指令
├── cli.py                   # 命令行解析和任务分派
├── config.json              # 共享设置与 C/T 独立训练配置
├── settings.py              # 配置解析、校验和缓存签名
├── models/
│   ├── efficientphys_cnn.py         # 卷积版网络
│   ├── efficientphys_transformer.py # Transformer 版网络
│   └── factory.py                   # 模型创建及批次输入适配
├── data/
│   ├── preprocessing.py     # 视频、PPG 标签与预处理缓存
│   ├── ubfc_dataset.py       # 缓存片段 Dataset
│   └── loaders.py            # DataLoader 与 worker 随机种子
├── engine/
│   ├── trainer.py           # 训练、验证及逐轮保存
│   ├── evaluator.py         # 测试集评估与结果输出
│   ├── heart_rate.py        # 波形转心率的后处理
│   ├── exporter.py          # 纯权重与元数据导出
│   ├── checkpoints.py       # 权重加载、原子保存和恢复约束
│   ├── runtime.py           # 设备选择及随机状态
│   └── history.py           # 损失 CSV 与曲线
├── tests/
│   └── test_models.py       # 合成数据检查，不执行训练
├── check_models.py          # 兼容原有检查入口
├── training_dataset/        # 原始数据
├── cache/                   # 预处理缓存
└── runs/                    # 已有及新训练结果
```

模型文件只定义网络；数据模块不依赖模型；engine 通过统一模型工厂调用 C/T。配置仍保留在原位置，所有相对路径、缓存签名和检查点参数名不因目录调整而改变。原有 `python -m training ...` 和 `python training/train.py ...` 指令继续可用。

## 1. 两个模型的配置

`training/config.json` 的 `active_model` 决定默认模型，也可通过 `--model` 覆盖：

- `efficientphys_c`：保留已有卷积版训练参数、预训练权重兼容性和缓存。
- `efficientphys_t`：Transformer 版，默认 T1。

顶层 `paths`、`subjects`、`preprocess`、`test` 是共享设置。`models` 内每个模型有独立的 `model`、`train`，也可覆盖上述共享字段；不覆盖另一个模型的设置。不同模型可分别设置数据划分、缓存目录、输入处理、epoch、batch size、学习率和测试参数。

| 项目 | C 默认 | T 默认 |
|---|---|---|
| 输入处理 | 首帧 Haar 固定人脸框，RGB，整段像素标准化 | 整帧缩放，原始 RGB 数值，不进行外部像素标准化 |
| 分辨率 | 72×72 | 72×72 |
| 标签 | PPG 差分除以差分标准差，末尾补零 | 对齐 PPG 波形减均值、除以标准差 |
| 网络输出 | PPG 差分预测 | PPG 波形预测 |
| 损失、优化器 | MSE、AdamW | MSE、AdamW |
| 学习率 | OneCycleLR，最大 0.009 | 恒定 0.0001 |
| weight decay | 0 | 0.01（本地实现选择） |
| epoch / batch size | 5 / 4 | 10 / 1 |
| 片段长度 | 180 帧 | 180 帧 |
| 测试心率范围 | 42–180 BPM | 45–150 BPM |

标签标准化方式、片段长度、batch size、weight decay 数值和恒定学习率计划均为本地实现设置，不能说这些细节全部由原论文指定。T 使用 MSE、AdamW、1e-4 学习率及 10 epoch 与 WACV 2023 论文报告一致；C 保持之前 Toolbox 训练设置，没有改成原论文的负 Pearson 损失。

旧单模型 JSON 和已有 C 检查点仍可使用。C/T 预处理不同，因此缓存签名不同；T 首次运行需要单独 prepare，不能直接使用 C 的差分标签缓存。

## 2. Transformer 网络

`models/efficientphys_transformer.py` 使用现有 PyTorch，不增加 timm 或 torchvision 依赖：

```text
相邻帧差分 → BatchNorm → Patch Partition / Linear Embedding
→ TSM + Swin Block → Patch Merging → 后续阶段
→ LayerNorm → 空间平均 → 单输出线性层 → 逐帧 PPG 波形
```

每个 Swin Block 前沿时间轴移动通道的前三分之一和第二个三分之一；不同时间组不交换帧。窗口注意力含相对位置偏置，交替使用普通窗口和空间移位窗口。对 72px 输入产生的奇数 token 网格进行补齐，注意力屏蔽填充键，输出裁去空间填充。

| 变体 | 各阶段 Block 数 | 默认各阶段注意力头数 |
|---|---|---|
| T1 | [2, 2, 6, 2] | [3, 6, 12, 24] |
| T2 | [2, 1] | [3, 6] |

切换 T2 时，同时将 `models.efficientphys_t.model.variant` 改为 `"t2"`，`num_heads` 改为 `[3,6]`。其他默认设置为 patch size 4、初始 embedding 96、window size 7、MLP ratio 4、dropout 0。这些采用标准 Swin 的结构选择，论文未明确给出全部参数。空间平均与线性输出层也是本地重建选择，不能保证与作者未提供的训练代码完全相同。

默认启用 `gradient_checkpointing` 以减少训练激活占用，仍应按实际 GPU 显存调整 batch size。C 与 T 的参数结构不同，C 的 `PURE_EfficientPhys.pth` 不能加载到 T。

## 3. 预处理与训练

每个 `training/training_dataset/subjectN/` 下放置 `vid.avi` 和 `ground_truth.txt`。后者三行依次为 PPG、设备心率和时间戳；训练损失使用 PPG，不使用设备心率作为网络回归标签。

在项目根目录执行：

```powershell
# C：已有缓存有效时跳过
& 'python' -m training prepare --model efficientphys_c
& 'python' -m training train --model efficientphys_c

# T：先生成独立缓存，再训练
& 'python' -m training prepare --model efficientphys_t
& 'python' -m training train --model efficientphys_t
```

按受试者划分训练/验证/测试数据。每段视频切成互不重叠的完整片段，尾部不足片段长度的帧丢弃。训练打乱片段，执行前向、MSE、反向和 AdamW 更新；验证关闭梯度，按最低验证 MSE 保存最佳权重。C 按批次更新 OneCycleLR；T 默认无调度器。T 每个片段独立前向，避免批次中不同片段之间的帧差分；C 保留 Toolbox 展平批次并追加末帧的处理方式。

新运行分别保存到：

```text
training/runs/efficientphys_c/<运行目录>/
training/runs/efficientphys_t/<运行目录>/
```

保存配置与环境、日志、history.csv、loss.png、每轮检查点、last.pt、best.pt 和 best_weights.pth。模型种类、架构和输入处理记录在配置及权重元数据中。best.pt 是完整训练状态，best_weights.pth 是同一最佳模型的纯网络权重。

## 4. 初始化与恢复

```powershell
# C 使用现有 PURE 权重开始新训练
& 'python' -m training train --model efficientphys_c --init-weights .\models\PURE_EfficientPhys.pth

# T 从同结构 T 权重开始新训练或微调
& 'python' -m training train --model efficientphys_t --init-weights '.\training\runs\efficientphys_t\实际运行目录\best_weights.pth'

# T 恢复上次完整 epoch 的训练状态
& 'python' -m training train --model efficientphys_t --resume '.\training\runs\efficientphys_t\实际运行目录\last.pt'
```

`--init-weights` 只载入网络参数，优化器重新初始化，全部网络参数参与更新；`--resume` 恢复优化器、调度器和随机状态，数据及训练计划必须一致。改变模型种类、T1/T2、数据或总 epoch 时不能原样 resume。所有 checkpoint 保存在完整 epoch 结束处。

## 5. 测试与导出

```powershell
& 'python' -m training test --model efficientphys_t --checkpoint '.\training\runs\efficientphys_t\实际运行目录\best.pt'
& 'python' -m training export --model efficientphys_t --checkpoint '.\training\runs\efficientphys_t\实际运行目录\best.pt' --output '.\training\exports\transformer_weights.pth'
```

C 的差分预测先积分再估计心率；T 的波形预测直接去趋势、滤波和频谱估计。测试保存波形、窗口心率 CSV、波形 MSE 与 BPM MAE/RMSE。仍采用本地非重叠窗口协议，不是原论文整段视频评估。Demo 目前只加载卷积版，导出的 Transformer 权重用于本训练模块，不能直接替换 Demo 的 C 权重。

## 6. 不启动训练的检查

```powershell
& 'python' -m training.check_models
```

仅检查配置兼容性、时间移位边界、奇数网格合并、C/T 前向和合成数据梯度、已存在的 C 纯权重兼容性。不会预处理视频、执行 epoch 或更新优化器。

来源与实现选择见 `THIRD_PARTY.md`。
