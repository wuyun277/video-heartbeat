> 此仓库仅提供代码与配置；数据、权重、运行结果及实验记录均需自行准备或生成。

# 方向一：轻量化生理信号注入基础实验

独立于现有摄像头演示和 EfficientPhys 训练流程。原始数据只读，默认读取 `../training/training_dataset`。当前用于开发可行性验证，不是 rePPG 或其他论文的完整复现。

## 已实现

- 扫描本地 UBFC：视频首帧/元数据、PPG/HR/时间戳、重复标签时间、标签时长与视频时长差异。
- B0：原视频；B1：三区域 Mask 内绿色通道调制；B2：相同 Mask 内固定 RGB 权重调制。
- 相同目标分别输入 B1/B2；目标包括 72 bpm 正弦波、subject11 的实测 PPG 片段。目标均去均值并归一化为单位标准差。
- GREEN、POS、CHROM、既有 PURE 预训练 EfficientPhys；保存逐窗口状态、ROI RGB、预测波形、原始 PPG、目标波形及视觉误差。
- 原始帧分辨率、帧顺序与 FPS 保留；示例视频使用 FFV1，并逐帧 CRC 校验无损往返。主实验评估解码后的 uint8 像素，不含有损编码。
- 每次 `inspect` / `run`（包括失败）及 `report` 自动更新两份记录文档；每次运行有独立 ID、配置、环境、代码快照和数据指纹。

## 运行

在项目根目录 `项目根目录` 执行：

```powershell
# 数据清单；不代表整段视频逐帧检查
& 'python' -m physio_baselines inspect

# 默认三位开发受试者，每段开头30秒，B0/B1/B2和四个读取器
& 'python' -m physio_baselines run

# 较小规模检查：20秒、单个受试者、传统读取器
& 'python' -m physio_baselines run --subjects subject8 --clip-seconds 20 --no-neural --no-example-videos

# 仅验证原视频读取器
& 'python' -m physio_baselines run --original-only

# 从某个已完成运行重新汇总；自动更新两份文档
& 'python' -m physio_baselines report --run-dir '.\physio_baselines\runs\具体运行ID'

# 基础科学计算、数据边界、无损编码的回归检查
& 'python' -m unittest physio_baselines.test_core -v
```

新实验修改本目录 `config.json` 或另存配置并传 `--config`，不要修改旧运行配置来伪装重新运行。依赖复用项目现有环境，无额外安装。

## 当前固定协议

1. **受试者**：subject8/9/10 为开发源视频；subject11 仅提供目标波形；subject14/15 保留。这里不是重新定义已有训练流程的训练/验证/测试划分。后续正式实验需要独立冻结协议；使用过的开发或 donor 数据应如实披露。
2. **时间轴**：视频采用帧序号/文件 FPS，标签时间归零后按时间戳插值；相同标签时间的值取平均，越界拒绝。未逐帧验证视频 PTS，脸部与手指 PPG 的真实生理延迟也未校准。记录标签与视频的总时长差异，不能将相关性直接当作同步精度。
3. **ROI**：MediaPipe 每帧追踪额头、左颊、右颊三块多边形，沿用现有 app 的索引。它们是几何皮肤候选 ROI，尚无语义皮肤分割、肤色自适应或遮挡模型。注入采用三块 ROI 的并集并做 sigma=3 像素的高斯柔边；各变体复用同一源视频 ROI，避免检测变化混入比较。
4. **注入**：`I'_RGB = round(clip(I_RGB * exp(a*s(t)*mask*w_RGB), 0, 255))`。B1 权重 `(0,1,0)`，B2 `(0.6,1,0.3)`；a 为 0.005/0.01。正号使正目标提高颜色值，仅为工程基线，不声称是真实组织光学模型。实拍源视频原有生理信号没有被擦除，实验检验的是叠加后的目标恢复。
5. **读取器**：POS/CHROM 按原论文投影公式独立实现，1.6 秒局部窗；POS 步长1帧、CHROM 半窗 Hann 重叠并按权重归一化。局部 CHROM 滤波及公共滤波均使用 0.7–3 Hz 三阶 Butterworth；该统一协议与各论文/Toolbox 默认实现不同，不能直接对照其已发表数字。EfficientPhys 复用本项目模型类与 PURE 权重，不调用带 GUI/质量门槛的 demo。固定首个有效人脸框放大1.5倍，RGB resize 72，整片段标量 mean/std，180帧推理块，末帧补齐后做差分预测，累计求和恢复波形。没有使用当前 UBFC 本地训练权重。
6. **评测窗口**：10秒非重叠窗，不足一窗的尾帧只保存不评测；每窗额外公共去趋势/带通，每端裁0.8秒后计算指标。读取器提供的波形仍保留其自身预处理。短窗频谱分辨率有限，FFT补零只插值频率网格，不增加真实频率分辨率。
7. **参考**：原视频同时对照其原始 PPG、各待注入目标；编辑视频对照指定目标。主要 HR 误差相对相同窗口 PPG 的频谱主峰；原视频另保存相对设备 HR 均值的误差，两者不可混称。
8. **波形与 SNR**：保存零延迟带符号 Pearson r、其绝对值、允许 ±0.5秒非循环平移后的最大绝对相关及最佳延迟。绝对值忽略读取器极性，平移结果只是诊断，不证明绝对相位正确。SNR 为评测带内目标基频和二次谐波 ±0.1Hz 功率/其余功率。未评估逐搏形态、HRV或临床合理性。
9. **有效性**：每个窗口都写一行；缺 ROI、非有限、过短或平坦信号记 invalid。没有以“相关性低/HR误差大”剔除窗口。coverage 是可计算比例，不是准确率或信号可信概率。按受试者先平均再总体平均，未有效的受试者计入数量但无误差均值，需同时查看覆盖率。
10. **视觉**：报告相对输入的全图/ROI像素 MAE、PSNR、编辑支持外误差和饱和比例。未变化图像的无限 PSNR 在 JSON 中记 null（同时 MAE=0）；高 PSNR 本身不证明没有可见闪烁。背景角落只在固定人脸框外且无编辑覆盖时采样，作为负对照，不代替独立生理真实性检验。

## 输出

`runs/<运行ID>/` 包含：

- `config.json`、`environment.json`、`source/`：配置、包版本、权重哈希和源代码快照。视频采用路径/大小/mtime 标识，未计算十余 GB 视频的完整内容哈希。
- `inventory.csv/json`：全体本地受试者清单。
- `window_metrics.csv`：逐窗口、逐读取器、逐 ROI 的有效/无效结果。
- `summary.csv/json/md`：按受试者平均的摘要；Markdown 只显示 pooled ROI 与神经读取器，完整结果以 CSV 为准。
- `visual_metrics.csv`：逐变体视觉误差；`clip_metadata.json` 记录实际帧数、尾帧与 ROI 覆盖。
- `<subject>/*signals.npz`：时间轴、原始 ROI RGB、源 PPG、目标、预测；`source_geometry.npz` 保存多边形与神经读取器固定框。
- `<subject>/roi_preview.png`：原图、ROI轮廓和注入 Mask；首位开发受试者另外保存 B0/B1/B2 的一组无损 AVI。
- `status.json`：running/completed/failed。失败的 traceback 在 `error.txt`，已完成的逐窗口结果保留。

## 两份持续维护文档

- `实验记录.md`：按运行追加操作、参数、结果、异常和解释，不覆盖历史。
- `当前进展与下一步.md`：自动区跟随最近运行更新；人工区记录结论、限制、决策和下一步。每次改变算法或协议后要更新人工区。

## 尚未完成

rePPG 权重推理、真正生成模型输出上的实验、有损压缩/缩放压力测试、区域自适应、SSIM/感知质量/闪烁评测、正式独立测试集与统计区间。现有 `input/ai_face_30s.mp4` 是 AI 图片动画，仅可称合成负对照。

## 方法来源

- POS：Wang et al., [Algorithmic Principles of Remote PPG](https://doi.org/10.1109/TBME.2016.2609282)。
- CHROM：de Haan & Jeanne, [Robust Pulse Rate From Chrominance-Based rPPG](https://doi.org/10.1109/TBME.2013.2266196)。
- 对照过 [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) 的方法与评测说明；本目录未直接复制其传统算法源码。EfficientPhys 模型类来自项目原有 `efficientphys.py`，原始来源仍为该工具箱；新目录没有重新分发预训练权重。
- 待加入的编辑对照：[rePPG 官方实现](https://github.com/Sangmyung-University-PrLab/rePPG-Rewriting-Photoplethysmography-Signal-to-Video)。
