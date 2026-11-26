# 实现来源与复现范围

## EfficientPhys-C

网络和原有训练方式参考 [ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)：

- neural_methods/model/EfficientPhys.py
- neural_methods/trainer/EfficientPhysTrainer.py
- dataset/data_loader/BaseLoader.py
- dataset/data_loader/UBFCrPPGLoader.py
- configs/train_configs/PURE_PURE_UBFC-rPPG_EFFICIENTPHYS.yaml

本地保留卷积层名称和 Toolbox 的批次展平、追加末帧方法。AttentionMask 增加分母保护。原有 C 采用 MSE、AdamW、OneCycleLR，训练设置保留；C 的 DiffNormalized 标签测试前需要积分。

上游许可为 Responsible Artificial Intelligence Source Code License，Version 1.1，Nov 20, 2022，完整文本见 LICENSE.rPPG-Toolbox。

## EfficientPhys-T

论文：Xin Liu, Brian Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff.
[EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf)，WACV 2023。
[arXiv 版本](https://arxiv.org/abs/2110.04447)。

本地 models/efficientphys_transformer.py 根据论文第 3.2 节重建，不是从官方 T 模型文件复制，也没有作者 T 预训练权重。实现帧差分与 BatchNorm，在每个 2D Swin Block 前加入通道三等分时间移位，支持 T1 [2,2,6,2] 和 T2 [2,1]。Swin 原理参考 [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer) 和其论文；本地仅使用 PyTorch，自行实现窗口注意力、相对位置偏置、空间移位掩码与 patch merging。

MSE、AdamW、1e-4 学习率和 10 epoch 对应 EfficientPhys 最终论文报告。论文未明确给出全部实现参数；patch size 4、embedding 96、窗口 7、注意力头数、MLP ratio 4、空间平均与线性输出头、weight decay 0.01、恒定学习率、180 帧片段、batch size 1、PPG 标准化及边界补齐是本地实现选择。不能声称与作者源码完全一致或已经复现论文结果。

T 默认将整帧缩放为 72×72 RGB，外部不进行像素标准化，网络内差分与 BatchNorm 处理；监督为标准化 PPG 波形。C/T 的标签与输入处理不同，缓存按配置隔离。

## 数据与本地流程

当前只支持本地 UBFC-rPPG Dataset-2 目录格式，使用下载的受试者子集，不是原论文 AFRL＋合成数据或 PURE→UBFC 实验。

UBFC-rPPG 来源：[官方页面](https://sites.google.com/view/ybenezeth/ubfcrppg)。
数据引用：Serge Bobbia, Richard Macwan, Yannick Benezeth, Alamin Mansouri, Julien Dubois.
Unsupervised skin tissue segmentation for remote photoplethysmography, Pattern Recognition Letters。

本地增加独立模型配置、显式受试者划分、uint8 缓存、预处理签名、检查点恢复和模型分目录保存。T 默认原始整帧输入；C 首帧人脸检测失败时要求手工框，不使用背景代替人脸。测试采用配置指定的非重叠窗口，比较预测心率与设备 HR 行的平均值，因此数值不等于论文或 Toolbox 官方 benchmark。

模块整理：卷积版位于 models/efficientphys_cnn.py；模型工厂与批次适配位于 models/factory.py；数据与执行流程分别位于 data/ 和 engine/。只调整组织方式，模型参数键、预处理签名、损失及训练协议保持不变。
