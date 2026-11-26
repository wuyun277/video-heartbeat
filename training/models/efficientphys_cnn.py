"""EfficientPhys-C convolution network, adapted from rPPG-Toolbox.

State-dict layer names stay unchanged; see ../THIRD_PARTY.md.
"""
from __future__ import annotations

import torch
from torch import nn

class AttentionMask(nn.Module):
    def forward(self, x):
        return x / x.sum(dim=(2, 3), keepdim=True).clamp_min(1e-12) * x.shape[2] * x.shape[3] * 0.5

class TemporalShift(nn.Module):
    def __init__(self, segments):
        super().__init__()
        self.segments = segments

    def forward(self, x):
        nt, c, h, w = x.shape
        if nt % self.segments:
            raise ValueError("Temporal length must be divisible by frame_depth")
        x = x.reshape(nt // self.segments, self.segments, c, h, w)
        fold = c // 3
        shifted = torch.zeros_like(x)
        shifted[:, :-1, :fold] = x[:, 1:, :fold]
        shifted[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        shifted[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return shifted.reshape(nt, c, h, w)

class EfficientPhysCNN(nn.Module):
    def __init__(self, frame_depth=10, image_size=72):
        super().__init__()
        dense_sizes = {36: 3136, 72: 16384, 96: 30976}
        if image_size not in dense_sizes:
            raise ValueError("Unsupported image size")
        self.TSM_1 = TemporalShift(frame_depth)
        self.TSM_2 = TemporalShift(frame_depth)
        self.TSM_3 = TemporalShift(frame_depth)
        self.TSM_4 = TemporalShift(frame_depth)
        self.motion_conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.motion_conv2 = nn.Conv2d(32, 32, 3)
        self.motion_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.motion_conv4 = nn.Conv2d(64, 64, 3)
        self.apperance_att_conv1 = nn.Conv2d(32, 1, 1)
        self.apperance_att_conv2 = nn.Conv2d(64, 1, 1)
        self.attn_mask_1 = AttentionMask()
        self.attn_mask_2 = AttentionMask()
        self.avg_pooling_1 = nn.AvgPool2d(2)
        self.avg_pooling_2 = nn.AvgPool2d(2)
        self.avg_pooling_3 = nn.AvgPool2d(2)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.5)
        self.final_dense_1 = nn.Linear(dense_sizes[image_size], 128)
        self.final_dense_2 = nn.Linear(128, 1)
        self.batch_norm = nn.BatchNorm2d(3)

    def forward(self, inputs):
        # 网络内完成帧差分，输入需要比输出多一帧。
        inputs = self.batch_norm(torch.diff(inputs, dim=0))
        d1 = torch.tanh(self.motion_conv1(self.TSM_1(inputs)))
        d2 = torch.tanh(self.motion_conv2(self.TSM_2(d1)))
        g1 = d2 * self.attn_mask_1(torch.sigmoid(self.apperance_att_conv1(d2)))
        d4 = self.TSM_3(self.dropout_1(self.avg_pooling_1(g1)))
        d5 = torch.tanh(self.motion_conv3(d4))
        d6 = torch.tanh(self.motion_conv4(self.TSM_4(d5)))
        g2 = d6 * self.attn_mask_2(torch.sigmoid(self.apperance_att_conv2(d6)))
        d8 = self.dropout_3(self.avg_pooling_3(g2))
        return self.final_dense_2(self.dropout_4(torch.tanh(self.final_dense_1(d8.flatten(1)))))
