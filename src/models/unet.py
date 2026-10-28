#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量 U-Net，用于前列腺活检组织学图像的 Gleason 模式语义分割。

任务：对每个像素分类为 5 类
    0 background / 1 GG3 / 2 GG4 / 3 GG5 / 4 NC(良性腺体)

选择语义分割而非实例分割的理由见 docs/methodology.md：SICAPv2 的掩码是
区域级（region-level）标注，同一类别通常只有一个连通域，不存在"实例"概念。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """标准 U-Net，通道数可通过 base_channels 调整以适配显存。

    Args:
        n_classes: 输出类别数（含背景）
        base_channels: 第一层卷积通道数（32 适合 8GB 显存，64 更充裕）
        depth: 下采样次数
        dropout: 瓶颈层 dropout 概率
    """

    def __init__(self, n_classes: int = 5, base_channels: int = 32,
                 depth: int = 4, dropout: float = 0.2):
        super().__init__()
        ch = base_channels
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        in_ch = 3
        for i in range(depth):
            self.downs.append(DoubleConv(in_ch, ch))
            in_ch = ch
            ch = min(ch * 2, 512)

        self.bottleneck = DoubleConv(in_ch, ch, dropout=dropout)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
            )
            self.up_convs.append(DoubleConv(ch, ch // 2))
            ch = ch // 2

        self.head = nn.Conv2d(ch, n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # 处理奇数尺寸导致的 1 像素误差
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
