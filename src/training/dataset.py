#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义分割数据集与同步增强。

图像块由 scripts/prepare_segmentation_dataset.py 生成，索引见 dataset.json。
增强对图像与掩码**同步**施加，避免空间错位。
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 5


def imread_unicode(path: str, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if img is None:
        raise IOError(f"无法读取: {path}")
    return img


class PatchSegmentationDataset(Dataset):
    def __init__(self, root: str, subset: str = "train", augment: bool = False,
                 seed: int = 42, normalize: bool = True):
        self.root = root
        self.subset = subset
        self.augment = augment
        self.normalize = normalize
        self.rng = random.Random(seed)

        with open(os.path.join(root, "dataset.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.items = meta["index"][subset]
        self.classes = meta["classes"]

    def __len__(self):
        return len(self.items)

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        # 水平 / 垂直翻转
        if self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if self.rng.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])

        # 90 度倍数旋转
        k = self.rng.choice([0, 1, 2, 3])
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))

        # 颜色抖动（仅图像）：亮度 / 对比度
        if self.rng.random() < 0.5:
            alpha = self.rng.uniform(0.85, 1.15)   # 对比度
            beta = self.rng.uniform(-12, 12)       # 亮度
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 轻微 HED 风格色相扰动：对 BGR 通道做独立缩放，模拟染色差异
        if self.rng.random() < 0.3:
            scales = [self.rng.uniform(0.9, 1.1) for _ in range(3)]
            img = np.clip(
                img.astype(np.float32) * np.array(scales, dtype=np.float32), 0, 255
            ).astype(np.uint8)

        return img, mask

    def __getitem__(self, idx):
        it = self.items[idx]
        stem = it["id"]

        img_path = os.path.join(self.root, "images", self.subset, stem + ".jpg")
        mask_path = os.path.join(self.root, "masks", self.subset, stem + ".png")

        img = imread_unicode(img_path, cv2.IMREAD_COLOR)      # HWC, BGR
        mask = imread_unicode(mask_path)                      # HW, 类别 id

        if self.augment:
            img, mask = self._augment(img, mask)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std

        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        mask = torch.from_numpy(mask.astype(np.int64))
        return img, mask


def compute_class_weights(root: str, subset: str = "train",
                          strategy: str = "median") -> torch.Tensor:
    """依据类别像素频率计算损失权重，缓解 GG5/NC 等稀有类被忽略的问题。

    strategy:
        inverse  -> 1 / freq
        median   -> median(freq) / freq（对极端类更温和，推荐）
    """
    with open(os.path.join(root, "dataset.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    by_id = meta.get("class_pixel_counts_by_id")
    if by_id:
        counts = np.array([int(by_id.get(str(i), 0)) for i in range(NUM_CLASSES)], dtype=np.float64)
    else:
        # 兼容以类别名为键的旧索引
        names = meta.get("classes", {})
        counts = np.array(
            [int(meta.get("class_pixel_counts", {}).get(names.get(str(i), str(i)), 0))
             for i in range(NUM_CLASSES)],
            dtype=np.float64,
        )
    present = counts > 0
    if not present.any():
        return torch.ones(NUM_CLASSES, dtype=torch.float32)

    freq = np.zeros_like(counts)
    freq[present] = counts[present] / counts[present].sum()

    w = np.zeros_like(counts)
    if strategy == "inverse":
        w[present] = 1.0 / freq[present]
    else:  # median：对极端稀有类比 inverse 温和
        w[present] = np.median(freq[present]) / freq[present]

    # 空类权重置 0（数据集中不存在该类，不参与损失）；其余裁剪到合理区间
    w[present] = np.clip(w[present], 0.05, 20.0)
    w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)
