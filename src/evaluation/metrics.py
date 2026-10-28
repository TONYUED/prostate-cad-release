#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语义分割评价指标：混淆矩阵、逐类 IoU / Dice、像素准确率。

像素级指标是本项目**独立于 Gleason 推导规则**的主评价指标。
"""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 5
CLASS_NAMES = {0: "background", 1: "GG3", 2: "GG4", 3: "GG5", 4: "NC"}


def confusion_matrix(pred: np.ndarray, target: np.ndarray, n_classes: int = NUM_CLASSES) -> np.ndarray:
    """混淆矩阵 cm[i, j]：真实类别 i 被预测为 j 的像素数。"""
    mask = (target >= 0) & (target < n_classes)
    hist = np.bincount(
        target[mask].astype(np.int64) * n_classes + pred[mask].astype(np.int64),
        minlength=n_classes ** 2,
    )
    return hist.reshape(n_classes, n_classes)


def iou_from_cm(cm: np.ndarray) -> np.ndarray:
    """逐类 IoU，未出现的类返回 NaN。"""
    inter = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1e-12), np.nan)
    return iou.astype(np.float64)


def dice_from_cm(cm: np.ndarray) -> np.ndarray:
    """逐类 Dice，未出现的类返回 NaN。"""
    inter = np.diag(cm)
    denom = cm.sum(axis=1) + cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1e-12), np.nan)
    return dice.astype(np.float64)


def pixel_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total > 0 else float("nan")


def summarize(cm: np.ndarray, class_names: dict | None = None) -> dict:
    """把混淆矩阵汇总为可读指标字典。"""
    class_names = class_names or CLASS_NAMES
    iou = iou_from_cm(cm)
    dice = dice_from_cm(cm)

    valid_iou = iou[~np.isnan(iou)]
    valid_dice = dice[~np.isnan(dice)]

    def _fmt(arr):
        return {
            class_names.get(i, str(i)): (None if np.isnan(v) else round(float(v), 6))
            for i, v in enumerate(arr)
        }

    # 前景类（不含背景）平均
    fg = [i for i in class_names if i != 0]
    fg_iou = [iou[i] for i in fg if not np.isnan(iou[i])]
    fg_dice = [dice[i] for i in fg if not np.isnan(dice[i])]

    return {
        "per_class_iou": _fmt(iou),
        "per_class_dice": _fmt(dice),
        "mIoU": round(float(valid_iou.mean()), 6) if valid_iou.size else None,
        "mDice": round(float(valid_dice.mean()), 6) if valid_dice.size else None,
        "foreground_mIoU": round(float(np.mean(fg_iou)), 6) if fg_iou else None,
        "foreground_mDice": round(float(np.mean(fg_dice)), 6) if fg_dice else None,
        "pixel_accuracy": round(pixel_accuracy(cm), 6),
        "pixel_counts": {class_names.get(i, str(i)): int(cm[i].sum()) for i in range(cm.shape[0])},
    }
