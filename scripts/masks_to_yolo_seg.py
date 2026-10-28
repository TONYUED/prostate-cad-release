#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 SICAPv2 语义掩码转换为 YOLOv8-seg **多类多边形**标注。

与旧脚本 mask_to_yolo.py 的区别
-------------------------------
旧脚本做了两步有损处理，导致分级信息全部丢失：

    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)   # 所有非 0 值混为一类
    x, y, w, h = cv2.boundingRect(cnt)                            # 轮廓退化为外接矩形

结果：570 条标注全部是类别 0、每行 5 列（bbox），而 dataset.yaml 声明的却是
3 类 + task: segment。

本脚本：
  * 按掩码像素值区分类别，保留 Gleason 分级；
  * 输出真实轮廓多边形（YOLO-seg 需要的 `cls x1 y1 x2 y2 ...` 格式）；
  * 提供面积过滤与多边形抽稀，避免标签文件爆炸。

类别映射
--------
    掩码值 3 (GG3) -> 类别 0
    掩码值 4 (GG4) -> 类别 1
    掩码值 5 (GG5) -> 类别 2
    掩码值 6 (NC)  -> 类别 3
    掩码值 1       -> 默认忽略（语义待确认，--include-unknown 可加入为类别 4）

用法
----
    python masks_to_yolo_seg.py ^
        --mask-dir  <掩码目录> ^
        --image-dir <HE 图像目录> ^
        --out       <输出数据集根目录>

    # 沿用已有的 train/val 划分
    python masks_to_yolo_seg.py ... --split-from <旧 data 根目录>

    # 只处理前 10 张做自检
    python masks_to_yolo_seg.py ... --limit 10
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np

VAL_GG3, VAL_GG4, VAL_GG5, VAL_NC, VAL_UNKNOWN = 3, 4, 5, 6, 1

# 掩码值 -> (YOLO 类别 id, 类别名)，--include-unknown 时追加值 1
BASE_CLASSES = [(VAL_GG3, "GG3"), (VAL_GG4, "GG4"), (VAL_GG5, "GG5"), (VAL_NC, "NC")]
UNKNOWN_CLASS = (VAL_UNKNOWN, "unconfirmed")

MASK_SUFFIX_RE = re.compile(r"_classimg_(nonconvex|convex)$", re.IGNORECASE)


def imread_unicode(path: str, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flag)
    if img is None:
        raise IOError(f"无法读取: {path}")
    return img


def case_id_from_name(path: str) -> str:
    return MASK_SUFFIX_RE.sub("", os.path.splitext(os.path.basename(path))[0])


def simplify_contour(cnt: np.ndarray, max_points: int) -> np.ndarray | None:
    """Douglas-Peucker 抽稀，保证点数不超过 max_points 且至少 3 点。"""
    peri = cv2.arcLength(cnt, True)
    eps = max(1.0, 0.001 * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    # 点数仍过多则逐步加大 epsilon
    factor = 2.0
    while len(approx) > max_points and factor <= 64.0:
        approx = cv2.approxPolyDP(cnt, max(1.0, 0.001 * peri * factor), True)
        factor *= 2.0
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)


def mask_to_polygons(mask: np.ndarray, value: int, min_area: int, max_points: int):
    """提取某一类别的所有轮廓多边形（归一化坐标）。"""
    binary = (mask == value).astype(np.uint8)
    if binary.sum() == 0:
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape[:2]
    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        pts = simplify_contour(cnt, max_points)
        if pts is None:
            continue
        norm = pts.astype(np.float64)
        norm[:, 0] /= w
        norm[:, 1] /= h
        # 数值裁剪，防止越界
        norm = np.clip(norm, 0.0, 1.0)
        polys.append(norm)
    return polys


def write_label(path: str, items: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for cls, poly in items:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
        lines.append(f"{cls} {coords}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_existing_split(data_root: str):
    """从旧数据集目录读取 train/val 划分（沿用可保证与既有实验可比）。"""
    split = {}
    for subset in ("train", "val"):
        d = os.path.join(data_root, subset, "labels")
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.txt")):
            split[case_id_from_name(p)] = subset
    return split


def main() -> int:
    p = argparse.ArgumentParser(
        description="SICAPv2 掩码 -> YOLOv8-seg 多类多边形标注",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--image-dir", required=True, help="对应 HE 图像目录（slideXXX_coreXXX.jpg）")
    p.add_argument("--out", required=True, help="输出数据集根目录")
    p.add_argument("--split-from", default=None,
                   help="旧 data 根目录，沿用其 train/val 划分")
    p.add_argument("--val-ratio", type=float, default=0.3, help="无 --split-from 时的验证集比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-area", type=int, default=200, help="轮廓最小面积（像素）")
    p.add_argument("--max-points", type=int, default=200, help="单个多边形最大顶点数")
    p.add_argument("--include-unknown", action="store_true", help="把掩码值 1 作为类别 4")
    p.add_argument("--no-copy-images", action="store_true", help="不复制图像，仅生成标签")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 张")
    args = p.parse_args()

    classes = list(BASE_CLASSES)
    if args.include_unknown:
        classes.append(UNKNOWN_CLASS)
    value_to_cls = {v: i for i, (v, _) in enumerate(classes)}
    names = {i: n for i, (_, n) in enumerate(classes)}

    masks = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    if not masks:
        raise FileNotFoundError(f"掩码目录无 .png: {args.mask_dir}")

    # 划分
    if args.split_from:
        split_map = read_existing_split(args.split_from)
        print(f"[划分] 沿用 {args.split_from}: "
              f"train {sum(1 for v in split_map.values() if v == 'train')} / "
              f"val {sum(1 for v in split_map.values() if v == 'val')}")
    else:
        split_map = {}
        cases = [case_id_from_name(m) for m in masks]
        random.Random(args.seed).shuffle(cases)
        n_val = int(len(cases) * args.val_ratio)
        for c in cases[:n_val]:
            split_map[c] = "val"
        for c in cases[n_val:]:
            split_map[c] = "train"
        print(f"[划分] 随机划分 (seed={args.seed}): val {n_val} / train {len(cases) - n_val}")

    if args.limit:
        masks = masks[: args.limit]

    print(f"[类别] {names}")
    print(f"[参数] min_area={args.min_area} max_points={args.max_points}")

    out = os.path.abspath(args.out)
    stats = {"train": 0, "val": 0, "instances": {n: 0 for _, n in classes}}
    warnings = []
    empty = []

    for i, mpath in enumerate(masks, 1):
        case_id = case_id_from_name(mpath)
        subset = split_map.get(case_id)
        if subset is None:
            warnings.append(f"{case_id}: 未在任何划分中，跳过")
            continue

        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            cand = os.path.join(args.image_dir, case_id + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            warnings.append(f"{case_id}: 找不到对应 HE 图像，跳过")
            continue

        mask = imread_unicode(mpath)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # 尺寸一致性校验：标签按掩码尺寸归一化，若与图像不符会整体错位
        if not args.no_copy_images:
            probe = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) if img_path.isascii() \
                else imread_unicode(img_path)
            if probe is not None and probe.shape[:2] != mask.shape[:2]:
                warnings.append(
                    f"{case_id}: 图像 {probe.shape[:2]} 与掩码 {mask.shape[:2]} 尺寸不一致"
                )

        items = []
        for value, _name in classes:
            polys = mask_to_polygons(mask, value, args.min_area, args.max_points)
            cls = value_to_cls[value]
            for poly in polys:
                items.append((cls, poly))
                stats["instances"][dict(classes)[value]] += 1

        if not items:
            empty.append(case_id)

        write_label(os.path.join(out, "labels", subset, case_id + ".txt"), items)

        if not args.no_copy_images:
            dst = os.path.join(out, "images", subset, os.path.basename(img_path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)

        stats[subset] += 1
        if i % 20 == 0 or i == len(masks):
            print(f"  处理 {i}/{len(masks)}")

    # dataset.yaml
    yaml_path = os.path.join(out, "dataset.yaml")
    lines = [
        "# YOLOv8-seg 数据集（由 scripts/masks_to_yolo_seg.py 生成）",
        f"# 生成时间: {datetime.now().isoformat(timespec='seconds')}",
        f"path: {out.replace(chr(92), '/')}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for i, n in names.items():
        lines.append(f"  {i}: {n}")
    lines += [
        f"nc: {len(names)}",
        "task: segment",
        "img_size: 640",
    ]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[完成]")
    print(f"  train {stats['train']} / val {stats['val']}")
    print(f"  实例数: {stats['instances']}")
    print(f"  dataset.yaml: {yaml_path}")
    if empty:
        print(f"  [注意] {len(empty)} 个 core 无任何达标轮廓: {empty[:8]}")
    if warnings:
        print(f"  [警告] {len(warnings)} 条:")
        for w in warnings[:10]:
            print(f"    - {w}")

    with open(os.path.join(out, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "classes": names,
                "stats": stats,
                "empty_cases": empty,
                "warnings": warnings,
                "args": {k: v for k, v in vars(args).items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
