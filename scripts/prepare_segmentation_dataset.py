#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 5120x5120 的 SICAPv2 core 图像与语义掩码切分为可训练的图像块（patch）。

为什么需要切分
--------------
原始 core 图像为 5120x5120（约 26M 像素/张，单张 RGB 解压后约 78 MB），
无法整体送入分割网络。本脚本按固定网格切分为 512x512（默认）小块，
并丢弃几乎全是背景的块，从而在显存可控的前提下保留绝大部分标注信息。

类别重映射（掩码原始值 -> 训练标签 id）
--------------------------------------
    0 (背景)      -> 0  background
    1 (语义待确认) -> 0  归入背景（默认忽略）
    3 (GG3)       -> 1
    4 (GG4)       -> 2
    5 (GG5)       -> 3
    6 (NC 良性)   -> 4

划分原则
--------
**按 core 划分**，而不是按 patch 随机划分：同一张 core 切出的所有 patch
必须落在同一子集，否则验证集会泄漏训练样本，指标虚高。

用法
----
    python prepare_segmentation_dataset.py ^
        --image-dir <HE 图像目录> ^
        --mask-dir  <掩码目录> ^
        --out       <输出目录> ^
        --split-from <旧 data 根目录>      # 沿用既有 train/val 划分

    python prepare_segmentation_dataset.py ... --limit 2    # 小样本自检
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from datetime import datetime

import cv2
import numpy as np

# 掩码原始值 -> 训练标签 id
VALUE_MAP = {0: 0, 1: 0, 3: 1, 4: 2, 5: 3, 6: 4}
CLASS_NAMES = {0: "background", 1: "GG3", 2: "GG4", 3: "GG5", 4: "NC"}
MASK_SUFFIX_RE = re.compile(r"_classimg_(nonconvex|convex)$", re.IGNORECASE)


def imread_unicode(path: str, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if img is None:
        raise IOError(f"无法读取: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)
    return True


def case_id_from_name(path: str) -> str:
    return MASK_SUFFIX_RE.sub("", os.path.splitext(os.path.basename(path))[0])


def remap_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape[:2], dtype=np.uint8)
    for src, dst in VALUE_MAP.items():
        if dst == 0:
            continue
        out[mask == src] = dst
    return out


def read_existing_split(data_root: str) -> dict:
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
        description="SICAPv2 core 图像切分为语义分割训练 patch",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split-from", default=None, help="旧 data 根目录，沿用其 train/val 划分")
    p.add_argument("--val-ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--stride", type=int, default=None, help="默认等于 patch-size（不重叠）")
    p.add_argument("--min-foreground", type=float, default=0.3,
                   help="patch 中非背景像素占比下限，低于则丢弃")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    stride = args.stride or args.patch_size
    ps = args.patch_size

    masks = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))
    if not masks:
        raise FileNotFoundError(f"掩码目录无 .png: {args.mask_dir}")

    if args.split_from:
        split_map = read_existing_split(args.split_from)
        print(f"[划分] 沿用 {args.split_from}: "
              f"train {sum(1 for v in split_map.values() if v == 'train')} / "
              f"val {sum(1 for v in split_map.values() if v == 'val')}")
    else:
        cases = [case_id_from_name(m) for m in masks]
        random.Random(args.seed).shuffle(cases)
        n_val = int(len(cases) * args.val_ratio)
        split_map = {c: "val" for c in cases[:n_val]}
        split_map.update({c: "train" for c in cases[n_val:]})
        print(f"[划分] 随机划分 (seed={args.seed}): val {n_val} / train {len(cases) - n_val}")

    if args.limit:
        masks = masks[: args.limit]

    out = os.path.abspath(args.out)
    for subset in ("train", "val"):
        os.makedirs(os.path.join(out, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(out, "masks", subset), exist_ok=True)

    index = {"train": [], "val": []}
    skipped_bg = 0
    warnings = []
    class_pixels = {i: 0 for i in CLASS_NAMES}

    for i, mpath in enumerate(masks, 1):
        case_id = case_id_from_name(mpath)
        subset = split_map.get(case_id)
        if subset is None:
            warnings.append(f"{case_id}: 无划分信息，跳过")
            continue

        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            cand = os.path.join(args.image_dir, case_id + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            warnings.append(f"{case_id}: 找不到 HE 图像，跳过")
            continue

        img = imread_unicode(img_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mpath)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if img.shape[:2] != mask.shape[:2]:
            warnings.append(
                f"{case_id}: 图像 {img.shape[:2]} 与掩码 {mask.shape[:2]} 尺寸不一致，"
                f"按掩码尺寸裁剪"
            )
            h, w = min(img.shape[0], mask.shape[0]), min(img.shape[1], mask.shape[1])
            img, mask = img[:h, :w], mask[:h, :w]

        label = remap_mask(mask)
        H, W = label.shape

        rows = list(range(0, H, stride))
        cols = list(range(0, W, stride))
        for r in rows:
            for c in cols:
                if r + ps > H or c + ps > W:
                    continue
                m_patch = label[r:r + ps, c:c + ps]
                fg_ratio = float((m_patch > 0).mean())
                if fg_ratio < args.min_foreground:
                    skipped_bg += 1
                    continue
                i_patch = img[r:r + ps, c:c + ps]

                stem = f"{case_id}__r{r}_c{c}"
                imwrite_unicode(
                    os.path.join(out, "images", subset, stem + ".jpg"),
                    i_patch,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
                imwrite_unicode(
                    os.path.join(out, "masks", subset, stem + ".png"), m_patch
                )
                vals, cnts = np.unique(m_patch, return_counts=True)
                for v, n in zip(vals, cnts):
                    class_pixels[int(v)] += int(n)
                index[subset].append(
                    {"id": stem, "core": case_id, "fg_ratio": round(fg_ratio, 4)}
                )

        if i % 10 == 0 or i == len(masks):
            print(f"  处理 {i}/{len(masks)}  (累计 patch: "
                  f"train {len(index['train'])} / val {len(index['val'])})")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/prepare_segmentation_dataset.py",
        "classes": CLASS_NAMES,
        "patch_size": ps,
        "stride": stride,
        "min_foreground": args.min_foreground,
        "n_patches": {k: len(v) for k, v in index.items()},
        "n_cores": {k: len({p['core'] for p in v}) for k, v in index.items()},
        "class_pixel_counts": {CLASS_NAMES[k]: v for k, v in class_pixels.items()},
        "class_pixel_counts_by_id": {str(k): int(v) for k, v in class_pixels.items()},
        "skipped_low_foreground": skipped_bg,
        "warnings": warnings,
        "index": index,
    }
    with open(os.path.join(out, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"  patch 数: train {payload['n_patches']['train']} / val {payload['n_patches']['val']}")
    print(f"  core 数 : train {payload['n_cores']['train']} / val {payload['n_cores']['val']}")
    print(f"  丢弃低前景 patch: {skipped_bg}")
    print(f"  类别像素: {payload['class_pixel_counts']}")
    print(f"  索引: {os.path.join(out, 'dataset.json')}")
    if warnings:
        print(f"  [警告] {len(warnings)} 条，例如: {warnings[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
