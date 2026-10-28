#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整张 core 图像的滑窗语义分割推理，并由分割结果推导 core 级 Gleason 评分。

推理链路（与金标准完全对称）
----------------------------
SICAPv2 真实掩码 --面积统计--> gleason_rules --> 金标准
模型分割输出    --面积统计--> gleason_rules --> 预测

两侧使用 src/core/gleason_rules.py 中的同一套规则。

用法
----
    # 单张
    python src/inference/segment_infer.py --weights outputs/unet/<ts>/best.pth ^
        --input <图像目录或单张图像> --out-dir outputs/predictions

    # 同时保存预测掩码（供像素级 Dice/IoU 评估）
    python src/inference/segment_infer.py --weights ... --input ... --save-masks
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.unet import UNet                                   # noqa: E402
from src.core import gleason_rules                                 # noqa: E402
from src.evaluation.metrics import NUM_CLASSES                     # noqa: E402


def imread_unicode(path: str, flag=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)
    if img is None:
        raise IOError(f"无法读取: {path}")
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)


def load_model(weights: str, base_channels: int, depth: int, device: torch.device) -> UNet:
    try:
        ckpt = torch.load(weights, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13 无该参数
        ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("model", ckpt)
    # 允许从 checkpoint 的 args 恢复结构
    if "args" in ckpt:
        base_channels = ckpt["args"].get("base_channels", base_channels)
        depth = ckpt["args"].get("depth", depth)
    model = UNet(n_classes=NUM_CLASSES, base_channels=base_channels, depth=depth)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(model: UNet, img: np.ndarray, patch_size: int, stride: int,
                 device: torch.device, batch_size: int = 8,
                 normalize: bool = True) -> np.ndarray:
    """对整图滑窗推理，返回与输入同尺寸的类别索引图。"""
    H, W = img.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)

    # 边界不足一个 patch 的位置向内回退，保证窗口落在图内
    rows = list(range(0, max(H - patch_size, 0) + 1, stride))
    cols = list(range(0, max(W - patch_size, 0) + 1, stride))
    if not rows or (H > patch_size and rows[-1] != H - patch_size):
        rows.append(H - patch_size)
    if not cols or (W > patch_size and cols[-1] != W - patch_size):
        cols.append(W - patch_size)
    rows = sorted(set(max(r, 0) for r in rows))
    cols = sorted(set(max(c, 0) for c in cols))

    tiles, coords = [], []
    for r in rows:
        for c in cols:
            tiles.append(img[r:r + patch_size, c:c + patch_size])
            coords.append((r, c))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for i in range(0, len(tiles), batch_size):
        batch = tiles[i:i + batch_size]
        arr = np.stack([
            cv2.cvtColor(t, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 for t in batch
        ])
        if normalize:
            arr = (arr - mean) / std
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for (r, c), p in zip(coords[i:i + batch_size], pred):
            h = min(patch_size, H - r)
            w = min(patch_size, W - c)
            out[r:r + h, c:c + w] = p[:h, :w]

    return out


def areas_from_mask(pred: np.ndarray) -> dict:
    """统计预测图中各训练标签 id 的像素数，并映射回 Gleason 规则用的原始值键。"""
    counts = np.bincount(pred.ravel(), minlength=NUM_CLASSES)
    # 训练标签 id -> 掩码原始值：0->0, 1->3, 2->4, 3->5, 4->6
    id_to_value = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6}
    return {id_to_value[i]: int(counts[i]) for i in range(NUM_CLASSES)}


def main() -> int:
    p = argparse.ArgumentParser(
        description="U-Net 整图滑窗分割推理 + core 级 Gleason 评分推导",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--input", required=True, help="图像文件或目录")
    p.add_argument("--out-dir", default=None, help="默认 outputs/predictions/<时间戳>")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--overlap", type=float, default=0.0, help="滑窗重叠比例 0~0.5")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--save-masks", action="store_true", help="保存预测掩码 PNG")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.weights, args.base_channels, args.depth, device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "outputs", "predictions", stamp)
    os.makedirs(out_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)

    if os.path.isdir(args.input):
        files = sorted(
            f for f in glob.glob(os.path.join(args.input, "*"))
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
        )
    else:
        files = [args.input]
    if args.limit:
        files = files[: args.limit]

    stride = max(1, int(args.patch_size * (1 - args.overlap)))

    results = {}
    for i, path in enumerate(files, 1):
        case_id = os.path.splitext(os.path.basename(path))[0]
        img = imread_unicode(path, cv2.IMREAD_COLOR)
        pred = predict_mask(model, img, args.patch_size, stride, device, args.batch_size)

        areas = areas_from_mask(pred)
        gleason = gleason_rules.derive_gleason(areas)
        burden = gleason_rules.cancer_burden(areas)

        results[case_id] = {
            "image": os.path.basename(path),
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "areas_px": {gleason_rules.VALUE_NAMES.get(k, str(k)): v for k, v in areas.items()},
            **gleason,
            **burden,
        }

        if args.save_masks:
            imwrite_unicode(os.path.join(out_dir, "masks", case_id + ".png"), pred)

        if i % 10 == 0 or i == len(files):
            print(f"  推理 {i}/{len(files)}")

    payload = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "weights": args.weights,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "device": str(device),
            "n_images": len(files),
            "rule_source": "src/core/gleason_rules.py（与金标准推导同源）",
        },
        **results,
    }
    out_json = os.path.join(out_dir, "predictions.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"  处理图像: {len(files)}")
    print(f"  预测结果: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
