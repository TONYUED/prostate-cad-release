#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 U-Net 做 Gleason 模式语义分割。

输入数据由 scripts/prepare_segmentation_dataset.py 生成（512x512 patch，
按 core 划分 train/val）。

用法
----
    # 完整训练
    python src/training/train_unet.py --data-dir data/seg_patches --epochs 60

    # 冒烟测试（少量样本、1 个 epoch，验证链路可跑通）
    python src/training/train_unet.py --data-dir data/seg_patches --epochs 1 ^
        --limit-samples 64 --batch-size 8

产出
----
    outputs/unet/<时间戳>/
        best.pth        验证集前景 mIoU 最优权重
        last.pth        最后一轮权重
        history.json    每轮 loss / mIoU / mDice
        history.csv
        config.json     本次训练的完整配置（便于复现）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.unet import UNet, count_parameters                # noqa: E402
from src.training.dataset import PatchSegmentationDataset, compute_class_weights  # noqa: E402
from src.evaluation.metrics import confusion_matrix, summarize, NUM_CLASSES  # noqa: E402


def autocast_ctx(enabled: bool, device_type: str = "cuda"):
    """兼容 torch>=2.4 与旧版的 autocast 写法。"""
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """多类 soft dice 损失（对类别不平衡比纯 CE 更稳健）。"""
    prob = torch.softmax(logits, dim=1)
    onehot = nn.functional.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + onehot.sum(dim=(2, 3))
    loss = 1 - (2 * inter + eps) / (denom + eps)
    return loss.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module, use_dice: bool) -> tuple:
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    total_loss = 0.0
    n_batches = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = criterion(logits, masks)
        if use_dice:
            loss = loss + dice_loss(logits, masks)
        total_loss += float(loss.item())
        n_batches += 1

        pred = logits.argmax(dim=1).cpu().numpy()
        gt = masks.cpu().numpy()
        cm += confusion_matrix(pred, gt, NUM_CLASSES)

    return total_loss / max(n_batches, 1), summarize(cm), cm


def main() -> int:
    p = argparse.ArgumentParser(
        description="训练 U-Net 进行 Gleason 模式语义分割",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--data-dir", required=True, help="patch 数据集根目录（含 dataset.json）")
    p.add_argument("--output-dir", default=None, help="默认 outputs/unet/<时间戳>")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--class-weight", default="median", choices=["median", "inverse", "none"])
    p.add_argument("--dice-weight", type=float, default=1.0, help="0 表示不使用 dice 损失")
    p.add_argument("--num-workers", type=int, default=0, help="Windows 建议保持 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / cpu，默认自动选择")
    p.add_argument("--amp", action="store_true", help="启用混合精度（需 CUDA）")
    p.add_argument("--patience", type=int, default=15, help="早停轮数，0 表示不早停")
    p.add_argument("--limit-samples", type=int, default=None, help="每子集最多用多少样本（冒烟测试）")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(REPO_ROOT, "outputs", "unet", stamp)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "dataset.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    n_train = len(meta["index"]["train"])
    n_val = len(meta["index"]["val"])
    if n_train == 0 or n_val == 0:
        raise RuntimeError(
            f"train/val 样本数为 {n_train}/{n_val}，无法训练。"
            f"请检查 dataset.json（若用 --limit 生成过数据，需要全量重新生成）。"
        )

    train_ds = PatchSegmentationDataset(args.data_dir, "train", augment=True, seed=args.seed)
    val_ds = PatchSegmentationDataset(args.data_dir, "val", augment=False)

    if args.limit_samples:
        train_ds.items = train_ds.items[: args.limit_samples]
        val_ds.items = val_ds.items[: args.limit_samples]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    if args.class_weight == "none":
        weights = None
    else:
        weights = compute_class_weights(args.data_dir, "train", args.class_weight)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    model = UNet(
        n_classes=NUM_CLASSES,
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    amp_enabled = args.amp and device.type == "cuda"
    try:  # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # 旧版本
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    print("=" * 66)
    print("U-Net Gleason 模式语义分割训练")
    print("=" * 66)
    print(f"设备        : {device}")
    print(f"数据        : {args.data_dir}")
    print(f"样本        : train {len(train_ds)} / val {len(val_ds)}")
    print(f"参数量      : {count_parameters(model):,}")
    print(f"类别权重    : {weights.tolist() if weights is not None else 'none'}")
    print(f"输出        : {out_dir}")

    history = []
    best_metric = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    use_dice = args.dice_weight > 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(amp_enabled, device.type):
                logits = model(imgs)
                loss = criterion(logits, masks)
                if use_dice:
                    loss = loss + args.dice_weight * dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            n_batches += 1

        scheduler.step()
        train_loss = running / max(n_batches, 1)
        val_loss, val_metrics, _ = evaluate(model, val_loader, device, criterion, use_dice)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "mIoU": val_metrics["mIoU"],
            "foreground_mIoU": val_metrics["foreground_mIoU"],
            "mDice": val_metrics["mDice"],
            "foreground_mDice": val_metrics["foreground_mDice"],
            "pixel_accuracy": val_metrics["pixel_accuracy"],
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)

        metric = val_metrics["foreground_mIoU"] or val_metrics["mIoU"] or 0.0
        is_best = metric > best_metric
        if is_best:
            best_metric, best_epoch = metric, epoch
            epochs_no_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics,
                 "args": vars(args)},
                os.path.join(out_dir, "best.pth"),
            )
        else:
            epochs_no_improve += 1

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "metrics": val_metrics,
             "args": vars(args)},
            os.path.join(out_dir, "last.pth"),
        )

        print(
            f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"mIoU {val_metrics['mIoU']} | fg_mIoU {val_metrics['foreground_mIoU']} | "
            f"fg_mDice {val_metrics['foreground_mDice']} | {row['seconds']}s"
            + ("  *best*" if is_best else "")
        )

        if args.patience and epochs_no_improve >= args.patience:
            print(f"[早停] {args.patience} 轮无提升，停止训练。")
            break

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"config": vars(args), "best_epoch": best_epoch,
             "best_foreground_mIoU": best_metric, "history": history},
            f, ensure_ascii=False, indent=2,
        )
    with open(os.path.join(out_dir, "history.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("\n[完成]")
    print(f"  最优轮次: {best_epoch}  前景 mIoU: {best_metric:.4f}")
    print(f"  权重: {os.path.join(out_dir, 'best.pth')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
