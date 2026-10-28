#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端冒烟测试：用极少样本跑通"金标准 -> 数据 -> 训练 -> 推理 -> 评估"全链路。

目的是验证**链路可运行**，不产出任何可用模型或有效指标。

用法
----
    python tests/smoke_test.py --image-dir <IMG_DIR> --mask-dir <MASK_DIR>

    # 指定工作目录与样本量
    python tests/smoke_test.py --image-dir ... --mask-dir ... \
        --work-dir .smoke --cores 4 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list, desc: str) -> None:
    print("\n" + "=" * 66)
    print(f"[步骤] {desc}")
    print("  " + " ".join(cmd[1:]))
    print("=" * 66)
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise SystemExit(f"[失败] {desc}（退出码 {res.returncode}）")


def main() -> int:
    p = argparse.ArgumentParser(description="端到端冒烟测试")
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--work-dir", default=".smoke", help="临时工作目录（结束后默认删除）")
    p.add_argument("--cores", type=int, default=4, help="使用的 core 数")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--samples", type=int, default=32, help="训练/验证各取多少 patch")
    p.add_argument("--keep", action="store_true", help="保留临时目录以便检查")
    args = p.parse_args()

    py = sys.executable
    work = os.path.abspath(os.path.join(REPO_ROOT, args.work_dir))
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)

    gold = os.path.join(work, "gold_standards.json")
    data = os.path.join(work, "patches")
    train_out = os.path.join(work, "model")
    pred_out = os.path.join(work, "predictions")
    eval_out = os.path.join(work, "eval")

    started = datetime.now()
    try:
        run([py, "scripts/build_gold_standards.py", "--mask-dir", args.mask_dir,
             "--out", gold, "--limit", str(args.cores)], "生成金标准")

        run([py, "scripts/prepare_segmentation_dataset.py", "--image-dir", args.image_dir,
             "--mask-dir", args.mask_dir, "--out", data,
             "--limit", str(args.cores)], "切分训练 patch")

        run([py, "src/training/train_unet.py", "--data-dir", data,
             "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
             "--limit-samples", str(args.samples), "--output-dir", train_out], "训练（冒烟）")

        weights = os.path.join(train_out, "best.pth")
        if not os.path.exists(weights):
            raise SystemExit(f"[失败] 未找到权重: {weights}")

        run([py, "src/inference/segment_infer.py", "--weights", weights,
             "--input", args.image_dir, "--limit", "2",
             "--save-masks", "--out-dir", pred_out], "滑窗推理")

        run([py, "src/evaluation/evaluate.py",
             "--predictions", os.path.join(pred_out, "predictions.json"),
             "--gold", gold,
             "--pred-masks", os.path.join(pred_out, "masks"),
             "--true-masks", args.mask_dir,
             "--out-dir", eval_out], "评估")

        # 校验产物
        report = os.path.join(eval_out, "evaluation_report.json")
        with open(report, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        assert "core_level" in data_json, "评估报告缺少 core_level"
        assert os.path.exists(os.path.join(eval_out, "per_core.csv")), "缺少 per_core.csv"

        print("\n" + "=" * 66)
        print(f"[通过] 全链路冒烟测试完成，用时 {(datetime.now() - started).total_seconds():.1f}s")
        print("  注意：本测试只验证链路可运行，指标无意义（模型仅训练 "
              f"{args.epochs} 轮、样本 {args.samples} 个）。")
        print("=" * 66)
    finally:
        if not args.keep and os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)
            print(f"[清理] 已删除临时目录: {work}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
