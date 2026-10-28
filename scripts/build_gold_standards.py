#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 SICAPv2 语义掩码推导病例级（core 级）Gleason 金标准。

背景
----
SICAPv2 官方为每张活检核心（core）提供像素级掩码 `*_classimg_nonconvex.png`，
单通道 uint8 语义图，取值与含义如下：

    0 = 背景 / 非腺体区域
    1 = 语义待确认的少量标注（本脚本默认忽略，可用 --keep-value-1 保留）
    3 = Gleason pattern 3 (GG3)
    4 = Gleason pattern 4 (GG4)
    5 = Gleason pattern 5 (GG5)
    6 = 良性 / 非癌腺体 (NC)

本脚本把像素级标注汇总为 core 级 Gleason 评分，供 ROC / 一致性分析作为
**真实金标准**使用，取代此前"由模型输出反推金标准"的做法。

推导规则
--------
1. 仅统计 GG3/GG4/GG5 的像素面积；若三者面积均为 0，判定为良性（benign）。
2. primary pattern   = 面积最大的模式。
3. secondary pattern = 面积第二大的模式；
   若其占癌区面积的比例 < --min-secondary-ratio（默认 5%）,
   则视为不存在明确次要模式，令 secondary = primary。
4. Gleason score = primary + secondary。
5. ISUP Grade Group 按 2014 ISUP 共识映射（见 isup_from_patterns）。

局限
----
该金标准由像素级掩码**推导**得出，与病理医师出具的病例级报告并不等同：
医师判读会综合结构形态（如筛状、融合腺体）而非单纯面积占比。
因此本文件应用于"与掩码同源"的评价；若需权威病例级评分，应从 SICAPv2
官方发布渠道获取其标注表并与本文件交叉核对。

用法
----
    python build_gold_standards.py --mask-dir <掩码目录> --out gold_standards.json
    python build_gold_standards.py --mask-dir <掩码目录> --limit 5   # 小样本自检
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

import cv2
import numpy as np

# SICAPv2 掩码取值
VAL_BG = 0
VAL_UNKNOWN = 1
VAL_GG3, VAL_GG4, VAL_GG5 = 3, 4, 5
VAL_NC = 6

CANCER_VALUES = (VAL_GG3, VAL_GG4, VAL_GG5)
LABEL_NAMES = {
    VAL_BG: "background",
    VAL_UNKNOWN: "unconfirmed",
    VAL_GG3: "GG3",
    VAL_GG4: "GG4",
    VAL_GG5: "GG5",
    VAL_NC: "NC",
}

# 掩码文件名形如 slide001_core003_classimg_nonconvex.png
MASK_SUFFIX_RE = re.compile(r"_classimg_(nonconvex|convex)$", re.IGNORECASE)


def imread_unicode(path: str) -> np.ndarray:
    """读取含非 ASCII 字符路径的图像（Windows 下 cv2.imread 会失败）。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"无法读取掩码: {path}")
    return img


def case_id_from_mask(path: str) -> str:
    """slide001_core003_classimg_nonconvex.png -> slide001_core003"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return MASK_SUFFIX_RE.sub("", stem)


def isup_from_patterns(primary: int, secondary: int) -> int:
    """按 2014 ISUP 共识把 (primary, secondary) 映射为 Grade Group 1-5。"""
    score = primary + secondary
    if score <= 6:
        return 1
    if score == 7:
        return 2 if primary == 3 else 3
    if score == 8:
        return 4
    return 5  # 9 或 10


def derive_gleason(areas: dict, min_secondary_ratio: float, keep_value_1: bool):
    """由各类像素面积推导 core 级 Gleason 评分。"""
    cancer = {v: int(areas.get(v, 0)) for v in CANCER_VALUES}
    cancer_total = sum(cancer.values())

    result = {
        "areas_px": {LABEL_NAMES[k]: int(v) for k, v in areas.items()},
        "cancer_area_px": cancer_total,
        "is_benign": cancer_total == 0,
    }

    if keep_value_1:
        result["areas_px"]["unconfirmed_kept"] = True

    if cancer_total == 0:
        # 良性：不参与 Gleason 评分
        result.update(
            {
                "primary_pattern": None,
                "secondary_pattern": None,
                "gleason_score": 0,
                "isup_grade": 0,
                "pattern_ratios": {},
                "note": "无 GG3/4/5 标注，判为良性/非癌",
            }
        )
        return result

    ranked = sorted(cancer.items(), key=lambda kv: -kv[1])
    primary, primary_area = ranked[0]
    secondary, secondary_area = ranked[1]

    if secondary_area / cancer_total < min_secondary_ratio:
        secondary = primary

    result.update(
        {
            "primary_pattern": int(primary),
            "secondary_pattern": int(secondary),
            "gleason_score": int(primary + secondary),
            "isup_grade": isup_from_patterns(primary, secondary),
            "pattern_ratios": {
                LABEL_NAMES[v]: round(a / cancer_total, 6) for v, a in cancer.items() if a > 0
            },
        }
    )
    return result


def build(mask_dir: str, min_secondary_ratio: float, keep_value_1: bool, limit: int | None):
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not masks:
        raise FileNotFoundError(f"掩码目录中没有 .png: {mask_dir}")
    if limit:
        masks = masks[:limit]

    gold = {}
    for i, path in enumerate(masks, 1):
        img = imread_unicode(path)
        counts = np.bincount(img.ravel(), minlength=7)
        areas = {v: int(counts[v]) for v in range(7) if counts[v] > 0}
        if not keep_value_1:
            areas.pop(VAL_UNKNOWN, None)

        case_id = case_id_from_mask(path)
        record = derive_gleason(areas, min_secondary_ratio, keep_value_1)
        record["mask_file"] = os.path.basename(path)
        gold[case_id] = record

        if i % 20 == 0 or i == len(masks):
            print(f"  处理 {i}/{len(masks)}")

    return gold


def summarize(gold: dict) -> dict:
    n = len(gold)
    benign = sum(1 for r in gold.values() if r["is_benign"])
    isup_dist = {}
    primary_dist = {}
    for r in gold.values():
        if r["is_benign"]:
            continue
        isup_dist[r["isup_grade"]] = isup_dist.get(r["isup_grade"], 0) + 1
        primary_dist[r["primary_pattern"]] = primary_dist.get(r["primary_pattern"], 0) + 1

    return {
        "n_cores": n,
        "n_benign": benign,
        "n_cancer": n - benign,
        "isup_grade_distribution": {
            f"ISUP {k}": v for k, v in sorted(isup_dist.items())
        },
        "primary_pattern_distribution": {
            f"GG{k}": v for k, v in sorted(primary_dist.items())
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="从 SICAPv2 掩码推导 core 级 Gleason 金标准",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--mask-dir", required=True, help="掩码目录（含 *_classimg_nonconvex.png）")
    p.add_argument("--out", default=None, help="输出 gold_standards.json 路径")
    p.add_argument(
        "--min-secondary-ratio",
        type=float,
        default=0.05,
        help="次要模式占癌区面积的最小比例，低于则令 secondary=primary（默认 0.05）",
    )
    p.add_argument("--keep-value-1", action="store_true", help="保留掩码值 1（语义待确认）")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 张（自检用）")
    args = p.parse_args()

    out = args.out or os.path.join(os.getcwd(), "gold_standards.json")

    print("=" * 66)
    print("SICAPv2 掩码 -> core 级 Gleason 金标准")
    print("=" * 66)
    print(f"掩码目录: {args.mask_dir}")
    print(f"次要模式面积阈值: {args.min_secondary_ratio:.0%}")

    gold = build(
        args.mask_dir,
        min_secondary_ratio=args.min_secondary_ratio,
        keep_value_1=args.keep_value_1,
        limit=args.limit,
    )

    payload = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "scripts/build_gold_standards.py",
            "source": "SICAPv2 pixel-level masks (*_classimg_nonconvex.png)",
            "derivation": (
                "primary = 面积最大的 GG3/4/5；secondary = 第二大模式，"
                "占比低于阈值时取 primary；ISUP 按 2014 ISUP 共识映射。"
            ),
            "min_secondary_ratio": args.min_secondary_ratio,
            "value_1_kept": args.keep_value_1,
            "caveat": (
                "由像素级掩码推导，不等同于病理医师出具的病例级报告；"
                "发表前建议与 SICAPv2 官方标注表核对。"
            ),
            **summarize(gold),
        },
        **gold,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    s = payload["_metadata"]
    print("\n[汇总]")
    print(f"  core 总数 : {s['n_cores']}  (癌 {s['n_cancer']} / 良性 {s['n_benign']})")
    print(f"  ISUP 分布 : {s['isup_grade_distribution']}")
    print(f"  主要模式  : {s['primary_pattern_distribution']}")
    print(f"\n[OK] 已写出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
