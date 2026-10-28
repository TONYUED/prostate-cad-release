#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gleason 评分规则的**唯一实现**。

金标准推导（scripts/build_gold_standards.py，作用于 SICAPv2 真实掩码）
与模型预测推导（src/inference/segment_infer.py，作用于模型分割输出）
必须调用同一套规则，否则两者之间的比较不成立。

规则
----
1. 仅统计 GG3/GG4/GG5 的像素面积；三者均为 0 则判为良性。
2. primary   = 面积最大的模式。
3. secondary = 面积第二大的模式；若占癌区面积比例 < min_secondary_ratio，
   则令 secondary = primary。
4. Gleason score = primary + secondary。
5. ISUP Grade Group 按 2014 ISUP 共识映射。

注意：这是**面积规则**，不等同于病理医师的病例级判读。像素级 Dice / IoU
才是独立于该规则的主评价指标，core 级评分一致性应作为次要指标报告。
"""

from __future__ import annotations

VAL_BACKGROUND = 0
VAL_UNCONFIRMED = 1
VAL_GG3, VAL_GG4, VAL_GG5 = 3, 4, 5
VAL_NC = 6

CANCER_VALUES = (VAL_GG3, VAL_GG4, VAL_GG5)

VALUE_NAMES = {
    VAL_BACKGROUND: "background",
    VAL_UNCONFIRMED: "unconfirmed",
    VAL_GG3: "GG3",
    VAL_GG4: "GG4",
    VAL_GG5: "GG5",
    VAL_NC: "NC",
}

DEFAULT_MIN_SECONDARY_RATIO = 0.05


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


def isup_from_score(score: int, primary: int | None = None) -> int:
    """从总分（必要时结合主要模式）得到 ISUP 分级。"""
    if score <= 0:
        return 0
    if score <= 6:
        return 1
    if score == 7:
        return 2 if primary == 3 else 3
    if score == 8:
        return 4
    return 5


def derive_gleason(areas_px: dict, min_secondary_ratio: float = DEFAULT_MIN_SECONDARY_RATIO) -> dict:
    """由各类别像素面积推导 core 级 Gleason 评分。

    Args:
        areas_px: {掩码值或类别名 -> 像素数}。接受掩码原始值 (3/4/5/6) 为键。
        min_secondary_ratio: 次要模式的最小面积占比。

    Returns:
        dict，含 primary_pattern / secondary_pattern / gleason_score / isup_grade /
        is_benign / cancer_area_px / pattern_ratios。
    """
    cancer = {}
    for v in CANCER_VALUES:
        n = int(areas_px.get(v, areas_px.get(VALUE_NAMES[v], 0)))
        cancer[v] = n
    cancer_total = sum(cancer.values())

    if cancer_total == 0:
        return {
            "is_benign": True,
            "primary_pattern": None,
            "secondary_pattern": None,
            "gleason_score": 0,
            "isup_grade": 0,
            "cancer_area_px": 0,
            "pattern_ratios": {},
            "note": "无 GG3/4/5 区域，判为良性/非癌",
        }

    ranked = sorted(cancer.items(), key=lambda kv: -kv[1])
    primary, primary_area = ranked[0]
    secondary, secondary_area = ranked[1]

    secondary_is_effective = secondary_area >= min_secondary_ratio * cancer_total
    if not secondary_is_effective:
        secondary = primary

    return {
        "is_benign": False,
        "primary_pattern": int(primary),
        "secondary_pattern": int(secondary),
        "gleason_score": int(primary + secondary),
        "isup_grade": isup_from_patterns(primary, secondary),
        "cancer_area_px": cancer_total,
        "pattern_ratios": {
            VALUE_NAMES[v]: round(a / cancer_total, 6) for v, a in cancer.items() if a > 0
        },
        "secondary_from_single_pattern": not secondary_is_effective,
    }


def cancer_burden(areas_px: dict) -> dict:
    """计算用于 ROC 的连续型预测分数。

    - gg4plus_ratio: (GG4 + GG5) / (GG3 + GG4 + GG5)，ISUP >= 3 的常用替代指标
    - high_grade_ratio: GG5 / 癌区
    - cancer_ratio_in_tissue: 癌区 / (癌区 + 良性腺体)，即恶性组织占比
    """
    g3 = int(areas_px.get(VAL_GG3, areas_px.get("GG3", 0)))
    g4 = int(areas_px.get(VAL_GG4, areas_px.get("GG4", 0)))
    g5 = int(areas_px.get(VAL_GG5, areas_px.get("GG5", 0)))
    nc = int(areas_px.get(VAL_NC, areas_px.get("NC", 0)))

    cancer_total = g3 + g4 + g5
    if cancer_total == 0:
        return {
            "gg4plus_ratio": 0.0,
            "high_grade_ratio": 0.0,
            "cancer_ratio_in_tissue": 0.0,
        }
    return {
        "gg4plus_ratio": round((g4 + g5) / cancer_total, 6),
        "high_grade_ratio": round(g5 / cancer_total, 6),
        "cancer_ratio_in_tissue": round(cancer_total / max(cancer_total + nc, 1), 6),
    }
