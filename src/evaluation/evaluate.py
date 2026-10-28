#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三层次评估：像素级分割 / core 级 Gleason 评分 / ISUP 分级 ROC。

主指标是**像素级 Dice 与 IoU**（不依赖任何评分推导规则）；
core 级评分一致性与 ROC 是次要指标，用于回答临床关心的问题。

只依赖 numpy + matplotlib，AUC 与二次加权 Kappa 均为本地实现，
避免不同 sklearn 版本带来的数值差异。

用法
----
    python src/evaluation/evaluate.py ^
        --predictions outputs/predictions/<ts>/predictions.json ^
        --gold       data/gold_standards.json ^
        --out-dir    results/eval_<时间戳>

    # 同时做像素级评估
    python src/evaluation/evaluate.py ... ^
        --pred-masks outputs/predictions/<ts>/masks ^
        --true-masks <SICAPv2 掩码目录>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.evaluation.metrics import confusion_matrix, summarize, NUM_CLASSES  # noqa: E402

CLASS_NAMES = {0: "background", 1: "GG3", 2: "GG4", 3: "GG5", 4: "NC"}
VALUE_MAP = {0: 0, 1: 0, 3: 1, 4: 2, 5: 3, 6: 4}


# ------------------------------------------------------------------ ROC
def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray):
    """返回 (fpr, tpr, thresholds)，按 score 降序遍历。"""
    order = np.argsort(-y_score, kind="mergesort")
    y_true = np.asarray(y_true)[order]
    y_score = np.asarray(y_score)[order]

    n_pos = float(y_true.sum())
    n_neg = float((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    tpr, fpr, thr = [], [], []
    tp = fp = 0
    prev = None
    for i in range(len(y_true)):
        if prev is not None and y_score[i] != prev:
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
            thr.append(prev)
        prev = y_score[i]
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(tp / n_pos)
    fpr.append(fp / n_neg)
    thr.append(prev)
    return np.array(fpr), np.array(tpr), np.array(thr)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U 统计量等价形式，数值稳定且不依赖 sklearn。"""
    y_true = np.asarray(y_true)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = np.asarray(y_score)[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 平均秩，含并列
        i = j + 1

    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = auc_score(y_true[idx], y_score[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def youden_point(fpr, tpr, thr):
    j = tpr - fpr
    k = int(np.argmax(j))
    return {
        "threshold": float(thr[k]),
        "sensitivity": float(tpr[k]),
        "specificity": float(1 - fpr[k]),
        "youden": float(j[k]),
    }


# --------------------------------------------------------- 加权 Kappa
def quadratic_weighted_kappa(a: np.ndarray, b: np.ndarray, n_labels: int | None = None) -> float:
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    k = n_labels or int(max(a.max(), b.max())) + 1
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[x, y] += 1
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    hist_a = np.bincount(a, minlength=k).astype(float)
    hist_b = np.bincount(b, minlength=k).astype(float)
    e = np.outer(hist_a, hist_b) / max(len(a), 1)
    num = (o * w).sum()
    den = (e * w).sum()
    if den <= 0:
        # 完全一致（或完全无差异）时无法定义kappa，记为完美一致
        return 1.0 if num == 0 else float("nan")
    return float(1 - num / den)


# --------------------------------------------------------------- 主流程
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _r(x):
    """保留 6 位小数；NaN/None 转为 None，避免写出非法 JSON 字面量 NaN。"""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 6)


def _fmt(x, nd: int = 3) -> str:
    """打印用：样本不足导致指标无定义时显示 n/a，而不是伪造一个 0。"""
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def pixel_level_eval(pred_dir: str, true_dir: str) -> dict | None:
    """像素级评估：预测掩码目录 vs SICAPv2 原始掩码目录。"""
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
    if not preds:
        return None

    lut = np.zeros(256, dtype=np.uint8)
    for src, dst in VALUE_MAP.items():
        lut[src] = dst

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    matched = 0
    for pp in preds:
        case_id = os.path.splitext(os.path.basename(pp))[0]
        tpath = None
        for cand in (
            os.path.join(true_dir, case_id + "_classimg_nonconvex.png"),
            os.path.join(true_dir, case_id + ".png"),
        ):
            if os.path.exists(cand):
                tpath = cand
                break
        if tpath is None:
            continue

        pred = cv2_imdecode(pp)
        true = cv2_imdecode(tpath)
        if true.ndim == 3:
            true = true[:, :, 0]
        if pred.shape != true.shape:
            h, w = min(pred.shape[0], true.shape[0]), min(pred.shape[1], true.shape[1])
            pred, true = pred[:h, :w], true[:h, :w]
        true = lut[true]
        cm += confusion_matrix(pred.astype(np.int64), true.astype(np.int64), NUM_CLASSES)
        matched += 1

    if matched == 0:
        return None
    out = summarize(cm, CLASS_NAMES)
    out["n_images"] = matched
    return out


def cv2_imdecode(path: str) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"无法读取: {path}")
    return img


def core_level_eval(preds: dict, gold: dict, isup_threshold: int = 3) -> dict:
    """core 级评估：Gleason 总分一致性、ISUP QWK、ROC。"""
    common = sorted(set(preds) & set(gold) & {"_metadata"} - {"_metadata"})
    common = sorted((set(preds) & set(gold)) - {"_metadata"})
    if not common:
        raise RuntimeError("预测与金标准没有可匹配的 case_id，请检查命名是否一致。")

    p_score, g_score = [], []
    p_isup, g_isup = [], []
    p_bin, g_bin = [], []
    y_true, y_score = [], []

    rows = []
    for cid in common:
        pr, gd = preds[cid], gold[cid]
        ps, gs = int(pr.get("gleason_score", 0)), int(gd.get("gleason_score", 0))
        pi, gi = int(pr.get("isup_grade", 0)), int(gd.get("isup_grade", 0))
        pb, gb = int(not pr.get("is_benign", True)), int(not gd.get("is_benign", True))

        p_score.append(ps); g_score.append(gs)
        p_isup.append(pi); g_isup.append(gi)
        p_bin.append(pb); g_bin.append(gb)

        y_true.append(1 if gi >= isup_threshold else 0)
        y_score.append(float(pr.get("gg4plus_ratio", 0.0)))

        rows.append({
            "case_id": cid,
            "pred_gleason": ps, "gold_gleason": gs,
            "pred_isup": pi, "gold_isup": gi,
            "exact": int(ps == gs),
            "within_1": int(abs(ps - gs) <= 1),
        })

    p_score, g_score = np.array(p_score), np.array(g_score)
    p_isup, g_isup = np.array(p_isup), np.array(g_isup)
    p_bin, g_bin = np.array(p_bin), np.array(g_bin)
    y_true, y_score = np.array(y_true), np.array(y_score)

    exact = float((p_score == g_score).mean())
    within1 = float((np.abs(p_score - g_score) <= 1).mean())

    if len(np.unique(y_true)) >= 2:
        auc = auc_score(y_true, y_score)
        lo, hi = bootstrap_auc_ci(y_true, y_score)
        curve = roc_curve_points(y_true, y_score)
        opt = youden_point(*curve) if curve else None
        fpr, tpr, thr = curve
    else:
        auc, lo, hi, opt = float("nan"), float("nan"), float("nan"), None
        fpr = tpr = thr = None

    # 良恶性二分类
    tp = int(((p_bin == 1) & (g_bin == 1)).sum())
    tn = int(((p_bin == 0) & (g_bin == 0)).sum())
    fp = int(((p_bin == 1) & (g_bin == 0)).sum())
    fn = int(((p_bin == 0) & (g_bin == 1)).sum())
    # 某一类无样本时指标无定义，返回 NaN（写入 JSON 时转为 null），不用 0 掩盖
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "n_cores": len(common),
        "gleason_exact_accuracy": _r(exact),
        "gleason_within_1_accuracy": _r(within1),
        "isup_exact_accuracy": _r(float((p_isup == g_isup).mean())),
        "isup_quadratic_weighted_kappa": _r(
            quadratic_weighted_kappa(p_isup, g_isup, n_labels=6)),
        "gleason_quadratic_weighted_kappa": _r(
            quadratic_weighted_kappa(p_score, g_score, n_labels=11)),
        "malignancy_sensitivity": _r(sens),
        "malignancy_specificity": _r(spec),
        "confusion_malignancy": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "roc": {
            "task": f"ISUP >= {isup_threshold} (Gleason >= 4+3) 的检出",
            "score": "gg4plus_ratio = (GG4+GG5)/(GG3+GG4+GG5)",
            "auc": _r(auc),
            "auc_ci95": [_r(lo), _r(hi)],
            "n_bootstrap": 1000,
            "optimal_point": opt,
            "n_positive": int(y_true.sum()),
            "n_negative": int((1 - y_true).sum()),
        },
        "_raw": {"fpr": fpr, "tpr": tpr, "thr": thr},
        "_rows": rows,
    }


def plot_roc(core: dict, out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    raw = core.get("_raw", {})
    fpr, tpr = raw.get("fpr"), raw.get("tpr")
    if fpr is None:
        return False

    roc = core["roc"]
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=300)
    ax.plot(fpr, tpr, color="#1f4e79", lw=2,
            label=f"AUC = {roc['auc']:.3f} (95% CI {roc['auc_ci95'][0]:.3f}-{roc['auc_ci95'][1]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    if roc.get("optimal_point"):
        o = roc["optimal_point"]
        ax.scatter([1 - o["specificity"]], [o["sensitivity"]], color="#a4243b", zorder=5,
                   label=f"Youden point (Se {o['sensitivity']:.2f}, Sp {o['specificity']:.2f})")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(f"ROC: {roc['task']}\n(n = {core['n_cores']} cores)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gleason 分割与分级的三层次评估",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--predictions", required=True, help="segment_infer.py 产出的 predictions.json")
    ap.add_argument("--gold", required=True, help="build_gold_standards.py 产出的金标准")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pred-masks", default=None, help="预测掩码目录（像素级评估）")
    ap.add_argument("--true-masks", default=None, help="SICAPv2 原始掩码目录")
    ap.add_argument("--isup-threshold", type=int, default=3)
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", f"eval_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    preds = {k: v for k, v in load_json(args.predictions).items() if k != "_metadata"}
    gold = {k: v for k, v in load_json(args.gold).items() if k != "_metadata"}

    print("=" * 66)
    print("评估：像素级 / core 级 / ROC")
    print("=" * 66)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "predictions": args.predictions,
        "gold": args.gold,
    }

    if args.pred_masks and args.true_masks:
        px = pixel_level_eval(args.pred_masks, args.true_masks)
        report["pixel_level"] = px
        if px:
            print("\n[像素级]")
            print(f"  匹配图像: {px['n_images']}")
            print(f"  mIoU {px['mIoU']} | 前景 mIoU {px['foreground_mIoU']}")
            print(f"  mDice {px['mDice']} | 前景 mDice {px['foreground_mDice']}")
            print(f"  逐类 Dice: {px['per_class_dice']}")
        else:
            print("\n[像素级] 无可匹配的掩码，跳过")

    core = core_level_eval(preds, gold, args.isup_threshold)
    report["core_level"] = {
        k: v for k, v in core.items() if k not in ("_raw", "_rows")
    }

    print("\n[core 级]")
    print(f"  样本数: {core['n_cores']}")
    print(f"  Gleason 完全一致率: {_fmt(core['gleason_exact_accuracy'])}")
    print(f"  Gleason ±1 容差率 : {_fmt(core['gleason_within_1_accuracy'])}")
    print(f"  ISUP 完全一致率   : {_fmt(core['isup_exact_accuracy'])}")
    print(f"  ISUP 二次加权Kappa: {_fmt(core['isup_quadratic_weighted_kappa'])}")
    print(f"  恶性检出 Se/Sp    : {_fmt(core['malignancy_sensitivity'])} / "
          f"{_fmt(core['malignancy_specificity'])}")
    print(f"\n[ROC] {core['roc']['task']}")
    print(f"  AUC = {_fmt(core['roc']['auc'])}  95% CI "
          f"[{_fmt(core['roc']['auc_ci95'][0])}, {_fmt(core['roc']['auc_ci95'][1])}]")

    roc_png = os.path.join(out_dir, "roc_curve.png")
    if plot_roc(core, roc_png):
        print(f"  ROC 图: {roc_png}")

    rows_csv = os.path.join(out_dir, "per_core.csv")
    with open(rows_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(core["_rows"][0].keys()))
        w.writeheader()
        w.writerows(core["_rows"])

    out_json = os.path.join(out_dir, "evaluation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] 报告: {out_json}")
    print(f"[OK] 逐例: {rows_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
