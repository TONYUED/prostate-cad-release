# Source Code Export — Prostate CAD (Gleason pattern segmentation)

Generated: 2026-10-28T11:39:38

Complete source of every module used to generate the reported results, for technical verification. Hashes are SHA-256 of the exact files in the repository.

## 1. Environment

| item | value |
|---|---|
| OS | Windows 10 |
| Python | 3.11.9 |
| torch | 2.5.1+cu121 |
| torch_cuda_build | 12.1 |
| cuda_available | True |
| gpu | NVIDIA GeForce RTX 4080 Laptop GPU |
| numpy | 2.2.6 |
| opencv-python | 4.12.0 |
| matplotlib | 3.10.6 |

## 2. File manifest (repository files, excluding data/outputs/results/legacy)

| file | bytes | lines | sha256 |
|---|---:|---:|---|
| `.gitignore` | 739 | 52 | `7eec0c856218a3b62821ac5e92f9b78a92dc7ac01d6a5f15306913d92a2c1e63` |
| `.zenodo.json` | 926 | 28 | `d297fc2bbbbd03df536dda252ef4332cfb81271a64cabde0cf702a85682ec2fe` |
| `CITATION.cff` | 1367 | 37 | `a1c738e41fe700894493870fed00c29ae1c71ddcc86b703ebc26436f6826b54a` |
| `LICENSE` | 2646 | 45 | `22e3340049d5b923ee517829407d8e83109f4f5e572c2cb842e3c6a92a749ada` |
| `README.md` | 5884 | 162 | `33768965ab1ee144723921b4415e8e514bda39d8fc292bd42a9a2007f2180461` |
| `docs/DATA.md` | 2798 | 76 | `f2570da824e78055ed0ecba474315c42c860baed4a7e542f43b3629002b0684b` |
| `docs/LIMITATIONS.md` | 2887 | 53 | `540df468a24ca51422f6b4379c273f81df31aff3585270d4a836056a63a35a1b` |
| `docs/methodology.md` | 5308 | 125 | `94a4a029c5dc391c59412956fe9c63eae8f81bffbba634497d3aa06a94add000` |
| `requirements.txt` | 329 | 16 | `c98424d9cb635329dc9cd1cd2aca3343b8f3e000fa533d907ca1220e65035e31` |
| `scripts/build_gold_standards.py` | 9382 | 269 | `d461d06215ffb34dcd5b6dcb461f3ba591a9594215dd8ecb82e2ba350b48d773` |
| `scripts/masks_to_yolo_seg.py` | 11058 | 301 | `e2ee3508ceaf614ce7385eef353ecb31ca7d012510b148a7d457dc9853962059` |
| `scripts/prepare_segmentation_dataset.py` | 9288 | 244 | `7a647d4619ce3ac13245b8937316eb9bbc0206019d07a372b80862a416be5a7d` |
| `src/core/gleason_rules.py` | 4845 | 143 | `c3a29124e7a9dcfd7851ce1f45f5d1141bc13b3af1c69bec2f7eeb990a27a757` |
| `src/evaluation/evaluate.py` | 15776 | 428 | `410e432857480b6912cdb08f49c64a0f2eb87052399d2cc3b4e1349ab66173a6` |
| `src/evaluation/metrics.py` | 2955 | 79 | `d5cf4028d22e9eeb7ec4121c0d004ea45cac132a76950a1aeb69f6b9aa20d92e` |
| `src/inference/segment_infer.py` | 8312 | 214 | `530aa08cefd47e2611ea477e37bd75bef6ce211a71983d13e1fac9ccd44d5c67` |
| `src/models/unet.py` | 3173 | 97 | `a34dd62508cfd50d32a9ea8116ff091e7b98175d50ef724ea20933b888790316` |
| `src/training/dataset.py` | 5166 | 141 | `a054352a42137661f7f0995dfe26cdd8625b6dcc24dc2a005f1119a845787f3c` |
| `src/training/train_unet.py` | 11379 | 294 | `ad27f3c4ee96f12fc0dbc48b3ebe19f19d9c0b13d3a06198e5848a2abb9db521` |
| `tests/smoke_test.py` | 4477 | 113 | `dd23c23ea77135ea6b44bd8332d4065cf4b7203eb5b39170c3c447cafd39589c` |

## 3. Full source


---

### `scripts/build_gold_standards.py`  (269 lines, sha256 `d461d06215ffb34dcd5b6dcb461f3ba591a9594215dd8ecb82e2ba350b48d773`)

```python
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
```

---

### `scripts/prepare_segmentation_dataset.py`  (244 lines, sha256 `7a647d4619ce3ac13245b8937316eb9bbc0206019d07a372b80862a416be5a7d`)

```python
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
```

---

### `scripts/masks_to_yolo_seg.py`  (301 lines, sha256 `e2ee3508ceaf614ce7385eef353ecb31ca7d012510b148a7d457dc9853962059`)

```python
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
```

---

### `src/core/gleason_rules.py`  (143 lines, sha256 `c3a29124e7a9dcfd7851ce1f45f5d1141bc13b3af1c69bec2f7eeb990a27a757`)

```python
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
```

---

### `src/models/unet.py`  (97 lines, sha256 `a34dd62508cfd50d32a9ea8116ff091e7b98175d50ef724ea20933b888790316`)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量 U-Net，用于前列腺活检组织学图像的 Gleason 模式语义分割。

任务：对每个像素分类为 5 类
    0 background / 1 GG3 / 2 GG4 / 3 GG5 / 4 NC(良性腺体)

选择语义分割而非实例分割的理由见 docs/methodology.md：SICAPv2 的掩码是
区域级（region-level）标注，同一类别通常只有一个连通域，不存在"实例"概念。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """标准 U-Net，通道数可通过 base_channels 调整以适配显存。

    Args:
        n_classes: 输出类别数（含背景）
        base_channels: 第一层卷积通道数（32 适合 8GB 显存，64 更充裕）
        depth: 下采样次数
        dropout: 瓶颈层 dropout 概率
    """

    def __init__(self, n_classes: int = 5, base_channels: int = 32,
                 depth: int = 4, dropout: float = 0.2):
        super().__init__()
        ch = base_channels
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        in_ch = 3
        for i in range(depth):
            self.downs.append(DoubleConv(in_ch, ch))
            in_ch = ch
            ch = min(ch * 2, 512)

        self.bottleneck = DoubleConv(in_ch, ch, dropout=dropout)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
            )
            self.up_convs.append(DoubleConv(ch, ch // 2))
            ch = ch // 2

        self.head = nn.Conv2d(ch, n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # 处理奇数尺寸导致的 1 像素误差
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

### `src/training/dataset.py`  (141 lines, sha256 `a054352a42137661f7f0995dfe26cdd8625b6dcc24dc2a005f1119a845787f3c`)

```python
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
```

---

### `src/training/train_unet.py`  (294 lines, sha256 `ad27f3c4ee96f12fc0dbc48b3ebe19f19d9c0b13d3a06198e5848a2abb9db521`)

```python
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
```

---

### `src/inference/segment_infer.py`  (214 lines, sha256 `530aa08cefd47e2611ea477e37bd75bef6ce211a71983d13e1fac9ccd44d5c67`)

```python
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
```

---

### `src/evaluation/metrics.py`  (79 lines, sha256 `d5cf4028d22e9eeb7ec4121c0d004ea45cac132a76950a1aeb69f6b9aa20d92e`)

```python
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
```

---

### `src/evaluation/evaluate.py`  (428 lines, sha256 `410e432857480b6912cdb08f49c64a0f2eb87052399d2cc3b4e1349ab66173a6`)

```python
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
```

---

### `tests/smoke_test.py`  (113 lines, sha256 `dd23c23ea77135ea6b44bd8332d4065cf4b7203eb5b39170c3c447cafd39589c`)

```python
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
```

---

### `requirements.txt`  (16 lines, sha256 `c98424d9cb635329dc9cd1cd2aca3343b8f3e000fa533d907ca1220e65035e31`)

```text
# 核心依赖（训练/推理/评估全链路）
numpy>=1.24
opencv-python>=4.8
torch>=2.0
matplotlib>=3.7

# 数据准备与报告
PyYAML>=6.0

# 可选：仅在运行 legacy/ 早期原型时需要
# tensorflow==2.15.0
# ultralytics>=8.0.0
# scikit-learn>=1.3
# pandas>=2.0
# seaborn>=0.12
# ttkbootstrap>=1.10
```

## 4. Exact commands that produce the reported results

```bash
python scripts/build_gold_standards.py --mask-dir <SICAPv2_MASK_DIR> \
       --out data/gold_standards.json
python scripts/prepare_segmentation_dataset.py \
       --image-dir <HE_IMG_DIR> --mask-dir <SICAPv2_MASK_DIR> \
       --out data/seg_patches --split-from <EXISTING_YOLO_DATA_ROOT>
python src/training/train_unet.py --data-dir data/seg_patches --epochs 60 \
       --batch-size 8 --amp
python src/inference/segment_infer.py --weights outputs/unet/<ts>/best.pth \
       --input <HE_IMG_DIR> --save-masks --out-dir outputs/predictions
python src/evaluation/evaluate.py \
       --predictions outputs/predictions/predictions.json \
       --gold data/gold_standards.json \
       --pred-masks outputs/predictions/masks --true-masks <SICAPv2_MASK_DIR> \
       --out-dir results/eval
python tests/smoke_test.py --image-dir <HE_IMG_DIR> --mask-dir <SICAPv2_MASK_DIR>
```

Note: `scripts/masks_to_yolo_seg.py` is retained for provenance only and is NOT part of the reported-result pipeline. The masks were verified to be region-level (1-2 connected components per class, 20-60% of image area), so instance segmentation (YOLOv8-seg) is not applicable. See `docs/methodology.md` and `docs/LIMITATIONS.md`.
