# 前列腺活检 Gleason 模式分割 —— 可复现代码

对前列腺穿刺活检全切片图像做**像素级 Gleason 模式语义分割**（GG3 / GG4 / GG5 / 良性腺体），
并由分割结果推导 core 级 Gleason 评分与 ISUP 分级。

> **非临床声明**：本项目是研究原型，**不是医疗器械**，未通过任何临床验证，
> 不得用于诊断、筛查、分期、治疗或任何临床决策。详见 [LICENSE](LICENSE)。

---

## 1. 这个仓库实际做了什么

| 环节 | 实现 |
|---|---|
| 金标准 | 由 SICAPv2 官方像素级掩码**推导** core 级 Gleason / ISUP（`scripts/build_gold_standards.py`） |
| 建模 | U-Net 语义分割，5 类（背景 / GG3 / GG4 / GG5 / NC） |
| 训练 | 512×512 patch，按 core 划分 train/val，CE(类别权重) + Dice 损失 |
| 推理 | 整图滑窗分割，输出预测掩码与 core 级评分 |
| 评估 | 三层次：像素级 Dice/IoU、core 级评分一致性、ISUP 分级 ROC |

**为什么是语义分割而不是实例分割**：SICAPv2 的掩码是区域级标注。实测 20 张掩码的
连通域统计显示，每个类别通常只有 1–2 个连通域，中位面积 500 万–1700 万像素（占整图
20–60%），不存在"独立腺体实例"的概念。因此本仓库不使用 YOLOv8-seg 等实例分割模型。
详见 [docs/methodology.md](docs/methodology.md)。

---

## 2. 环境

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

需要 NVIDIA GPU（CUDA）以在合理时间内完成训练；CPU 亦可运行但会很慢。

---

## 3. 完整复现流程

假设已按 [docs/DATA.md](docs/DATA.md) 获取 SICAPv2，并记下两个路径：

- `<IMG_DIR>`：HE 图像目录（`slideXXX_coreXXX.jpg`）
- `<MASK_DIR>`：掩码目录（`slideXXX_coreXXX_classimg_nonconvex.png`）

### 3.1 生成 core 级金标准

```bash
python scripts/build_gold_standards.py \
    --mask-dir <MASK_DIR> \
    --out data/gold_standards.json
```

输出 `data/gold_standards.json`，含每个 core 的 `primary_pattern` /
`secondary_pattern` / `gleason_score` / `isup_grade`。

### 3.2 切分训练 patch

```bash
python scripts/prepare_segmentation_dataset.py \
    --image-dir <IMG_DIR> \
    --mask-dir  <MASK_DIR> \
    --out       data/seg_patches \
    --patch-size 512 --min-foreground 0.3
```

按 **core** 划分 train/val（默认随机 70/30，可用 `--split-from` 沿用既有划分），
避免同一 core 的 patch 同时出现在两侧。

### 3.3 训练

```bash
python src/training/train_unet.py \
    --data-dir data/seg_patches \
    --epochs 60 --batch-size 8 --amp
```

产出 `outputs/unet/<时间戳>/best.pth`（按验证集前景 mIoU 选择）、`history.csv`。

冒烟测试（验证链路，不产出可用模型）：

```bash
python src/training/train_unet.py --data-dir data/seg_patches \
    --epochs 1 --limit-samples 64 --batch-size 4
```

### 3.4 推理

```bash
python src/inference/segment_infer.py \
    --weights outputs/unet/<时间戳>/best.pth \
    --input   <IMG_DIR> \
    --out-dir outputs/predictions \
    --save-masks
```

产出 `predictions.json`（每 core 的各类面积与推导评分）与 `masks/`（预测掩码）。

### 3.5 评估

```bash
python src/evaluation/evaluate.py \
    --predictions outputs/predictions/<时间戳>/predictions.json \
    --gold        data/gold_standards.json \
    --pred-masks  outputs/predictions/<时间戳>/masks \
    --true-masks  <MASK_DIR> \
    --out-dir     results/eval
```

产出 `evaluation_report.json`、`per_core.csv`、`roc_curve.png`。

---

## 4. 仓库结构

```
├── scripts/
│   ├── build_gold_standards.py          # 掩码 -> core 级 Gleason 金标准
│   ├── prepare_segmentation_dataset.py  # 5120x5120 -> 512x512 patch
│   └── masks_to_yolo_seg.py             # 掩码 -> 多类多边形 YOLO 标签（可选，见下文）
├── src/
│   ├── core/gleason_rules.py            # ★ 评分规则唯一实现（金标准与预测共用）
│   ├── models/unet.py                   # U-Net
│   ├── training/{dataset.py,train_unet.py}
│   ├── inference/segment_infer.py       # 滑窗推理 + 评分推导
│   └── evaluation/{metrics.py,evaluate.py}
├── docs/
│   ├── methodology.md                   # 方法学（与代码一致）
│   ├── LIMITATIONS.md                   # ★ 已知限制，投稿前必读
│   └── DATA.md                          # 数据获取与许可
├── legacy/                              # 早期原型（仅供历史参考，勿引用其结果）
├── tests/smoke_test.py
├── LICENSE / CITATION.cff / .zenodo.json / requirements.txt
└── README.md
```

`scripts/masks_to_yolo_seg.py` 保留用于需要实例级标签的场景（导出多类多边形）；
但基于上文连通域分析，**不建议**用它训练 YOLOv8-seg 作为本任务的主方法。

---

## 5. 已知限制（投稿前必读）

见 [docs/LIMITATIONS.md](docs/LIMITATIONS.md)。最重要的三条：

1. **金标准由掩码面积推导**，不完全等同于病理医师出具的病例级报告。
2. **单中心、单次划分**，无外部验证、无交叉验证、无多专家一致性评估。
3. 本仓库**不包含**早期原型中的随机模拟/伪造代码，那部分结果不可引用。

---

## 6. 引用

代码引用见 [CITATION.cff](CITATION.cff)；
数据必须引用 SICAPv2（Silva-Rodríguez 等，Mendeley Data）。

---

## 7. 许可

代码 MIT，附非临床使用免责声明。数据需遵循 SICAPv2 自身许可，见 [docs/DATA.md](docs/DATA.md)。
