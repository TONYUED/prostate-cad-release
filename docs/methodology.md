# 方法学说明

本文件描述的内容与仓库代码**逐项对应**。任何与本文不符的表述都不代表当前实现。

## 1. 数据

- **来源**：SICAPv2（Silva-Rodríguez et al., Mendeley Data）
- **实际使用规模**：244 个活检 core，每个 core 一张 HE 图像 + 一张像素级掩码
- **图像尺寸**：5120×5120（个别为 5632×5120）
- **掩码格式**：单通道 uint8 语义图，取值 `{0, 1, 3, 4, 5, 6}`

| 掩码值 | 含义 | 出现 core 数 /244 | 像素占比 |
|---|---|---|---|
| 0 | 背景 / 非腺体区域 | 244 | 52.6% |
| 3 | Gleason pattern 3 (GG3) | 83 | 10.4% |
| 4 | Gleason pattern 4 (GG4) | 170 | 28.3% |
| 5 | Gleason pattern 5 (GG5) | 33 | 2.7% |
| 6 | 良性 / 非癌腺体 (NC) | 29 | 4.5% |
| 1 | 语义待确认（**默认忽略**） | 10 | 1.5% |

**关于值 1**：仅出现在 10 个 core 且面积很小（1.5%），官方语义无法从数据本身确认，
默认归入背景。若后续获得官方说明，可修改 `VALUE_MAP` 重新生成数据与标签。

## 2. 任务定义：语义分割，而非实例分割

对 20 张掩码做连通域统计（8-连通）：

- 每个类别的连通域数量中位数为 **1**（最大 5）
- 单个连通域面积中位数约 **500 万–1700 万像素**（占整图 20–60%）

即：同一类别在掩码中通常构成一整片连续区域，掩码标注的是**区域**而非**独立腺体实例**。
因此：

- 实例分割（YOLOv8-seg 等）在本数据上无明确实例定义，mask mAP 不可解释；
- 本仓库采用 **逐像素语义分类**，评价指标为 Dice / IoU。

> 早期版本曾把掩码二值化后取外接矩形，得到 570 条"实例"标注（平均每个 core 2.4 个框），
> 并据此训练 YOLOv8-seg。该做法丢失了全部 Gleason 分级信息与轮廓形状，
> 相关代码**未纳入**本仓库。

## 3. 数据准备

（`scripts/prepare_segmentation_dataset.py`）

- 切分：512×512 patch，默认步长 512（不重叠），可用 `--stride` 调整
- 丢弃前景（非背景）像素占比 < 30% 的 patch
- 类别重映射：`0→0, 1→0, 3→1, 4→2, 5→3, 6→4`
- **划分粒度为 core**：同一 core 的所有 patch 落在同一子集，避免泄漏
- 参考量级：244 个 core 约生成 1.4 万个有效 patch

## 4. 模型

（`src/models/unet.py`）

- U-Net，`base_channels=32`，`depth=4`，bottleneck dropout 0.2
- 输入 3 通道 RGB，输出 5 类 logits
- 参数量 7,763,173
- 输入归一化使用 ImageNet 均值/方差

## 5. 训练配置

（`src/training/train_unet.py`）

| 项目 | 取值 |
|---|---|
| 损失 | CrossEntropyLoss(类别权重) + 1.0 × Dice |
| 类别权重 | `median_freq / freq`，空类置 0，裁剪到 [0.05, 20] 后归一化 |
| 优化器 | AdamW，lr 1e-3，weight_decay 1e-4 |
| 调度 | CosineAnnealingLR，T_max = epochs |
| 批次 | 8（可按显存调整） |
| 轮数 | 60（含 patience=15 早停） |
| 精度 | 可选 AMP（`--amp`） |
| 增强 | 水平/垂直翻转、0/90/180/270° 旋转、亮度对比度抖动、BGR 通道扰动（模拟染色差异） |
| 模型选择 | 验证集**前景 mIoU** 最优 |

## 6. 推理

（`src/inference/segment_infer.py`）

- 整图滑窗，窗口 512×512，步长按 `--overlap` 确定（默认不重叠）
- 边界不足一个窗口的位置从末尾回退，保证窗口落在图内
- 输出：与输入同尺寸的类别索引图（可选保存 PNG）+ 各类像素面积

## 7. core 级 Gleason 评分推导

（`src/core/gleason_rules.py`，**金标准与预测共用同一实现**）

1. 仅统计 GG3 / GG4 / GG5 的像素面积；三者均为 0 判为良性
2. primary = 面积最大的模式
3. secondary = 面积第二大的模式；若占癌区面积 < 5%，令 secondary = primary
4. Gleason score = primary + secondary
5. ISUP Grade Group 按 2014 ISUP 共识映射：
   ≤6→1；7 且 primary=3→2；7 且 primary=4→3；8→4；9/10→5

金标准侧（`build_gold_standards.py`）与预测侧（`segment_infer.py`）调用同一函数，
消除"两套规则"带来的系统性偏差。

## 8. 评价

（`src/evaluation/evaluate.py`）

**主指标（不依赖评分推导规则）**

- 逐类 IoU 与 Dice、mIoU、前景 mIoU、前景 mDice、像素准确率
- 由混淆矩阵计算，预测掩码与 SICAPv2 原始掩码逐像素比对

**次要指标（依赖推导规则）**

- Gleason 总分完全一致率、±1 容差率
- ISUP 完全一致率、二次加权 Cohen's Kappa（本地实现，6 个等级）
- 良恶性二分类灵敏度 / 特异度

**ROC**

- 阳性定义：金标准 ISUP ≥ 3（对应 Gleason ≥ 4+3）
- 评分：`gg4plus_ratio = (GG4 + GG5) / (GG3 + GG4 + GG5)`
- AUC 用 Mann-Whitney U 等价形式计算（含并列秩修正）
- 95% CI 用 1000 次 bootstrap（种子固定 42）
- 最佳工作点按 Youden 指数

## 9. 与旧版文档的关系

原项目中的 `methodology_documentation.md` 声称使用 U-Net + ResNet34、1000+ 张全切片、
800/100/100 划分、5 折交叉验证、多专家标注 ICC 0.85–0.92。
这些描述与任何现存代码与数据**均不对应**，该文档未纳入本仓库，也不应被引用。
