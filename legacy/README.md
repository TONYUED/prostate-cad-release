# legacy：早期原型（仅供历史参考）

## 这些代码是什么

`legacy/` 收录的是本项目早期的 TCGA / CNN 原型代码，与 `src/` 下的当前实现
**没有继承关系**，也不共享任何数据接口。

## 必须知道的三件事

1. **训练数据是合成的**。早期 CNN（`train_prostate_cnn.py`）使用
   `create_synthetic_data` 生成的**合成圆点图像**训练，没有使用任何真实病理图像。
   其准确率、损失曲线等指标**不能**作为任何方法学结论的依据。

2. **未做真实病理验证**。该分支从未在 SICAPv2 或任何其他真实数据集上完成训练
   与评估闭环。

3. **仅作历史记录保留**。如果你要复现论文结果，请使用 `src/` 与 `scripts/` 下的
   当前实现，不要运行本目录的代码。

## 与当前实现的关系

| 项目 | 早期原型 (legacy/) | 当前实现 (src/) |
|---|---|---|
| 任务 | 图像级分类 / 合成数据检测 | 像素级 Gleason 模式语义分割 |
| 模型 | 简单 CNN | U-Net |
| 训练数据 | 合成圆点图像 | SICAPv2 真实掩码 |
| 金标准 | 无 | 由官方掩码推导（有明确规则与局限） |
| 可否引用结果 | **否** | 是（须遵守 docs/LIMITATIONS.md） |

## 文件说明

| 文件 | 说明 |
|---|---|
| `prostate_cancer_cnn.py` | CNN 模型定义 |
| `train_prostate_cnn.py` | 训练脚本（合成数据） |
| `tcga_data_processor.py` | TCGA 数据处理 |
| `immune_cell_detector.py` | 免疫细胞检测原型 |
| `tumor_infiltration_analyzer.py` | 肿瘤浸润分析原型 |
| `visualization_module.py` | 可视化 |
| `dashboard_generator.py` | 报表生成 |

许可沿用原 `101/LICENSE`（MIT）。
