# 数据获取与许可

本仓库**不包含任何图像数据**。所有实验数据需单独获取。

## 1. SICAPv2

- 名称：SICAPv2 — Prostate Whole Slide Images with Gleason Grades Annotations
- 作者：Silva-Rodríguez J., Colomer A., Sales M.A., Molina R., Naranjo V.
- 获取：https://data.mendeley.com/datasets/9xxm58dvs3/1
- 内容：前列腺穿刺活检 HE 全切片图像 + 像素级 Gleason 分级掩码 + 病例级评分标注表

引用（论文中必须引用）：

> Silva-Rodríguez, J., Colomer, A., Sales, M. A., Molina, R., & Naranjo, V. (2020).
> Going deeper through the Gleason scoring scale: An automatic end-to-end system for
> histology prostate grading and cribriform pattern detection.

## 2. 期望的目录结构

脚本只需要两个目录，文件名按 SICAPv2 原始命名：

```
<IMG_DIR>/                                  # HE 图像
    slide001_core003.jpg
    slide001_core004.jpg
    ...

<MASK_DIR>/                                 # 掩码
    slide001_core003_classimg_nonconvex.png
    slide001_core004_classimg_nonconvex.png
    ...
```

`case_id` 由文件名去掉 `_classimg_nonconvex` 得到（如 `slide001_core003`），
金标准与预测结果均以此为键对齐。

一个已知的实际布局示例：

```
<IMG_DIR>  = 图片·/Train Imgs/Train Imgs
<MASK_DIR> = test_deep_learning_data/mask_images
```

## 3. 许可注意

- SICAPv2 通常仅授权**非商业研究使用**，且**不允许第三方再分发**。
- 因此本仓库：
  - 不包含任何图像或掩码；
  - 不提供下载脚本（需自行在官方页面注册并同意条款后下载）；
  - 若需随论文分发，建议单独建立**受限访问**存档（如 Zenodo restricted access），
    由申请者在同意数据条款后获取。
- 不要在本仓库的 Git 历史中提交图像文件（`.gitignore` 已排除常见图像扩展名）。

## 4. 掩码取值速查

| 值 | 含义 |
|---|---|
| 0 | 背景 / 非腺体 |
| 1 | 语义待确认（默认忽略） |
| 3 | Gleason pattern 3 |
| 4 | Gleason pattern 4 |
| 5 | Gleason pattern 5 |
| 6 | 良性 / 非癌腺体 |

校验方法（确认你手上的掩码版本与本仓库假设一致）：

```bash
python -c "import cv2,numpy as np,glob,sys;\
p=sorted(glob.glob(sys.argv[1]+'/*.png'))[:30];\
print(sorted(np.unique(cv2.imdecode(np.fromfile(p[0],dtype=np.uint8),-1)).tolist()))" <MASK_DIR>
```

应输出形如 `[0, 3, 4, 5, 6]` 的子集。若出现大量连续灰度值（如 0–255），
说明拿到的是彩色掩码的灰度转换版本而非语义标签图，需要改用颜色阈值处理，
`build_gold_standards.py` 与 `prepare_segmentation_dataset.py` 的 `VALUE_MAP`
都需要相应调整。
