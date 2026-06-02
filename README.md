# 血细胞 13 分类

基于 **MobileNetV3-Small + 预训练权重 + 类别平衡损失** 的 13 类血细胞自动分类。  
数据集共 22,106 张单细胞图片，存在严重的类别不均衡（最少类仅 44 张）。  
通过离线预处理与类别权重优化，最终测试准确率达到 **82.34%**。

## 文件说明

| 文件 | 作用 |
|------|------|
| `preprocess.py` | 将原始图片处理为 128×128 float16 张量 |
| `merge_data.py` | 合并零散 `.pt` 文件为 `all_images.pt`，加速加载 |
| `train_multiclass.py` | 主训练脚本（含混淆矩阵、分类报告） |
| `debug_val.py` | 诊断验证集分布与模型预测偏好 |
| `requirements.txt` | 项目依赖 |
| `.gitignore` | 排除大文件及临时文件 |

## 环境配置

```bash
pip install -r requirements.txt
核心依赖：torch, torchvision, numpy, tqdm, matplotlib, scikit-learn, seaborn
数据准备（本地运行）
将原始图片按类别放在 D:\12blood\cropped_cells 下

运行 preprocess.py → 生成 Dataset/ 中的 22105 个 .pt 文件

运行 merge_data.py → 生成 all_images.pt 和 all_labels.pt
训练与评估
bash
python train_multiclass.py
使用 ImageNet 预训练的 MobileNetV3-Small

自动计算类别权重（越少见的类权重越高）

训练 4 个 epoch，输出训练/验证曲线和混淆矩阵

最佳模型保存为 best_model.pth

实验结果（4 Epochs）
Epoch	Train Loss	Train Acc	Val Acc
1	1.0745	63.92%	47.26%
2	0.5626	78.02%	72.00%
3	0.4045	82.09%	82.59%
4	0.3000	85.52%	80.37%
最终测试集准确率：82.34%

分类报告（部分关键类别）
类别	样本数	精确率	召回率	F1-score
0	1879	0.82	0.81	0.82
1	210	0.64	0.87	0.74
2 (极少类)	44	0.51	0.68	0.58
4	318	1.00	1.00	1.00
10 (极少类)	101	0.59	0.78	0.67
加权平均	6632	0.84	0.82	0.83
项目亮点
混合精度训练：加速训练并降低显存

类别平衡损失：有效缓解 1:40 的不均衡

离线预处理：图片提前转成张量，训练时零 IO 等待

轻量模型：MobileNetV3-Small 仅 153 万参数，推理快

作者
姓名：2311050125 贾悦冬

用途：课程设计

日期：2026年6月2日
