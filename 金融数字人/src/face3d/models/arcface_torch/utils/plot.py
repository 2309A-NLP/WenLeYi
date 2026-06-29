"""
ROC 曲线绘制工具

该模块用于绘制人脸识别模型在 IJB-C 数据集上的 ROC（Receiver Operating Characteristic）曲线。
ROC 曲线是评估二分类模型性能的重要工具，展示了不同阈值下
TPR（真正例率）和 FPR（假正例率）的关系。

主要功能：
1. 加载预计算的验证分数
2. 计算 ROC 曲线和 AUC（曲线下面积）
3. 生成 TPR@FPR 指标表格
4. 绘制可视化 ROC 曲线图
"""

# coding: utf-8

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, auc

# IJB-C 数据集路径
image_path = "/data/anxiang/IJB_release/IJBC"
# 需要比较的模型分数文件列表
files = [
        "./ms1mv3_arcface_r100/ms1mv3_arcface_r100/ijbc.npy"
]


def read_template_pair_list(path):
    """
    读取模板验证对列表
    
    从文本文件中读取验证对信息，每行格式：模板1_ID 模板2_ID 标签
    
    参数:
        path: 验证对文件路径
        
    返回:
        t1: 第一个模板 ID 数组
        t2: 第二个模板 ID 数组
        label: 标签数组（1=同一人, 0=不同人）
    """
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, p2, label


# 读取验证对标签
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % image_path,
                 '%s_template_pair_label.txt' % 'ijbc'))

# 加载各模型的预测分数
methods = []
scores = []
for file in files:
    methods.append(file.split('/')[-2])  # 从路径提取模型名称
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))

# 为每个方法分配不同的颜色
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))

# FPR 目标值：从 1e-6 到 1e-1
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]

# 创建 TPR@FPR 指标表格
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])

# 创建绘图窗口
fig = plt.figure()

for method in methods:
    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)  # 计算 AUC（曲线下面积）
    fpr = np.flipud(fpr)     # 翻转以从高 FPR 到低 FPR
    tpr = np.flipud(tpr)     # select largest tpr at same fpr
    
    # 绘制 ROC 曲线
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    
    # 填充 TPR@FPR 表格
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, "IJBC"))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)

# 设置图表样式
plt.xlim([10 ** -6, 0.1])      # X 轴范围（对数刻度）
plt.ylim([0.3, 1.0])            # Y 轴范围
plt.grid(linestyle='--', linewidth=1)  # 网格线
plt.xticks(x_labels)            # X 轴刻度
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))  # Y 轴刻度
plt.xscale('log')               # X 轴使用对数刻度
plt.xlabel('False Positive Rate')  # X 轴标签：假正例率
plt.ylabel('True Positive Rate')   # Y 轴标签：真正例率
plt.title('ROC on IJB')            # 图表标题
plt.legend(loc="lower right")      # 图例位置

# 打印 TPR@FPR 指标表格
print(tpr_fpr_table)
