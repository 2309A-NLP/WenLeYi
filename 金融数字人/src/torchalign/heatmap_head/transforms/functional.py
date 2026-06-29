# 热图到坐标转换的函数式工具模块
# 提供 heatmap2coord 函数，将热图解码为关键点坐标
# 使用 soft-argmax 策略：选取top-k个最大值点，用softmax加权平均得到亚像素精度的坐标

import torch
import torch.nn as nn
import torch.nn.functional as F 


def heatmap2coord(heatmap, topk=9):
    """将热图解码为关键点坐标
    
    使用 soft-argmax 方法进行热图解码：
    1. 找到热图中top-k个最大值位置
    2. 用softmax对这些位置的值进行归一化作为权重
    3. 用权重对位置坐标进行加权平均，得到亚像素精度的坐标
    
    这种方法比简单的 argmax（取最大值位置）精度更高，
    因为它考虑了多个高响应点的分布，可以得到连续的坐标值。
    
    参数:
        heatmap: 输入热图 [N, C, H, W]
                 N: 批量大小
                 C: 关键点数量
                 H, W: 热图高度和宽度
        topk: 选取的最大值点数量，默认为9
    返回:
        coord: 关键点坐标 [N, C, 2]，最后一维为 (x, y)
    """
    N, C, H, W = heatmap.shape
    # 将热图展平为 [N, C, 1, H*W]
    # topk选取每个关键点通道中最大的topk个值及其索引
    # score: [N, C, 1, topk] 最大值分数
    # index: [N, C, 1, topk] 最大值的一维索引
    score, index = heatmap.view(N,C,1,-1).topk(topk, dim=-1)
    # 将一维索引转换为二维坐标 (x, y)
    # x = index % W（列号），y = index // W（行号）
    coord = torch.cat([index%W, index//W], dim=2)
    # 用softmax归一化分数作为权重，对坐标进行加权平均
    # 这样得到的是亚像素精度的连续坐标
    return (coord*F.softmax(score, dim=-1)).sum(-1)
